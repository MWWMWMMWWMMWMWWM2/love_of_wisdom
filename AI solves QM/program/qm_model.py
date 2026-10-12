"""
qm_model.py -- a potential-conditioned neural wave-function model with pre-training and fine-tuning.

Architecture ("operator" network):  psi_k(x | V) = NN( x, onehot(k), d(V) ) * envelope(x),
  d(V) = the potential sampled at K points of the box, scaled to [-1, 1]  (a fixed-length descriptor),
  k    = state index (0 = ground state, ...).
Stage 1 - pre-training (train_pretrained.py): on random potentials, supervised on the EXACT grid eigenfunctions
          (sparse diagonalisation) -> the network learns the map  potential -> low eigenstates.
Stage 2 - fine-tuning (this file, fit()): for a new potential the pre-trained prediction is the starting
          point and the energy functional <psi|H|psi>/<psi|psi> is minimised (variational principle, exact
          gradients by back-propagation), state by state with orthogonal projection.
Baselines reported next to the model: exact grid diagonalisation (the truth on the grid) and WKB /
Bohr-Sommerfeld quantisation  int sqrt(2(E-V)) dx = (n + 1/2) pi,  which the model should beat.
Continuum: energies above the smaller asymptotic value of V are scattering states; transmission T(E) is
computed by direct numerical integration (transfer-matrix / shooting), see scattering().

Units: hbar = m = 1.  Files: pretrained_model.npz (weights + normalisation constants).
"""
from __future__ import annotations
import os
import time
from collections import deque
import numpy as np
from scipy import sparse
from scipy.integrate import simpson
from scipy.sparse.linalg import eigsh  # noqa: F401 (kept for external callers)
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, "pretrained_model.npz")

K_DESC = 32          # descriptor length
N_STATES = 4         # states the model knows (k = 0..3)
X_HALF = 8.0         # box is [-X_HALF, X_HALF]
N_GRID = 320
V_SCALE = 6.0        # potentials are scaled so |V| <~ V_SCALE before entering the network


# ----------------------------------------------------------------------------------------------
# network (numpy, manual backprop)
# ----------------------------------------------------------------------------------------------
class MLP:
    def __init__(self, sizes, seed=0, W=None, b=None):
        if W is None:
            rng = np.random.default_rng(seed)
            W = [rng.standard_normal((a, c)) * np.sqrt(2.0 / (a + c)) for a, c in zip(sizes[:-1], sizes[1:])]
            b = [np.zeros(c) for c in sizes[1:]]
        self.W, self.b = [np.array(w, float) for w in W], [np.array(x, float) for x in b]
        self.params = self.W + self.b

    def copy(self):
        return MLP(None, W=[w.copy() for w in self.W], b=[x.copy() for x in self.b])

    def forward(self, X):
        self.cache = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.tanh(z) if i < len(self.W) - 1 else z
            self.cache.append(h)
        return h[:, 0]

    def backward(self, dout):
        g = dout[:, None]
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            gW[i] = self.cache[i].T @ g
            gb[i] = g.sum(axis=0)
            if i > 0:
                g = (g @ self.W[i].T) * (1.0 - self.cache[i] ** 2)
        return gW + gb

    def save(self, path, **extra):
        np.savez(path, n_layers=len(self.W), **{f"W{i}": w for i, w in enumerate(self.W)},
                 **{f"b{i}": x for i, x in enumerate(self.b)}, **extra)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        n = int(d["n_layers"])
        return cls(None, W=[d[f"W{i}"] for i in range(n)], b=[d[f"b{i}"] for i in range(n)]), d


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            p -= self.lr * (m / (1 - self.b1 ** self.t)) / (np.sqrt(v / (1 - self.b2 ** self.t)) + self.eps)


# ----------------------------------------------------------------------------------------------
# grid, Hamiltonian, exact and WKB references
# ----------------------------------------------------------------------------------------------
def grid():
    return np.linspace(-X_HALF, X_HALF, N_GRID)


def hamiltonian(x, V):
    n, dx = len(x), x[1] - x[0]
    lap = sparse.diags([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], [-1, 0, 1]) / dx ** 2
    return (-0.5 * lap + sparse.diags(V)).tocsr()


def exact_states(x, V, n=N_STATES):
    """Lowest n eigenpairs of the tridiagonal grid Hamiltonian by direct (LAPACK) tridiagonal diagonalisation:
    deterministic, no iterative convergence that could fail (ARPACK eigsh occasionally does)."""
    dx = x[1] - x[0]
    diag = 1.0 / dx ** 2 + np.asarray(V, float)
    off = np.full(len(x) - 1, -0.5 / dx ** 2)
    vals, vecs = eigh_tridiagonal(diag, off, select="i", select_range=(0, n - 1))
    o = np.argsort(vals)
    vecs = vecs[:, o] / np.sqrt(dx)
    for j in range(vecs.shape[1]):             # sign convention: first significant lobe positive
        i = np.argmax(np.abs(vecs[:, j]) > 0.2 * np.abs(vecs[:, j]).max())
        if vecs[i, j] < 0:
            vecs[:, j] *= -1
    return vals[o], vecs


def continuum_threshold(V):
    """Bound states have E < min(V at the two box edges)."""
    return float(min(V[0], V[-1]))


def wkb_energies(x, V, n_max=N_STATES):
    """Bohr-Sommerfeld: int_{x1}^{x2} sqrt(2 (E - V)) dx = (n + 1/2) pi between classical turning points."""
    Vmin, Vth = V.min(), continuum_threshold(V)
    dx = x[1] - x[0]

    def action(E):
        return np.sum(np.sqrt(np.clip(2 * (E - V), 0, None))) * dx

    out = []
    for n in range(n_max):
        target = (n + 0.5) * np.pi
        f = lambda E: action(E) - target
        if f(Vth - 1e-9) < 0:
            break
        out.append(brentq(f, Vmin + 1e-9, Vth - 1e-9))
    return np.array(out)


# ----------------------------------------------------------------------------------------------
# descriptor and model input
# ----------------------------------------------------------------------------------------------
def descriptor(x, V):
    xs = np.linspace(-X_HALF, X_HALF, K_DESC)
    d = np.interp(xs, x, V) / V_SCALE
    return np.clip(d, -3, 3)


def model_input(x, V, k):
    d = descriptor(x, V)
    n = len(x)
    X = np.zeros((n, 1 + N_STATES + K_DESC))
    X[:, 0] = x / X_HALF
    X[:, 1 + k] = 1.0
    X[:, 1 + N_STATES:] = d[None, :]
    return X


def envelope(x):
    return np.exp(-(x / (0.75 * X_HALF)) ** 4)       # kills psi at the box edges only


def new_model(seed=0):
    return MLP([1 + N_STATES + K_DESC, 96, 96, 64, 1], seed)


def load_model():
    if os.path.exists(MODEL_FILE):
        net, meta = MLP.load(MODEL_FILE)
        return net, dict(meta)
    return None, {}


# ----------------------------------------------------------------------------------------------
# fine-tuning on one potential (variational)
# ----------------------------------------------------------------------------------------------
def fit(x, V, levels, net=None, steps=3000, lr=1e-3, verbose=False, callback=None, explore=3):
    """Return dict with energies/wave functions for the requested bound levels, plus baselines.
    net: pre-trained MLP (copied, then fine-tuned per state).  If None, trains from scratch.
    Drives LiveFit (same optimiser, stall detection and noise kicks) to completion."""
    lf = LiveFit(x, V, levels, net=net, lr=lr, max_steps=steps, explore=explore)
    while not lf.done:
        lf.step(50)
        if callback and lf.current is not None:
            callback(lf.current[0], lf.it, lf.current[1])
    res = lf.result()
    if verbose:
        for k, lv in sorted(res["levels"].items()):
            if lv["bound"]:
                print(f"  level {k}: start {lv['E_start']:.5f} -> model {lv['E_model']:.6f} | exact {lv['E_exact']:.6f} | "
                      f"WKB {lv['E_wkb'] if lv['E_wkb'] is not None else float('nan'):.5f} | "
                      f"{lv['steps']} steps, {lv['kicks']} kicks, {lv['escapes']} escapes")
    return res


# ----------------------------------------------------------------------------------------------
# scattering states (continuum): transmission by numerical integration
# ----------------------------------------------------------------------------------------------
def scattering(x, V, E):
    """Integrate psi'' = 2 (V - E) psi from the right with an outgoing wave e^{i k_R x}, read off the
    incident and reflected amplitudes on the left.  Returns T, R and psi(x) (complex, incident amp = 1).
    Requires E > V at both edges."""
    VL, VR = V[0], V[-1]
    if E <= max(VL, VR):
        return None
    kL, kR = np.sqrt(2 * (E - VL)), np.sqrt(2 * (E - VR))
    dx = x[1] - x[0]
    psi = np.zeros(len(x), complex)
    psi[-1] = np.exp(1j * kR * x[-1])
    psi[-2] = np.exp(1j * kR * x[-2])
    for i in range(len(x) - 2, 0, -1):        # Numerov-free simple 3-point recursion (fine at this dx)
        psi[i - 1] = 2 * psi[i] - psi[i + 1] + dx ** 2 * 2 * (V[i] - E) * psi[i]
    # on the left psi = A e^{ikx} + B e^{-ikx}
    x0, x1 = x[0], x[1]
    M = np.array([[np.exp(1j * kL * x0), np.exp(-1j * kL * x0)], [np.exp(1j * kL * x1), np.exp(-1j * kL * x1)]])
    A, B = np.linalg.solve(M, psi[:2])
    T = (kR / kL) * abs(1.0 / A) ** 2
    R = abs(B / A) ** 2
    return {"T": float(T), "R": float(R), "psi": psi / A, "E": E}


# ----------------------------------------------------------------------------------------------
# time evolution: spectral (from the model's eigenstates) and full split-operator (ground truth)
# ----------------------------------------------------------------------------------------------
def gaussian_packet(x, x0=0.0, sigma=0.7, k0=0.0):
    psi = np.exp(-(x - x0) ** 2 / (4 * sigma ** 2) + 1j * k0 * x)
    return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * (x[1] - x[0]))


class SpectralEvolver:
    """psi(x,t) = sum_n c_n psi_n(x) exp(-i E_n t) using the (AI) eigenstates found for the potential.
    Exact for any initial state inside the span of the known bound states; the 'missing' probability
    1 - sum |c_n|^2 lies in states not computed (higher levels / continuum)."""
    def __init__(self, x, energies, states, psi0):
        self.x, self.dx = x, x[1] - x[0]
        self.E = np.asarray(energies)
        self.U = np.stack(states, axis=1)                     # (n_x, n_states), orthonormal
        self.c = (self.U.T @ psi0) * self.dx                  # expansion coefficients
        self.prob = np.abs(self.c) ** 2
        self.captured = float(self.prob.sum())

    def psi(self, t):
        return self.U @ (self.c * np.exp(-1j * self.E * t))

    def energy_mean(self):
        return float(np.sum(self.prob * self.E) / max(self.captured, 1e-12))


class SplitOperator:
    """Full time-dependent Schroedinger equation on the grid (no eigenstates needed): ground truth."""
    def __init__(self, x, V, psi0, dt=0.004):
        from scipy import fft as sfft
        self.sfft = sfft
        self.x, self.dt = x, dt
        n, dx = len(x), x[1] - x[0]
        k = 2 * np.pi * sfft.fftfreq(n, d=dx)
        self.expV = np.exp(-0.5j * V * dt)
        self.expT = np.exp(-0.5j * k ** 2 * dt)
        self.psi = psi0.astype(complex)
        self.t = 0.0

    def advance(self, t_target):
        while self.t < t_target - 1e-12:
            p = self.expV * self.psi
            p = self.sfft.ifft(self.expT * self.sfft.fft(p))
            self.psi = self.expV * p
            self.t += self.dt
        return self.psi


def fidelity(x, a, b):
    dx = x[1] - x[0]
    return float(abs(np.vdot(a, b) * dx) ** 2 / ((np.vdot(a, a).real * dx) * (np.vdot(b, b).real * dx)))


# ----------------------------------------------------------------------------------------------
# semiclassical time-evolution baselines (the "WKB of dynamics")
# ----------------------------------------------------------------------------------------------
class ThawedGaussian:
    """Heller's thawed-Gaussian approximation (J. Chem. Phys. 62, 1544 (1975)): the time-dependent WKB
    for a Gaussian wave packet.  psi = exp(i[a (x-q)^2 + p (x-q) + gamma]) with the centre on the classical
    trajectory (q_dot = p, p_dot = -V'(q)) and the width following the local curvature V''(q):
        a_dot = -2 a^2 - V''(q)/2,   gamma_dot = i a + p^2/2 - V(q)       (hbar = m = 1).
    Exact for quadratic potentials; fails when the packet spreads over non-quadratic structure
    (tunnelling, splitting at a barrier, reflection from steps)."""
    def __init__(self, x, V, x0, sigma, k0):
        self.x, self.V = x, V
        dx = x[1] - x[0]
        self.V1 = np.gradient(V, dx)
        self.V2 = np.gradient(self.V1, dx)
        self.y = np.array([x0, k0, 1j / (4 * sigma ** 2), 0.0], complex)     # q, p, a, gamma
        self.t = 0.0

    def _f(self, y):
        q, p, a, g = y
        qr = float(q.real)
        V, V1, V2 = (np.interp(qr, self.x, arr) for arr in (self.V, self.V1, self.V2))
        return np.array([p, -V1, -2 * a * a - 0.5 * V2, 1j * a + 0.5 * p * p - V], complex)

    def advance(self, t_target, dt=0.002):
        while self.t < t_target - 1e-12:
            h = min(dt, t_target - self.t)
            y = self.y
            k1 = self._f(y); k2 = self._f(y + 0.5 * h * k1); k3 = self._f(y + 0.5 * h * k2); k4 = self._f(y + h * k3)
            self.y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self.t += h
        q, p, a, g = self.y
        psi = np.exp(1j * (a * (self.x - q) ** 2 + p * (self.x - q) + g))
        return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * (self.x[1] - self.x[0]))


class ClassicalEnsemble:
    """Classical (Liouville) propagation of the Wigner distribution of the initial Gaussian: N trajectories
    q ~ N(x0, sigma), p ~ N(k0, 1/(2 sigma)) follow Hamilton's equations; the position histogram is the
    classical prediction for |psi(x,t)|^2.  No interference, no tunnelling."""
    def __init__(self, x, V, x0, sigma, k0, n=20000, seed=0):
        rng = np.random.default_rng(seed)
        self.x, self.dx = x, x[1] - x[0]
        self.V1 = np.gradient(V, self.dx)
        self.q = rng.normal(x0, sigma, n)
        self.p = rng.normal(k0, 1.0 / (2 * sigma), n)
        self.t = 0.0

    def advance(self, t_target, dt=0.004):
        while self.t < t_target - 1e-12:
            h = min(dt, t_target - self.t)
            self.p -= 0.5 * h * np.interp(self.q, self.x, self.V1)
            self.q += h * self.p
            self.p -= 0.5 * h * np.interp(self.q, self.x, self.V1)
            self.t += h
        edges = np.concatenate([self.x - self.dx / 2, [self.x[-1] + self.dx / 2]])
        rho, _ = np.histogram(self.q, bins=edges)
        rho = rho / (len(self.q) * self.dx)
        k = np.exp(-0.5 * (np.arange(-6, 7) * self.dx / 0.15) ** 2); k /= k.sum()
        return np.convolve(rho, k, mode="same")


def density_overlap(x, rho_a, rho_b):
    """Bhattacharyya overlap of two probability densities (1 = identical), the density analogue of fidelity."""
    dx = x[1] - x[0]
    return float(np.sum(np.sqrt(np.clip(rho_a, 0, None) * np.clip(rho_b, 0, None))) * dx)


# ----------------------------------------------------------------------------------------------
# incremental (real-time) fine-tuning: the UI calls step() a few times per frame and draws the
# wave function while it converges, instead of blocking in fit().
# ----------------------------------------------------------------------------------------------
class LiveFit:
    """Resumable variational fine-tuning.  Solves the requested bound levels one after the other;
    call step(n) repeatedly; .current gives (k, E, psi) of the level being optimised; .done when finished.

    Getting stuck.  The energy functional is not convex in the network weights, so plain gradient descent
    can (a) creep for ever on a flat plateau (Adam's jitter is ~1e-6, so a fixed 'energy change < 1e-7'
    criterion never fires) or (b) settle in a LOCAL minimum -- typically the wave function sitting in the
    wrong well of a multi-well potential, where the gradient that would move amplitude across the barrier
    is exponentially small.  Both are common in neural wave-function optimisation.
    Remedy used here (basin hopping / annealed noise):
      * stall detector: the best energy has not improved by more than tol*(1+|E|) over `window` steps;
      * on a stall, if kicks are left, restore the best weights and perturb, alternating two kinds of move:
          - 'bump' (noise in FUNCTION space, physics-guided): add a Gaussian lump of amplitude to psi in a
            classically allowed region (V < E) that psi does not yet occupy -- i.e. propose "try the other
            well" -- fit the network to that proposal for a few supervised steps, then continue variationally;
          - 'noise' (noise in WEIGHT space): Gaussian noise on every weight, relative scale growing per kick;
        Adam is reset after a kick.  If the energy later beats the old best we have 'escaped' (the move is
        accepted); otherwise the next stall restores the best weights before the next move (rejected);
      * after `explore` kicks without improvement (or max_steps) the level is finished with the best state.
    The energy variance <H^2> - <H>^2 is also tracked: it is zero only for a true eigenstate, so it is a
    reference-free quality measure (small variance + stall = converged, not stuck)."""
    def __init__(self, x, V, levels, net=None, lr=1e-3, max_steps=10 ** 7, tol=2e-6, window=10 ** 7, explore=0,
                 kick=0.3, time_limit=90.0, stable_time=15.0, psi_tol=1e-6, check_every=10):
        # Stopping rule = STABILITY OF THE WAVE FUNCTION (not of the energy):
        #   every `check_every` steps  d = int (psi - psi_ema)^2 dx  (Simpson), psi_ema = exponential moving
        #   average of psi (reference that does not jump when Adam throws a transient spike);
        #   the level counts as stable when the MEDIAN of d over the last 5 checks is below psi_tol
        #   (median = one spike cannot reset the clock).  Once stable, a `stable_time` (15 s) timer starts while
        #   the optimisation continues, and the learning rate is reduced to damp swings; if psi stays stable for
        #   the whole window the level is accepted (best-energy state kept); if it becomes unstable the timer is
        #   cancelled and re-armed only when stability returns.
        # time_limit: hard wall-clock backstop per level (best state kept, flagged timed_out).
        # window/explore are the old step-based stall + kick machinery (off by default).
        self.x, self.V, self.dx = x, V, x[1] - x[0]
        self.H = hamiltonian(x, V)
        self.env = envelope(x)
        self.Eth = continuum_threshold(V)
        n_req = max(N_STATES, max(levels) + 1 if levels else 1)
        self.E_exact, self.psi_exact = exact_states(x, V, n_req)
        self.n_bound = int(np.sum(self.E_exact < self.Eth))
        try:
            self.E_wkb = wkb_energies(x, V, n_req)
        except Exception:  # noqa: BLE001 -- WKB is only a baseline; never let it block the solver
            self.E_wkb = np.array([])
        self.net, self.lr, self.max_steps, self.tol = net, lr, max_steps, tol
        self.window, self.explore, self.kick0, self.time_limit = window, explore, kick, time_limit
        self.stable_time, self.psi_tol, self.check_every = stable_time, psi_tol, check_every
        self.rng = np.random.default_rng(0)
        self.queue = [k for k in sorted(levels) if k < self.n_bound]
        self.unbound = [k for k in sorted(levels) if k >= self.n_bound]
        self.results = {k: {"bound": False, "E_exact": self.E_exact[k] if k < len(self.E_exact) else None}
                        for k in self.unbound}
        self.found = []
        self.current = None
        self.phase = ""
        self.variance = np.nan
        self.done = not self.queue
        self._start_level()

    def _start_level(self):
        if not self.queue:
            self.done = True; self.current = None; self._relabel(); return
        k = self.queue.pop(0)
        self.k = k
        self.model = self.net.copy() if self.net is not None else new_model(seed=k)
        self.X = model_input(self.x, self.V, min(k, N_STATES - 1))
        self.opt = Adam(self.model.params, self.lr if self.net is not None else 3e-3)
        self.proj = [self.psi_exact[:, j] for j in range(k) if j not in [f[0] for f in self.found]] + [f[1] for f in self.found]
        self.it, self.best, self.E0, self.hist = 0, (np.inf, None), None, []
        self.best_params, self.best_it = None, 0
        self.kicks, self.escapes, self.E_before_kick = 0, 0, None
        self.t_best, self.timed_out = time.time(), False
        self.t_level = time.time(); self.resets = 0
        self.psi_ema, self.dists, self.stable_since, self.stable_for, self.psi_change = None, deque(maxlen=15), None, 0.0, np.nan
        self.phase = "descending"

    def _project(self, p):
        for q in self.proj:
            p = p - q * (q @ p) * self.dx
        return p

    def _kick(self):
        """Basin hopping move: restore the best weights, perturb (bump or noise), reset the optimiser."""
        if self.best_params is not None:
            for p, bp in zip(self.model.params, self.best_params):
                p[...] = bp
        self.kicks += 1
        self.E_before_kick = self.best[0]
        self.best_it = self.it
        if self.kicks % 2 == 1 and self._bump(greedy=(self.kicks == 1)):
            return
        scale = self.kick0 * (1.0 + 0.5 * (self.kicks // 2))          # 0.3, 0.45, ... of the weight rms
        for p in self.model.params:
            rms = np.sqrt(np.mean(p ** 2)) if p.size else 0.0
            p += self.rng.standard_normal(p.shape) * scale * (rms if rms > 0 else 0.05)
        self.opt = Adam(self.model.params, self.lr if self.net is not None else 3e-3)
        self.phase = f"exploring: weight-noise kick {self.kicks}/{self.explore} (scale {scale:.2f})"

    def _bump(self, greedy=True, width=0.6, sup_steps=150):
        """Function-space move: psi -> psi + lump in an unoccupied classically allowed region, then teach
        the network that proposal (supervised) so the variational search restarts from the other basin."""
        E, psi = self.best
        if psi is None:
            return False
        occ = psi ** 2 / max(float(np.max(psi ** 2)), 1e-12)
        w = np.clip(E - self.V, 0, None) * (1.0 - occ) * self.env
        if w.sum() <= 1e-9:
            return False
        w = w / w.sum()
        xc = self.x[int(np.argmax(w))] if greedy else float(self.rng.choice(self.x, p=w))
        bump = np.exp(-(self.x - xc) ** 2 / (2 * width ** 2))
        bump /= np.sqrt(np.sum(bump ** 2) * self.dx)
        cand = self._project(psi + bump * float(self.rng.choice([-1.0, 1.0])))
        cand /= np.sqrt(np.sum(cand ** 2) * self.dx)
        target = cand / np.maximum(self.env, 1e-6)              # network output is multiplied by the envelope
        target = np.clip(target, -5 * np.abs(cand).max(), 5 * np.abs(cand).max())
        opt = Adam(self.model.params, 3e-3)
        for _ in range(sup_steps):
            out = self.model.forward(self.X)
            opt.step(self.model.backward(2.0 * (out - target) * self.env ** 2 / len(self.x)))
        self.opt = Adam(self.model.params, self.lr if self.net is not None else 3e-3)
        self.phase = f"exploring: bump kick {self.kicks}/{self.explore} -- trying amplitude near x = {xc:.1f}"
        return True

    def step(self, n=25):
        for _ in range(n):
            if self.done:
                return
            raw = self.model.forward(self.X) * self.env
            psi = self._project(raw)
            norm = psi @ psi * self.dx
            Hpsi = self.H @ psi
            E = (psi @ Hpsi) * self.dx / norm
            if not np.isfinite(E) or norm < 1e-14:          # weights blew up (e.g. after a noise kick)
                if self.best_params is not None:
                    for p, bp in zip(self.model.params, self.best_params):
                        p[...] = bp
                self.opt = Adam(self.model.params, self.lr if self.net is not None else 3e-3)
                self.it += 1
                if self.kicks >= self.explore:
                    self._finish_level()
                else:
                    self._kick()
                continue
            self.variance = float((Hpsi @ Hpsi) * self.dx / norm - E * E)     # <H^2> - E^2  (0 for an eigenstate)
            if self.E0 is None:
                self.E0 = E
            self.hist.append(E)
            if E < self.best[0] - self.tol * (1.0 + abs(E)):
                self.best_it = self.it
                self.t_best = time.time()
                if self.E_before_kick is not None and E < self.E_before_kick - 1e-4 * (1.0 + abs(E)) and self.phase.startswith("exploring"):
                    self.escapes += 1
                    self.phase = f"escaped a local minimum (kick {self.kicks}): E {self.E_before_kick:.4f} -> {E:.4f}"
            if E < self.best[0]:
                self.best = (E, psi / np.sqrt(norm))
                self.best_params = [p.copy() for p in self.model.params]
            self.opt.step(self.model.backward(self._project(2.0 * (Hpsi - E * psi) * self.dx / norm) * self.env))
            self.it += 1
            psin = psi / np.sqrt(norm)
            self.current = (self.k, E, psin)
            if self.it % self.check_every == 0 and self._stability_check(psin):
                self.phase = f"accepted: stable for {self.stable_time:.0f} s"
                self._finish_level()
                continue
            stalled = self.it > 200 and self.it - self.best_it >= self.window
            if time.time() - self.t_level > self.time_limit:
                self.timed_out = True
                self.phase = f"stopped: hard limit {self.time_limit:.0f} s"
                self._finish_level()
            elif self.it >= self.max_steps or (stalled and self.kicks >= self.explore):
                self._finish_level()
            elif stalled:
                self._kick()

    def _stability_check(self, psin):
        """Swing-tolerant stability test of the wave function; returns True when the level is accepted."""
        now = time.time()
        if self.psi_ema is None:
            self.psi_ema = psin.copy(); return False
        if psin @ self.psi_ema < 0:
            psin = -psin
        d = float(simpson((psin - self.psi_ema) ** 2, x=self.x))
        self.psi_ema = 0.8 * self.psi_ema + 0.2 * psin
        self.dists.append(d)
        self.psi_change = float(np.median(self.dists))
        stable = len(self.dists) >= 5 and self.psi_change < self.psi_tol
        if stable:
            if self.stable_since is None:
                self.stable_since = now
                self.opt.lr = max(self.opt.lr * 0.3, self.lr * 1e-3)   # lr decay on plateau (monotone): damps Adam swings
            self.stable_for = now - self.stable_since
            self.phase = f"stable for {self.stable_for:.1f}/{self.stable_time:.0f} s  (psi change {self.psi_change:.1e})"
            return self.stable_for >= self.stable_time
        if self.stable_since is not None:
            self.resets += 1
            self.phase = f"unstable again (psi change {self.psi_change:.1e}) -- timer reset"
        else:
            self.phase = f"optimising (psi change {self.psi_change:.1e}, needs < {self.psi_tol:.0e})"
        self.stable_since, self.stable_for = None, 0.0
        return False

    def stop_level(self, reason="time limit"):
        """Stop the level in progress now, keep its best state (whatever its accuracy), go on to the next."""
        if self.done or self.best[1] is None:
            return
        self.timed_out = True
        self.phase = f"stopped: {reason}"
        self._finish_level()

    def abort(self, reason="time budget"):
        """Absolute stop: keep the best state of the level in progress, mark the remaining levels unsolved."""
        if self.done:
            return
        self.timed_out = True
        self.phase = f"stopped: {reason}"
        rest = list(self.queue); self.queue = []
        if self.best[1] is not None:
            self._finish_level()
        else:
            self.done = True; self.current = None
        for k in rest:
            self.results[k] = {"bound": False, "skipped": True, "E_exact": self.E_exact[k]}
        self.done = True; self.current = None

    def _finish_level(self):
        k = self.k
        E_fit, psi_fit = self.best
        if psi_fit @ self.psi_exact[:, k] < 0:
            psi_fit = -psi_fit
        self.found.append((k, psi_fit))
        self.results[k] = {"bound": True, "E_model": E_fit, "E_start": self.E0, "psi": psi_fit, "steps": self.it,
                           "kicks": self.kicks, "escapes": self.escapes, "timed_out": self.timed_out,
                           "stable_for": self.stable_for, "psi_change": self.psi_change, "resets": self.resets,
                           "seconds": time.time() - self.t_level,
                           "E_exact": self.E_exact[k], "E_wkb": self.E_wkb[k] if k < len(self.E_wkb) else None,
                           "history": np.array(self.hist)}
        self._start_level()

    def _relabel(self):
        """The found states are mutually orthogonal, so if a later level escaped to a LOWER energy than an
        earlier one (earlier level was stuck in the wrong well) the requested indices are reassigned in
        energy order; the exact/WKB references are updated accordingly."""
        keys = sorted(k for k, lv in self.results.items() if lv.get("bound"))
        if len(keys) < 2:
            return
        order = sorted(keys, key=lambda k: self.results[k]["E_model"])
        if order == keys:
            return
        old = {k: self.results[k] for k in keys}
        for k_new, k_old in zip(keys, order):
            lv = dict(old[k_old])
            lv["E_exact"] = self.E_exact[k_new]
            lv["E_wkb"] = self.E_wkb[k_new] if k_new < len(self.E_wkb) else None
            if lv["psi"] @ self.psi_exact[:, k_new] < 0:
                lv["psi"] = -lv["psi"]
            if k_old != k_new:
                lv["relabelled_from"] = k_old
            self.results[k_new] = lv
        self.found = [(k, self.results[k]["psi"]) for k in keys]

    def result(self):
        """Same dict layout as fit() (usable once done)."""
        return {"x": self.x, "V": self.V, "threshold": self.Eth, "n_bound_exact": self.n_bound,
                "E_exact": self.E_exact, "psi_exact": self.psi_exact, "E_wkb": self.E_wkb,
                "levels": dict(self.results), "pretrained": self.net is not None}
