"""
schrodinger.py -- time-dependent Schroedinger equation in 1, 2 or 3 dimensions.

Method: split-operator (Fourier) propagation,  psi(t+dt) = e^{-iV dt/2} F^{-1} e^{-iT dt} F e^{-iV dt/2} psi(t),
second order in dt, unitary to machine precision, exact for a free particle.  Units: hbar = m = 1.
Imaginary-time propagation (t -> -i tau) plus Gram-Schmidt gives the lowest eigenstates, which is how the
"stationary state" scenarios are prepared (no analytic formulae are assumed).

MODIFY: add a potential in POTENTIALS (a function of the coordinate arrays), an initial state in INITIAL,
or a scenario in SCENARIOS (potential + initial state + box + dt).  Everything works in any dimension.
"""
from __future__ import annotations
import numpy as np
from scipy import fft as sfft

_FFT = dict(workers=-1)


class Grid:
    def __init__(self, dim: int, n: int, L: float):
        self.dim, self.n, self.L = dim, n, L
        x1 = np.linspace(-L / 2, L / 2, n, endpoint=False)
        self.dx = x1[1] - x1[0]
        self.axes = [x1] * dim
        self.X = np.meshgrid(*self.axes, indexing="ij")           # list of coordinate arrays
        k1 = 2 * np.pi * sfft.fftfreq(n, d=self.dx)
        K = np.meshgrid(*([k1] * dim), indexing="ij")
        self.k2 = sum(k * k for k in K)
        self.r2 = sum(x * x for x in self.X)

    @property
    def dV(self):
        return self.dx ** self.dim


# ----------------------------------------------------------------------------------------------
# potentials  V(grid, **params)
# ----------------------------------------------------------------------------------------------
def V_free(g, **p):
    return np.zeros_like(g.r2)


def V_harmonic(g, omega=1.0, **p):
    return 0.5 * omega ** 2 * g.r2


def V_barrier(g, height=2.0, width=1.0, **p):
    return np.where(np.abs(g.X[0]) < width / 2, height, 0.0)


def V_double_well(g, a=2.0, depth=1.0, **p):
    return depth * ((g.X[0] / a) ** 2 - 1.0) ** 2


def V_double_slit(g, wall_x=0.0, thickness=0.6, slit_sep=3.0, slit_width=0.8, height=50.0, **p):
    x, y = g.X[0], g.X[1]
    wall = np.abs(x - wall_x) < thickness / 2
    slits = (np.abs(y - slit_sep / 2) < slit_width / 2) | (np.abs(y + slit_sep / 2) < slit_width / 2)
    return np.where(wall & ~slits, height, 0.0)


def V_coulomb_soft(g, Z=1.0, soft=0.3, **p):
    """Soft-core Coulomb -Z / sqrt(r^2 + soft^2): the standard regularisation on a grid."""
    return -Z / np.sqrt(g.r2 + soft ** 2)


POTENTIALS = {"free": V_free, "harmonic": V_harmonic, "barrier": V_barrier, "double_well": V_double_well,
              "double_slit": V_double_slit, "coulomb": V_coulomb_soft}


# ----------------------------------------------------------------------------------------------
# initial states
# ----------------------------------------------------------------------------------------------
def gaussian_packet(g, x0=None, k0=None, sigma=1.0):
    x0 = np.zeros(g.dim) if x0 is None else np.asarray(x0, float)
    k0 = np.zeros(g.dim) if k0 is None else np.asarray(k0, float)
    r2 = sum((X - a) ** 2 for X, a in zip(g.X, x0))
    phase = sum(k * X for k, X in zip(k0, g.X))
    psi = np.exp(-r2 / (4 * sigma ** 2) + 1j * phase)
    return normalize(psi, g)


def normalize(psi, g):
    return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * g.dV)


# ----------------------------------------------------------------------------------------------
class Solver:
    def __init__(self, grid: Grid, V: np.ndarray, dt: float):
        self.g, self.V, self.dt = grid, V, dt
        self.t = 0.0
        self.psi = None
        self._expV_half = np.exp(-0.5j * V * dt)
        self._expT = np.exp(-0.5j * grid.k2 * dt)

    def set_state(self, psi):
        self.psi = normalize(psi.astype(complex), self.g)
        self.t = 0.0

    def step(self, nsteps=1):
        psi = self.psi
        ax = tuple(range(self.g.dim))
        for _ in range(nsteps):
            psi = self._expV_half * psi
            psi = sfft.ifftn(self._expT * sfft.fftn(psi, axes=ax, **_FFT), axes=ax, **_FFT)
            psi = self._expV_half * psi
        self.psi = psi
        self.t += nsteps * self.dt

    # --- observables ----------------------------------------------------------------------------
    def density(self):
        return np.abs(self.psi) ** 2

    def energy(self):
        g, psi = self.g, self.psi
        ax = tuple(range(g.dim))
        pk = sfft.fftn(psi, axes=ax, **_FFT)
        T = np.sum(0.5 * g.k2 * np.abs(pk) ** 2) / np.sum(np.abs(pk) ** 2)
        Vm = np.sum(self.V * np.abs(psi) ** 2) * g.dV
        return float(T + Vm), float(T), float(Vm)

    def expectation_x(self):
        rho = self.density()
        return np.array([np.sum(X * rho) * self.g.dV for X in self.g.X])

    def width(self):
        rho = self.density()
        x = self.expectation_x()
        return np.sqrt(np.array([np.sum((X - a) ** 2 * rho) * self.g.dV for X, a in zip(self.g.X, x)]))

    def transmission(self, x_edge=0.0):
        """Probability to the right of x_edge (1D barrier scenarios)."""
        return float(np.sum(self.density()[self.g.X[0] > x_edge]) * self.g.dV)


def imaginary_time_eigenstates(grid: Grid, V: np.ndarray, n_states: int = 1, dtau: float = 0.02,
                               iters: int = 3000, seed: int = 0):
    """Lowest n_states eigenpairs by imaginary-time propagation with Gram-Schmidt orthogonalisation
    (a standard, robust way to get bound states on a grid without diagonalising anything)."""
    rng = np.random.default_rng(seed)
    ax = tuple(range(grid.dim))
    expV = np.exp(-0.5 * V * dtau)
    expT = np.exp(-0.5 * grid.k2 * dtau)
    states = []
    for s in range(n_states):
        psi = rng.standard_normal(grid.k2.shape) * np.exp(-grid.r2 / (2 * (grid.L / 6) ** 2))
        for _ in range(iters):
            psi = expV * psi
            psi = sfft.ifftn(expT * sfft.fftn(psi, axes=ax, **_FFT), axes=ax, **_FFT).real
            psi = expV * psi
            for phi in states:
                psi -= phi * np.sum(phi * psi) * grid.dV
            psi = normalize(psi, grid).real
        states.append(psi)
    energies = []
    for psi in states:
        s = Solver(grid, V, 1e-3)
        s.set_state(psi)
        energies.append(s.energy()[0])
    return states, energies


# ----------------------------------------------------------------------------------------------
# scenarios: name -> dict(dim, n, L, dt, potential, params, init)
# ----------------------------------------------------------------------------------------------
SCENARIOS = {
    # ---- 1D
    "1d_free_spreading": dict(dim=1, n=1024, L=60.0, dt=0.01, potential="free", params={},
                              init=lambda g: gaussian_packet(g, x0=[-10], k0=[2.0], sigma=1.0),
                              note="free Gaussian packet: moves at v = k0 and spreads as sigma(t) = sigma sqrt(1 + (t/2 sigma^2)^2)"),
    "1d_harmonic_coherent": dict(dim=1, n=512, L=24.0, dt=0.01, potential="harmonic", params=dict(omega=1.0),
                                 init=lambda g: gaussian_packet(g, x0=[3.0], k0=[0.0], sigma=1 / np.sqrt(2)),
                                 note="coherent state: a displaced ground state oscillates without changing shape"),
    "1d_harmonic_squeezed": dict(dim=1, n=512, L=24.0, dt=0.01, potential="harmonic", params=dict(omega=1.0),
                                 init=lambda g: gaussian_packet(g, x0=[0.0], k0=[0.0], sigma=1.6),
                                 note="wrong width: the packet breathes at 2 omega (squeezed state)"),
    "1d_tunneling": dict(dim=1, n=2048, L=80.0, dt=0.005, potential="barrier", params=dict(height=2.0, width=1.0),
                         init=lambda g: gaussian_packet(g, x0=[-12], k0=[1.6], sigma=2.0),
                         note="E = k0^2/2 = 1.28 < V0 = 2: part of the packet tunnels through the barrier"),
    "1d_double_well": dict(dim=1, n=512, L=16.0, dt=0.01, potential="double_well", params=dict(a=2.0, depth=1.5),
                           init="left_well",
                           note="packet in the left well tunnels back and forth: period pi/(E1 - E0)"),
    # ---- 2D
    "2d_double_slit": dict(dim=2, n=384, L=40.0, dt=0.01, potential="double_slit",
                           params=dict(wall_x=-2.0, thickness=0.6, slit_sep=4.0, slit_width=0.9, height=60.0),
                           init=lambda g: gaussian_packet(g, x0=[-12, 0], k0=[3.0, 0], sigma=2.5),
                           note="two-slit interference of a single wave packet"),
    "2d_harmonic_orbit": dict(dim=2, n=192, L=20.0, dt=0.01, potential="harmonic", params=dict(omega=1.0),
                              init=lambda g: gaussian_packet(g, x0=[3.0, 0.0], k0=[0.0, 3.0], sigma=1 / np.sqrt(2)),
                              note="2D coherent state on a circular orbit: the classical limit inside QM"),
    "2d_free_spreading": dict(dim=2, n=256, L=40.0, dt=0.01, potential="free", params={},
                              init=lambda g: gaussian_packet(g, x0=[-8, -8], k0=[1.5, 1.5], sigma=1.2),
                              note="free 2D packet"),
    # ---- 3D
    "3d_harmonic_superposition": dict(dim=3, n=48, L=14.0, dt=0.01, potential="harmonic", params=dict(omega=1.0),
                                      init="eig_superposition", n_eig=3,
                                      note="superposition of the three lowest 3D oscillator eigenstates: |psi|^2 sloshes at the Bohr frequencies"),
    "3d_hydrogenlike": dict(dim=3, n=64, L=40.0, dt=0.02, potential="coulomb", params=dict(Z=1.0, soft=0.4),
                            init="eig_superposition", n_eig=2,
                            note="soft-Coulomb 3D atom: ground state + first excited state superposition (charge oscillation)"),
    "3d_free_packet": dict(dim=3, n=48, L=30.0, dt=0.01, potential="free", params={},
                           init=lambda g: gaussian_packet(g, x0=[-6, 0, 0], k0=[1.5, 0.3, 0], sigma=1.5),
                           note="free 3D packet moving and spreading"),
}


def build(name: str):
    sc = SCENARIOS[name]
    g = Grid(sc["dim"], sc["n"], sc["L"])
    V = POTENTIALS[sc["potential"]](g, **sc["params"])
    solver = Solver(g, V, sc["dt"])
    init = sc["init"]
    info = {}
    if init in ("left_well", "eig_superposition"):
        print(f"[{name}] preparing eigenstates by imaginary-time propagation (a few seconds in 3D)...", flush=True)
    if init == "left_well":
        states, E = imaginary_time_eigenstates(g, V, 2, iters=4000)
        psi = (states[0] + states[1]) / np.sqrt(2)     # localised in one well
        info["energies"] = E
        info["tunnel_period"] = np.pi / abs(E[1] - E[0])
    elif init == "eig_superposition":
        states, E = imaginary_time_eigenstates(g, V, sc["n_eig"], iters=1500 if g.dim == 3 else 3000)
        psi = sum(s for s in states)
        info["energies"] = E
    else:
        psi = init(g)
    solver.set_state(psi)
    return solver, sc, info
