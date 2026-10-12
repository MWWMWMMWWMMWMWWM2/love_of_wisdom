# Neural-network wave functions: the variational principle as a learning problem

## Abstract

We implement, in plain numpy, the core of the "neural quantum state" approach to quantum problems that have
no analytic solution: a small multilayer perceptron multiplied by a Gaussian envelope serves as a trial wave
function ψ_θ, and the Rayleigh quotient E[θ] = ⟨ψ_θ|H|ψ_θ⟩/⟨ψ_θ|ψ_θ⟩ is minimised by gradient descent
(Adam) with gradients obtained by back-propagation through the energy functional and the network.  By the
variational theorem every E[θ] is an upper bound on the ground-state energy, so the training curve is a
rigorous, monotonically improvable bound; excited states follow by projection onto the orthogonal complement
of the states already found.  On six benchmark problems — harmonic, quartic and double-well oscillators, a
random rough potential, a non-separable 2D oscillator and the Hénon–Heiles potential — a network with
~1 200 parameters reaches the exact grid energies to 10⁻³–10⁻⁴ relative accuracy in seconds, reproducing the
tunnel splitting of the double well, Anderson-like localisation in the disordered potential and the
degenerate pair of the coupled 2D oscillator.  We explain why this is the small-dimensional prototype of
the methods (FermiNet, PauliNet, neural VMC) that now compete with the best quantum-chemistry techniques,
and where the grid quadrature used here must be replaced by Monte Carlo sampling.

## 1. Why "AI" for quantum problems

Beyond a handful of textbook cases (particle in a box, harmonic oscillator, hydrogen) the Schrödinger
equation has no closed-form solutions.  The classic numerical routes are (i) basis-set diagonalisation, which
scales exponentially with the number of particles, and (ii) the variational method with a hand-designed
trial function (Hartree–Fock, Jastrow factors), whose accuracy is limited by human ingenuity.  Neural
networks are universal function approximators with cheap gradients; Carleo & Troyer (2017) showed that a
restricted Boltzmann machine trained variationally reaches state-of-the-art energies for spin chains, and
Hermann et al. (2020) and Pfau et al. (2020) extended the idea to electrons in molecules with antisymmetric
networks.  The ingredient common to all of them is the one implemented here: *the energy functional is the
loss*.  No labelled data are needed; the Hamiltonian is the teacher.

## 2. Method

**Ansatz.**  ψ_θ(x) = NN_θ(x) · exp(−|x|²/2s²), NN_θ a 2-hidden-layer tanh network (d → 32 → 32 → 1).  The
envelope guarantees normalisability and the correct decay far from the origin; the network learns the
shape, including nodes (it is unconstrained in sign).

**Energy and gradient.**  On a grid with spacing dV, the discretised Hamiltonian H is the sparse second-order
finite-difference operator −½∇² + V.  With ψ the vector of network outputs times the envelope,

    E = (ψᵀHψ)/(ψᵀψ),        ∂E/∂ψ = 2(Hψ − Eψ)/(ψᵀψ),

and ∂E/∂θ follows by the chain rule through the envelope and the network (hand-written back-propagation,
`MLP.backward`).  The quadrature is exact to the grid accuracy; H is the same operator whose lowest
eigenpairs `scipy.sparse.linalg.eigsh` provides for validation.

**Optimisation.**  Adam with learning rate 3·10⁻³, up to 3 000 steps, early stopping when the energy
changes by less than 10⁻⁷ per 100 steps; the best energy encountered is reported (the variational bound).
The spikes visible in the training curves are Adam overshoots; the bound is never violated.

**Excited states.**  After state k has converged its normalised vector φ_k is stored and every subsequent ψ
is replaced by Pψ, P = 1 − Σ_j |φ_j⟩⟨φ_j|.  The Rayleigh quotient of Pψ is bounded below by E_{k+1}
provided the φ_j are exact; with approximate φ_j the bound can be violated at the 10⁻⁴ level, which is
what the tables show (the "neural" energy occasionally lies a hair below the exact grid value for k ≥ 1).

## 3. Benchmarks (default settings, one CPU)

| problem | states | rel. error | what it shows |
|---|---|---|---|
| `harmonic_1d` | 0.5, 1.5, 2.5 | ≤ 3·10⁻⁴ | sanity check against E_n = n + ½ |
| `quartic_1d` V = x⁴/2 | 0.5302, 1.8998, 3.7278 | ~10⁻³ | no closed form; agrees with the literature value E₀ = 0.530181 |
| `double_well_1d` | 0.9004, 0.9482, 2.1851 | ≤ 6·10⁻⁴ | tunnel splitting ΔE = 0.048 resolved |
| `disordered_1d` | −1.201, 0.085, 1.333 | ≤ 10⁻² | low states localise in random wells (Anderson) |
| `coupled_2d` V = ½r² + 0.4x²y² | 1.0796, 2.2160 (×2) | ≤ 10⁻³ | non-separable; the degenerate pair is found |
| `henon_heiles_2d` | 0.9968, 1.9850 | ≤ 3·10⁻⁴ | quasi-bound states of a classically chaotic potential |

## 4. From this toy to the real thing

Two things change in many dimensions.  (i) The grid is replaced by Monte Carlo sampling of |ψ_θ|² (Metropolis),
and the energy by the average of the local energy E_L = Hψ/ψ over samples — variational Monte Carlo, whose
cost grows polynomially, not exponentially, with the number of particles; the gradient estimator is the
standard ∂E/∂θ = 2⟨(E_L − E) ∂ln ψ/∂θ⟩.  (ii) For fermions the ansatz must be antisymmetric: a Slater
determinant of network orbitals (FermiNet/PauliNet), which also removes the need to learn nodes from
scratch.  Everything else — envelope, energy-as-loss, variational bound, projection for excited states — is
exactly what this code does.

## 5. Limitations

Bosonic single-particle problems in ≤ 2 dimensions; grid quadrature (exact but exponential in dimension);
no sampling noise; excited-state projection uses approximate lower states; a fixed small architecture.
All energies are compared with exact diagonalisation of the identical grid Hamiltonian, so "exact" here
means "converged numerics", not analytic.

## References

* G. Carleo, M. Troyer, *Science* **355**, 602 (2017) — neural-network quantum states.
* J. Hermann, Z. Schätzle, F. Noé, *Nat. Chem.* **12**, 891 (2020) — PauliNet.
* D. Pfau, J. S. Spencer, A. G. D. G. Matthews, W. M. C. Foulkes, *Phys. Rev. Res.* **2**, 033429 (2020) — FermiNet.
* W. M. C. Foulkes et al., *Rev. Mod. Phys.* **73**, 33 (2001) — quantum Monte Carlo methods.
* D. P. Kingma, J. Ba, arXiv:1412.6980 (2014) — Adam.
* K. Banerjee et al., *Proc. R. Soc. A* **360**, 575 (1978) — quartic oscillator eigenvalues (E₀ = 0.530181).

## 6. Part II — a stored, potential-conditioned model for user-drawn potentials, with real-time dynamics

(The time evolution is the purpose; the energy levels and probabilities are computed because the full
Schrödinger equation needs them.)

**Problem.**  The solver of §2 trains one network per potential from scratch.  For an interactive tool we want
a model that *already knows* what low-lying eigenfunctions look like and only has to adapt — and the user
wants to see the full time-dependent Schrödinger equation, not just energy levels.

**Model.**  ψ_k(x | V) = NN(x, e_k, d(V)) · exp(−(x/6)⁴), where e_k is a one-hot state index (k = 0..3) and
d(V) the potential sampled at 32 points of the box, scaled by 1/6 (a fixed-length descriptor — a crude
"operator network" in the spirit of DeepONet, Lu et al., Nat. Mach. Intell. 3, 218 (2021)).  Hidden layers
96–96–64, tanh, ~16 000 parameters, stored in `pretrained_model.npz` (158 kB).

**Stage 1, pre-training on known numerics** (`train_pretrained.py`): 1 600 random potentials built from the
same primitives the builder offers (linear, polynomial, exponential, step, smooth-step segments; bowls;
Gaussian wells) are diagonalised exactly; the network is fitted by least squares to the normalised bound
eigenfunctions (≈ 5 700 states, 10 epochs, 3.5 min on one CPU).  Held-out: the raw prediction gives the
energy to a median relative error of 9 % — a good starting point, not an answer.

**Stage 2, fine-tuning on the drawn potential** (`qm_model.fit`): starting from the stored weights with the
descriptor frozen to the new potential, the Rayleigh quotient is minimised as in §2, state by state with
projection.  The variational principle guarantees that fine-tuning can only improve the pre-trained guess;
1 200 Adam steps take ~1 s per state.

**Baselines shown to the user.**  (i) Exact sparse diagonalisation of the same grid Hamiltonian; (ii) the
WKB / Bohr–Sommerfeld rule ∫√(2(E−V)) dx = (n + ½)π between turning points.  On an asymmetric step-well
with a bump the model's energy errors are 10⁻⁶–2·10⁻⁴ while WKB's are 3·10⁻²–2·10⁻¹: three to four orders
of magnitude better, as it must be — WKB is an asymptotic approximation, the variational energy is an exact
upper bound.

**Discrete vs continuous spectrum.**  A state is bound only if E < min(V(−∞), V(+∞)); the tool counts the
bound states and refuses to call a level above the threshold an eigenstate (the grid would return a
box-quantised artefact).  For such energies it switches to scattering: the stationary equation is
integrated inward from an outgoing wave on the right, incident and reflected amplitudes are read off on
the left, and T(E) = (k_R/k_L)|1/A|², R = |B/A|² (T + R = 1 to 10⁻³).  Peaks of T(E) are resonances —
what "a level in the continuum" physically means.

**Real-time dynamics — the actual deliverable.**  Given the computed levels, any initial state ψ₀ (a wave
packet or a chosen superposition) evolves as ψ(x,t) = Σₙ cₙ ψₙ(x) e^{−iEₙt} with probabilities
|cₙ|² = |⟨ψₙ|ψ₀⟩|².  This *is* the solution of the time-dependent Schrödinger equation within the span of
the computed states, and the energies and probabilities are exactly the ingredients the full equation needs.
The tool runs, side by side and in real time:

* the full split-operator solution of the TDSE (the referee; §1 of `qm_wavefunctions`);
* the AI-level spectral propagation above;
* Heller's **thawed-Gaussian approximation** (J. Chem. Phys. 62, 1544 (1975)) — the time-dependent WKB
  for a wave packet: centre on the classical trajectory, width following the local curvature V″(q), exact
  for quadratic potentials;
* a **classical trajectory ensemble** (Liouville propagation of the initial Wigner distribution), the
  classical-mechanics prediction for |ψ|².

The fidelities with the full equation are shown live.  On the asymmetric step-well example with a packet
released off-centre: AI-levels 0.999 at all times; thawed Gaussian 0.996 → 0.70 (t = 3) → 0.30 (t = 10)
as the packet spreads over non-quadratic parts of the well and reflects; classical ensemble 0.95–0.98 in
density overlap but with no interference structure.  The "captured probability" Σ|cₙ|² tells the user
when more levels (or the continuum) are needed: a packet near the threshold captures 0.93 and the AI
fidelity drops accordingly — physics, not numerics.  (The static WKB energies are still listed in the
solve table for reference, but the comparison that matters for dynamics is the one above.)

**Real-time training and continual learning.**  The fine-tuning is not a batch job: `LiveFit` performs
the same Adam updates in slices of 30 steps between screen refreshes, so the user watches the variational
wave function converge (energy and error updating live) and the run stops itself when the energy changes
by less than 10⁻⁷ over 100 steps.  Independently, `ContinualTrainer` keeps improving the stored weights
whenever the interface is idle: each tick draws one new random potential (exact states by one sparse
diagonalisation), takes a dozen supervised steps on a mix of random and user-solved potentials, and every
300 steps evaluates a fixed held-out set, keeping the new weights only if the median energy error of the raw
prediction did not increase (otherwise it rolls back).  This is a minimal form of experience replay with
validation gating — the standard guard against catastrophic drift in continual learning.

**Limitations of Part II.**  One dimension; four pre-trained state slots (higher levels are fine-tuned
from slot 3 and still converge, but start further away); the descriptor resolves features no finer than
0.5 length units; pre-training is supervised on grid numerics and inherits the grid's accuracy — the point
is speed and a warm start, not new physics.
