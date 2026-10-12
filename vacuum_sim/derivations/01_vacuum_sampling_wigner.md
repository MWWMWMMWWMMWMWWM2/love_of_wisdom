# 01 — Sampling the vacuum exactly: the Wigner-function argument

## The lattice field is a set of harmonic oscillators

A real scalar field on a periodic N³ lattice with spacing a has the Hamiltonian

    H = a³ Σ_x [ ½ π_x² + ½ Σ_i (φ_{x+i} − φ_x)²/a² + ½ m² φ_x² ] .

With the orthonormal discrete Fourier transform F_k = N^{-3/2} Σ_x φ_x e^{−ik·x} and
Q_k = √(a³) F_k, P_k = √(a³) (transform of π), Parseval gives

    H = ½ Σ_k ( |P_k|² + ω_k² |Q_k|² ),     ω_k² = m² + (2/a)² Σ_i sin²(k_i a/2).        (1)

The sum runs over all N³ lattice momenta; for a real field Q_{−k} = Q_k*, so the N³ complex
coefficients carry exactly N³ real oscillators.  (For a complex field the coefficients are
independent and carry 2N³ oscillators: particles and antiparticles.)

## Ground state and Wigner function

Each oscillator's ground state is Gaussian.  Its Wigner function is the positive Gaussian

    W₀(q,p) ∝ exp[ −ω q²/ħ − p²/(ħω) ],   ⟨q²⟩ = ħ/(2ω),  ⟨p²⟩ = ħω/2,  ⟨{q,p}⟩/2 = 0.  (2)

Two facts make the simulation exact:

1. **Moyal = Poisson for quadratic Hamiltonians.**  The quantum Liouville (Moyal) equation for W
   contains, beyond the classical Poisson bracket, only terms with third and higher derivatives of
   H.  For H quadratic in (q,p) they vanish identically, so W(t) is obtained by transporting each
   phase-space point along the *classical* trajectory.  This holds for any time dependence of the
   quadratic H (masses m(t), external fields A(t), boundary conditions), i.e. for every perturbation
   in this program.

2. **Gaussian W ⇒ Wick's theorem for symmetric ordering.**  Sampling (q,p) from (2) and averaging
   any polynomial reproduces the *symmetrically ordered* (Weyl-ordered) quantum expectation value.
   For quadratic observables symmetric ordering is the physical one: ⟨H⟩ = Σ ½ħω_k, the zero-point
   energy (Vol.1 p.221 "the sum of the zero-point energies of the modes of the field").

So one sample of the Gaussian, evolved classically, is one exact "history" of the vacuum: its
statistics over samples are the vacuum's symmetrically ordered correlators.  What a single sample
looks like is what a probe with the lattice resolution a (or the smoothing σ, see below) would
register.  This is the *truncated Wigner approximation* of quantum optics (Polkovnikov, Ann. Phys.
325 (2010) 1790; Blakie et al., Adv. Phys. 57 (2008) 363), which is not an approximation here.

## Normalisation used in the code (ħ = 1)

    ⟨|Q_k|²⟩_vac = 1/(2ω_k),   ⟨|P_k|²⟩_vac = ω_k/2                                          (3)

Check: ⟨φ_x²⟩ = N^{-3} Σ_k ⟨|F_k|²⟩ = (N a)^{-3} Σ_k 1/(2ω_k) = (1/V) Σ_k 1/(2ω_k), the standard
result (∫d³k/(2π)³ · 1/(2ω_k) in the continuum).  Implementation: transform unit white noise ξ_x
(⟨|Ξ_k|²⟩ = 1 for an orthonormal transform, Hermitian symmetry automatic) and multiply by
√(1/2ω_k) and √(ω_k/2).  `tests/test_physics.py::test_vacuum_variance_matches_mode_sum` checks (3).

The zero mode of a massless field (ω = 0, the spatially constant mode in a periodic box) is a free
particle without a normalisable ground state; it is dropped (standard zero-mode subtraction).

## Time evolution

For constant ω each mode rotates: Q(t) = Q cos ωt + (P/ω) sin ωt, P(t) = P cos ωt − ωQ sin ωt.
Exact, symplectic, no CFL condition, energy conserved to round-off
(`test_energy_conservation_and_zero_occupation`).  When ω depends on time the frame is split into
sub-steps with ω frozen at each sub-step midpoint ("exponential midpoint" scheme, second order,
symplectic).  Its accuracy is checked against the exact Landau–Zener/Sauter result in
`test_sauter_landau_zener_single_mode` (3 % with the default sub-step).

## Ensemble bookkeeping (why the read-outs have no sampling noise)

Alongside the sample we propagate, per mode, the covariance (C_qq, C_pp, C_qp) with the same
linear map, C → R C Rᵀ.  Because the state stays Gaussian, this is the *complete* description of
the ensemble.  The exact occupation number relative to the instantaneous vacuum is

    n_k = (C_pp + ω_k² C_qq)/(2ω_k) − ½ = |β_k|²,                                            (4)

which is the Bogoliubov coefficient (derivation in 03).  Energy per mode: ω_k (n_k + ½).

## Probe resolution σ

The rendered field is optionally smoothed with exp(−k²σ²/2) in mode space before transforming
back.  Physically this is the field seen by a detector of size σ (a smeared operator φ_σ(x)); its
fluctuations are finite even in the continuum, ⟨φ_σ²⟩ ∼ 1/σ² for a massless field.  The energy
density then shown is that of the smeared field; the exact numbers in the panel are always the
unsmoothed lattice quantities.
