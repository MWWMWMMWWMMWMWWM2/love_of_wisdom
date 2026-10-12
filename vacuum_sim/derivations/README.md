# Derivations and sources

Everything the simulation computes is either (a) a textbook result with a reference, or (b) derived
here.  Page numbers "Vol.1 p.N" refer to Polchinski, *String Theory Vol.1: An Introduction to the
Bosonic String*; "Vol.2" to *Superstring Theory and Beyond* (print pagination, as in the epub/pdf
you supplied).

| file | content | status |
|---|---|---|
| `01_vacuum_sampling_wigner.md` | why drawing the vacuum from the ground-state Wigner function and evolving classically is **exact** for free fields; normalisation used in the code | derived (standard result) |
| `02_zero_point_energy_and_string_tower.md` | zero-point sum = vacuum energy density (Vol.1 §7.3), the string mass tower and degeneracies (Vol.1 §1.3-1.4, Vol.2 ch.10), the proper-time cutoff, the "52 orders of magnitude" | textbook + derived |
| `03_mass_quench_bogoliubov.md` | particle creation by a time-dependent mass: exact Bogoliubov coefficients, sudden limit, adiabatic limit | textbook (Birrell & Davies) |
| `04_schwinger_on_the_lattice.md` | uniform electric field as a parametric oscillator per mode; lattice gauge link; when the lattice result equals the continuum Schwinger/Sauter result; the pulse-length limit | derived |
| `05_casimir_on_the_lattice.md` | Dirichlet chains, DST-I diagonalisation, lattice Casimir energy vs. −π²/1440d³, massive K₂ formula, the sudden-insertion (dynamical Casimir) covariance transfer | derived |
| `06_supersymmetry_and_the_mean_vacuum_energy.md` | (−1)^F sums (Vol.1 eq. 7.3.24), level-by-level cancellation, SUSY breaking, what SUSY does and does not do to *fluctuations* | textbook + derived |
| `07_virtual_particles_what_is_real.md` | what the popular "particles popping in and out" picture means, what the simulation shows, and which processes really turn vacuum fluctuations into particles | conceptual |

## What is exact and what is approximate

Exact (no approximation beyond the lattice regularisation itself):

* the vacuum state of every free field on the lattice (Gaussian ground state),
* its real-time evolution, energy density, and all symmetrically-ordered correlation functions,
* particle creation by mass quenches / ramps and by the uniform electric field (no back-reaction),
* the static Casimir energy and force on the lattice,
* the ensemble-mean particle numbers created by the *first* sudden change of boundary conditions.

Approximate / idealised:

* the lattice is the UV cutoff (Λ = π/a).  Numbers such as ⟨h⟩_vac depend on it -- they must; that is
  the cosmological-constant problem, not a bug (Vol.1 p.222).
* "perfect mirrors" (Dirichlet on every field) and "sudden" changes are idealisations.
* after the second geometry change the ensemble covariance neglects correlations between modes
  that the first change created (the *sample* is still exact).
* the electric field is external (no back-reaction of the created pairs on the field).
* interactions between fields are absent.  This is what makes everything above exact, and it is
  also why the Standard Model itself cannot be simulated this way (see README).

## Where to look in the two volumes

* Vol.1 §1.3 pp.21-22: world-sheet zero-point sum, ζ-regularisation, Casimir interpretation.
* Vol.1 §7.3 pp.219-222: torus vacuum amplitude, absence of UV divergence, sum of zero-point
  energies (7.3.18-7.3.22), Coleman-Weinberg (7.3.23), (−1)^F generalisation (7.3.24),
  cosmological constant problem.
* Vol.1 §9.8 p.320: r.m.s. size of the string ground state (grows with resolution).
* Vol.1 ch.8, Vol.2 p.145: Casimir energies of compact dimensions.
* Vol.2 ch.10 (superconformal algebra, GSO projection: no tachyon, bosons = fermions),
  p.282 (order-by-order cancellation of the vacuum energy), p.371 (no-scale models), App. B.
