# 05 — Casimir plates on the lattice, static and dynamical

References: Casimir, Proc. K. Ned. Akad. Wet. 51 (1948) 793; Milonni, *The Quantum Vacuum*
(1994) ch. 2–3; Bordag, Mohideen & Mostepanenko, Phys. Rep. 353 (2001) 1 (massive case);
Moore, J. Math. Phys. 11 (1970) 2679 and Wilson et al., Nature 479 (2011) 376 (dynamical Casimir
effect, predicted and observed).  Polchinski uses the word for the finite-size energy of the
string world-sheet (Vol.1 p.22, p.73) and of compact dimensions (Vol.1 ch.8, Vol.2 p.145) — same
physics, different space.

## Geometry

Two perfect mirrors = Dirichlet planes φ = 0 at the z-layers z₁ < z₂ of the periodic lattice.
The remaining sites form two chains along z: inside (n_A = z₂ − z₁ − 1 sites) and outside
(n_B = N − n_A − 2 sites, wrapping through the periodic boundary).  In x, y everything stays
periodic.  A Dirichlet chain of n sites is diagonalised by the type-I discrete sine transform,

    S_j(s) = √(2/(n+1)) sin(π j s/(n+1)),  s, j = 1..n,   −Δ S_j = (2/a)² sin²(π j/(2(n+1))) S_j ,

so the modes are (k_x, k_y, j) with

    ω² = m² + (2/a)²[ sin²(k_x a/2) + sin²(k_y a/2) + sin²(π j /(2(n+1))) ] .

`geometry.DirichletBlock` uses `scipy.fft.dst(type=1, norm='ortho')` (orthonormal, involutory),
so the vacuum normalisation of file 01 is unchanged.  The continuum-equivalent length of a chain
is (n+1)a: the distance between its two zero-planes.  The energy density on the grid uses the
same finite differences including the links to the plate sites, so the grid energy equals the
mode energy exactly (`smoke test` in the session log; `ensemble_energy_profile` sums to the
zero-point energy to 1e-12).

## Static Casimir energy and force

E₀(d) = Σ_modes ½ω over both chains depends on the plate separation d = (n_A + 1)a while the
total number of degrees of freedom does not; the d-dependent part is the Casimir energy.
Continuum reference for one massless real scalar per unit area, both regions:

    E(d)/A = −π²/1440 [ 1/d³ + 1/(L − d)³ ],     F/A = −∂_d E/A = −π²/(480 d⁴) + π²/(480 (L−d)⁴).

(EM field: twice these numbers.)  Massive field:  E/A = −(m²/8π²d) Σ_{n≥1} K₂(2nmd)/n²,
exponentially small for md ≫ 1.  The lattice force (central difference of E₀ over one lattice
spacing, plate 2 moved) agrees with the same central difference of the continuum formula to
2–7 % for d ≥ 6a and m ≤ 0.35/l_s, with O(a²/d²) corrections (`test_casimir_lattice_vs_continuum`;
numbers in the session log: ratio 1.03–1.07 for N = 48).  The panel shows lattice and continuum
side by side, in string units and in newtons for the chosen l_s.

The energy-density *profile* ⟨h⟩(z) between the plates (plotted live) is computed exactly from
the mode covariances: it is lower in the middle of the gap and piles up at the mirrors (the
lattice keeps the surface divergence of the continuum finite).

## Sudden insertion / removal = dynamical Casimir effect (sudden limit)

Inserting the mirrors instantaneously clamps φ = π = 0 on the plate layers and re-expands the
field in the new mode basis.  The old vacuum is not the new one: the new modes are populated.
Exact ensemble result (first change after a reset): with the z-mixing matrix

    M_{ji} = Σ_{z ∈ chain} S_j(z) conj(S^{old}_i(z))          (S^{old} = plane waves e^{−ik_z z}/√N),

the diagonal covariances transfer as  C^{new}_{jj} = Σ_i |M_{ji}|² C^{old}_{ii}  for C_qq, C_pp
and C_qp separately (the transverse FFT is common to both bases and drops out), and eq. F1 gives
the occupations.  `test_plate_insertion_exact_transfer` checks the transferred ensemble energy
against the mean over 120 independently sampled configurations.  Removal is the same with the
roles exchanged; moving = remove + insert.  After the first change the neglected inter-mode
correlations make later transfers approximate; the *sample* remains exact throughout.

The experimentally observed version (Wilson et al. 2011) uses a mirror oscillating at GHz
frequencies; here a periodic "move" would emulate it crudely.  MODIFY: a smoothly moving mirror
needs a time-dependent basis — a genuine extension.
