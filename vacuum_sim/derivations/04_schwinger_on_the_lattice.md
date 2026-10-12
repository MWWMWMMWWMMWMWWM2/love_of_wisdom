# 04 — The Schwinger effect as a parametric oscillator per mode (lattice version)

References: Schwinger, Phys. Rev. 82 (1951) 664; Sauter, Z. Phys. 69 (1931) 742; for the
quantum-kinetic / mode-by-mode formulation Kluger, Mottola & Eisenberg, PRD 58 (1998) 125015.
Not covered in Polchinski (the only "Schwinger" there is the Schwinger term of current algebras).

## Continuum

Charged scalar φ (mass m, charge e) in a uniform electric field E(t) along x.  Temporal gauge
A₀ = 0, A_x(t) = −∫₀ᵗ E dt′.  The covariant derivative turns a plane wave e^{ik·x} into a mode with
*kinetic* momentum k_x − eA_x(t) = k_x + e∫E dt, so every Fourier mode is an independent
oscillator with the time-dependent frequency

    ω_k(t)² = m² + k_⊥² + (k_x + e∫₀ᵗE dt′)².                                                  (1)

The vacuum state of each mode is the Gaussian of file 01; the evolution is exactly the mode
rotation with ω_k(t) (quadratic Hamiltonian, no back-reaction).  Pairs are created when a mode is
swept through k_kin = 0, where ω is minimal and changes fastest relative to itself.

Exact single-mode result for a constant field sweeping from k_kin = −∞ to +∞ (Landau–Zener /
Sauter):

    |β_k|² = exp( −π m_⊥² / eE ),      m_⊥² = m² + k_⊥².                                       (2)

Summing over the modes that cross zero during a time T (density V d³k/(2π)³, sweep eE T in k_x):

    N_pairs = V T (eE)/(2π)³ ∫ d²k_⊥ e^{−π m_⊥²/eE} = V T (eE)² e^{−πm²/eE} / (8π³),

the leading term of Schwinger's scalar-QED rate

    Γ = (eE)²/(8π³) Σ_{n≥1} (−1)^{n+1} n^{−2} e^{−nπm²/eE}     (spinor QED: (eE)²/(4π³) Σ n^{−2} e^{…}).   (3)

The effect is non-perturbative (e^{−π m²/eE}, essential singularity in e) and switches on around
the critical field E_c = m²/e (1.3×10¹⁸ V/m for electrons).

## Lattice implementation (exact lattice gauge theory)

On the lattice the field couples to the link variable U_x = e^{−i e a A_x}; the covariant
difference (φ_{x+a} e^{i e a ∫E} − φ_x)/a is used in the energy density, and in Fourier space it
multiplies Φ_k by (e^{i(k_x + e∫E)a} − 1), so

    ω_k(t)² = m² + (2/a)²[ sin²((k_x + e∫E)a/2) + sin²(k_y a/2) + sin²(k_z a/2) ].               (4)

`test_complex_field_energy_with_gauge_link` checks that the grid energy with the link equals the
mode-space energy with (4).  The pulse is flat-topped with raised-cosine ramps; the frame is
sub-stepped with ω frozen at each sub-step midpoint.

### Bloch wrapping and the pulse-length limit

On a lattice k_x + e∫E is periodic with period 2π/a: a uniform field makes every mode Bloch-
oscillate instead of accelerating forever.  A mode near the positive zone edge wraps to −π/a and
would cross k_kin = 0 a second time — an artefact absent in the continuum.  As long as the total
sweep S = e∫E dt ≤ π/a, every mode crosses zero at most once and only if it would also do so in
the continuum, so the set of pair-creation events is identical to the continuum one (the modes
that matter sit near k_kin = 0 where (4) ≈ (1) if m_⊥ a ≪ 1).  `Simulation.max_pulse_duration`
returns the corresponding maximal pulse length; the GUI warns above it.

### Checks

* `test_sauter_landau_zener_single_mode`: the actual `evolve()` code with continuum ω(t) sweeping
  through zero reproduces (2) for eight values of m_⊥² to 3 %.
* `test_schwinger_rate_on_lattice`: total exact pairs from a lattice pulse agree with Γ V τ_eff
  within the tolerance set by the finite pulse/ramps.

### Energy is not conserved here — correctly

The external field does work on the pairs.  The sample's total energy rises during the pulse;
the "excess over vacuum" view acquires a positive spatial mean; the exact pair number is in the
panel.  Because the created particles carry kinetic momentum ±eE t along x (they are accelerated
by the field), they are delocalised plane waves, not little balls appearing at points: the
spatially uniform statistics you see is the correct quantum picture.
