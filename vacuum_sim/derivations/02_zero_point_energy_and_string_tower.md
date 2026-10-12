# 02 — Zero-point energy, the string tower, and what "vacuum energy" means here

## 2.1 The vacuum energy density is the sum of zero-point energies (Vol.1 §7.3)

Polchinski derives the one-loop vacuum amplitude first for a point particle (sum over closed
paths, eq. 7.3.18-7.3.20) and identifies the vacuum energy density of a free field of mass m as

    ρ₀ = (1/V) Σ_k ½ ħ ω_k  →  ∫ d³k/(2π)³ ½ ω_k,      ω_k = √(k² + m²)          (Vol.1 eq. 7.3.22)

"which is just the sum of the zero-point energies of the modes of the field" (p.221).  On our
lattice this is a finite sum (cutoff Λ = π/a).  The sampled fields' energy density averages to
exactly this value (`test_sample_energy_equals_mode_energy_and_zero_point`).

Continuum with a sharp cutoff |k| < Λ (one real scalar):

    ρ₀(m,Λ) = [ Λ(2Λ² + m²)√(Λ² + m²) − m⁴ asinh(Λ/m) ] / (32π²)
            = Λ⁴/(16π²) + m²Λ²/(16π²) − m⁴ ln(2Λ/m)/(64π²) + O(m⁴).                          (1)

The Λ⁴ and m²Λ² terms are regulator dependent; the m⁴ ln m² term is universal — this is the
Coleman–Weinberg formula (Vol.1 eq. 7.3.23; Coleman & Weinberg, PRD 7 (1973) 1888).  The
generalisation to particles of any spin is (Vol.1 eq. 7.3.24)

    ρ₀ = Σ_i (−1)^{F_i} g_i ∫ d³k/(2π)³ ½ ω_{k,i},        g_i = number of polarisations,

fermions entering with the opposite sign.  `observables.py` implements (1), the Coleman–Weinberg
log, and a proper-time regulated version.

## 2.2 Why string theory is different (Vol.1 pp.219–220)

Summing (7.3.22) over the whole string spectrum, eq. (7.3.12), "only makes things worse, as all
the states of the string contribute with the same sign".  The actual string one-loop amplitude
has the *same integrand* but is integrated over the modular fundamental region instead of the
field-theory region: "the UV divergent region is simply absent".  The torus amplitude is finite
without a cutoff.  Nothing in a lattice field simulation can reproduce this — it is a statement
about the world-sheet, not about spacetime fields.  What we *can* do honestly:

* use the string spectrum (masses and degeneracies) for the fields we sample,
* show the field-theory zero-point sum with an explicit cutoff, as Polchinski does for
  "rough insight" (p.220: cut the proper-time integral at l ≥ ε and drop divergent terms),
* show the supersymmetric cancellation that the superstring actually uses (file 06).

## 2.3 The tower (Vol.1 §1.3–1.4, Vol.2 ch.10)

Light-cone quantisation gives  m² = (N − 1)/α′ (open) and 4(N − 1)/α′ (closed) for the bosonic
string, where the −1 is the regularised zero-point energy of the world-sheet oscillators,
Σ_{n≥1} n → ζ(−1) = −1/12 per transverse direction, 24 directions → −1  (Vol.1 §1.3, p.21–22;
Polchinski calls the finite remainder "an example of a Casimir energy").  For the GSO-projected
superstring the tachyon is absent and  m² = N/α′ (open), 4N/α′ (closed type II).

Degeneracies (number of physical states at level N) from the partition functions:

    bosonic open       Σ d_N q^N = Π_{n≥1} (1 − qⁿ)^{−24}          = 1, 24, 324, 3200, 25650, …
    open superstring   Σ d_N q^N = 8 Π_{n≥1} ((1+qⁿ)/(1−qⁿ))⁸       = 8, 128, 1152, 7680, 42112, …  (bosons = fermions)
    closed             d_N^{closed} = d_N^{L} · d_N^{R};  type II: bosons = fermions = 2 d_N².

(`spectrum.py`, checked in `test_string_degeneracies`.)  The growth d_N ∼ e^{c√N} (Hagedorn) is
what makes the naive field-theory sum over the tower diverge.

In string units (α′ = 1) the sampled fields are: a massless neutral scalar (one polarisation of the
massless level — think dilaton / modulus), a light charged scalar of the open-string sector (it
carries the Chan–Paton U(1) charge, Vol.1 §6.5 — the Schwinger field acts on it), and the first
massive levels (m = 2, 2√2 for closed type II).  Each sampled field stands for g_N states in the
mean-energy read-outs; sampling all g_N copies would cost g_N× the CPU and change nothing but the
relative size of fluctuations of the *sum* (∝ 1/√g_N).

## 2.4 Proper-time cutoff at the string scale

Polchinski's prescription (p.220) applied to (7.3.22):

    ρ(m, ε) = −(1/32π²) ∫_ε^∞ dl l^{−3} e^{−m² l}  =  −m⁴ Γ(−2, m²ε)/(32π²) .

With ε = α′ the weight e^{−m²_N ε} beats the Hagedorn growth, so the tower sum converges and is
dominated by the first few levels (`Simulation.tower_summary`).  Note the sign: with this
regulator the power divergences are negative; only the m⁴ ln m² term is universal.  We display
the number as an illustration of *how string theory organises the sum*, not as a prediction.

## 2.5 Sizes and the "52 orders of magnitude" (Vol.1 p.222)

With one massless boson and a cutoff at the electroweak scale Λ ≈ 100 GeV, (1) gives
ρ ≈ Λ⁴/16π² ≈ 6×10⁵ GeV⁴, versus the observed ρ_Λ ≈ (2.25 meV)⁴ ≈ 2.6×10⁻⁴⁷ GeV⁴: 10^{52.4}
too large, exactly Polchinski's remark.  The panel converts the simulation's ⟨h⟩_vac to J/m³ and
GeV⁴ for the chosen string length; at l_s ∼ 10⁻³⁴ m the number is ∼10^{120} times the observed
dark-energy density — the famous "worst prediction in physics".  It is there to make the scale
tangible, not to be believed literally.
