# 06 — Supersymmetry and the mean vacuum energy

## The (−1)^F sum (Vol.1 eq. 7.3.24)

    ρ₀ = Σ_i (−1)^{F_i} g_i (1/V) Σ_k ½ ω_{k,i}

Each bosonic state contributes +½ω per mode, each fermionic state −½ω (the fermionic zero-point
energy has the opposite sign because of the anticommutators).  If bosons and fermions come in
degenerate pairs — exact supersymmetry — the sum vanishes *mode by mode*, for any cutoff.  For
the superstring this is Jacobi's "aequatio identica satis abstrusa" applied to the one-loop
partition function: the GSO-projected spectrum has d_B(N) = d_F(N) at every level (Vol.2 ch.10;
`test_supersymmetric_cancellation` verifies the level-by-level cancellation of our tower sums),
and "the order-by-order supersymmetric cancellation of the vacuum energy means that there are no
tadpole divergences" (Vol.2 p.282).

## Broken supersymmetry

Supersymmetry is not observed, so the partners must be heavier: m_F² = m_B² + M_susy².  Then

    ρ₀ ≈ Σ g_i [ ρ(m_B) − ρ(√(m_B² + M²)) ] ≈ −Σ g_i M² Λ²/(16π²) + O(M⁴ ln Λ),

i.e. the quartic divergence cancels but a quadratic one survives, leaving a remainder "of at
least (M_susy)⁴" after renormalisation (Vol.1 p.222).  The panel's "SUSY-breaking scale" slider
shows this: with M_susy = 0 the weighted sum is exactly 0; with M_susy > 0 it is of order
g M² Λ² (negative with a sharp cutoff — the sign is regulator dependent, the magnitude is not).

Sum rules that make the vacuum energy finite and cutoff independent:
Σ(−1)^F g = 0 (removes Λ⁴), Σ(−1)^F g m² = 0 (removes Λ²), Σ(−1)^F g m⁴ = 0 (removes the log);
what remains is Σ(−1)^F g m⁴ ln m²/(64π²).  Broken supersymmetry with a universal mass shift
satisfies the first but not the second.  Vol.2 p.371 shows a no-scale model in which
supersymmetry is broken yet the tree-level vacuum energy is still exactly zero — one of several
mechanisms discussed for the cosmological-constant problem.

## What SUSY does NOT do

Supersymmetry cancels the *mean* energy ⟨0|H|0⟩.  It does not remove the fluctuations of the
bosonic fields: ⟨0|φ(x)²|0⟩ is unchanged, the Casimir force is unchanged (it comes from the
d-dependence of the bosonic and fermionic sums separately, which do not cancel for Dirichlet
bosons vs. their fermionic partners with the appropriate boundary conditions), and a detector
still registers noise.  That is why the simulation keeps rendering the same field when you tick
the SUSY box and only the numbers in the panel change.  Fermionic fields themselves cannot be
sampled as classical configurations (they are Grassmann-valued; their Wigner function is not a
probability density), which is why they enter only through the exact bookkeeping of ½ω sums.

MODIFY: to visualise a fermionic *bilinear* such as ⟨ψ̄ψ⟩(x) fluctuations one would need the
fermionic two-point function on the lattice (Wilson or staggered fermions) — a substantial
extension, and the fermion-doubling problem awaits.
