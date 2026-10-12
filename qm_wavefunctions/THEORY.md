# Real-time wave-function dynamics in one, two and three dimensions by split-operator propagation

## Abstract

We present a compact numerical laboratory for the time-dependent Schrödinger equation (TDSE) of a single
non-relativistic particle in 1, 2 and 3 spatial dimensions.  The propagator is the symmetric split-operator
(Fourier) scheme, which is unitary, second order in the time step and exact for free motion.  Stationary
states are obtained without any analytic input by imaginary-time propagation with Gram–Schmidt
orthogonalisation.  The complex amplitude is represented by phase colouring (brightness = |ψ|, hue = arg ψ),
which makes momentum, interference and superposition beats visible directly.  Eleven scenarios reproduce
textbook results — dispersion of a free packet, coherent and squeezed oscillator states, barrier tunnelling,
double-well oscillation, two-slit interference, a circular coherent orbit, and "sloshing" superpositions of
3D oscillator and hydrogen-like eigenstates — each with a quantitative check against the known result.

## 1. Physical model

A particle of mass m in a potential V(**x**) obeys (ħ = m = 1 throughout)

    i ∂ψ/∂t = H ψ,     H = −½ ∇² + V(**x**),                                                    (1)

and |ψ(**x**,t)|² is the probability density (Born rule).  Equation (1) is linear and norm preserving; its
formal solution ψ(t) = e^{−iHt} ψ(0) is what we compute.  Expectation values ⟨x⟩, ⟨p⟩ follow Newton's
equations for the mean force (Ehrenfest theorem), which is why the harmonic coherent state moves on the
classical trajectory while a free packet spreads — an interference effect between the momentum components
contained in the packet, absent classically.

## 2. Numerical method

**Split-operator propagation.**  H = T + V with T = −½∇² diagonal in momentum space and V diagonal in
position space.  The Strang splitting

    e^{−iH dt} = e^{−iV dt/2} e^{−iT dt} e^{−iV dt/2} + O(dt³)                                  (2)

is applied with two FFTs per step.  Each factor is a pure phase, so the norm is conserved to round-off, the
scheme is time reversible, and for V = 0 it is exact for any dt.  The momentum grid k = 2πn/L imposes
periodic boundaries; boxes are chosen large enough that nothing reaches the edge during a run (the
free-packet scenarios are the only ones that eventually wrap around).  Accuracy check: the 1D coherent state
conserves ⟨E⟩ = 5.0000 to 2·10⁻⁶ over 600 steps and keeps its width 1/√2 to 10⁻⁶ (see `--selftest`
numbers in the README).

**Imaginary time.**  Replacing t → −iτ in (2) turns the propagator into e^{−Hτ}, which damps every component
of a trial function by e^{−E_n τ}; the ground state survives.  Projecting out previously converged states
after each step (Gram–Schmidt) yields excited states in turn.  For the 1D and 3D harmonic oscillator this
reproduces E = 0.5, 1.5, 2.5 and E = 1.5, 2.5, 2.5 to four digits on the default grids.

**Soft-core Coulomb.**  A point −1/r cannot be represented on a Cartesian grid; the standard regularisation
−1/√(r² + a²) with a = 0.4 gives E₀ = −0.386 and E₁ = −0.116 (exact hydrogen: −0.5, −0.125).  The
qualitative physics (bound states, 1s–2p charge oscillation at the Bohr frequency) is unchanged.

**Phase colouring.**  ψ is complex; showing only |ψ|² hides the momentum.  Mapping arg ψ to hue turns a
travelling packet into a rainbow whose stripe spacing is the de Broglie wavelength 2π/k, and a standing
superposition into regions of constant colour that flip sign at nodes.

## 3. Scenarios and what they demonstrate

| scenario | result reproduced |
|---|---|
| `1d_free_spreading` | σ(t) = σ₀√(1 + (t/2σ₀²)²), ⟨x⟩ = x₀ + k₀t; checked to 10⁻¹³ |
| `1d_harmonic_coherent` | Glauber coherent state: rigid oscillation, ⟨E⟩ = ½ + x₀²/2 |
| `1d_harmonic_squeezed` | wrong width ⇒ breathing at 2ω (squeezed state) |
| `1d_tunneling` | E = 1.28 < V₀ = 2: partial transmission; the read-out gives the transmitted probability |
| `1d_double_well` | the packet localised in one well oscillates with period π/(E₁ − E₀) (tunnel splitting) |
| `2d_double_slit` | two-slit interference of a single packet; fringes appear behind the wall |
| `2d_harmonic_orbit` | 2D coherent state on a circular orbit — the classical limit embedded in QM |
| `3d_harmonic_superposition` | |ψ|² of a superposition of eigenstates oscillates at the Bohr frequencies (E_n − E_m) |
| `3d_hydrogenlike` | 1s + 2p-like superposition of a soft-Coulomb atom: the dipole oscillates (this is what radiates) |

## 4. Limitations

Single particle, no spin, no relativity, no measurement/decoherence; the grid box is periodic; the soft-core
Coulomb is a regularisation.  All of these are the conventional choices for an introductory TDSE code
(Feit, Fleck & Steiger 1982; Kosloff 1988), and every quoted number is checked against the analytic result
where one exists.

## References

* M. D. Feit, J. A. Fleck, A. Steiger, *J. Comput. Phys.* **47**, 412 (1982) — split-operator method.
* R. Kosloff, *J. Phys. Chem.* **92**, 2087 (1988) — Fourier grid methods for quantum dynamics.
* D. J. Griffiths, *Introduction to Quantum Mechanics* (3rd ed.), ch. 2 — free packet, oscillator, tunnelling.
* R. J. Glauber, *Phys. Rev.* **131**, 2766 (1963) — coherent states.
* J. Javanainen, J. H. Eberly, Q. Su, *Phys. Rev. A* **38**, 3430 (1988) — soft-core Coulomb potential.
