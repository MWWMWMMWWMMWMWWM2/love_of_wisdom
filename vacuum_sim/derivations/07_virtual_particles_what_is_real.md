# 07 — "Virtual particles": what the pictures mean and what the simulation shows

## The popular picture and the actual statement

"Particle–antiparticle pairs pop in and out of existence" is a cartoon of perturbation theory:
internal lines of Feynman diagrams ("virtual particles") are integration variables, not events.
The rigorous statements are:

1. The vacuum |0⟩ is a stationary state: ⟨0|H|0⟩ = E₀ is constant, and no observable's
   expectation value depends on time.  Nothing "happens" in the vacuum.
2. Nevertheless the field is not zero: ⟨0|φ(x)²|0⟩ > 0 and, for a detector of finite resolution
   σ, the smeared field φ_σ has finite, nonzero variance ∼ 1/σ² (massless).  A detector coupled
   to the field registers noise — Lamb shift, spontaneous emission and the Casimir force are its
   measurable consequences (Milonni, *The Quantum Vacuum*; Vol.1 p.222 counts the zero-point
   energy of the known fields as real for gravity).
3. The vacuum is Lorentz invariant.  An inertial observer moving at any speed sees the same
   statistics and no particles.  An *accelerated* observer, however, sees a thermal bath at
   T = ħa/(2πck_B) (Unruh 1976) — "particle" is observer dependent.  The panel converts your
   chosen acceleration into that temperature; setting the simulation temperature to the same
   value and re-sampling shows what that observer would see.

## How this maps onto the simulation

* The rendered field is one exact history of the vacuum state (file 01).  Its "boiling" is real in
  the sense of (2): it is what a probe would register.  It is *not* a movie of particles being
  created, because in the undisturbed vacuum none are (the exact particle-number read-out stays
  at zero while the picture boils).
* Real particles appear only when the vacuum is disturbed, and the program contains the three
  textbook mechanisms, all treated exactly:
  - a parameter of the field changes in time (mass quench / ramp; cosmological expansion is the
    same mathematics) — file 03;
  - an external field strong enough to pay 2mc² over a Compton wavelength (Schwinger) — file 04;
  - boundary conditions change in time (dynamical Casimir effect) — file 05.
  Hawking radiation is the horizon version of the same Bogoliubov story and is not included
  (it needs a curved background).
* "Dematerialisation" (annihilation): in a free theory created particles never disappear.  What
  *can* happen is that a second, reversed disturbance undoes the first — quench the mass back
  after exactly a half-period of a mode and that mode's occupation returns to zero (the
  Bogoliubov map is invertible).  Try: quench m: 0 → 1, wait, quench back to 0 — the occupation
  numbers oscillate rather than simply add.  Genuine annihilation into other species requires
  interactions, which are outside the exactly solvable framework (README, "limitations").

## Why the Standard Model cannot be done this way

The Standard Model is an interacting theory.  Its vacuum is not Gaussian (gluon condensates,
chiral condensate, confinement), its fields cannot be sampled as classical configurations
(fermions, non-abelian gauge constraints), and the only first-principles numerical method —
Euclidean lattice QCD — computes imaginary-time correlators, not real-time movies.  What this
program does is the exactly solvable part that every one of those theories contains: the free
vacuum, its fluctuations, and linear-response particle creation.  Everything it shows is a
theorem of that free theory; nothing is an approximation of the interacting one.
