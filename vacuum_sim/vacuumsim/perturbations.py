"""
perturbations.py -- time schedules for the linear perturbations of the vacuum.

All perturbations here keep the Hamiltonian quadratic, so the simulation stays *exact*
(see field.py).  They are the honest mechanisms by which vacuum fluctuations become real particles:

  * MassRamp   : m(t) changes -> Bogoliubov mixing of positive/negative frequencies.
                 Sudden (tau=0):  n_k = (w1-w0)^2 / (4 w0 w1)  exactly.  Slow ramp: adiabatic, n_k -> 0.
                 (Physically: a scalar whose mass comes from a rolling modulus / Higgs-like field;
                 or the effective mass of a field in an expanding universe -> cosmological creation.)
  * EFieldPulse: uniform electric field E(t) along x acting on a charged field.  In the temporal
                 gauge A_x = -int E dt every Fourier mode is a parametric oscillator with
                 omega_k(t)^2 = m^2 + k_perp^2 + (k_x + e int E dt)^2 (lattice version in geometry.py).
                 Sweeping k_x through 0 creates pairs: the Schwinger effect, exactly (scalar QED,
                 no back-reaction).  The pulse is flat-topped with raised-cosine ramps.
  * Plates     : handled in simulation.py (geometry change), i.e. the dynamical Casimir effect in
                 the sudden limit.

MODIFY:
 * Sauter pulse E(t) = E0 sech^2(t/tau): replace EFieldPulse.window.
 * Oscillating mirror (the experimentally observed dynamical Casimir effect, Wilson et al. 2011):
   move a plate periodically -- each move is a sudden geometry change in this framework.
 * Expanding universe: scale factor a(t) multiplying the lattice spacing -> omega_k(t) = k/a(t).
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class MassRamp:
    field_index: int
    m0: float
    m1: float
    t0: float
    duration: float = 0.0        # 0 -> sudden quench
    smooth: bool = True

    def mass(self, t: float) -> float:
        if self.duration <= 0.0 or t >= self.t0 + self.duration:
            return self.m1 if t >= self.t0 else self.m0
        if t <= self.t0:
            return self.m0
        s = (t - self.t0) / self.duration
        if self.smooth:
            s = 0.5 - 0.5 * math.cos(math.pi * s)
        return self.m0 + (self.m1 - self.m0) * s

    def active(self, t: float) -> bool:
        return self.t0 <= t < self.t0 + max(self.duration, 0.0)

    def finished(self, t: float) -> bool:
        return t >= self.t0 + max(self.duration, 0.0)


@dataclass
class EFieldPulse:
    E0: float                    # field strength (natural units; eE0 has dimension mass^2)
    t0: float
    duration: float
    ramp_fraction: float = 0.15  # fraction of the duration spent ramping up (and down)

    def window(self, t: float) -> float:
        if t <= self.t0 or t >= self.t0 + self.duration:
            return 0.0
        s = (t - self.t0) / self.duration
        r = min(max(self.ramp_fraction, 1e-6), 0.5)
        if s < r:
            return 0.5 - 0.5 * math.cos(math.pi * s / r)
        if s > 1.0 - r:
            return 0.5 - 0.5 * math.cos(math.pi * (1.0 - s) / r)
        return 1.0

    def E(self, t: float) -> float:
        return self.E0 * self.window(t)

    def active(self, t: float) -> bool:
        return self.t0 <= t < self.t0 + self.duration

    def finished(self, t: float) -> bool:
        return t >= self.t0 + self.duration

    @property
    def effective_duration(self) -> float:
        """int window dt  (flat top + two half ramps)."""
        r = min(max(self.ramp_fraction, 1e-6), 0.5)
        return self.duration * (1.0 - r)

    def total_sweep(self, e: float) -> float:
        """Total kinetic-momentum shift e * int E dt."""
        return e * self.E0 * self.effective_duration
