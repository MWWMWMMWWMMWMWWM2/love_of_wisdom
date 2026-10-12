"""
observables.py -- closed-form reference results the simulation is checked against / annotated with.

References (page numbers are to Polchinski, String Theory Vol.1 unless noted):
  * Vacuum energy density = sum of zero-point energies of the field modes: sec. 7.3, pp. 220-221,
    eqs. (7.3.18)-(7.3.22); Coleman-Weinberg formula (7.3.23); arbitrary spin with (-1)^F (7.3.24).
  * Divergence structure and why the string torus amplitude has no UV region: pp. 219-220.
  * Cosmological-constant problem, "52 orders of magnitude": p. 222.
  * Casimir energy from a finite-size system: sec. 1.3 p. 22 and sec. 2.6 p. 73 (world-sheet
    version); spacetime plates are standard QFT (Milonni, The Quantum Vacuum, ch. 2-3).
  * Schwinger pair production: Schwinger, Phys. Rev. 82, 664 (1951) -- not in Polchinski.
  * Sudden-quench Bogoliubov coefficients: Birrell & Davies, QFT in Curved Space, sec. 3.4.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import exp1, kv

# ----------------------------------------------------------------------------------------------
# Zero-point energy densities of one real scalar field (per state; multiply by multiplicities)
# ----------------------------------------------------------------------------------------------

def zero_point_density_lattice(mass: float, geom, nosc: int = 1) -> float:
    """(1/V) sum_modes 1/2 omega  -- exactly what the sampled field's energy density averages to."""
    tot = 0.0
    for b in geom.blocks:
        tot += 0.5 * np.sqrt(mass ** 2 + b.k2_spatial(0.0)).sum()
    return nosc * tot / geom.lat.volume


def zero_point_density_sharp_cutoff(mass: float, Lambda: float) -> float:
    """Continuum, 3 spatial dims, sharp momentum cutoff |k| < Lambda:
         rho = 1/(4 pi^2) int_0^Lambda k^2 sqrt(k^2+m^2) dk
             = 1/(32 pi^2) [ Lambda (2 Lambda^2 + m^2) sqrt(Lambda^2+m^2) - m^4 asinh(Lambda/m) ]
       ~ Lambda^4/(16 pi^2) + m^2 Lambda^2/(16 pi^2) - m^4 ln(2 Lambda/m)/(64 pi^2) + ...   (eq. 7.3.22-type)"""
    L, m = float(Lambda), float(mass)
    if m == 0.0:
        return L ** 4 / (16 * math.pi ** 2)
    return (L * (2 * L * L + m * m) * math.sqrt(L * L + m * m) - m ** 4 * math.asinh(L / m)) / (32 * math.pi ** 2)


def coleman_weinberg_finite(mass: float, mu: float = 1.0) -> float:
    """Regulator-independent (log) part of the one-loop vacuum energy of one real scalar,
    rho_CW = m^4/(64 pi^2) ln(m^2/mu^2)  (Coleman & Weinberg 1973; Vol.1 eq. 7.3.23 up to scheme
    constants).  In a theory obeying the supertrace sum rules  sum (-1)^F g = sum (-1)^F g m^2 =
    sum (-1)^F g m^4 = 0  this is the *whole* answer and it is cutoff independent."""
    m = float(mass)
    if m <= 0:
        return 0.0
    return m ** 4 / (64 * math.pi ** 2) * math.log(m * m / (mu * mu))


def proper_time_regulated_density(mass: float, eps: float) -> float:
    """Schwinger proper-time regularisation, cut off at l >= eps (Polchinski's "rough insight"
    prescription, p. 220):  rho(m) = -1/(32 pi^2) int_eps^inf dl l^-3 e^{-m^2 l}.
    Note the sign: with this regulator the power divergences come out negative; only the m^4 log
    term is scheme independent.  Returned in units where eps has dimension length^2 (alpha')."""
    m2 = float(mass) ** 2
    if m2 == 0.0:
        return -1.0 / (64 * math.pi ** 2 * eps ** 2)
    x = m2 * eps
    # Gamma(-2, x) via recursion from E1(x): Gamma(-1,x) = e^-x/x - E1(x);
    # Gamma(-2,x) = (e^-x/x^2 - Gamma(-1,x)) / 2
    g_m1 = math.exp(-x) / x - float(exp1(x))
    g_m2 = (math.exp(-x) / x ** 2 - g_m1) / 2.0
    return -m2 ** 2 * g_m2 / (32 * math.pi ** 2)


def supertrace_tower(levels, regulator, **kw) -> dict:
    """sum_N (-1)^F g_N rho(m_N)  over a string tower (Vol.1 eq. 7.3.24 applied to the string
    spectrum).  `regulator` is one of the density functions above.  Tachyonic levels are skipped
    and reported (their contribution is complex/divergent: p. 219)."""
    total_b = total_f = 0.0
    skipped = []
    for lv in levels:
        if lv.tachyonic:
            skipped.append(lv.N)
            continue
        r = regulator(lv.mass, **kw)
        total_b += lv.bosons * r
        total_f += lv.fermions * r
    return {"bosons": total_b, "fermions": total_f, "net": total_b - total_f, "skipped_tachyon_levels": skipped}


# ----------------------------------------------------------------------------------------------
# Casimir effect (massless real scalar, Dirichlet plates, continuum reference)
# ----------------------------------------------------------------------------------------------

def casimir_energy_per_area_continuum(d: float, mass: float = 0.0, nmax: int = 60) -> float:
    """One real scalar between Dirichlet plates a distance d apart (continuum, per unit area):
         massless:  E/A = - pi^2 / (1440 d^3)      (half the electromagnetic value -pi^2/(720 d^3))
         massive :  E/A = - m^2/(8 pi^2 d) sum_{n>=1} K_2(2 n m d) / n^2   -> exponentially small for m d >> 1
       (Milonni, The Quantum Vacuum; Bordag, Mohideen & Mostepanenko, Phys. Rep. 353 (2001))."""
    if mass <= 0.0:
        return -math.pi ** 2 / (1440.0 * d ** 3)
    x = 2.0 * mass * d
    if x > 600:
        return 0.0
    return -(mass ** 2 / (8 * math.pi ** 2 * d)) * sum(float(kv(2, n * x)) / n ** 2 for n in range(1, nmax + 1))


def casimir_pressure_continuum(d: float) -> float:
    """F/A = -dE/dd / A = - pi^2 / (480 d^4)  (attractive)."""
    return -math.pi ** 2 / (480.0 * d ** 4)


def lattice_zero_point_energy(mass: float, geom, nosc: int = 1) -> float:
    return nosc * sum(0.5 * np.sqrt(mass ** 2 + b.k2_spatial(0.0)).sum() for b in geom.blocks)


# ----------------------------------------------------------------------------------------------
# Particle creation reference formulae
# ----------------------------------------------------------------------------------------------

def sudden_quench_occupation(omega0, omega1):
    """|beta_k|^2 = (omega1 - omega0)^2 / (4 omega0 omega1) for an instantaneous frequency jump."""
    return (omega1 - omega0) ** 2 / (4.0 * omega0 * omega1)


def sauter_single_mode_probability(m_perp2: float, eE: float) -> float:
    """Exact asymptotic pair number per mode for a constant field sweeping a mode through
    k_parallel = 0:  |beta|^2 = exp(-pi m_perp^2 / (eE)),  m_perp^2 = m^2 + k_perp^2  (scalar QED)."""
    return math.exp(-math.pi * m_perp2 / eE)


def schwinger_rate_scalar(mass: float, eE: float, nterms: int = 20) -> float:
    """Pairs per unit volume per unit time for scalar QED in a constant field:
         Gamma = (eE)^2/(8 pi^3) sum_{n>=1} (-1)^{n+1}/n^2 exp(-n pi m^2/(eE))
       (spinor QED: (eE)^2/(4 pi^3) sum 1/n^2 exp(...) )."""
    if eE <= 0:
        return 0.0
    s = 0.0
    for n in range(1, nterms + 1):
        s += (-1) ** (n + 1) / n ** 2 * math.exp(-n * math.pi * mass ** 2 / eE)
    return eE ** 2 / (8 * math.pi ** 3) * s


def schwinger_critical_field(mass: float, e: float) -> float:
    """E_c = m^2 / e  (in natural units; for the electron 1.3e18 V/m)."""
    return mass ** 2 / e if e > 0 else float("inf")


def unruh_temperature_natural(accel: float) -> float:
    """T = a / (2 pi)  (hbar = c = k_B = 1)."""
    return accel / (2 * math.pi)
