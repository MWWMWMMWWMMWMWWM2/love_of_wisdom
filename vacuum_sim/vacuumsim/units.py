"""
units.py -- natural units <-> SI conversions.

Everything inside the simulation is in *string units*:
    hbar = c = 1,   length unit = l_s = sqrt(alpha')   (the string length, Polchinski Vol.1 eq. 1.2.?, alpha' = Regge slope)
so a mass m = 2 means "2 / l_s = 2 * M_s" where M_s = hbar c / l_s is the string mass scale.

The physical size of l_s is NOT fixed by the theory alone (it depends on the string coupling and
the compactification), so the user picks it.  Presets:
    * l_s = 1e-34 m      ~ 6 Planck lengths  (classic perturbative heterotic-string expectation)
    * l_s = l_Planck     = 1.616e-35 m
    * l_s = 1e-19 m      ~ TeV-scale strings (large-extra-dimension scenarios)

MODIFY: add presets to STRING_LENGTH_PRESETS; everything else derives from l_s automatically.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

# --- CODATA-2018 style constants (SI) ---------------------------------------------------------
HBAR = 1.054571817e-34      # J s
C = 299792458.0             # m/s
G_NEWTON = 6.67430e-11      # m^3 kg^-1 s^-2
K_B = 1.380649e-23          # J/K
EV = 1.602176634e-19        # J
GEV = 1e9 * EV

L_PLANCK = math.sqrt(HBAR * G_NEWTON / C**3)         # 1.616e-35 m
T_PLANCK = L_PLANCK / C                              # 5.39e-44 s
E_PLANCK = math.sqrt(HBAR * C**5 / G_NEWTON)         # 1.956e9 J  (= 1.22e19 GeV)

# Observed dark-energy (cosmological constant) density, Planck-2018 numbers:
#   rho_crit = 3 H0^2 / (8 pi G) ~ 8.5e-27 kg/m^3 ;  Omega_Lambda ~ 0.69
RHO_LAMBDA_SI = 0.69 * 8.5e-27 * C**2                # ~ 5.3e-10 J/m^3
# Same number in GeV^4 (natural units):  (2.25 meV)^4
RHO_LAMBDA_GEV4 = (2.25e-12) ** 4                    # ~ 2.6e-47 GeV^4

STRING_LENGTH_PRESETS = {
    "l_s = 1e-34 m (~6 l_Planck, perturbative string)": 1e-34,
    "l_s = l_Planck (1.6e-35 m)": L_PLANCK,
    "l_s = 1e-19 m (TeV-scale strings)": 1e-19,
    "l_s = 1e-18 m (100 GeV strings, for the '52 orders' comparison)": 1e-18,
}


@dataclass
class Units:
    """Conversion helper. All simulation quantities are dimensionless multiples of these."""
    l_s: float = 1e-34  # metres

    # --- base units -------------------------------------------------------------------------
    @property
    def length(self) -> float:        # m
        return self.l_s

    @property
    def time(self) -> float:          # s
        return self.l_s / C

    @property
    def energy(self) -> float:        # J  (string mass scale M_s c^2)
        return HBAR * C / self.l_s

    @property
    def energy_GeV(self) -> float:
        return self.energy / GEV

    @property
    def energy_density(self) -> float:  # J/m^3
        return self.energy / self.l_s**3

    @property
    def energy_density_GeV4(self) -> float:
        # (hbar c / l_s)^4 / (hbar c)^3 = E^4/(hbar c)^3 ; in natural units simply E_GeV^4
        return self.energy_GeV**4

    @property
    def efield(self) -> float:        # V/m for unit "e*E" in natural units with e = sqrt(4 pi alpha)
        # In natural units the combination eE has dimension mass^2.  A field strength of 1 (string
        # units) means eE = M_s^2.  E_SI = eE * (M_s c^2)^2 / (hbar c e_SI)  with e = electron charge.
        return (self.energy) ** 2 / (HBAR * C * EV)

    # --- helpers ----------------------------------------------------------------------------
    def describe_length(self, x_sim: float) -> str:
        return f"{x_sim:.3g} l_s = {x_sim*self.length:.3e} m"

    def describe_time(self, t_sim: float) -> str:
        return f"{t_sim:.3g} l_s/c = {t_sim*self.time:.3e} s"

    def describe_energy_density(self, rho_sim: float) -> str:
        si = rho_sim * self.energy_density
        gev4 = rho_sim * self.energy_density_GeV4
        ratio = si / RHO_LAMBDA_SI if si != 0 else 0.0
        if ratio > 0:
            return (f"{rho_sim:.4g} M_s^4 = {si:.3e} J/m^3 = {gev4:.3e} GeV^4  "
                    f"(10^{math.log10(ratio):.1f} x observed dark energy)")
        return f"{rho_sim:.4g} M_s^4 = {si:.3e} J/m^3 = {gev4:.3e} GeV^4"

    def unruh_temperature_K(self, accel_sim: float) -> float:
        """Unruh temperature for proper acceleration `accel_sim` in string units (c^2/l_s).
        T_U = hbar a / (2 pi c k_B)."""
        a_si = accel_sim * C**2 / self.l_s
        return HBAR * a_si / (2 * math.pi * C * K_B)


def electroweak_cutoff_estimate() -> dict:
    """Polchinski Vol.1 p.222: zero-point energy of the known particles up to the electroweak scale
    is already ~52 orders of magnitude above the observed cosmological constant.
    One massless boson with sharp cutoff Lambda:  rho = Lambda^4 / (16 pi^2)."""
    Lambda_GeV = 100.0
    rho = Lambda_GeV**4 / (16 * math.pi**2)
    return {"Lambda_GeV": Lambda_GeV, "rho_GeV4": rho,
            "orders_above_observed": math.log10(rho / RHO_LAMBDA_GEV4)}
