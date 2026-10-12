"""
field.py -- one quantum field on the lattice: exact vacuum sampling + exact linear evolution.

THE CENTRAL IDEA (see derivations/01_vacuum_sampling_wigner.md)
---------------------------------------------------------------
A free field is a collection of independent harmonic oscillators, one per mode k.  The vacuum
|0> is the product of their ground states.  The ground state of an oscillator is a Gaussian, and
its Wigner function W(q, p) is a *positive* Gaussian with
        <q^2> = hbar / (2 omega),     <p^2> = hbar omega / 2,     <q p + p q>/2 = 0.
For a quadratic Hamiltonian the quantum evolution of the Wigner function is *exactly* the
classical Liouville flow (the Moyal bracket reduces to the Poisson bracket).  Therefore:

    drawing (q_k, p_k) from these Gaussians and evolving them with the classical equations of
    motion produces an ensemble whose statistics reproduce ALL symmetrically-ordered vacuum
    correlation functions exactly, at all times, including after any *linear* perturbation
    (time-dependent masses, external electric fields, moving perfect mirrors).

What we render is one member of that ensemble -- one "history" of the vacuum -- exactly what a
finite-resolution probe coupled to the field would register.  The ensemble means (zero-point
energy density, created-particle numbers) are tracked separately and exactly through the
covariance of every mode (self.Cqq, self.Cpp, self.Cqp), so the numbers in the read-out panel do
not suffer from sampling noise.

Normalisation used everywhere (hbar = 1):
        Q_k = sqrt(a^3) * (orthonormal transform of phi)_k      <|Q_k|^2>_vac = 1/(2 omega_k)
        P_k = sqrt(a^3) * (orthonormal transform of pi )_k      <|P_k|^2>_vac = omega_k/2
        H = f * sum_k (|P_k|^2 + omega_k^2 |Q_k|^2),  f = 1/2 (real field), 1 (complex field)
Each coefficient of a real field carries one oscillator (1/2 omega of zero-point energy); each
coefficient of a complex field carries two (particle + antiparticle), i.e. nosc = 1 or 2.

Particle number per coefficient, relative to the instantaneous vacuum (exact, Bogoliubov):
        n_k = (Cpp + omega^2 Cqq) / (2 omega) - 1/2                                (eq. F1)

MODIFY:
 * Thermal initial state: multiply the vacuum variances by (2 n_BE(omega) + 1) in sample_vacuum
   and reset_covariance (n_BE = 1/(e^{omega/T} - 1)).  Everything else is unchanged.
 * Squeezed / coherent initial states: add a mean (coherent) or unequal q/p variances (squeezed).
 * An interacting field (phi^4 etc.) breaks the exactness: you would need a real-space leapfrog
   step instead of the mode rotation (see evolve()).  The classical-statistical approximation is
   then only valid for highly occupied modes -- not for the vacuum itself.
"""
from __future__ import annotations
from dataclasses import dataclass, field as dc_field
import numpy as np

from .geometry import Geometry, Block, z_mixing_matrix


@dataclass
class FieldSpec:
    name: str
    mass: float                      # in units 1/l_s  (M_s)
    is_complex: bool = False          # charged complex scalar (two real components)
    charge: float = 0.0              # coupling e (dimensionless in natural units); used if is_complex
    multiplicity: int = 1            # how many physical states this field stands for (read-outs only)
    fermion_partners: int = 0        # SUSY partners of the same mass (read-outs only, (-1)^F sums)
    level: int = 0                   # string level N (informational)
    theory: str = ""                 # informational
    color: tuple = (0.9, 0.9, 0.9)

    @property
    def nosc(self) -> int:
        return 2 if self.is_complex else 1

    @property
    def f(self) -> float:
        return 1.0 if self.is_complex else 0.5


@dataclass
class BlockState:
    block: Block
    Q: np.ndarray                    # complex coefficients (sample)
    P: np.ndarray
    Cqq: np.ndarray                  # ensemble covariance (exact bookkeeping), real arrays
    Cpp: np.ndarray
    Cqp: np.ndarray
    omega: np.ndarray = dc_field(default=None)


class FieldState:
    def __init__(self, spec: FieldSpec, geom: Geometry, rng: np.random.Generator):
        self.spec = spec
        self.geom = geom
        self.rng = rng
        self.mass = float(spec.mass)          # current (possibly quenched) mass
        self.kshift = 0.0                     # kinetic-momentum shift  -e A_x(t)  (charged fields)
        self.blocks: list[BlockState] = []
        self._build_blocks()
        self.sample_vacuum()

    # ------------------------------------------------------------------------------------------
    def _omega(self, block: Block) -> np.ndarray:
        ks = self.kshift if self.spec.is_complex else 0.0
        return np.sqrt(self.mass ** 2 + block.k2_spatial(ks))

    def _build_blocks(self):
        self.blocks = []
        for b in self.geom.blocks:
            om = self._omega(b)
            z = np.zeros(b.shape)
            self.blocks.append(BlockState(b, np.zeros(b.shape, complex), np.zeros(b.shape, complex),
                                          z.copy(), z.copy(), z.copy(), om))

    def refresh_omega(self):
        for bs in self.blocks:
            bs.omega = self._omega(bs.block)

    @staticmethod
    def _inv(om: np.ndarray) -> np.ndarray:
        """1/omega with the massless zero mode (omega = 0, the spatially constant mode of a
        massless field in a periodic box) excluded.  That mode is a free particle without a
        normalisable ground state; standard practice is to drop it (zero-mode subtraction)."""
        with np.errstate(divide="ignore"):
            return np.where(om > 0, 1.0 / np.where(om > 0, om, 1.0), 0.0)

    # ------------------------------------------------------------------------------------------
    def _white_noise_coeffs(self, block: Block) -> np.ndarray:
        """Orthonormal-transform coefficients of unit white noise on the block's slab.
        Real field: real noise -> Hermitian-symmetric coefficients, <|Xi|^2> = 1.
        Complex field: complex noise (xi1 + i xi2)/sqrt2 -> independent coefficients, <|Xi|^2> = 1."""
        lat = self.geom.lat
        n_layers = len(block.zindex)
        if self.spec.is_complex:
            xi = (self.rng.standard_normal((lat.N, lat.N, n_layers))
                  + 1j * self.rng.standard_normal((lat.N, lat.N, n_layers))) / np.sqrt(2.0)
        else:
            xi = self.rng.standard_normal((lat.N, lat.N, n_layers))
        return block.forward(xi)

    def sample_vacuum(self, temperature: float = 0.0):
        """Draw a fresh vacuum (or thermal, T>0) configuration and reset the exact covariance."""
        for bs in self.blocks:
            om = bs.omega
            boost = self._thermal_factor(om, temperature)
            bs.Q = self._white_noise_coeffs(bs.block) * np.sqrt(0.5 * boost * self._inv(om))
            bs.P = self._white_noise_coeffs(bs.block) * np.sqrt(boost * om / 2.0)
        self.reset_covariance(temperature)

    @staticmethod
    def _thermal_factor(om, T):
        if T <= 0:
            return np.ones_like(om)
        with np.errstate(over="ignore", divide="ignore"):
            x = om / T
            fac = 1.0 / np.tanh(0.5 * x)      # 2 n_BE + 1
        return np.where(np.isfinite(fac), fac, 1.0)

    def reset_covariance(self, temperature: float = 0.0):
        """Declare the *current instantaneous vacuum* as the ensemble state (n_k = 0)."""
        for bs in self.blocks:
            boost = self._thermal_factor(bs.omega, temperature)
            bs.Cqq = 0.5 * boost * self._inv(bs.omega)
            bs.Cpp = boost * bs.omega / 2.0
            bs.Cqp = np.zeros_like(bs.omega)

    # ------------------------------------------------------------------------------------------
    def evolve(self, dt: float):
        """Exact rotation of every mode by its (current) frequency.  Exact for constant omega;
        for time-dependent omega the caller sub-steps (frequency frozen within a sub-step =
        2nd-order symplectic 'exponential midpoint' scheme when omega is evaluated mid-step)."""
        for bs in self.blocks:
            om = bs.omega
            x = om * dt
            c = np.cos(x)
            s = np.sin(x)
            sig = dt * np.sinc(x / np.pi)          # sin(om dt)/om, safe at om -> 0
            Q, P = bs.Q, bs.P
            bs.Q = c * Q + sig * P
            bs.P = -om * s * Q + c * P
            # covariance:  C' = R C R^T  with R = [[c, sig], [-om s, c]]
            Cqq, Cpp, Cqp = bs.Cqq, bs.Cpp, bs.Cqp
            bs.Cqq = c * c * Cqq + 2 * c * sig * Cqp + sig * sig * Cpp
            bs.Cpp = (om * s) ** 2 * Cqq - 2 * om * s * c * Cqp + c * c * Cpp
            bs.Cqp = -c * om * s * Cqq + (c * c - s * s) * Cqp + c * sig * Cpp

    # ------------------------------------------------------------------------------------------
    def to_grid(self, smoothing: float = 0.0):
        """Return (phi, pi) on the full grid (real arrays for real fields, complex otherwise).
        `smoothing` sigma (in l_s): Gaussian low-pass exp(-k^2 sigma^2 / 2) applied in mode space =
        what a probe of finite resolution sigma would see (the energy density is then that of the
        smeared field)."""
        a3 = self.geom.lat.a ** 3
        ks = self.kshift if self.spec.is_complex else 0.0
        slabs_q, slabs_p = [], []
        for bs in self.blocks:
            Q, P = bs.Q, bs.P
            if smoothing > 0:
                filt = np.exp(-0.5 * bs.block.k2_spatial(ks) * smoothing ** 2)
                Q = Q * filt
                P = P * filt
            slabs_q.append(bs.block.inverse(Q))
            slabs_p.append(bs.block.inverse(P))
        phi = self.geom.assemble(slabs_q) / np.sqrt(a3)
        pi = self.geom.assemble(slabs_p) / np.sqrt(a3)
        if not self.spec.is_complex:
            phi, pi = phi.real, pi.real
        return phi, pi

    def energy_density(self, phi: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """Classical energy density of the configuration (per site, energy per unit volume).
        Real:    1/2 pi^2 + 1/2 m^2 phi^2 + 1/2 |grad phi|^2   (link energies split half/half)
        Complex: |pi|^2 + m^2 |phi|^2 + |D phi|^2  with the lattice gauge link e^{i kshift a} in x.
        Summing a^3 * h over the grid reproduces sum_k f (|P|^2 + omega^2 |Q|^2) exactly."""
        a = self.geom.lat.a
        f = self.spec.f
        h = np.abs(pi) ** 2 + self.mass ** 2 * np.abs(phi) ** 2
        for axis in range(3):
            nb = np.roll(phi, -1, axis=axis)               # phi(x + a e_axis)
            if axis == 0 and self.spec.is_complex and self.kshift != 0.0:
                nb = nb * np.exp(1j * self.kshift * a)      # covariant difference
            e_link = np.abs(nb - phi) ** 2 / a ** 2        # link energy density, stored at site x
            h += 0.5 * (e_link + np.roll(e_link, 1, axis=axis))
        return f * h

    # ------------------------------------------------------------------------------------------
    def total_energy_sample(self) -> float:
        """H of the sampled configuration from the mode data (cheap, exact)."""
        e = 0.0
        for bs in self.blocks:
            e += np.sum(np.abs(bs.P) ** 2 + bs.omega ** 2 * np.abs(bs.Q) ** 2)
        return float(self.spec.f * e)

    def mode_occupations(self):
        """Exact ensemble occupation per coefficient relative to the instantaneous vacuum (eq. F1)."""
        out = []
        for bs in self.blocks:
            inv = self._inv(bs.omega)
            n = 0.5 * (bs.Cpp + bs.omega ** 2 * bs.Cqq) * inv - 0.5
            # roundoff from the rotation accumulates at ~1e-15 per step; below 1e-9 it is not physics
            out.append(np.where((bs.omega > 0) & (n > 1e-9), n, 0.0))
        return out

    def particle_numbers(self) -> dict:
        """Ensemble-exact totals: quanta, particles(=pairs for charged), energies, densities."""
        n_tot = 0.0
        e_part = 0.0
        e_zp = 0.0
        for bs, n in zip(self.blocks, self.mode_occupations()):
            n = np.clip(n, 0.0, None)            # roundoff guard
            n_tot += n.sum()
            e_part += (bs.omega * n).sum()
            e_zp += 0.5 * bs.omega.sum()
        V = self.geom.lat.volume
        nosc = self.spec.nosc
        return {
            "quanta": nosc * n_tot,                       # total number of particles (+antiparticles)
            "pairs": n_tot if self.spec.is_complex else None,
            "particle_energy": nosc * e_part,
            "particle_energy_density": nosc * e_part / V,
            "number_density": nosc * n_tot / V,
            "zero_point_energy": nosc * e_zp,             # sum over modes of 1/2 omega (this field)
            "zero_point_density": nosc * e_zp / V,
            "ensemble_energy_density": nosc * (e_zp + e_part) / V,
        }

    def zero_point_energy(self) -> float:
        return float(self.spec.nosc * sum(0.5 * bs.omega.sum() for bs in self.blocks))

    # ------------------------------------------------------------------------------------------
    def ensemble_energy_profile(self, vacuum: bool = False) -> np.ndarray:
        """Exact ensemble-mean energy density as a function of z (uniform in x, y), length N.
        vacuum=True: use the instantaneous-vacuum covariance instead of the tracked state.
        Includes the plate layers (their link energies).  Used for the Casimir energy-density
        profile between the plates and for the 'excess over vacuum' rendering."""
        lat = self.geom.lat
        N, a = lat.N, lat.a
        f = self.spec.f
        prof = np.zeros(N)
        link_after = np.zeros(N)       # energy of the link z -> z+1 (stored at z)
        norm = 1.0 / (N * N * a ** 3)
        ks = self.kshift if self.spec.is_complex else 0.0
        for bs in self.blocks:
            b = bs.block
            S = b.z_basis()                         # (n_modes_z, n_layers)
            S2 = np.abs(S) ** 2
            Cqq = (0.5 * self._inv(bs.omega)) if vacuum else bs.Cqq
            Cpp = (bs.omega / 2.0) if vacuum else bs.Cpp
            # transverse lattice k^2 (x,y only) per coefficient
            k2xy = b.k2_transverse(ks)
            # sums over transverse momenta -> arrays over the z-mode index
            sq = Cqq.sum(axis=(0, 1))               # sum_{kperp} Cqq
            sp = Cpp.sum(axis=(0, 1))
            sg = (Cqq * k2xy).sum(axis=(0, 1))      # sum_{kperp} Cqq (kx^2 + ky^2)_lattice
            phi2 = norm * (sq @ S2)                 # <|phi|^2>(z_local)
            pi2 = norm * (sp @ S2)
            gradT = norm * (sg @ S2)
            zi = b.zindex
            prof[zi] += f * (pi2 + self.mass ** 2 * phi2 + gradT)
            # z-links: pad chain with the Dirichlet zeros (or wrap for periodic)
            if hasattr(b, "n"):                     # Dirichlet chain
                Spad = np.concatenate([np.zeros((S.shape[0], 1)), S, np.zeros((S.shape[0], 1))], axis=1)
                dS2 = np.abs(np.diff(Spad, axis=1)) ** 2          # (n_modes, n+1 links)
                e_links = norm * (sq @ dS2) / a ** 2              # link energies along the chain
                # link i connects chain position i-1 -> i  (position 0 = left plate, n+1 = right plate)
                left_plate = (zi[0] - 1) % N
                layers = np.concatenate([[left_plate], zi])       # sites where links start
                link_after[layers] += f * e_links
            else:                                                  # periodic chain
                dS2 = np.abs(np.roll(S, -1, axis=1) - S) ** 2
                e_links = norm * (sq @ dS2) / a ** 2
                link_after[zi] += f * e_links
        # each site gets half of the link before it and half of the link after it
        prof += 0.5 * (link_after + np.roll(link_after, 1))
        return prof

    # ------------------------------------------------------------------------------------------
    def set_geometry(self, new_geom: Geometry, exact_transfer: bool = True):
        """Sudden change of boundary conditions (insert / remove / move perfect mirrors).
        Sample: project the current real-space configuration onto the new modes (plates clamp
        phi = pi = 0).  Ensemble covariance: transferred with the z-mixing matrices
        (exact if the old modes were uncorrelated, i.e. for the first change after a reset)."""
        phi, pi = self.to_grid()
        old_blocks = self.blocks
        old_geom = self.geom
        self.geom = new_geom
        a3 = new_geom.lat.a ** 3
        mask = new_geom.plate_mask()
        phi = np.where(mask, 0.0, phi)
        pi = np.where(mask, 0.0, pi)
        self._build_blocks()
        for bs in self.blocks:
            b = bs.block
            bs.Q = b.forward(new_geom.slab(phi, b)) * np.sqrt(a3)
            bs.P = b.forward(new_geom.slab(pi, b)) * np.sqrt(a3)
            if exact_transfer:
                Cqq = np.zeros(b.shape)
                Cpp = np.zeros(b.shape)
                Cqp = np.zeros(b.shape)
                for ob in old_blocks:
                    M2 = np.abs(z_mixing_matrix(b, ob.block)) ** 2        # (n_new, n_old)
                    Cqq += np.einsum("ji,xyi->xyj", M2, ob.Cqq)
                    Cpp += np.einsum("ji,xyi->xyj", M2, ob.Cpp)
                    Cqp += np.einsum("ji,xyi->xyj", M2, ob.Cqp)
                bs.Cqq, bs.Cpp, bs.Cqp = Cqq, Cpp, Cqp
            else:
                bs.Cqq, bs.Cpp, bs.Cqp = 0.5 * self._inv(bs.omega), bs.omega / 2.0, np.zeros(b.shape)
        return old_geom

    # ------------------------------------------------------------------------------------------
    def state_arrays(self) -> dict:
        d = {"mass": self.mass, "kshift": self.kshift}
        for i, bs in enumerate(self.blocks):
            for nm in ("Q", "P", "Cqq", "Cpp", "Cqp"):
                d[f"b{i}_{nm}"] = getattr(bs, nm)
        return d

    def load_state_arrays(self, d: dict):
        self.mass = float(d["mass"])
        self.kshift = float(d["kshift"])
        self.refresh_omega()
        for i, bs in enumerate(self.blocks):
            for nm in ("Q", "P", "Cqq", "Cpp", "Cqp"):
                setattr(bs, nm, np.array(d[f"b{i}_{nm}"]))
