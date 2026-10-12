"""
simulation.py -- orchestrates fields, time, perturbations and read-outs (no GUI code here).

The GUI (vacuumsim/gui) and the headless renderer (run_headless.py) both drive this class, so
anything you change here shows up in both.

Default field content = a small slice of the string tower (spectrum.py):
    * level-0 massless neutral scalar   (think: dilaton / a modulus -- one polarisation)
    * level-0 massless-or-light CHARGED scalar (open-string sector, carries the Chan-Paton U(1)
      charge, Vol.1 sec. 6.5) -- this is the field the Schwinger electric field acts on
    * level-1 and level-2 massive neutral scalars (m = 2 and 2 sqrt2 for the closed superstring)
Each sampled field stands for `multiplicity` physical states of that level in the read-outs.

MODIFY:
 * More/other fields: edit default_specs() or pass your own list of FieldSpec.
 * Different lattice: Simulation(N=..., a=...).  Cost per frame ~ (number of fields) x N^3 log N.
 * Add a perturbation: give it a schedule (perturbations.py), apply it inside _substep().
"""
from __future__ import annotations
import math
import json
import numpy as np

from .geometry import Lattice, PeriodicGeometry, PlateGeometry
from .field import FieldSpec, FieldState
from .perturbations import MassRamp, EFieldPulse
from .spectrum import tower, THEORIES
from . import observables as obs
from .units import Units


def default_specs(theory: str = "closed type II superstring (D=10)", n_massive: int = 2,
                  charged_mass: float = 0.35, charge: float = 1.0) -> list[FieldSpec]:
    lv = tower(theory, n_massive + 3)
    # the massless level: N = 0 for superstrings, N = 1 for the bosonic string (N = 0 is the tachyon)
    massless = next(L for L in lv if (not L.tachyonic) and L.mass2 == 0.0)
    specs = [FieldSpec(f"neutral scalar (level {massless.N}, massless)", 0.0, False, 0.0, massless.bosons,
                       massless.fermions, massless.N, theory, (0.55, 0.80, 1.00)),
             FieldSpec("charged scalar (open-string U(1))", charged_mass, True, charge, 2,
                       2 if massless.fermions > 0 else 0, massless.N, theory, (1.00, 0.75, 0.35))]
    palette = [(0.80, 0.55, 1.00), (0.55, 1.00, 0.65), (1.00, 0.55, 0.55), (0.9, 0.9, 0.5)]
    first_massive = massless.N + 1
    n_added = 0
    for L in lv[first_massive:]:
        if L.tachyonic or n_added >= n_massive:
            continue
        specs.append(FieldSpec(f"neutral scalar (level {L.N}, m={L.mass:.3g})", L.mass, False, 0.0,
                               L.bosons, L.fermions, L.N, theory, palette[n_added % len(palette)]))
        n_added += 1
    return specs


class Simulation:
    def __init__(self, N: int = 48, a: float = 1.0, seed: int = 1, specs: list[FieldSpec] | None = None,
                 theory: str = "closed type II superstring (D=10)", units: Units | None = None):
        self.lat = Lattice(N, a)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.theory = theory
        self.geom = PeriodicGeometry(self.lat)
        self.specs = specs if specs is not None else default_specs(theory)
        self.fields = [FieldState(s, self.geom, self.rng) for s in self.specs]
        self.units = units or Units()
        self.t = 0.0
        self.frame = 0
        self.ramps: list[MassRamp] = []
        self.pulse: EFieldPulse | None = None
        self.A_int = 0.0                       # int_0^t E(t') dt'   (kinetic shift = e * A_int)
        self.substep = 0.02                    # max sub-step while frequencies are time dependent
        self.susy_on = False
        self.susy_breaking = 0.0               # M_susy: fermion partners get m_F^2 = m_B^2 + M_susy^2
        self.events: list[str] = []
        self.history = {"t": [], "quanta": [], "particle_energy_density": [], "sample_energy_density": []}
        self.plate_positions: tuple[int, int] | None = None
        self.pulse_baseline: dict[int, float] = {}
        self.log(f"vacuum sampled on {N}^3 lattice, a = {a} l_s, seed {seed}")

    # ------------------------------------------------------------------------------------------
    def log(self, msg: str):
        self.events.append(f"t={self.t:8.3f}  {msg}")
        if len(self.events) > 400:
            del self.events[:100]

    @property
    def plates_on(self) -> bool:
        return self.geom.kind == "plates"

    def charged_fields(self):
        return [f for f in self.fields if f.spec.is_complex]

    # ------------------------------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------------------------------
    def time_dependent(self, t: float) -> bool:
        if self.pulse is not None and self.pulse.active(t):
            return True
        return any(r.active(t) for r in self.ramps)

    def step(self, dt: float):
        """Advance by dt (one rendered frame).  Exact single rotation when nothing depends on time;
        otherwise sub-steps with frequencies frozen at each sub-step midpoint."""
        if dt <= 0:
            return
        t_end = self.t + dt
        while self.t < t_end - 1e-12:
            remaining = t_end - self.t
            if self.time_dependent(self.t) or self.time_dependent(min(t_end, self.t + self.substep)):
                h = min(self.substep, remaining)
            else:
                # jump to the next scheduled event or the end of the frame
                nxt = self._next_event_time(self.t)
                h = min(remaining, max(nxt - self.t, 1e-9)) if nxt is not None else remaining
            self._substep(h)
        self._finalize_events()
        self.frame += 1
        self._record_history()

    def _next_event_time(self, t):
        times = []
        if self.pulse is not None and self.pulse.t0 > t:
            times.append(self.pulse.t0)
        for r in self.ramps:
            if r.t0 > t:
                times.append(r.t0)
        return min(times) if times else None

    def _substep(self, h: float):
        t_mid = self.t + 0.5 * h
        changed = False
        # masses from ramps (evaluated at the midpoint)
        for r in self.ramps:
            if not r.finished(self.t):
                fs = self.fields[r.field_index]
                m_mid = r.mass(t_mid)
                if m_mid != fs.mass:
                    fs.mass = m_mid
                    changed = True
        # electric field: kinetic shift e * A_int, A_int advanced with the midpoint rule
        if self.pulse is not None and not self.pulse.finished(self.t):
            E_mid = self.pulse.E(t_mid)
            A_mid = self.A_int + 0.5 * h * E_mid
            for fs in self.charged_fields():
                fs.kshift = fs.spec.charge * A_mid
            changed = True
            self.A_int += h * E_mid
        if changed:
            for fs in self.fields:
                fs.refresh_omega()
        for fs in self.fields:
            fs.evolve(h)
        self.t += h
        # end-of-step: put the exact end values in (masses / kshift) so read-outs are consistent
        if changed:
            for r in self.ramps:
                fs = self.fields[r.field_index]
                fs.mass = r.mass(self.t)
            if self.pulse is not None:
                for fs in self.charged_fields():
                    fs.kshift = fs.spec.charge * self.A_int
            for fs in self.fields:
                fs.refresh_omega()

    def _finalize_events(self):
        done = [r for r in self.ramps if r.finished(self.t)]
        for r in done:
            fs = self.fields[r.field_index]
            fs.mass = r.m1
            fs.refresh_omega()
            pn = fs.particle_numbers()
            self.log(f"mass ramp of '{fs.spec.name}' {r.m0:.3g}->{r.m1:.3g} finished: "
                     f"created quanta = {pn['quanta']:.4g} (exact), density {pn['number_density']:.3e}")
            self.ramps.remove(r)
        if self.pulse is not None and self.pulse.finished(self.t):
            tot = self.pairs_since_pulse()
            self.log(f"E-field pulse finished: pairs created by the pulse = {tot:.4g} (exact), "
                     f"continuum Schwinger estimate = {self.schwinger_estimate_pairs():.4g}")
            self.pulse = None

    def pairs_since_pulse(self) -> float:
        """Exact pairs created since the last pulse was fired (all charged fields)."""
        tot = 0.0
        for i, fs in enumerate(self.fields):
            if fs.spec.is_complex:
                tot += (fs.particle_numbers()["pairs"] or 0.0) - self.pulse_baseline.get(i, 0.0)
        return max(tot, 0.0)

    def _record_history(self):
        q = sum(f.particle_numbers()["quanta"] for f in self.fields)
        e = sum(f.particle_numbers()["particle_energy_density"] for f in self.fields)
        s = sum(f.total_energy_sample() for f in self.fields) / self.lat.volume
        for k, v in (("t", self.t), ("quanta", q), ("particle_energy_density", e), ("sample_energy_density", s)):
            self.history[k].append(float(v))
        if len(self.history["t"]) > 3000:
            for k in self.history:
                del self.history[k][:500]

    # ------------------------------------------------------------------------------------------
    # Perturbation controls
    # ------------------------------------------------------------------------------------------
    def resample_vacuum(self, seed: int | None = None, temperature: float = 0.0):
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        for fs in self.fields:
            fs.rng = self.rng
            fs.sample_vacuum(temperature)
        self.A_int = 0.0
        for fs in self.charged_fields():
            fs.kshift = 0.0
            fs.refresh_omega()
        self.log(f"vacuum re-sampled (seed {self.seed}" + (f", T={temperature}" if temperature > 0 else "") + ")")

    def quench_mass(self, field_index: int, new_mass: float, duration: float = 0.0, delay: float = 0.0):
        fs = self.fields[field_index]
        self.ramps = [r for r in self.ramps if r.field_index != field_index]
        self.ramps.append(MassRamp(field_index, fs.mass, float(new_mass), self.t + delay, duration))
        kind = "sudden quench" if duration <= 0 else f"ramp over {duration:.3g}"
        self.log(f"{kind}: '{fs.spec.name}' mass {fs.mass:.3g} -> {new_mass:.3g}")

    def apply_efield(self, E0: float, duration: float, delay: float = 0.0, ramp_fraction: float = 0.15):
        if not self.charged_fields():
            self.log("no charged field present: E-field has nothing to act on")
            return
        self.pulse = EFieldPulse(float(E0), self.t + delay, float(duration), ramp_fraction)
        self.A_int = 0.0
        # pairs already present (from earlier events) are remembered so the read-out can report
        # what THIS pulse created without discarding the exact bookkeeping
        self.pulse_baseline = {i: (fs.particle_numbers()["pairs"] or 0.0)
                               for i, fs in enumerate(self.fields) if fs.spec.is_complex}
        e = self.charged_fields()[0].spec.charge
        self.log(f"E-field pulse E0={E0:.3g}, duration {duration:.3g}: sweep e*int E dt = "
                 f"{self.pulse.total_sweep(e):.3g} (lattice limit pi/a = {math.pi/self.lat.a:.3g})")

    def max_pulse_duration(self, E0: float) -> float:
        """Longest pulse for which no mode crosses the Brillouin-zone edge before crossing k=0
        (keeps the lattice Schwinger effect identical to the continuum one).  See derivations/04."""
        cf = self.charged_fields()
        if not cf or E0 <= 0:
            return float("inf")
        e = cf[0].spec.charge
        r = 0.15
        return (math.pi / self.lat.a) / (e * E0) / (1.0 - r)

    def set_plates(self, z1: int | None, z2: int | None):
        """Insert (z1, z2 given) or remove (None) two perfect mirrors.  Sudden = dynamical Casimir."""
        if z1 is None or z2 is None:
            if not self.plates_on:
                return
            new = PeriodicGeometry(self.lat)
            self.plate_positions = None
            msg = "plates removed"
        else:
            new = PlateGeometry(self.lat, z1, z2)
            self.plate_positions = (new.z1, new.z2)
            msg = f"plates inserted at z-layers {new.z1}, {new.z2} (d = {new.separation:.3g} l_s)"
        before = sum(f.particle_numbers()["quanta"] for f in self.fields)
        for fs in self.fields:
            fs.set_geometry(new, exact_transfer=True)
        self.geom = new
        after = sum(f.particle_numbers()["quanta"] for f in self.fields)
        self.log(f"{msg}; quanta created by the sudden change = {after - before:.4g} (exact ensemble mean)")

    # ------------------------------------------------------------------------------------------
    # Rendering data
    # ------------------------------------------------------------------------------------------
    def grids(self, indices=None, smoothing: float = 0.0):
        """(phi, pi) per field for the requested indices."""
        idx = range(len(self.fields)) if indices is None else indices
        return {i: self.fields[i].to_grid(smoothing) for i in idx}

    def energy_density(self, indices=None, smoothing: float = 0.0) -> np.ndarray:
        h = np.zeros(self.lat.shape)
        for i, (phi, pi) in self.grids(indices, smoothing).items():
            h += self.fields[i].energy_density(phi, pi)
        return h

    def field_amplitude(self, indices=None, smoothing: float = 0.0) -> np.ndarray:
        """Sum of the (real) field values; charged field contributes sqrt2*Re(phi)."""
        out = np.zeros(self.lat.shape)
        for i, (phi, pi) in self.grids(indices, smoothing).items():
            out += np.sqrt(2.0) * phi.real if self.fields[i].spec.is_complex else phi
        return out

    def vacuum_profile(self, indices=None) -> np.ndarray:
        """Exact zero-point energy density vs z of the current geometry (sum over fields)."""
        idx = range(len(self.fields)) if indices is None else indices
        return sum(self.fields[i].ensemble_energy_profile(vacuum=True) for i in idx)

    def ensemble_profile(self, indices=None) -> np.ndarray:
        idx = range(len(self.fields)) if indices is None else indices
        return sum(self.fields[i].ensemble_energy_profile(vacuum=False) for i in idx)

    def excess_energy(self, indices=None, smoothing: float = 0.0) -> np.ndarray:
        """Sample energy density minus the exact vacuum expectation (per z-layer).  Averages to
        zero for the undisturbed vacuum; its spatial mean after a perturbation is the energy
        density of the real particles that were created."""
        h = self.energy_density(indices, smoothing)
        if smoothing > 0:
            # the vacuum expectation of the *smoothed* energy density is lower; estimate it from
            # the sample's own mean over the plate-free layers (unbiased for a translation-invariant state)
            return h - h.mean()
        return h - self.vacuum_profile(indices)[None, None, :]

    # ------------------------------------------------------------------------------------------
    # Read-outs
    # ------------------------------------------------------------------------------------------
    def schwinger_estimate_pairs(self) -> float:
        """Continuum Schwinger rate x volume x effective pulse duration (leading estimate)."""
        if self.pulse is None:
            return 0.0
        tot = 0.0
        for fs in self.charged_fields():
            eE = fs.spec.charge * self.pulse.E0
            tot += obs.schwinger_rate_scalar(fs.mass, eE) * self.lat.volume * self.pulse.effective_duration
        return tot

    def zero_point_summary(self) -> dict:
        """Mean vacuum energy densities: sampled fields, weighted by multiplicities, with/without SUSY."""
        rho_sampled = 0.0
        rho_weighted = 0.0
        rho_susy = 0.0
        per_field = []
        for fs in self.fields:
            zp = obs.zero_point_density_lattice(fs.mass, self.geom, fs.spec.nosc)
            g = fs.spec.multiplicity
            gF = fs.spec.fermion_partners
            mF = math.sqrt(fs.mass ** 2 + self.susy_breaking ** 2)
            zpF = obs.zero_point_density_lattice(mF, self.geom, fs.spec.nosc)
            rho_sampled += zp
            rho_weighted += g * zp
            rho_susy += g * zp - gF * zpF
            per_field.append({"name": fs.spec.name, "mass": fs.mass, "zp_density": zp, "multiplicity": g,
                              "fermion_partners": gF, "partner_mass": mF, "partner_zp_density": zpF})
        return {"sampled": rho_sampled, "weighted": rho_weighted,
                "susy": rho_susy if self.susy_on else None, "per_field": per_field,
                "cutoff": math.pi / self.lat.a}

    def tower_summary(self, n_levels: int = 40) -> dict:
        """Polchinski-style 'sum over the whole string tower' with a proper-time cutoff at the
        string scale (eps = alpha' = 1), eq. (7.3.24) with (-1)^F.  For the superstring the net
        result is identically zero level by level (bosons = fermions)."""
        levels = tower(self.theory, n_levels)
        pt = obs.supertrace_tower(levels, obs.proper_time_regulated_density, eps=1.0)
        return {"levels": levels[:8], "proper_time": pt,
                "n_levels": n_levels, "theory": self.theory}

    def casimir_summary(self) -> dict | None:
        """Lattice zero-point energy of the plate geometry and its d-dependence (Casimir energy and
        force), compared with the continuum formulae.  Both the lattice and the continuum forces
        are central differences over one lattice spacing, so they are directly comparable."""
        if not self.plates_on:
            return None
        g = self.geom
        d = g.separation
        D = g.outside_length
        A = self.lat.L ** 2
        a = self.lat.a

        def E_lat(z2):
            try:
                gg = PlateGeometry(self.lat, g.z1, z2)
            except ValueError:
                return None
            return sum(obs.lattice_zero_point_energy(f.mass, gg, f.spec.nosc) for f in self.fields)

        def E_cont(dd):
            DD = self.lat.L - dd
            return sum(f.spec.nosc * A * (obs.casimir_energy_per_area_continuum(dd, f.mass)
                                          + obs.casimir_energy_per_area_continuum(DD, f.mass))
                       for f in self.fields)

        E_now = E_lat(g.z2)
        Ep, Em = E_lat(g.z2 + 1), E_lat(g.z2 - 1)
        force = None if (Ep is None or Em is None) else -(Ep - Em) / (2 * a)
        force_cont = -(E_cont(d + a) - E_cont(d - a)) / (2 * a)
        # symmetric-position reference (inside length = outside length) for the energy difference
        n_free = self.lat.N - 2
        n_sym = n_free // 2
        E_ref = E_lat(g.z1 + n_sym + 1)
        d_sym = (n_sym + 1) * a
        return {"d": d, "D_out": D, "area": A,
                "E_zero_point": E_now,
                "E_casimir_vs_symmetric": None if E_ref is None else E_now - E_ref,
                "E_casimir_continuum_vs_symmetric": E_cont(d) - E_cont(d_sym),
                "force": force, "force_continuum": force_cont,
                "pressure": None if force is None else force / A,
                "pressure_continuum": force_cont / A}

    def readout(self) -> dict:
        pn = [f.particle_numbers() for f in self.fields]
        return {
            "t": self.t, "frame": self.frame, "N": self.lat.N, "a": self.lat.a,
            "fields": [{"name": f.spec.name, "mass": f.mass, "kshift": f.kshift, **p}
                       for f, p in zip(self.fields, pn)],
            "total_quanta": sum(p["quanta"] for p in pn),
            "total_particle_energy_density": sum(p["particle_energy_density"] for p in pn),
            "sample_energy_density": sum(f.total_energy_sample() for f in self.fields) / self.lat.volume,
            "zero_point": self.zero_point_summary(),
            "casimir": self.casimir_summary(),
            "pulse": None if self.pulse is None else {
                "E0": self.pulse.E0, "t0": self.pulse.t0, "duration": self.pulse.duration,
                "E_now": self.pulse.E(self.t), "A_int": self.A_int,
                "estimate_pairs": self.schwinger_estimate_pairs()},
            "ramps": [{"field": r.field_index, "m0": r.m0, "m1": r.m1, "t0": r.t0, "duration": r.duration}
                      for r in self.ramps],
        }

    # ------------------------------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------------------------------
    def save(self, path: str):
        params = {
            "N": self.lat.N, "a": self.lat.a, "seed": self.seed, "t": self.t, "theory": self.theory,
            "susy_on": self.susy_on, "susy_breaking": self.susy_breaking, "A_int": self.A_int,
            "plates": list(self.plate_positions) if self.plate_positions else None,
            "l_s": self.units.l_s,
            "specs": [s.__dict__ for s in self.specs],
            "events": self.events[-50:],
        }
        arrays = {"params_json": np.array(json.dumps(params))}
        for i, fs in enumerate(self.fields):
            for k, v in fs.state_arrays().items():
                arrays[f"f{i}_{k}"] = np.asarray(v)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "Simulation":
        data = np.load(path, allow_pickle=False)
        params = json.loads(str(data["params_json"]))
        specs = [FieldSpec(**{**s, "color": tuple(s["color"])}) for s in params["specs"]]
        sim = cls(N=params["N"], a=params["a"], seed=params["seed"], specs=specs,
                  theory=params.get("theory", "closed type II superstring (D=10)"),
                  units=Units(params.get("l_s", 1e-34)))
        if params.get("plates"):
            z1, z2 = params["plates"]
            sim.geom = PlateGeometry(sim.lat, z1, z2)
            sim.plate_positions = (z1, z2)
            for fs in sim.fields:
                fs.geom = sim.geom
                fs._build_blocks()
        for i, fs in enumerate(sim.fields):
            d = {k[len(f"f{i}_"):]: data[k] for k in data.files if k.startswith(f"f{i}_")}
            fs.load_state_arrays(d)
        sim.t = float(params["t"])
        sim.susy_on = bool(params.get("susy_on", False))
        sim.susy_breaking = float(params.get("susy_breaking", 0.0))
        sim.A_int = float(params.get("A_int", 0.0))
        sim.events = list(params.get("events", []))
        sim.log(f"state loaded from {path}")
        return sim
