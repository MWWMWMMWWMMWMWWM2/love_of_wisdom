"""
Physics validation tests.  Run with   python -m pytest tests/   or simply   python tests/test_physics.py
Every test compares the simulation against a closed-form quantum-field-theory result.
"""
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vacuumsim.geometry import Lattice, PeriodicGeometry, PlateGeometry          # noqa: E402
from vacuumsim.field import FieldSpec, FieldState                                 # noqa: E402
from vacuumsim.simulation import Simulation                                       # noqa: E402
from vacuumsim import observables as obs                                          # noqa: E402
from vacuumsim import spectrum                                                    # noqa: E402


def _field(mass=0.7, N=16, a=1.0, seed=0, is_complex=False, charge=0.0, geom=None):
    lat = Lattice(N, a)
    geom = geom or PeriodicGeometry(lat)
    return FieldState(FieldSpec("t", mass, is_complex, charge), geom, np.random.default_rng(seed))


# ------------------------------------------------------------------------------------------------
def test_vacuum_variance_matches_mode_sum():
    """<phi(x)^2> = (1/V) sum_k 1/(2 omega_k)   (Vol.1 p.221: sum of zero-point contributions)."""
    lat = Lattice(16, 1.0)
    geom = PeriodicGeometry(lat)
    theory = sum(1.0 / (2 * np.sqrt(0.7 ** 2 + b.k2_spatial(0.0))) for b in geom.blocks).sum() / lat.volume
    vals = []
    for s in range(40):
        phi, _ = _field(0.7, 16, seed=s, geom=geom).to_grid()
        vals.append((phi ** 2).mean())
    m, err = np.mean(vals), np.std(vals) / math.sqrt(len(vals))
    assert abs(m - theory) < 4 * err + 1e-3 * theory, (m, err, theory)


def test_sample_energy_equals_mode_energy_and_zero_point():
    """Grid energy density integrates to the mode-space energy exactly; its ensemble mean is the
    zero-point energy sum 1/2 omega (checked statistically)."""
    ratios = []
    for s in range(60):
        f = _field(0.5, 12, seed=s)
        phi, pi = f.to_grid()
        e_grid = f.energy_density(phi, pi).sum() * f.geom.lat.a ** 3
        assert abs(e_grid - f.total_energy_sample()) < 1e-8 * e_grid
        ratios.append(f.total_energy_sample() / f.zero_point_energy())
    m, err = np.mean(ratios), np.std(ratios) / math.sqrt(len(ratios))
    assert abs(m - 1.0) < 4 * err + 1e-3, (m, err)


def test_complex_field_energy_with_gauge_link():
    f = _field(0.4, 12, seed=3, is_complex=True, charge=1.0)
    f.kshift = 0.37
    f.refresh_omega()
    phi, pi = f.to_grid()
    e_grid = f.energy_density(phi, pi).sum() * f.geom.lat.a ** 3
    assert abs(e_grid - f.total_energy_sample()) < 1e-8 * e_grid


def test_energy_conservation_and_zero_occupation():
    f = _field(0.3, 12, seed=1)
    e0 = f.total_energy_sample()
    for _ in range(200):
        f.evolve(0.31)
    phi, pi = f.to_grid()
    e1 = f.energy_density(phi, pi).sum() * f.geom.lat.a ** 3
    assert abs(e1 - e0) < 1e-9 * e0
    assert max(np.abs(n).max() for n in f.mode_occupations()) < 1e-10


# ------------------------------------------------------------------------------------------------
def test_sudden_quench_bogoliubov():
    """n_k = (w1 - w0)^2 / (4 w0 w1)  exactly, and constant afterwards."""
    f = _field(0.5, 12, seed=2)
    w0 = f.blocks[0].omega.copy()
    f.mass = 1.5
    f.refresh_omega()
    w1 = f.blocks[0].omega
    n_exact = obs.sudden_quench_occupation(w0, w1)
    assert np.allclose(f.mode_occupations()[0], n_exact, atol=1e-12)
    for _ in range(50):
        f.evolve(0.17)
    assert np.allclose(f.mode_occupations()[0], n_exact, atol=1e-10)
    # ensemble-exact particle energy equals the statistical mean of the sample excess
    V = f.geom.lat.volume
    exact = f.particle_numbers()["particle_energy_density"]
    ex = []
    for s in range(150):
        g = _field(0.5, 12, seed=1000 + s)
        g.mass = 1.5
        g.refresh_omega()
        ex.append((g.total_energy_sample() - g.zero_point_energy()) / V)
    m, err = np.mean(ex), np.std(ex) / math.sqrt(len(ex))
    assert abs(m - exact) < 4 * err + 0.01 * exact, (m, err, exact)


def test_adiabatic_ramp_creates_fewer_particles():
    sim = Simulation(N=12, a=1.0, seed=4, specs=[FieldSpec("s", 0.5)])
    sim.quench_mass(0, 1.5, duration=0.0)
    sim.step(0.5)
    n_sudden = sim.fields[0].particle_numbers()["quanta"]
    sim2 = Simulation(N=12, a=1.0, seed=4, specs=[FieldSpec("s", 0.5)])
    sim2.quench_mass(0, 1.5, duration=12.0)
    sim2.step(13.0)
    n_slow = sim2.fields[0].particle_numbers()["quanta"]
    assert n_sudden > 20 * n_slow > 0, (n_sudden, n_slow)


# ------------------------------------------------------------------------------------------------
def test_sauter_landau_zener_single_mode():
    """Drive the actual evolve() code path with a continuum frequency omega^2 = m_perp^2 + (k + eEt)^2
    sweeping k from -K to +K.  Exact asymptotic pair number: |beta|^2 = exp(-pi m_perp^2 / eE)."""
    f = _field(1.0, 2, seed=0)                        # 8 coefficients used as 8 test oscillators
    bs = f.blocks[0]
    m_perp2 = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]).reshape(bs.omega.shape)
    eE = 2.0
    K = 60.0
    T = 2 * K / eE
    dt = 0.002
    t = 0.0
    def omega(tt):
        return np.sqrt(m_perp2 + (-K + eE * tt) ** 2)
    bs.omega = omega(0.0)
    f.reset_covariance()
    nsteps = int(round(T / dt))
    for i in range(nsteps):
        bs.omega = omega(t + 0.5 * dt)
        f.evolve(dt)
        t += dt
    bs.omega = omega(t)
    n = f.mode_occupations()[0]
    expected = np.exp(-np.pi * m_perp2 / eE)
    assert np.allclose(n, expected, rtol=0.03, atol=2e-4), (n, expected)


def test_schwinger_rate_on_lattice():
    """Total pairs created by a pulse vs. the continuum Schwinger estimate Gamma V tau."""
    sim = Simulation(N=24, a=0.5, seed=7, specs=[FieldSpec("c", 0.35, True, 1.0)])
    sim.substep = 0.01
    E0 = 1.2
    tau = 0.8 * sim.max_pulse_duration(E0)
    sim.apply_efield(E0, tau)
    while sim.pulse is not None:
        sim.step(0.2)
    pairs = sim.fields[0].particle_numbers()["pairs"]
    est = obs.schwinger_rate_scalar(0.35, E0) * sim.lat.volume * tau * 0.85   # effective duration
    assert 0.6 * est < pairs < 1.5 * est, (pairs, est)


# ------------------------------------------------------------------------------------------------
def test_casimir_lattice_vs_continuum():
    """Zero-point energy of a massless scalar between Dirichlet plates on the lattice reproduces
    the continuum -pi^2/(1440 d^3) per area (both regions) up to O(a^2/d^2) corrections."""
    N = 40
    lat = Lattice(N, 1.0)
    A = lat.L ** 2

    def E(z2):
        return obs.lattice_zero_point_energy(0.0, PlateGeometry(lat, 0, z2))

    def cont(z2):
        d = z2 * lat.a                      # (n_inside + 1) a with n_inside = z2 - 1
        D = lat.L - d
        return A * (obs.casimir_energy_per_area_continuum(d) + obs.casimir_energy_per_area_continuum(D))

    for z2 in (8, 10, 12):
        lattice = (E(z2 + 1) - E(z2 - 1)) / 2.0          # dE/dd * a  (central difference)
        continuum = (cont(z2 + 1) - cont(z2 - 1)) / 2.0
        assert abs(lattice - continuum) < 0.15 * abs(continuum), (z2, lattice, continuum)
        assert lattice > 0                                 # E decreases as d shrinks -> attractive


def test_plate_insertion_exact_transfer():
    """After a sudden insertion the transferred covariance must give the same ensemble energy as
    the average over many sampled configurations projected onto the new modes."""
    lat = Lattice(12, 1.0)
    pg = PlateGeometry(lat, 3, 8)
    f = _field(0.0, 12, seed=0)
    f.set_geometry(pg)
    ens = sum(f.spec.f * (bs.Cpp + bs.omega ** 2 * bs.Cqq).sum() for bs in f.blocks)
    created = f.particle_numbers()["quanta"]
    assert created > 0
    samples = []
    for s in range(120):
        g = _field(0.0, 12, seed=500 + s)
        g.set_geometry(PlateGeometry(lat, 3, 8), exact_transfer=False)
        samples.append(g.total_energy_sample())
    m, err = np.mean(samples), np.std(samples) / math.sqrt(len(samples))
    assert abs(m - ens) < 4 * err + 1e-3 * ens, (m, err, ens)


# ------------------------------------------------------------------------------------------------
def test_string_degeneracies():
    assert [int(x) for x in spectrum.open_bosonic_degeneracies(4)] == [1, 24, 324, 3200, 25650]
    assert [int(x) for x in spectrum.open_superstring_degeneracies(4)] == [8, 128, 1152, 7680, 42112]
    lv = spectrum.tower("closed type II superstring (D=10)", 3)
    assert lv[0].bosons == 128 and lv[0].fermions == 128 and lv[0].mass2 == 0.0
    assert lv[1].mass2 == 4.0
    lvb = spectrum.tower("open bosonic string (D=26)", 3)
    assert lvb[0].tachyonic and lvb[1].mass2 == 0.0 and lvb[1].bosons == 24


def test_supersymmetric_cancellation():
    for th in ("closed type II superstring (D=10)", "open superstring (D=10)"):
        levels = spectrum.tower(th, 30)
        r = obs.supertrace_tower(levels, obs.proper_time_regulated_density, eps=1.0)
        assert r["bosons"] != 0.0 and abs(r["net"]) < 1e-12 * abs(r["bosons"])
    # sharp-cutoff formula agrees with numerical integration
    L, m = 3.0, 0.8
    k = np.linspace(0, L, 200001)
    num = np.trapezoid(k ** 2 * np.sqrt(k ** 2 + m ** 2), k) / (4 * math.pi ** 2)
    assert abs(num - obs.zero_point_density_sharp_cutoff(m, L)) < 1e-6 * num


def test_save_load_roundtrip():
    sim = Simulation(N=8, a=1.0, seed=3)
    sim.step(0.7)
    sim.set_plates(1, 5)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "state.npz")
        sim.save(p)
        sim2 = Simulation.load(p)
    assert sim2.plates_on and abs(sim2.t - sim.t) < 1e-12
    for a, b in zip(sim.fields, sim2.fields):
        for ba, bb in zip(a.blocks, b.blocks):
            assert np.allclose(ba.Q, bb.Q) and np.allclose(ba.Cqq, bb.Cqq)


if __name__ == "__main__":
    import time
    names = [n for n in dir() if n.startswith("test_")]
    failed = 0
    for n in names:
        t0 = time.time()
        try:
            globals()[n]()
            print(f"PASS  {n}  ({time.time()-t0:.1f}s)")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {n}: {e}")
        except Exception as e:   # noqa: BLE001
            failed += 1
            print(f"ERROR {n}: {type(e).__name__}: {e}")
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
