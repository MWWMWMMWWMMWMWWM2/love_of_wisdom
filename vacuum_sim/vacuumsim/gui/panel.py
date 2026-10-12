"""
panel.py -- the control / read-out panel shown next to (not on top of) the 3D view.

All physics decisions live in vacuumsim.simulation; this file only builds widgets, forwards
button presses to callbacks supplied by main_window.py, and formats numbers for display.

MODIFY: to add a control, (1) add the widget in the matching _build_* method, (2) read its value in
main_window.py where the callback is handled.  Every group is a plain QGroupBox in a vertical
layout inside a QScrollArea, so the panel just grows.
"""
from __future__ import annotations
import math

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..render import SEQUENTIAL, DIVERGING
from ..spectrum import THEORIES
from ..units import STRING_LENGTH_PRESETS, RHO_LAMBDA_GEV4, electroweak_cutoff_estimate

Qt = QtCore.Qt


def _mono_font():
    f = QtGui.QFont("Consolas")
    f.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    f.setPointSize(8)
    return f


def _spin(lo, hi, val, step, decimals=3, suffix=""):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    return s


def _ispin(lo, hi, val):
    s = QtWidgets.QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setKeyboardTracking(False)
    return s


def _slider(lo, hi, val):
    s = QtWidgets.QSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue(val)
    return s


def _readout():
    lab = QtWidgets.QLabel("")
    lab.setFont(_mono_font())
    lab.setWordWrap(True)
    lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lab


def _fmt(x, spec=".4g"):
    return "n/a" if x is None else format(x, spec)


def _group(title):
    g = QtWidgets.QGroupBox(title)
    form = QtWidgets.QFormLayout(g)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    return g, form


class ControlPanel(QtWidgets.QWidget):
    def __init__(self, callbacks: dict, parent=None):
        super().__init__(parent)
        self.cb = callbacks
        self.field_checks: list[QtWidgets.QCheckBox] = []
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        self.vbox = outer
        self._build_simulation()
        self._build_render()
        self._build_plates()
        self._build_efield()
        self._build_quench()
        self._build_susy()
        self._build_units()
        self._build_history()
        self._build_saveload()
        self._build_log()
        outer.addStretch(1)

    # ------------------------------------------------------------------------------------------
    def _build_simulation(self):
        g, f = _group("Simulation")
        row = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("Pause")
        self.play_btn.setCheckable(True)
        self.play_btn.setChecked(True)
        self.play_btn.toggled.connect(lambda on: (self.play_btn.setText("Pause" if on else "Play"), self.cb["play"](on)))
        self.step_btn = QtWidgets.QPushButton("Step")
        self.step_btn.clicked.connect(lambda: self.cb["step"]())
        row.addWidget(self.play_btn)
        row.addWidget(self.step_btn)
        f.addRow(row)
        self.dt_spin = _spin(0.001, 0.5, 0.04, 0.005, 3, "  l_s/c per frame")
        f.addRow("time step", self.dt_spin)
        self.fps_spin = _ispin(5, 60, 30)
        f.addRow("target fps", self.fps_spin)
        self.seed_spin = _ispin(0, 10 ** 6, 1)
        f.addRow("random seed", self.seed_spin)
        self.temp_spin = _spin(0.0, 20.0, 0.0, 0.1, 2, "  M_s (0 = vacuum)")
        f.addRow("temperature", self.temp_spin)
        self.resample_btn = QtWidgets.QPushButton("Re-sample the vacuum (new random history)")
        self.resample_btn.clicked.connect(lambda: self.cb["resample"]())
        f.addRow(self.resample_btn)
        self.N_combo = QtWidgets.QComboBox()
        for n in (24, 32, 40, 48, 56, 64):
            self.N_combo.addItem(str(n), n)
        self.N_combo.setCurrentIndex(1)
        f.addRow("grid N (N^3 sites)", self.N_combo)
        self.a_spin = _spin(0.2, 4.0, 1.0, 0.1, 2, "  l_s")
        f.addRow("lattice spacing a", self.a_spin)
        self.theory_combo = QtWidgets.QComboBox()
        for name in THEORIES:
            self.theory_combo.addItem(name)
        f.addRow("string theory", self.theory_combo)
        self.nmassive_spin = _ispin(0, 4, 2)
        f.addRow("massive levels sampled", self.nmassive_spin)
        self.rebuild_btn = QtWidgets.QPushButton("Rebuild simulation with these settings")
        self.rebuild_btn.clicked.connect(lambda: self.cb["rebuild"]())
        f.addRow(self.rebuild_btn)
        self.perf_label = _readout()
        f.addRow(self.perf_label)
        self.vbox.addWidget(g)

    def _build_render(self):
        g, f = _group("Rendering")
        self.fields_box = QtWidgets.QVBoxLayout()
        f.addRow("fields shown", self.fields_box)
        self.quantity_combo = QtWidgets.QComboBox()
        self.quantity_combo.addItems(["vacuum energy density  h(x,t)",
                                      "field amplitude  phi(x,t)",
                                      "excess over vacuum  h - <h>_vac  (real particles)"])
        self.quantity_combo.currentIndexChanged.connect(lambda _: self.cb["render_changed"]())
        f.addRow("quantity", self.quantity_combo)
        self.smooth_slider = _slider(0, 40, 12)
        self.smooth_label = QtWidgets.QLabel("probe resolution sigma = 1.2 l_s")
        self.smooth_slider.valueChanged.connect(
            lambda v: (self.smooth_label.setText(f"probe resolution sigma = {v/10:.1f} l_s"
                                                 + ("  (raw lattice)" if v == 0 else "")),
                       self.cb["render_changed"]()))
        f.addRow(self.smooth_label, self.smooth_slider)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["volume", "slice", "isosurface"])
        self.mode_combo.currentIndexChanged.connect(lambda _: self.cb["render_changed"]())
        f.addRow("mode", self.mode_combo)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(list(SEQUENTIAL) + list(DIVERGING))
        self.cmap_combo.currentIndexChanged.connect(lambda _: self.cb["render_changed"]())
        f.addRow("colour map", self.cmap_combo)
        self.alpha_slider = _slider(5, 100, 85)
        f.addRow("opacity", self.alpha_slider)
        self.gamma_slider = _slider(5, 60, 22)
        f.addRow("contrast (alpha gamma)", self.gamma_slider)
        self.hi_slider = _slider(900, 1000, 995)
        f.addRow("upper percentile", self.hi_slider)
        self.slice_axis = QtWidgets.QComboBox()
        self.slice_axis.addItems(["x", "y", "z"])
        self.slice_axis.setCurrentIndex(2)
        self.slice_pos = _slider(0, 100, 50)
        f.addRow("slice axis / position", self._hbox(self.slice_axis, self.slice_pos))
        self.iso_slider = _slider(50, 999, 900)
        f.addRow("isosurface level (percentile)", self.iso_slider)
        self.range_label = _readout()
        f.addRow(self.range_label)
        self.vbox.addWidget(g)

    def _build_plates(self):
        g, f = _group("Casimir mirrors (two perfect plates, phi = 0)")
        self.z1_spin = _ispin(0, 63, 10)
        self.z2_spin = _ispin(0, 63, 20)
        f.addRow("plate layers z1 / z2", self._hbox(self.z1_spin, self.z2_spin))
        row = QtWidgets.QHBoxLayout()
        self.plates_btn = QtWidgets.QPushButton("Insert / move plates (sudden)")
        self.plates_btn.clicked.connect(lambda: self.cb["plates"](True))
        self.plates_rm_btn = QtWidgets.QPushButton("Remove")
        self.plates_rm_btn.clicked.connect(lambda: self.cb["plates"](False))
        row.addWidget(self.plates_btn)
        row.addWidget(self.plates_rm_btn)
        f.addRow(row)
        self.casimir_label = _readout()
        f.addRow(self.casimir_label)
        self.profile_plot = pg.PlotWidget(title="exact <h>(z): vacuum (blue), with created particles (orange), sample (dots)")
        self.profile_plot.setMinimumHeight(150)
        self.profile_plot.setLabel("bottom", "z layer")
        self.profile_plot.showGrid(x=True, y=True, alpha=0.2)
        self.prof_vac_curve = self.profile_plot.plot(pen=pg.mkPen((140, 197, 255), width=2))
        self.prof_ens_curve = self.profile_plot.plot(pen=pg.mkPen((255, 159, 67), width=1.5))
        self.prof_sample_curve = self.profile_plot.plot(pen=None, symbol="o", symbolSize=3,
                                                        symbolBrush=(240, 240, 240), symbolPen=None)
        f.addRow(self.profile_plot)
        self.vbox.addWidget(g)

    def _build_efield(self):
        g, f = _group("Uniform electric field on the charged scalar (Schwinger effect)")
        self.E0_spin = _spin(0.0, 20.0, 1.0, 0.1, 2, "  M_s^2 / e")
        f.addRow("field strength E0", self.E0_spin)
        self.Edur_spin = _spin(0.05, 200.0, 2.0, 0.1, 2, "  l_s/c")
        f.addRow("pulse duration", self.Edur_spin)
        self.fire_btn = QtWidgets.QPushButton("Fire pulse")
        self.fire_btn.clicked.connect(lambda: self.cb["efield"]())
        f.addRow(self.fire_btn)
        self.efield_label = _readout()
        f.addRow(self.efield_label)
        self.vbox.addWidget(g)

    def _build_quench(self):
        g, f = _group("Mass quench / ramp (Bogoliubov particle creation)")
        self.quench_field = QtWidgets.QComboBox()
        f.addRow("field", self.quench_field)
        self.quench_mass = _spin(0.0, 10.0, 1.0, 0.05, 3, "  M_s")
        f.addRow("new mass", self.quench_mass)
        self.quench_dur = _spin(0.0, 100.0, 0.0, 0.1, 2, "  l_s/c  (0 = sudden)")
        f.addRow("ramp duration", self.quench_dur)
        self.quench_btn = QtWidgets.QPushButton("Quench now")
        self.quench_btn.clicked.connect(lambda: self.cb["quench"]())
        f.addRow(self.quench_btn)
        self.quench_label = _readout()
        f.addRow(self.quench_label)
        self.vbox.addWidget(g)

    def _build_susy(self):
        g, f = _group("Supersymmetry and the mean vacuum energy  (Vol.1 eq. 7.3.24, Vol.2 ch.10)")
        self.susy_check = QtWidgets.QCheckBox("include fermionic superpartners: sum (-1)^F g_i (1/2 omega)")
        self.susy_check.toggled.connect(lambda on: self.cb["susy"](on, self.susy_break.value()))
        f.addRow(self.susy_check)
        self.susy_break = _spin(0.0, 10.0, 0.0, 0.05, 3, "  M_s")
        self.susy_break.valueChanged.connect(lambda v: self.cb["susy"](self.susy_check.isChecked(), v))
        f.addRow("SUSY-breaking scale M_susy", self.susy_break)
        self.susy_label = _readout()
        f.addRow(self.susy_label)
        self.vbox.addWidget(g)

    def _build_units(self):
        g, f = _group("Units, scales, and relativity")
        self.ls_combo = QtWidgets.QComboBox()
        for name in STRING_LENGTH_PRESETS:
            self.ls_combo.addItem(name)
        self.ls_combo.currentIndexChanged.connect(lambda _: self.cb["units"]())
        f.addRow("string length l_s", self.ls_combo)
        self.units_label = _readout()
        f.addRow(self.units_label)
        self.accel_spin = _spin(0.0, 1e6, 0.0, 0.1, 3, "  c^2/l_s")
        self.accel_spin.valueChanged.connect(lambda _: self.cb["units"]())
        f.addRow("observer proper acceleration", self.accel_spin)
        self.unruh_label = _readout()
        f.addRow(self.unruh_label)
        self.vbox.addWidget(g)

    def _build_history(self):
        g, f = _group("History")
        self.hist_plot = pg.PlotWidget(title="real particles created (exact ensemble mean)")
        self.hist_plot.setMinimumHeight(140)
        self.hist_plot.setLabel("bottom", "t (l_s/c)")
        self.hist_plot.showGrid(x=True, y=True, alpha=0.2)
        self.hist_curve = self.hist_plot.plot(pen=pg.mkPen((255, 107, 107), width=2))
        f.addRow(self.hist_plot)
        self.energy_plot = pg.PlotWidget(title="energy density of this sample (M_s^4)")
        self.energy_plot.setMinimumHeight(120)
        self.energy_plot.showGrid(x=True, y=True, alpha=0.2)
        self.energy_curve = self.energy_plot.plot(pen=pg.mkPen((140, 197, 255), width=2))
        f.addRow(self.energy_plot)
        self.vbox.addWidget(g)

    def _build_saveload(self):
        g, f = _group("Save / load")
        row = QtWidgets.QHBoxLayout()
        b1 = QtWidgets.QPushButton("Save state (.npz)")
        b1.clicked.connect(lambda: self.cb["save"]())
        b2 = QtWidgets.QPushButton("Load state")
        b2.clicked.connect(lambda: self.cb["load"]())
        b3 = QtWidgets.QPushButton("Screenshot")
        b3.clicked.connect(lambda: self.cb["screenshot"]())
        for b in (b1, b2, b3):
            row.addWidget(b)
        f.addRow(row)
        self.vbox.addWidget(g)

    def _build_log(self):
        g, f = _group("Event log")
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(_mono_font())
        self.log_box.setMinimumHeight(120)
        f.addRow(self.log_box)
        self.vbox.addWidget(g)

    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _hbox(*widgets):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        for x in widgets:
            h.addWidget(x)
        return w

    def set_fields(self, specs):
        while self.fields_box.count():
            item = self.fields_box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.field_checks = []
        self.quench_field.clear()
        for i, s in enumerate(specs):
            c = QtWidgets.QCheckBox(f"{s.name}   m = {s.mass:.3g}" + ("  (charged)" if s.is_complex else ""))
            c.setChecked(True)
            r, gg, b = [int(255 * x) for x in s.color]
            c.setStyleSheet(f"QCheckBox {{ color: rgb({r},{gg},{b}); }}")
            c.toggled.connect(lambda _on: self.cb["render_changed"]())
            self.fields_box.addWidget(c)
            self.field_checks.append(c)
            self.quench_field.addItem(f"{i}: {s.name}", i)

    def shown_field_indices(self):
        return [i for i, c in enumerate(self.field_checks) if c.isChecked()]

    # ------------------------------------------------------------------------------------------
    def update_readouts(self, sim, ro: dict, fps: float, frame_ms: float, render_range):
        u = sim.units
        zp = ro["zero_point"]
        self.perf_label.setText(f"t = {ro['t']:.3f} l_s/c   frame {ro['frame']}   {fps:4.1f} fps   "
                                f"physics+render {frame_ms:.0f} ms/frame\n"
                                f"lattice cutoff Lambda = pi/a = {zp['cutoff']:.3g} M_s;   "
                                f"omega_max = {sim.lat.omega_max:.3g} M_s  -> keep dt < {0.5/sim.lat.omega_max:.3g}")
        lo, hi = render_range if render_range else (0, 0)
        self.range_label.setText(f"colour range: {lo:.4g} .. {hi:.4g}")
        # Casimir
        cs = ro["casimir"]
        if cs:
            si_force = "n/a" if cs["force"] is None else f"{cs['force'] * u.energy / u.length:.3e} N"
            self.casimir_label.setText(
                f"d = {cs['d']:.3g} l_s = {cs['d']*u.length:.3e} m   (other side {cs['D_out']:.3g} l_s)\n"
                f"zero-point energy of the plate geometry (lattice, all fields): {cs['E_zero_point']:.6g} M_s\n"
                f"Casimir energy rel. to symmetric position: lattice {_fmt(cs['E_casimir_vs_symmetric'])}, "
                f"continuum {_fmt(cs['E_casimir_continuum_vs_symmetric'])} M_s\n"
                f"force on plate 2: lattice {_fmt(cs['force'])}, continuum {_fmt(cs['force_continuum'])} M_s/l_s  "
                f"(= {si_force});  pressure {_fmt(cs['pressure'], '.3e')} vs {_fmt(cs['pressure_continuum'], '.3e')} M_s^4\n"
                f"negative = attractive.  Continuum: -pi^2/(1440 d^3) per area (massless), "
                f"K_2-formula for massive fields.")
        else:
            self.casimir_label.setText("no plates.  Insert two mirrors: the mode spectrum between them changes, "
                                       "the zero-point energy depends on d, and -dE/dd is the Casimir force.\n"
                                       "The insertion itself is sudden, so it also creates real particles "
                                       "(dynamical Casimir effect) -- see the event log.")
        # E field
        cf = sim.charged_fields()
        if cf:
            fs = cf[0]
            e = fs.spec.charge
            E0 = self.E0_spin.value()
            eE = e * E0
            Ec = fs.mass ** 2 / e if e > 0 else float("inf")
            tmax = sim.max_pulse_duration(E0)
            from .. import observables as obs
            rate = obs.schwinger_rate_scalar(fs.mass, eE)
            pn = fs.particle_numbers()
            pulse = ro["pulse"]
            live = ""
            if pulse:
                live = (f"\nPULSE ACTIVE: E(t) = {pulse['E_now']:.3g}, sweep so far e*int E dt = {e*pulse['A_int']:.3g} "
                        f"(limit pi/a = {math.pi/sim.lat.a:.3g});  continuum estimate for whole pulse: "
                        f"{pulse['estimate_pairs']:.3g} pairs")
            self.efield_label.setText(
                f"charged field m = {fs.mass:.3g} M_s, e = {e:.3g};  critical field E_c = m^2/e = {Ec:.3g}\n"
                f"E0 = {E0:.3g} -> eE/m^2 = {eE/max(fs.mass**2,1e-30):.3g};  "
                f"Schwinger rate Gamma = {rate:.3e} pairs / (l_s^3 l_s/c)  ~ exp(-pi m^2/eE) = {math.exp(-math.pi*fs.mass**2/max(eE,1e-30)):.3e}\n"
                f"E0 in SI: {E0*u.efield:.3e} V/m  (electron E_c = 1.3e18 V/m for comparison)\n"
                f"max pulse before lattice Bloch artefacts: {tmax:.3g} l_s/c\n"
                f"pairs created by the last pulse (exact): {sim.pairs_since_pulse():.4g};  "
                f"all pairs present in this field: {pn['pairs'] or 0:.4g}, energy density "
                f"{pn['particle_energy_density']:.3e} M_s^4" + live)
        else:
            self.efield_label.setText("no charged field in this simulation")
        # quench
        lines = []
        for fd in ro["fields"]:
            lines.append(f"{fd['name'][:34]:34s} m={fd['mass']:.3g}  quanta={fd['quanta']:.4g}  "
                         f"n={fd['number_density']:.3e}/l_s^3  E_part={fd['particle_energy_density']:.3e}")
        if ro["ramps"]:
            for r in ro["ramps"]:
                lines.append(f"ramp running: field {r['field']} {r['m0']:.3g}->{r['m1']:.3g} over {r['duration']:.3g}")
        self.quench_label.setText("\n".join(lines))
        # SUSY
        ts = sim.tower_summary()
        pt = ts["proper_time"]
        susy_txt = (f"sampled fields (bosons, 1 state each): <h>_vac = {zp['sampled']:.4g} M_s^4  (lattice cutoff)\n"
                    f"weighted by level multiplicities g_i (bosons only): {zp['weighted']:.4g} M_s^4\n")
        if zp["susy"] is not None:
            susy_txt += (f"with fermionic partners, sum (-1)^F g_i (1/2 omega), M_susy = {sim.susy_breaking:.3g}: "
                         f"{zp['susy']:.4g} M_s^4\n")
        susy_txt += (f"whole tower ({ts['n_levels']} levels, proper-time cutoff eps = alpha'): bosons {pt['bosons']:.4g}, "
                     f"fermions {pt['fermions']:.4g}, net {pt['net']:.4g}"
                     + (f"  [tachyon levels {pt['skipped_tachyon_levels']} skipped]" if pt['skipped_tachyon_levels'] else "")
                     + "\nSUSY cancels the MEAN vacuum energy level by level; the fluctuations you see are untouched.")
        self.susy_label.setText(susy_txt)
        # units
        ew = electroweak_cutoff_estimate()
        self.units_label.setText(
            f"l_s = {u.l_s:.3e} m;  a = {u.describe_length(sim.lat.a)};  box = {u.describe_length(sim.lat.L)}\n"
            f"frame dt = {u.describe_time(self.dt_spin.value())};  M_s = {u.energy_GeV:.3e} GeV\n"
            f"<h>_vac (sampled fields) = {u.describe_energy_density(zp['sampled'])}\n"
            f"observed dark energy: {RHO_LAMBDA_GEV4:.2e} GeV^4.  Polchinski p.222: cutoff at {ew['Lambda_GeV']:.0f} GeV alone "
            f"gives 10^{ew['orders_above_observed']:.1f} x too much.\n"
            f"Relativity: the vacuum is Lorentz invariant -- an inertial observer at any speed sees the same "
            f"fluctuation statistics and no particles.")
        acc = self.accel_spin.value()
        if acc > 0:
            T = acc / (2 * math.pi)
            self.unruh_label.setText(f"Unruh temperature T = a/(2 pi) = {T:.3g} M_s = {u.unruh_temperature_K(acc):.3e} K: "
                                     f"an observer with this proper acceleration sees the vacuum as a thermal bath.\n"
                                     f"(Try: set 'temperature' above to {T:.3g} and re-sample to see what that observer sees.)")
        else:
            self.unruh_label.setText("acceleration 0: inertial observer, T_Unruh = 0.")
        # history / profile
        H = sim.history
        self.hist_curve.setData(H["t"], H["quanta"])
        self.energy_curve.setData(H["t"], H["sample_energy_density"])
        # log
        txt = "\n".join(sim.events[-14:])
        if txt != self.log_box.toPlainText():
            self.log_box.setPlainText(txt)
            self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())
