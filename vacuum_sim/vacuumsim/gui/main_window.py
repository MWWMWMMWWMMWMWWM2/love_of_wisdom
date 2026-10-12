"""
main_window.py -- ties the physics (Simulation), the 3D view and the control panel together.

Layout:  [ 3D OpenGL view (stretches) | scrollable control panel (fixed ~440 px) ]
The panel never overlaps the volume; drag the splitter handle to resize.

Frame loop (QTimer):  step the simulation by dt  ->  compute the chosen scalar field on the grid
                      ->  colour-map it to RGBA  ->  upload to the GL volume  ->  every 10th frame
                      refresh the numeric read-outs (they involve extra mode sums).

MODIFY: the callbacks dict is the single place where UI actions reach the physics; add a key there
and a button in panel.py to expose a new experiment.
"""
from __future__ import annotations
import os
import time
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ..simulation import Simulation, default_specs
from ..units import Units, STRING_LENGTH_PRESETS
from ..render import VolumeStyle, RangeSmoother, scalar_to_rgba, DIVERGING
from .. import observables as obs
from .view3d import VacuumGLView
from .panel import ControlPanel

Qt = QtCore.Qt


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, sim: Simulation):
        super().__init__()
        self.sim = sim
        self.setWindowTitle("Vacuum fluctuations of a string-tower of quantum fields -- exact lattice simulation")
        self.playing = True
        self.smoother = RangeSmoother(12)
        self.last_range = None
        self._frame_ms = 0.0
        self._fps = 0.0
        self._last_tick = time.perf_counter()

        callbacks = {
            "play": self.on_play, "step": self.on_step, "resample": self.on_resample,
            "rebuild": self.on_rebuild, "render_changed": self.on_render_changed,
            "plates": self.on_plates, "efield": self.on_efield, "quench": self.on_quench,
            "susy": self.on_susy, "units": self.on_units, "save": self.on_save, "load": self.on_load,
            "screenshot": self.on_screenshot,
        }
        self.view = VacuumGLView(sim.lat.N)
        self.panel = ControlPanel(callbacks)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(440)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.view)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1000, 460])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Left-drag: rotate | wheel: zoom | middle-drag / Ctrl+drag: pan | "
                                     "W A S D E Q: fly (Shift = fast) | R: reset camera | arrows: orbit")
        self._sync_panel_to_sim()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(int(1000 / self.panel.fps_spin.value()))
        self.view.setFocus()

    # ------------------------------------------------------------------------------------------
    def _sync_panel_to_sim(self):
        sim = self.sim
        p = self.panel
        p.set_fields(sim.specs)
        N = sim.lat.N
        for s in (p.z1_spin, p.z2_spin):
            s.setRange(0, N - 1)
        p.z1_spin.setValue(N // 3)
        p.z2_spin.setValue(2 * N // 3)
        p.seed_spin.setValue(sim.seed)
        p.a_spin.setValue(sim.lat.a)
        i = p.N_combo.findData(N)
        if i >= 0:
            p.N_combo.setCurrentIndex(i)
        j = p.theory_combo.findText(sim.theory)
        if j >= 0:
            p.theory_combo.setCurrentIndex(j)
        p.susy_check.setChecked(sim.susy_on)
        p.susy_break.setValue(sim.susy_breaking)
        self.view.set_plates(list(sim.geom.plate_layers))
        self.smoother.reset()

    # ------------------------------------------------------------------------------------------
    # frame loop
    # ------------------------------------------------------------------------------------------
    def _tick(self):
        t0 = time.perf_counter()
        now = t0
        dt_wall = now - self._last_tick
        self._last_tick = now
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt_wall if dt_wall > 0 else 0.0)
        try:
            if self.playing:
                self.sim.step(self.panel.dt_spin.value())
            self._render()
            if self.sim.frame % 10 == 0 or not self.playing:
                self._update_readouts()
        except Exception as e:      # noqa: BLE001 -- keep the GUI alive, show the problem
            self.statusBar().showMessage(f"error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            self.playing = False
            self.panel.play_btn.setChecked(False)
        self._frame_ms = 1000 * (time.perf_counter() - t0)
        want = int(1000 / self.panel.fps_spin.value())
        if self.timer.interval() != want:
            self.timer.setInterval(want)

    def _current_quantity(self):
        p = self.panel
        idx = p.shown_field_indices()
        if not idx:
            return None, False
        smoothing = p.smooth_slider.value() / 10.0
        which = p.quantity_combo.currentIndex()
        if which == 0:
            return self.sim.energy_density(idx, smoothing), False
        if which == 1:
            return self.sim.field_amplitude(idx, smoothing), True
        return self.sim.excess_energy(idx, smoothing), True

    def _style(self, diverging: bool) -> VolumeStyle:
        p = self.panel
        cmap = p.cmap_combo.currentText()
        if diverging and cmap not in DIVERGING:
            cmap = "ice-fire"
        if not diverging and cmap in DIVERGING:
            cmap = "inferno"
        return VolumeStyle(cmap=cmap, alpha_max=p.alpha_slider.value() / 100.0,
                           alpha_gamma=p.gamma_slider.value() / 10.0,
                           hi_percentile=p.hi_slider.value() / 10.0)

    def _render(self):
        q, diverging = self._current_quantity()
        if q is None:
            return
        p = self.panel
        mode = p.mode_combo.currentText()
        self.view.set_mode(mode)
        N = self.sim.lat.N
        if mode == "isosurface":
            data = np.abs(q) if diverging else q
            level = float(np.percentile(data, p.iso_slider.value() / 10.0))
            self.view.set_isosurface(data, level)
            self.last_range = (float(data.min()), float(data.max()))
            return
        rgba, rng = scalar_to_rgba(q, self._style(diverging), diverging, self.smoother,
                                   mask=self.sim.geom.plate_mask() if self.sim.plates_on else None)
        self.last_range = rng
        if mode == "slice":
            axis = p.slice_axis.currentIndex()
            layer = int(round(p.slice_pos.value() / 100.0 * (N - 1)))
            keep = np.zeros(N, dtype=bool)
            keep[layer] = True
            shape = [1, 1, 1]
            shape[axis] = N
            rgba = rgba.copy()
            rgba[..., 3] = np.where(keep.reshape(shape), np.maximum(rgba[..., 3], 40), 0)
        self.view.set_volume(rgba)
        pulse = self.sim.pulse
        self.view.show_efield(pulse is not None and pulse.active(self.sim.t),
                              strength=(pulse.window(self.sim.t) if pulse else 0.0))

    def _update_readouts(self):
        ro = self.sim.readout()
        self.panel.update_readouts(self.sim, ro, self._fps, self._frame_ms, self.last_range)
        idx = self.panel.shown_field_indices() or None
        prof_vac = self.sim.vacuum_profile(idx)
        prof_ens = self.sim.ensemble_profile(idx)
        z = np.arange(self.sim.lat.N)
        h = self.sim.energy_density(idx)
        layer_mean = h.mean(axis=(0, 1))
        if self.sim.plates_on:          # hide the plate layers themselves so the plot zooms on the gap
            pl = self.sim.geom.plate_layers
            prof_vac, prof_ens, layer_mean = (np.array(v, dtype=float) for v in (prof_vac, prof_ens, layer_mean))
            prof_vac[pl] = prof_ens[pl] = layer_mean[pl] = np.nan
        self.panel.prof_vac_curve.setData(z, prof_vac, connect="finite")
        self.panel.prof_ens_curve.setData(z, prof_ens, connect="finite")
        self.panel.prof_sample_curve.setData(z, layer_mean, connect="finite")

    # ------------------------------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------------------------------
    def on_play(self, on: bool):
        self.playing = bool(on)

    def on_step(self):
        self.sim.step(self.panel.dt_spin.value())
        self._render()
        self._update_readouts()

    def on_resample(self):
        self.sim.resample_vacuum(seed=self.panel.seed_spin.value(), temperature=self.panel.temp_spin.value())
        self.smoother.reset()

    def on_rebuild(self):
        p = self.panel
        N = p.N_combo.currentData()
        a = p.a_spin.value()
        theory = p.theory_combo.currentText()
        specs = default_specs(theory, n_massive=p.nmassive_spin.value())
        units = self.sim.units
        self.sim = Simulation(N=N, a=a, seed=p.seed_spin.value(), specs=specs, theory=theory, units=units)
        self.view.set_lattice_size(N)
        self._sync_panel_to_sim()
        self.statusBar().showMessage(f"rebuilt: N={N}, a={a}, {theory}, {len(specs)} fields")

    def on_render_changed(self):
        self.smoother.reset()

    def on_plates(self, insert: bool):
        p = self.panel
        try:
            if insert:
                z1, z2 = p.z1_spin.value(), p.z2_spin.value()
                if self.sim.plates_on:
                    self.sim.set_plates(None, None)         # move = remove + insert (two sudden changes)
                self.sim.set_plates(z1, z2)
            else:
                self.sim.set_plates(None, None)
        except ValueError as e:
            self.statusBar().showMessage(str(e))
            return
        self.view.set_plates(list(self.sim.geom.plate_layers))
        self.smoother.reset()
        self._update_readouts()

    def on_efield(self):
        p = self.panel
        E0 = p.E0_spin.value()
        dur = p.Edur_spin.value()
        tmax = self.sim.max_pulse_duration(E0)
        if dur > tmax:
            self.sim.log(f"WARNING: pulse duration {dur:.3g} exceeds the lattice limit {tmax:.3g}; "
                         f"modes will Bloch-wrap around the Brillouin zone (lattice artefact)")
        self.sim.apply_efield(E0, dur)
        self._update_readouts()

    def on_quench(self):
        p = self.panel
        i = p.quench_field.currentData()
        if i is None:
            return
        m_new = p.quench_mass.value()
        dur = p.quench_dur.value()
        fs = self.sim.fields[i]
        if dur <= 0:
            pred = 0.0
            for bs in fs.blocks:
                w0 = bs.omega
                w1 = np.sqrt(m_new ** 2 + bs.block.k2_spatial(fs.kshift if fs.spec.is_complex else 0.0))
                ok = w0 > 0
                pred += np.sum(obs.sudden_quench_occupation(w0[ok], w1[ok]))
            self.sim.log(f"sudden-quench prediction sum_k (w1-w0)^2/(4 w0 w1) = {fs.spec.nosc*pred:.4g} quanta")
        self.sim.quench_mass(i, m_new, duration=dur)
        self._update_readouts()

    def on_susy(self, on: bool, m_break: float):
        self.sim.susy_on = bool(on)
        self.sim.susy_breaking = float(m_break)
        self._update_readouts()

    def on_units(self):
        name = self.panel.ls_combo.currentText()
        self.sim.units = Units(STRING_LENGTH_PRESETS.get(name, 1e-34))
        self._update_readouts()

    def on_save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save simulation state", "vacuum_state.npz",
                                                        "NumPy archive (*.npz)")
        if path:
            if not path.endswith(".npz"):
                path += ".npz"
            self.sim.save(path)
            self.sim.log(f"state saved to {os.path.basename(path)}")

    def on_load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load simulation state", "", "NumPy archive (*.npz)")
        if not path:
            return
        try:
            sim = Simulation.load(path)
        except Exception as e:      # noqa: BLE001
            self.statusBar().showMessage(f"could not load: {e}")
            return
        self.sim = sim
        self.view.set_lattice_size(sim.lat.N)
        self._sync_panel_to_sim()
        self.view.set_plates(list(sim.geom.plate_layers))

    def on_screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save screenshot", "vacuum.png", "PNG image (*.png)")
        if path:
            self.view.screenshot(path)

    # forward flying keys to the view even when a control has focus (unless typing in a text box)
    def keyPressEvent(self, ev):
        fw = QtWidgets.QApplication.focusWidget()
        if isinstance(fw, (QtWidgets.QAbstractSpinBox, QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit, QtWidgets.QComboBox)):
            return super().keyPressEvent(ev)
        if ev.key() in self.view.FLY_KEYS or ev.key() == Qt.Key.Key_R:
            self.view.keyPressEvent(ev)
            return
        super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        if ev.key() in self.view.FLY_KEYS:
            self.view.keyReleaseEvent(ev)
            return
        super().keyReleaseEvent(ev)
