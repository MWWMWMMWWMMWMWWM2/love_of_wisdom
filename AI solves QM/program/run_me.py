#!/usr/bin/env python3
"""
potential_builder.py -- draw your own potential, let the stored AI model find the bound states.

    python potential_builder.py

How it works
  * Click in the upper panel to place anchor points from LEFT to RIGHT (x always increases; y may be negative:
    a negative potential is a well, i.e. a region where the particle is bound / has extra kinetic energy).
  * Before clicking the next point choose how the segment from the previous point will be interpolated:
      linear | quadratic | cubic | exponential | step (horizontal at the left height) | smooth step | spline (through all points)
  * Left of the first point and right of the last point the potential continues horizontally.
  * Type the levels you want (e.g. "0,1,2") and press  Solve with AI.  The pre-trained model
    (pretrained_model.npz) predicts each eigenfunction from the potential's shape and is then fine-tuned
    variationally.  The table compares: AI model | exact grid diagonalisation | WKB (Bohr-Sommerfeld).
  * Discrete vs continuous: only energies below the lower of the two asymptotic values of V are bound
    states (discrete).  Levels above that are flagged "continuum"; press  Scattering  and type an energy to
    see the scattering wave at that energy and its transmission T(E) (computed numerically), plus the T(E)
    curve.  Resonances show up as peaks.
  * Evolve in time: after solving, type an initial state -- "gauss x0,sigma,k0" (a wave packet) or
    "0:1,1:1" (a superposition of solved levels) -- and press Evolve.  Two propagators run live: the spectral
    one psi(t) = sum c_n psi_n e^{-i E_n t} built from the AI levels (with the probabilities |c_n|^2 shown),
    and the full time-dependent Schroedinger equation (split-operator).  Their fidelity shows how completely
    the solved levels describe the dynamics (unsolved levels / continuum = missing probability).
Buttons: Undo, Clear, Solve with AI, Scattering, Evolve in time, Save figure.   Units: hbar = m = 1, box x in [-8, 8].
"""
from __future__ import annotations
import os
import sys
import time
import textwrap
import logging
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox
from scipy.interpolate import CubicSpline

from qm_model import (grid, fit, LiveFit, scattering, load_model, continuum_threshold, X_HALF, gaussian_packet,
                      SpectralEvolver, SplitOperator, fidelity, ThawedGaussian, ClassicalEnsemble, density_overlap)
from matplotlib.animation import FuncAnimation
from continual import ContinualTrainer

BG = "#0b0d12"
STABLE_TIME = 15.0           # default for the "stable s" box: accept a level once psi has stayed stable this long
HERE = os.path.dirname(os.path.abspath(__file__))
# GIFs go to an easy-to-find "GIF" folder: beside "program" in the release layout, else beside the script
GIF_DIR = os.path.join(os.path.dirname(HERE) if os.path.basename(HERE) == "program" else HERE, "GIF")
HARD_LIMIT = 90.0            # backstop per level: stop anyway and keep the best state
PSI_TOL = 1e-6               # default for the "psi tol" box: psi is stable when median int (psi - psi_avg)^2 dx < tol
LOG = logging.getLogger("builder")
logging.basicConfig(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver_log.txt"),
                    level=logging.INFO, format="%(asctime)s %(message)s")
SEGMENTS = ["linear", "quadratic", "cubic", "exponential", "step", "smooth step", "spline"]


def build_potential(x, pts, kinds):
    """pts: list of (x, y) increasing in x; kinds[i]: interpolation of segment i (between pts[i], pts[i+1])."""
    if not pts:
        return np.zeros_like(x)
    V = np.full_like(x, pts[0][1])
    if "spline" in kinds and len(pts) >= 3:
        xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        cs = CubicSpline(xs, ys, bc_type="natural")
    for i in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        seg = (x >= x0) & (x < x1)
        t = (x[seg] - x0) / max(x1 - x0, 1e-9)
        k = kinds[i]
        if k == "linear":
            V[seg] = y0 + (y1 - y0) * t
        elif k == "quadratic":
            V[seg] = y0 + (y1 - y0) * t ** 2
        elif k == "cubic":
            V[seg] = y0 + (y1 - y0) * (3 * t ** 2 - 2 * t ** 3)
        elif k == "exponential":
            r = 3.0
            V[seg] = y0 + (y1 - y0) * (np.exp(r * t) - 1) / (np.exp(r) - 1)
        elif k == "step":
            V[seg] = y0
        elif k == "smooth step":
            V[seg] = y0 + (y1 - y0) * 0.5 * (1 + np.tanh(10 * (t - 0.5)))
        elif k == "spline" and len(pts) >= 3:
            V[seg] = cs(x[seg])
        else:
            V[seg] = y0 + (y1 - y0) * t
    V[x >= pts[-1][0]] = pts[-1][1]
    return V


class LearnThread(threading.Thread):
    """Runs the background learner OUTSIDE the GUI thread, so a slow tick (held-out evaluation, writing the
    model or the experience file) can never freeze the window.  It pauses while the UI is busy (solving or
    evolving) so it does not compete for the CPU, and takes `lock` around every change to the shared network."""
    def __init__(self, builder):
        super().__init__(daemon=True, name="auto-learn")
        self.b = builder
        self.stop = False

    def run(self):
        b = self.b
        while not self.stop:
            try:
                if b.auto_learn and b.trainer is not None and b.livefit is None and not b.evolving:
                    with b.net_lock:
                        while b.pending_memory:
                            b.trainer.remember(*b.pending_memory.pop(0))
                        b.trainer.tick()
                    time.sleep(0.25)
                else:
                    time.sleep(0.2)
            except Exception:  # noqa: BLE001
                LOG.exception("auto-learn error")
                time.sleep(2.0)


class Builder:
    def __init__(self):
        self.x = grid()
        self.pts, self.kinds = [], []
        self.result = None
        self.net, self.meta = load_model()
        self.fig = plt.figure(figsize=(13, 8), facecolor=BG)
        self.axV = self.fig.add_axes([0.06, 0.56, 0.62, 0.38], facecolor=BG)
        self.axPsi = self.fig.add_axes([0.06, 0.09, 0.60, 0.39], facecolor=BG)
        self.axTxt = self.fig.add_axes([0.70, 0.03, 0.29, 0.26], facecolor=BG); self.axTxt.set_axis_off()
        self.status_cols = 58       # characters per line in the status panel (7.5 pt monospace, 0.29 * 13 in)
        self.radio = RadioButtons(self.fig.add_axes([0.70, 0.60, 0.12, 0.33], facecolor="#161b22"), SEGMENTS)
        for lab in self.radio.labels:
            lab.set_color("#c9d1d9"); lab.set_fontsize(9)
        TB = dict(color="#161b22", hovercolor="#21262d", textalignment="left")
        self.levels_box = TextBox(self.fig.add_axes([0.90, 0.905, 0.085, 0.04]), "levels ", initial="0,1,2", **TB)
        self.energy_box = TextBox(self.fig.add_axes([0.90, 0.855, 0.085, 0.04]), "E scat ", initial="", **TB)
        self.stable_box = TextBox(self.fig.add_axes([0.90, 0.805, 0.085, 0.04]), "stable s ", initial=f"{STABLE_TIME:g}", **TB)
        self.tol_box = TextBox(self.fig.add_axes([0.90, 0.755, 0.085, 0.04]), "psi tol ", initial=f"{PSI_TOL:g}", **TB)
        self.b_solve = self._button([0.84, 0.69, 0.14, 0.05], "Solve with AI", self.solve, accent=True)
        self.b_scat = self._button([0.84, 0.627, 0.14, 0.05], "Scattering T(E)", self.scatter)
        self.b_undo = self._button([0.84, 0.564, 0.065, 0.05], "Undo", self.undo)
        self.b_clear = self._button([0.915, 0.564, 0.065, 0.05], "Clear", self.clear, danger=True)
        self.b_save = self._button([0.84, 0.50, 0.14, 0.05], "Save figure", self.save_fig)
        self.init_box = TextBox(self.fig.add_axes([0.90, 0.44, 0.085, 0.04]), "init ", initial="gauss -2,0.7,0", **TB)
        self.b_learn = self._button([0.84, 0.31, 0.14, 0.05], "Auto-learn: ON", self.toggle_learn)
        self.learn_label = self.fig.text(0.06, 0.012, "", color="#8b949e", fontsize=8, va="bottom")
        self.b_evolve = self._button([0.84, 0.38, 0.14, 0.05], "Evolve in time", self.evolve, accent=True)
        self.b_gif = self._button([0.70, 0.38, 0.12, 0.05], "Create GIF", self.request_gif)
        self.b_gif.ax.set_visible(False)        # shown only while an evolution is running
        self.gif_requested = False
        self.anim = None
        for tb in (self.levels_box, self.energy_box, self.init_box, self.stable_box, self.tol_box):
            tb.label.set_color("#c9d1d9"); tb.text_disp.set_color("#e6edf3"); tb.cursor.set_color("#e6edf3")
            tb.on_submit(lambda _t, tb=tb: self._flash(tb.ax, "#1f6feb", back="#161b22"))
        self.radio.on_clicked(lambda _l: self._radio_colours())
        self._radio_colours()
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        # background learning: the stored model trains on random + solved potentials whenever the UI is idle
        self.trainer = ContinualTrainer(self.net, self.x) if self.net is not None else None
        self.auto_learn = self.trainer is not None
        self.livefit = None
        self.cancel_requested = False
        self.evolving = False
        self.net_lock = threading.Lock()        # guards the shared network between the learner and the solver
        self.pending_memory = []                # solved potentials waiting to be stored by the learner thread
        self.learn_thread = LearnThread(self); self.learn_thread.start()
        self.learn_timer = self.fig.canvas.new_timer(interval=500)   # GUI side: only refreshes the status line
        self.learn_timer.add_callback(self._learn_tick)
        self.learn_timer.start()
        self.status("Click in the upper panel to place points left to right.  Model: "
                    + ("pretrained_model.npz loaded" if self.net else "NOT FOUND (run train_pretrained.py); training from scratch"))
        self.draw()

    # ------------------------------------------------------------------------------------------
    # widget styling: hover colour, press flash, "busy" state, active radio highlight
    def _button(self, rect, label, callback, accent=False, danger=False):
        base = "#1f6feb" if accent else ("#6e2a2a" if danger else "#30363d")
        hover = "#388bfd" if accent else ("#a12d2d" if danger else "#484f58")
        b = Button(self.fig.add_axes(rect), label, color=base, hovercolor=hover)
        b.label.set_fontsize(9); b.label.set_color("#e6edf3")
        b._base, b._hover = base, hover

        def wrapped(ev):
            self._flash(b.ax, "#e6edf3", back=b.color)
            callback(ev)
        b.on_clicked(wrapped)
        return b

    def _flash(self, ax, colour, back=None):
        """Briefly recolour a widget so every click is visibly acknowledged."""
        old = ax.get_facecolor() if back is None else back
        ax.set_facecolor(colour); self.fig.canvas.draw_idle()
        t = self.fig.canvas.new_timer(interval=160)
        t.single_shot = True
        t.add_callback(lambda: (ax.set_facecolor(old), self.fig.canvas.draw_idle()))
        t.start()
        self._timers = getattr(self, "_timers", [])[-5:] + [t]

    def _busy(self, button, busy):
        button.color = "#8b949e" if busy else button._base
        button.hovercolor = "#8b949e" if busy else button._hover
        button.ax.set_facecolor(button.color)
        button.label.set_text(button.label.get_text().replace(" ...", "") + (" ..." if busy else ""))
        self.fig.canvas.draw(); self.fig.canvas.flush_events()

    def _radio_colours(self):
        sel = self.radio.value_selected
        for lab in self.radio.labels:
            lab.set_color("#58a6ff" if lab.get_text() == sel else "#c9d1d9")
            lab.set_fontweight("bold" if lab.get_text() == sel else "normal")
        if self.pts or self.result:
            self.draw()
        else:
            self.fig.canvas.draw_idle()

    def save_fig(self, ev):
        self.fig.savefig("my_potential.png", dpi=120, facecolor=BG)
        self.status("saved my_potential.png")

    def status(self, msg):
        self.axTxt.cla(); self.axTxt.set_axis_off()
        lines = []
        for ln in msg.split("\n"):
            lines += textwrap.wrap(ln, self.status_cols, subsequent_indent="  ", break_long_words=True) or [""]
        self.axTxt.text(0, 1, "\n".join(lines[:18]), color="#e6edf3", fontsize=7.5, va="top", family="monospace")
        self.fig.canvas.draw_idle()

    def V(self):
        return build_potential(self.x, self.pts, self.kinds)

    def on_click(self, ev):
        if ev.inaxes != self.axV or ev.button != 1:
            return
        xn = float(ev.xdata)
        if self.pts and xn <= self.pts[-1][0]:
            xn = self.pts[-1][0] + 0.2
        if xn > X_HALF - 0.05:
            return
        self.stop_anim()
        if self.pts:
            self.kinds.append(self.radio.value_selected)
        self.pts.append((xn, float(ev.ydata)))
        self.result = None
        self.status(f"{len(self.pts)} point(s).  Next segment: {self.radio.value_selected}.  Press Solve with AI when done.")
        self.draw()

    def undo(self, ev):
        self.cancel_requested = True
        self.stop_anim()
        if self.pts:
            self.pts.pop()
            if self.kinds:
                self.kinds.pop()
        self.result = None
        self.status(f"removed last point ({len(self.pts)} left)")
        self.draw()

    def clear(self, ev):
        """Full reset, always allowed: cancels a running solve (the solve loop checks the flag), stops the
        time evolution / scattering animation, forgets points and results, redraws."""
        self.cancel_requested = True            # seen by the solve loop right after this handler returns
        self.stop_anim()
        self.pts, self.kinds, self.result = [], [], None
        self.status("cleared.  Click in the upper panel to place points left to right.")
        self.draw()

    def draw(self):
        ax = self.axV
        ax.cla()
        ax.set_facecolor(BG)
        V = self.V()
        ax.plot(self.x, V, color="#58a6ff", lw=2)
        ax.fill_between(self.x, V.min() - 1, V, color="#58a6ff", alpha=0.08)
        if self.pts:
            ax.plot(*zip(*self.pts), "o", color="#ffa657")
        Eth = continuum_threshold(V) if self.pts else 0.0
        ax.axhline(Eth, color="#8b949e", ls="--", lw=0.8)
        ax.text(X_HALF - 0.2, Eth, "continuum threshold", color="#8b949e", fontsize=8, ha="right", va="bottom")
        if self.result:
            for k, r in self.result["levels"].items():
                if r["bound"]:
                    ax.axhline(r["E_model"], color="#3fb950", lw=1)
                    ax.text(-X_HALF + 0.2, r["E_model"], f"E{k}={r['E_model']:.3f}", color="#3fb950", fontsize=8, va="bottom")
        ax.set_xlim(-X_HALF, X_HALF)
        lo, hi = (V.min() - 2, V.max() + 2) if self.pts else (-5, 5)
        ax.set_ylim(min(lo, -5), max(hi, 5))
        ax.set_ylabel("V(x)", color="#c9d1d9"); ax.tick_params(colors="#c9d1d9", labelsize=8)
        ax.set_title("potential  (click to add points; current segment type: " + self.radio.value_selected + ")",
                     color="#c9d1d9", fontsize=9)
        self._clear_psi()
        self.axPsi.set_xlim(-X_HALF, X_HALF)
        if self.result:
            self.draw_result()
        self.fig.canvas.draw_idle()

    def _clear_psi(self):
        for twin in [a for a in self.fig.axes if a is not self.axPsi and getattr(a, "_is_twin_of", None) is self.axPsi]:
            twin.remove()
        self.axPsi.cla(); self.axPsi.set_facecolor(BG)
        self.axPsi.tick_params(colors="#c9d1d9", labelsize=8)

    def draw_result(self):
        r = self.result
        ax = self.axPsi
        if r.get("scattering"):
            s = r["scattering"]
            ax.plot(self.x, np.abs(s["psi"]) ** 2, color="#ffa657", lw=1.5, label=f"|psi_E|^2, E={s['E']:.3f}, T={s['T']:.3f}")
            ax.plot(self.x, s["psi"].real, color="#58a6ff", lw=0.8, alpha=0.7, label="Re psi")
            ax2 = ax.twinx(); ax2._is_twin_of = ax
            ax2.plot(r["TE"][0], r["TE"][1], color="#3fb950", lw=1)
            ax2.set_ylim(0, 1.05); ax2.set_ylabel("T(E) (green)", color="#3fb950", fontsize=8)
            ax2.tick_params(colors="#3fb950", labelsize=8)
            ax.set_title("scattering state (continuum) and transmission curve", color="#c9d1d9", fontsize=9)
        else:
            offs = 0.0
            for k, lv in sorted(r["levels"].items()):
                if not lv["bound"]:
                    continue
                ax.plot(self.x, lv["psi"] + offs, color="#ffa657", lw=1.8, label="AI model" if offs == 0 else None)
                ax.plot(self.x, r["psi_exact"][:, k] + offs, color="#8b949e", lw=3, alpha=0.5,
                        label="exact (grid)" if offs == 0 else None)
                ax.text(-X_HALF + 0.2, offs + 0.05, f"n={k}", color="#c9d1d9", fontsize=8)
                offs += 1.2
            ax.set_title("eigenfunctions (offset for clarity): AI model vs exact numerics", color="#c9d1d9", fontsize=9)
        ax.legend(fontsize=8, facecolor="#161b22", labelcolor="#c9d1d9", loc="upper right")

    # ------------------------------------------------------------------------------------------
    def solve(self, ev):
        """Button handler: everything is inside try/except so a failure is shown in the panel and logged."""
        try:
            self._solve()
        except Exception as e:  # noqa: BLE001
            LOG.exception("solve failed before training started")
            self.livefit = None; self._busy(self.b_solve, False)
            self.status(f"solver error: {type(e).__name__}: {e}\n(details in solver_log.txt)")

    def _solve(self):
        """Start the fine-tuning as a real-time process: the wave function is drawn while it converges."""
        LOG.info("Solve pressed: %d points, levels '%s', busy=%s", len(self.pts), self.levels_box.text, self.livefit is not None)
        if self.livefit is not None:
            self.status("already solving -- wait for it to finish"); return
        if len(self.pts) < 2:
            self.status("Place at least two points first."); return
        try:
            levels = sorted({int(s) for s in self.levels_box.text.replace(";", ",").split(",") if s.strip() != ""})
        except ValueError:
            self.status("levels must be integers, e.g. 0,1,2"); return
        try:
            stable_time = float(self.stable_box.text.replace(",", "."))
            if not 0 <= stable_time <= 600:
                raise ValueError
        except ValueError:
            self.status("stable s must be a number of seconds between 0 and 600 (default 15)"); return
        try:
            psi_tol = float(self.tol_box.text.replace(",", "."))
            if not 1e-14 <= psi_tol <= 1e-1:
                raise ValueError
        except ValueError:
            self.status("psi tol must be a number between 1e-14 and 0.1, e.g. 1e-6 (default): the wave function "
                        "counts as stable when its change, median int (psi - psi_avg)^2 dx, is below it"); return
        self.stop_anim()
        self.result = None
        self.cancel_requested = False
        with self.net_lock:                     # the learner may be mid-update: copy the weights atomically
            lf = self.livefit = LiveFit(self.x, self.V(), levels, net=self.net, explore=0,
                                        time_limit=max(HARD_LIMIT, 4 * stable_time), stable_time=stable_time, psi_tol=psi_tol)
        self.solve_t0 = self.last_frame_t = time.time(); self.anim_restarts = 0
        LOG.info("solve: levels %s, %d points", levels, len(self.pts))
        self._busy(self.b_solve, True)
        self.status(f"training in real time: pre-trained prediction -> variational fine-tuning (watch the lower panel).\n"
                    f"A level is accepted when its wave function stays stable (median int dpsi^2 dx < {psi_tol:.0e}) "
                    f"for {stable_time:g} s while training continues; a swing resets the timer. "
                    f"Hard limit {max(HARD_LIMIT, 4 * stable_time):.0f} s per level.")
        # plain loop (no animation timer to depend on): draw every 0.25 s
        last_draw = 0.0
        try:
            if lf.done:                                   # nothing to optimise: all requested levels are unbound
                self._solve_draw(lf, 0.0)
            while not lf.done:
                lf.step(30)
                now = time.time()
                if now - last_draw > 0.25 or lf.done:
                    self._solve_draw(lf, now - self.solve_t0)
                    self.fig.canvas.draw(); self.fig.canvas.flush_events()   # processes button clicks (Clear)
                    last_draw = time.time()
                    if self.cancel_requested:
                        LOG.info("solve cancelled by Clear at step %d", lf.it)
                        break
        except Exception as e:  # noqa: BLE001
            LOG.exception("solver error")
            self.status(f"solver error: {type(e).__name__}: {e}\n(details in solver_log.txt)")
        finally:
            self.livefit = None; self._busy(self.b_solve, False)
            if self.cancel_requested:
                self.cancel_requested = False
                self.status("cleared (solver cancelled).  Click in the upper panel to place points left to right.")
                self.draw()

    def _solve_frame(self, _):
        lf = self.livefit
        if lf is None:
            return []
        try:
            self.last_frame_t = time.time()
            elapsed = time.time() - self.solve_t0
            if elapsed > SOLVE_BUDGET and not lf.done:
                LOG.info("budget exceeded at step %d level %s", lf.it, lf.k)
                lf.abort(f"absolute time limit {SOLVE_BUDGET:.0f} s reached")
            else:
                lf.step(30)
            self._solve_draw(lf, elapsed)
        except Exception as e:  # noqa: BLE001 -- never freeze silently: show the error and stop
            LOG.exception("solver error")
            self.stop_anim(); self.livefit = None; self._busy(self.b_solve, False)
            self.status(f"solver error: {type(e).__name__}: {e}\n(details in solver_log.txt)")
        return []

    def _solve_draw(self, lf, elapsed):
        ax = self.axPsi
        self._clear_psi(); ax.set_xlim(-X_HALF, X_HALF)
        offs = 0.0
        for k, fv in lf.found:
            ax.plot(self.x, fv + offs, color="#ffa657", lw=1.6)
            ax.plot(self.x, lf.psi_exact[:, k] + offs, color="#8b949e", lw=3, alpha=0.4)
            ax.text(-X_HALF + 0.2, offs + 0.05, f"n={k} done", color="#3fb950", fontsize=8)
            offs += 1.2
        if lf.current is not None:
            k, E, psi = lf.current
            if psi @ lf.psi_exact[:, k] < 0:
                psi = -psi
            ax.plot(self.x, psi + offs, color="#f778ba", lw=1.8, label=f"n={k} training: E = {E:.5f}  (exact {lf.E_exact[k]:.5f})")
            ax.plot(self.x, lf.psi_exact[:, k] + offs, color="#8b949e", lw=3, alpha=0.4, label="exact grid")
            ax.set_title(f"level n={k}  step {lf.it}  E = {E:.6f}  error {abs(E - lf.E_exact[k]):.1e}  "
                         f"var(H) {lf.variance:.1e}   {elapsed:.0f} s\n{lf.phase}",
                         color="#c9d1d9", fontsize=8.5)
            ax.legend(fontsize=8, facecolor="#161b22", labelcolor="#c9d1d9", loc="upper right")
        if lf.done:
            self.stop_anim()
            self.result = lf.result()
            self.livefit = None
            LOG.info("done in %.1f s: %s", elapsed, {k: (round(v["E_model"], 5), v["steps"], v["kicks"]) if v.get("bound") else "unbound/skipped"
                                                    for k, v in self.result["levels"].items()})
            self._busy(self.b_solve, False)
            if self.trainer is not None:
                self.pending_memory.append((self.result["V"], self.result["psi_exact"], self.result["E_exact"]))
            self._report()
            self.draw()
        return []

    def _report(self):
        r = self.result
        lines = [f"bound states in this potential: {r['n_bound_exact']}  (threshold E < {r['threshold']:.3f})"]
        if not any(lv.get("bound") for lv in r["levels"].values()):
            lines.append("none of the requested levels is a bound state: a bound state needs E below BOTH ends of the "
                         "potential (the dashed line = the lower end). Raise the ends of the potential, deepen the well, "
                         "or use Scattering T(E) for the continuum.")
        lines += [
                 f"{'n':>2} {'AI':>9} {'exact':>9} {'WKB':>9} {'AI err':>7} {'WKB err':>7} {'steps':>6}"]
        for k, lv in sorted(r["levels"].items()):
            if lv["bound"]:
                w = lv["E_wkb"]
                ws = f"{w:9.5f}" if w is not None else f"{'n/a':>9}"
                we = f"{abs(w - lv['E_exact']):7.1e}" if w is not None else f"{'':>7}"
                lines.append(f"{k:2d} {lv['E_model']:9.5f} {lv['E_exact']:9.5f} {ws} "
                             f"{abs(lv['E_model']-lv['E_exact']):7.1e} {we} {lv['steps']:6d}")
            else:
                lines.append(f"{k:2d}  -- not solved (time limit reached before this level) --" if lv.get("skipped") else
                             f"{k:2d}  -- continuum (E_grid {lv['E_exact']:.3f} is a box artefact); use Scattering --")
        if any(lv["bound"] for lv in r["levels"].values()):
            lines.append("pre-trained start energies: " + ", ".join(f"{lv['E_start']:.3f}" for lv in r["levels"].values() if lv["bound"]))
            esc = sum(lv.get("escapes", 0) for lv in r["levels"].values() if lv["bound"])
            kicks = sum(lv.get("kicks", 0) for lv in r["levels"].values() if lv["bound"])
            if kicks:
                lines.append(f"exploration: {kicks} kicks, {esc} escaped a local minimum")
            stuck = [k for k, lv in sorted(r["levels"].items()) if lv.get("timed_out")]
            if stuck:
                lines.append(f"level(s) {stuck}: never stable for the requested time -- stopped by the hard limit; "
                             f"the best state reached is kept (see AI err).")
            lines.append("time per level: " + ", ".join(f"n={k}: {lv['seconds']:.0f}s ({lv['resets']} timer resets)"
                                                       for k, lv in sorted(r["levels"].items()) if lv.get("bound")))
            rel = [(k, lv["relabelled_from"]) for k, lv in sorted(r["levels"].items()) if "relabelled_from" in lv]
            if rel:
                lines.append("levels re-ordered by energy: " + ", ".join(f"n={a} was solved as n={b}" for a, b in rel))
        self.status("\n".join(lines))

    # ------------------------------------------------------------------------------------------
    # background learning
    def toggle_learn(self, ev):
        if self.trainer is None:
            self.status("no stored model (run train_pretrained.py first)"); return
        self.auto_learn = not self.auto_learn
        self.b_learn.label.set_text("Auto-learn: ON" if self.auto_learn else "Auto-learn: OFF")

    def _learn_tick(self):
        """GUI timer: shows the learner's statistics; all training happens in LearnThread."""
        if not self.auto_learn or self.trainer is None or self.livefit is not None or self.evolving:
            return
        t = self.trainer
        txt = (f"background learning: {t.steps} steps, loss {t.loss:.3e}, "
               f"held-out err {t.heldout_err:.3f} (best {t.best_err if t.best_err is not None else float('nan'):.3f}), "
               f"{len(t.experience)} remembered potentials")
        self.learn_label.set_text(txt)
        self.fig.canvas.draw_idle()

    def stop_anim(self, ev=None):
        self.evolving = False                   # ends the evolve loop at its next frame
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
                self.anim._stop()
            except Exception:  # noqa: BLE001
                pass
            self.anim = None

    def evolve(self, ev):
        """Real-time solution of the full time-dependent Schroedinger equation for a chosen initial state.
        Two propagators run side by side: the spectral one built from the AI eigenstates (probabilities
        |c_n|^2 and energies E_n) and the split-operator solver of the full equation (ground truth).
        Their overlap (fidelity) tells how much of the dynamics the computed levels capture."""
        if not self.result or not any(lv.get("bound") for lv in self.result["levels"].values()):
            self.status("First press 'Solve with AI' (the levels you solve are the ones used for the dynamics)."); return
        self.stop_anim()
        x, V = self.x, self.V()
        txt = self.init_box.text.strip()
        levels = sorted(k for k, lv in self.result["levels"].items() if lv.get("bound"))
        tga = cls = None
        try:
            if txt.startswith("gauss"):
                x0, sig, k0 = [float(v) for v in txt[5:].split(",")]
                psi0 = gaussian_packet(x, x0, sig, k0)
                tga = ThawedGaussian(x, V, x0, sig, k0)          # semiclassical (time-dependent WKB) baseline
                cls = ClassicalEnsemble(x, V, x0, sig, k0, n=6000)  # classical-trajectory baseline
            else:                                   # e.g. "0:1,1:1" -> superposition of solved levels
                psi0 = np.zeros(len(x), complex)
                for item in txt.split(","):
                    k, c = item.split(":"); psi0 += complex(c) * self.result["levels"][int(k)]["psi"]
                psi0 /= np.sqrt(np.sum(np.abs(psi0) ** 2) * (x[1] - x[0]))
        except Exception as e:  # noqa: BLE001
            self.status(f"init must be 'gauss x0,sigma,k0' or 'n:c,n:c' with solved levels ({e})"); return
        spec = SpectralEvolver(x, [self.result["levels"][k]["E_model"] for k in levels],
                               [self.result["levels"][k]["psi"] for k in levels], psi0)
        full = SplitOperator(x, V, psi0)
        self.status(f"evolving.  probabilities |c_n|^2 of the solved levels: " +
                    ", ".join(f"n={k}: {p:.3f}" for k, p in zip(levels, spec.prob)) +
                    f"\ncaptured by solved levels: {spec.captured:.3f}  (rest = higher levels / continuum)\n"
                    f"<E> within solved levels = {spec.energy_mean():.4f}\n"
                    f"white = full Schroedinger eq., orange = AI levels, green = thawed Gaussian (semiclassical),\n"
                    f"purple = classical trajectories.  Title: fidelities with the full equation.")
        ax = self.axPsi
        # persistent artists (created once, data updated per frame): a cla() per frame costs ~100 ms in
        # matplotlib, set_data costs ~1 ms, so the loop stays responsive to Clear / Undo / Solve clicks
        self._clear_psi(); ax.set_xlim(-X_HALF, X_HALF)
        pf0 = full.psi; ymax = 1.2 * np.abs(pf0).max() ** 2
        ax.fill_between(x, 0, V / (abs(V).max() + 1e-9) * 0.5 * ymax, color="#3a4150", alpha=0.5)
        l_full, = ax.plot(x, np.abs(pf0) ** 2, color="white", lw=2.5, alpha=0.6, label="|psi|^2 full TDSE")
        l_spec, = ax.plot(x, np.abs(pf0) ** 2, color="#ffa657", lw=1.5, label="|psi|^2 spectral (AI levels)")
        l_re, = ax.plot(x, pf0.real * 0.8 * np.abs(pf0).max(), color="#58a6ff", lw=0.8, label="Re psi (spectral)")
        if tga is not None:
            l_tga, = ax.plot(x, np.abs(pf0) ** 2, "--", color="#3fb950", lw=1.2, label="|psi|^2 thawed Gaussian (semiclassical)")
            l_cls, = ax.plot(x, 0 * x, ":", color="#d2a8ff", lw=1.5, label="classical trajectory ensemble")
        ax.set_ylim(-0.9 * ymax, ymax)
        ax.legend(fontsize=8, facecolor="#161b22", labelcolor="#c9d1d9", loc="upper right")
        title = ax.set_title("", color="#c9d1d9", fontsize=8.5)
        state = {"t": 0.0}

        def frame():
            state["t"] += 0.03
            t = state["t"]
            ps = spec.psi(t); pf = full.advance(t)
            l_full.set_ydata(np.abs(pf) ** 2); l_spec.set_ydata(np.abs(ps) ** 2)
            l_re.set_ydata(ps.real * 0.8 * np.abs(ps).max())
            txt = f"t = {t:.2f}   fidelity AI-levels = {fidelity(x, ps, pf):.4f}"
            if tga is not None:
                pt = tga.advance(t); rc = cls.advance(t)
                l_tga.set_ydata(np.abs(pt) ** 2); l_cls.set_ydata(rc)
                txt += f"   thawed-Gaussian = {fidelity(x, pt, pf):.3f}   classical density overlap = {density_overlap(x, rc, np.abs(pf) ** 2):.3f}"
            title.set_text(txt + f"   captured prob. {spec.captured:.3f}   (Clear stops)")
        # plain loop instead of an animation timer: every frame pumps the GUI events, so Clear / Undo / Solve
        # stay clickable (each of them calls stop_anim, which drops the flag and ends this loop)
        self.evolving = True
        self.gif_requested = False
        LOG.info("evolve: %s", txt)
        canvas = self.fig.canvas
        self.b_gif.ax.set_visible(True)
        frames, stamps, n_frame, crop = [], [], 0, None
        try:
            canvas.draw()
            bg = canvas.copy_from_bbox(self.fig.bbox) if hasattr(canvas, "copy_from_bbox") else None
            region = ax.get_tightbbox(canvas.get_renderer()).padded(8) if bg is not None else None
            if hasattr(canvas, "buffer_rgba"):          # pixel rows/cols of the two plot panels, for the GIF
                r = canvas.get_renderer(); hgt = int(self.fig.bbox.height)
                bb = self.axV.get_tightbbox(r).union([self.axV.get_tightbbox(r), ax.get_tightbbox(r)]).padded(10)
                crop = (max(0, hgt - int(bb.y1)), min(hgt, hgt - int(bb.y0)), max(0, int(bb.x0)), int(bb.x1))
            while self.evolving:
                t0 = time.time()
                frame()
                if bg is not None:                      # blit only the lower panel
                    canvas.restore_region(bg)
                    ax.draw(canvas.get_renderer())
                    canvas.blit(region)
                else:
                    canvas.draw()
                n_frame += 1
                if crop is not None and n_frame % 2 == 0 and len(frames) < 400:     # ~12 fps, <= 400 frames
                    from PIL import Image
                    buf = np.asarray(canvas.buffer_rgba())[crop[0]:crop[1], crop[2]:crop[3], :3]
                    frames.append(Image.fromarray(buf.copy()).convert("P", palette=Image.ADAPTIVE, colors=128))
                    stamps.append(time.time())
                canvas.flush_events()
                if self.gif_requested:
                    self.gif_requested = False
                    path = self._write_gif(frames, np.diff(stamps))
                    LOG.info("gif: %s (%d frames)", path, len(frames))
                    self.status(f"saved GIF/{os.path.basename(path)} ({len(frames)} frames, from the start of this "
                                f"evolution until now).  Evolution continues; press again for a longer one."
                                if path else "no frames recorded yet")
                if not self.evolving:
                    break
                spare = 0.04 - (time.time() - t0)
                if spare > 0:
                    time.sleep(spare)
        except Exception as e:  # noqa: BLE001
            LOG.exception("evolve error")
            self.status(f"evolve error: {type(e).__name__}: {e}\n(details in solver_log.txt)")
        finally:
            self.evolving = False
            self.b_gif.ax.set_visible(False)
            self.fig.canvas.draw_idle()

    def request_gif(self, ev):
        """Pressed during an evolution: the frames recorded since it started (potential + lower panel) are
        written as an animated GIF next to the scripts; the evolution keeps running."""
        if self.evolving:
            self.gif_requested = True

    def _write_gif(self, frames, dts):
        from PIL import Image
        import datetime
        if not frames:
            return None
        os.makedirs(GIF_DIR, exist_ok=True)
        name = datetime.datetime.now().strftime("evolution_%Y%m%d_%H%M%S.gif")
        path = os.path.join(GIF_DIR, name)
        ms = int(max(20, min(200, 1000 * float(np.median(dts)) if len(dts) else 80)))
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=False)
        return path

    def scatter(self, ev):
        if len(self.pts) < 2:
            self.status("Place points first."); return
        V = self.V()
        Emin = max(V[0], V[-1]) + 1e-3
        Emax = Emin + max(4.0, V.max() - Emin + 3.0)
        Es = np.linspace(Emin, Emax, 300)
        Ts = np.array([scattering(self.x, V, E)["T"] for E in Es])
        try:
            E = float(self.energy_box.text) if self.energy_box.text.strip() else Es[np.argmax(Ts)]
        except ValueError:
            E = Es[np.argmax(Ts)]
        if E <= Emin:
            self.status(f"scattering needs E > {Emin:.3f} (both asymptotic values of V)"); return
        self.stop_anim()
        s = scattering(self.x, V, E)
        self.result = {"scattering": s, "levels": {}, "TE": (-X_HALF + (Es - Emin) / (Emax - Emin) * 2 * X_HALF, Ts),
                       "psi_exact": None}
        peaks = Es[1:-1][(Ts[1:-1] > Ts[:-2]) & (Ts[1:-1] > Ts[2:]) & (Ts[1:-1] > 0.5)]
        self.status(f"E = {E:.4f}: transmission T = {s['T']:.4f}, reflection R = {s['R']:.4f}\n"
                    f"T(E) curve: E from {Emin:.2f} to {Emax:.2f} mapped onto the x axis.\n"
                    f"resonance peaks (T>0.5) at E = {np.round(peaks, 3)}")
        self.draw()


if __name__ == "__main__":
    Builder()
    plt.show()
