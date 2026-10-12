#!/usr/bin/env python3
"""
run_wavefunction.py -- interactive time-dependent Schroedinger equation in 1D, 2D and 3D (matplotlib only).

    python run_wavefunction.py                      # menu of scenarios, default 1d_tunneling
    python run_wavefunction.py --scenario 2d_double_slit
    python run_wavefunction.py --scenario 3d_hydrogenlike
    python run_wavefunction.py --list
    python run_wavefunction.py --scenario 1d_tunneling --save tunnel.gif --frames 200   (no window)

Keys in the window:  space = pause/run,  + / - = faster/slower,  r = restart,  n / p = next/previous scenario,
                     q = quit.
Amplitude representation:
    1D : Re(psi) (blue), Im(psi) (orange), |psi|^2 filled, coloured by phase; potential in grey.
    2D : image with brightness = |psi| and hue = arg(psi)  ("phase-colouring"); potential walls drawn white.
    3D : three mid-plane slices (same colouring) plus a point cloud of the highest-|psi|^2 voxels.
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import animation                     # noqa: E402
from matplotlib.colors import hsv_to_rgb             # noqa: E402

from schrodinger import SCENARIOS, build             # noqa: E402

BG = "#0b0d12"


def phase_rgb(psi, gamma=0.6):
    """brightness = |psi| (normalised), hue = phase."""
    amp = np.abs(psi)
    v = (amp / (amp.max() + 1e-30)) ** gamma
    h = (np.angle(psi) + np.pi) / (2 * np.pi)
    s = np.ones_like(v)
    return hsv_to_rgb(np.stack([h, s, v], axis=-1))


class App:
    def __init__(self, name, steps_per_frame=4):
        self.names = list(SCENARIOS)
        self.idx = self.names.index(name)
        self.spf = steps_per_frame
        self.running = True
        self.fig = plt.figure(figsize=(11, 6.5), facecolor=BG)
        self.load()

    def load(self):
        self.fig.clf()
        self.name = self.names[self.idx]
        self.solver, self.sc, self.info = build(self.name)
        self.frame = 0
        d = self.sc["dim"]
        if d == 1:
            self.ax = self.fig.add_subplot(111, facecolor=BG)
        elif d == 2:
            self.ax = self.fig.add_subplot(111, facecolor=BG)
        else:
            self.axes = [self.fig.add_subplot(2, 2, i + 1, facecolor=BG) for i in range(3)]
            self.ax3 = self.fig.add_subplot(2, 2, 4, projection="3d", facecolor=BG)
        self.title = self.fig.suptitle("", color="#e6edf3", fontsize=10)
        self.fig.text(0.01, 0.01, "space: pause  +/-: speed  r: restart  n/p: scenario  q: quit",
                      color="#8b949e", fontsize=8)

    # ------------------------------------------------------------------------------------------
    def draw(self, _=None):
        if self.running:
            self.solver.step(self.spf)
        s, g = self.solver, self.solver.g
        E, T, Vm = s.energy()
        extra = ""
        if "tunnel_period" in self.info:
            extra = f"   tunnel period = {self.info['tunnel_period']:.2f}"
        if "energies" in self.info:
            extra += "   E_n = " + ", ".join(f"{e:.3f}" for e in self.info["energies"])
        if self.name == "1d_tunneling":
            extra += f"   transmitted = {s.transmission(0.5):.3f}"
        self.title.set_text(f"{self.name}: {self.sc['note']}\n t = {s.t:.2f}   <E> = {E:.4f} (T {T:.3f}, V {Vm:.3f})"
                            f"   <x> = {np.round(s.expectation_x(), 2)}   sigma = {np.round(s.width(), 2)}{extra}")
        d = g.dim
        if d == 1:
            ax = self.ax
            ax.cla()
            x = g.X[0]
            psi = s.psi
            rho = np.abs(psi) ** 2
            Vn = s.V / (np.abs(s.V).max() + 1e-30) * rho.max()
            ax.fill_between(x, 0, Vn, color="#3a4150", alpha=0.6, label="V (scaled)")
            cols = phase_rgb(psi, gamma=1.0)
            ax.scatter(x, rho, c=cols, s=4, label="|psi|^2 (colour = phase)")
            ax.plot(x, psi.real * np.sqrt(rho.max() / (np.abs(psi).max() + 1e-30)), color="#58a6ff", lw=1, label="Re psi")
            ax.plot(x, psi.imag * np.sqrt(rho.max() / (np.abs(psi).max() + 1e-30)), color="#ffa657", lw=1, label="Im psi")
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(-1.1 * rho.max(), 1.1 * rho.max())
            ax.legend(loc="upper right", fontsize=8, facecolor="#161b22", labelcolor="#c9d1d9", edgecolor="#3a4150")
            ax.tick_params(colors="#c9d1d9")
        elif d == 2:
            ax = self.ax
            ax.cla()
            rgb = phase_rgb(s.psi)
            wall = s.V > 0.5 * s.V.max() if s.V.max() > 0 else np.zeros_like(s.V, bool)
            rgb[wall] = (0.85, 0.85, 0.9)
            ax.imshow(rgb.transpose(1, 0, 2), origin="lower", extent=[-g.L / 2, g.L / 2, -g.L / 2, g.L / 2],
                      interpolation="bilinear")
            ax.set_xlabel("x", color="#c9d1d9"); ax.set_ylabel("y", color="#c9d1d9")
            ax.tick_params(colors="#c9d1d9")
        else:
            n = g.n
            psi = s.psi
            for i, (ax, sl, lab) in enumerate(zip(self.axes,
                                                  [psi[n // 2, :, :], psi[:, n // 2, :], psi[:, :, n // 2]],
                                                  ["y-z plane (x=0)", "x-z plane (y=0)", "x-y plane (z=0)"])):
                ax.cla()
                ax.imshow(phase_rgb(sl).transpose(1, 0, 2), origin="lower",
                          extent=[-g.L / 2, g.L / 2, -g.L / 2, g.L / 2], interpolation="bilinear")
                ax.set_title(lab, color="#c9d1d9", fontsize=9)
                ax.tick_params(colors="#c9d1d9", labelsize=7)
            ax = self.ax3
            ax.cla()
            ax.set_facecolor(BG)
            ax.set_axis_off()
            rho = np.abs(psi) ** 2
            thr = np.percentile(rho, 97)
            sel = rho >= thr
            X, Y, Z = g.X
            cols = phase_rgb(psi)[sel]
            alpha = 0.15 + 0.85 * np.clip((rho[sel] - thr) / (rho.max() - thr + 1e-30), 0, 1)
            cols = np.concatenate([cols, alpha[:, None]], axis=1)
            ax.scatter(X[sel], Y[sel], Z[sel], c=cols, s=6, marker="s", depthshade=False, linewidths=0)
            R = max(np.abs(np.concatenate([X[sel], Y[sel], Z[sel]])).max() * 1.3, g.L / 8)
            ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R)
            ax.view_init(elev=22, azim=-55 + 0.5 * self.frame)
            ax.set_title("top 3% of |psi|^2 (colour = phase)", color="#c9d1d9", fontsize=9)
        self.frame += 1
        return []

    def on_key(self, ev):
        if ev.key == " ":
            self.running = not self.running
        elif ev.key in ("+", "="):
            self.spf = min(self.spf * 2, 256)
        elif ev.key == "-":
            self.spf = max(self.spf // 2, 1)
        elif ev.key == "r":
            self.load()
        elif ev.key == "n":
            self.idx = (self.idx + 1) % len(self.names); self.load()
        elif ev.key == "p":
            self.idx = (self.idx - 1) % len(self.names); self.load()
        elif ev.key == "q":
            plt.close(self.fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="1d_tunneling", choices=list(SCENARIOS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--steps", type=int, default=4, help="solver steps per frame")
    ap.add_argument("--save", default=None, help="write a GIF/MP4 instead of opening a window")
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()
    if args.list:
        for k, v in SCENARIOS.items():
            print(f"{k:28s} {v['note']}")
        return
    app = App(args.scenario, args.steps)
    if args.save:
        anim = animation.FuncAnimation(app.fig, app.draw, frames=args.frames, blit=False)
        writer = animation.FFMpegWriter(fps=20) if args.save.endswith(".mp4") else animation.PillowWriter(fps=20)
        anim.save(args.save, writer=writer)
        print("wrote", args.save)
        return
    app.fig.canvas.mpl_connect("key_press_event", app.on_key)
    app.anim = animation.FuncAnimation(app.fig, app.draw, interval=30, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
