#!/usr/bin/env python3
"""
run_headless.py -- render the simulation to an animated GIF / MP4 with matplotlib (no Qt, no OpenGL).

Use this if the interactive GUI cannot start on your machine, or to produce a movie.

    python run_headless.py                       # default demo scenario -> vacuum_demo.gif
    python run_headless.py --N 32 --frames 160 --out my_movie.mp4
    python run_headless.py --scenario quench     # vacuum -> sudden mass quench
    python run_headless.py --scenario schwinger  # vacuum -> electric-field pulse on the charged field
    python run_headless.py --scenario casimir    # vacuum -> insert two mirrors
    python run_headless.py --scenario tour       # all of the above in sequence (default)

Layout of each frame:  left = 3D view (the brightest voxels of the chosen quantity, box, mirrors),
right = mid-plane slice, exact vacuum energy-density profile along z (Casimir profile) with the
sample's layer averages, and the history of created particles.

MODIFY: add a scenario in SCENARIOS (a list of (time, callable) events); change `quantity` to
'field' or 'excess' to render the field amplitude or the excess over the vacuum expectation.
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib import animation                      # noqa: E402
from matplotlib.colors import ListedColormap          # noqa: E402

from vacuumsim import Simulation                      # noqa: E402
from vacuumsim.render import lut, SEQUENTIAL          # noqa: E402


def mpl_cmap(name):
    return ListedColormap(lut(name) / 255.0, name=name)


def build_scenario(sim: Simulation, name: str):
    """Events as (time, description, function)."""
    ev = []
    if name in ("casimir", "tour"):
        N = sim.lat.N
        ev.append((2.0, "insert two perfect mirrors (sudden -> dynamical Casimir effect)",
                   lambda: sim.set_plates(N // 3, 2 * N // 3)))
    if name in ("quench", "tour"):
        ev.append((4.0, "sudden mass quench of the level-0 scalar: m 0 -> 1.2",
                   lambda: sim.quench_mass(0, 1.2, duration=0.0)))
    if name in ("schwinger", "tour"):
        def pulse():
            E0 = 1.0
            sim.apply_efield(E0, 0.8 * sim.max_pulse_duration(E0))
        ev.append((6.0, "electric-field pulse on the charged scalar (Schwinger pair creation)", pulse))
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--frames", type=int, default=140)
    ap.add_argument("--dt", type=float, default=0.06)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--scenario", default="tour", choices=["vacuum", "casimir", "quench", "schwinger", "tour"])
    ap.add_argument("--quantity", default="energy", choices=["energy", "field", "excess"])
    ap.add_argument("--smoothing", type=float, default=0.0, help="probe resolution sigma in l_s")
    ap.add_argument("--fields", default="all", help="comma-separated field indices to render, or 'all'")
    ap.add_argument("--top", type=float, default=6.0, help="percent of brightest voxels drawn in 3D")
    ap.add_argument("--out", default="vacuum_demo.gif")
    args = ap.parse_args()

    sim = Simulation(N=args.N, a=args.a, seed=args.seed)
    events = build_scenario(sim, args.scenario)
    idx = None if args.fields == "all" else [int(i) for i in args.fields.split(",")]
    N = sim.lat.N

    fig = plt.figure(figsize=(13, 7.2), facecolor="#0b0d12")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.6, 1.0], hspace=0.45, wspace=0.25,
                          left=0.03, right=0.98, top=0.93, bottom=0.07)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d", facecolor="#0b0d12")
    ax_slice = fig.add_subplot(gs[0, 1], facecolor="#0b0d12")
    ax_prof = fig.add_subplot(gs[1, 1], facecolor="#0b0d12")
    ax_hist = fig.add_subplot(gs[2, 1], facecolor="#0b0d12")
    for ax in (ax_slice, ax_prof, ax_hist):
        ax.tick_params(colors="#c9d1d9", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#3a4150")
    title = fig.suptitle("", color="#e6edf3", fontsize=11)
    cmap = mpl_cmap("inferno" if args.quantity == "energy" else "ice-fire")
    xs, ys, zs = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    banner = fig.text(0.03, 0.015, "", color="#ffd166", fontsize=9)
    state = {"banner": "", "banner_until": -1.0, "fired": set()}

    def quantity():
        if args.quantity == "energy":
            return sim.energy_density(idx, args.smoothing)
        if args.quantity == "field":
            return sim.field_amplitude(idx, args.smoothing)
        return sim.excess_energy(idx, args.smoothing)

    def draw(frame):
        # fire events
        for j, (t_ev, desc, fn) in enumerate(events):
            if j not in state["fired"] and sim.t <= t_ev < sim.t + args.dt:
                state["fired"].add(j)
                fn()
                state["banner"] = f"t = {t_ev:.1f}: {desc}"
                state["banner_until"] = sim.t + 1.5
        sim.step(args.dt)
        q = quantity()
        signed = args.quantity != "energy"
        # ---- 3D: brightest voxels
        ax3.cla()
        ax3.set_facecolor("#0b0d12")
        ax3.set_axis_off()
        ax3.set_xlim(0, N); ax3.set_ylim(0, N); ax3.set_zlim(0, N)
        ax3.view_init(elev=22, azim=-55 + 0.35 * frame)
        # box edges
        for s_, e_ in [((0, 0, 0), (N, 0, 0)), ((0, 0, 0), (0, N, 0)), ((0, 0, 0), (0, 0, N)),
                       ((N, N, N), (0, N, N)), ((N, N, N), (N, 0, N)), ((N, N, N), (N, N, 0)),
                       ((N, 0, 0), (N, N, 0)), ((N, 0, 0), (N, 0, N)), ((0, N, 0), (N, N, 0)),
                       ((0, N, 0), (0, N, N)), ((0, 0, N), (N, 0, N)), ((0, 0, N), (0, N, N))]:
            ax3.plot(*zip(s_, e_), color="#3a4150", lw=0.8)
        mag = np.abs(q) if signed else q
        thr = np.percentile(mag, 100 - args.top)
        sel = mag >= thr
        vals = q[sel]
        if signed:
            vmax = np.percentile(np.abs(q), 99.7) or 1.0
            colors = cmap(0.5 + 0.5 * np.clip(vals / vmax, -1, 1))
        else:
            lo, hi = np.percentile(q, 5), np.percentile(q, 99.7)
            colors = cmap(np.clip((vals - lo) / (hi - lo + 1e-30), 0, 1))
        colors[:, 3] = 0.35 + 0.65 * np.clip((mag[sel] - thr) / (mag.max() - thr + 1e-30), 0, 1)
        ax3.scatter(xs[sel] + 0.5, ys[sel] + 0.5, zs[sel] + 0.5, c=colors, s=9, marker="s",
                    depthshade=False, linewidths=0)
        if sim.plates_on:
            for z in sim.geom.plate_layers:
                xx, yy = np.meshgrid([0, N], [0, N])
                ax3.plot_surface(xx, yy, np.full_like(xx, z + 0.5, dtype=float), color="#8ec5ff", alpha=0.18)
        if sim.pulse is not None and sim.pulse.active(sim.t):
            ax3.quiver(2, N / 2, N + 3, 0.25 * N, 0, 0, color="#ffd166", arrow_length_ratio=0.25, lw=2)
            ax3.text(2, N / 2, N + 6, "E", color="#ffd166")
        # ---- slice
        ax_slice.cla()
        sl = q[:, :, N // 2].T
        if signed:
            vm = np.percentile(np.abs(q), 99.7) or 1.0
            ax_slice.imshow(sl, origin="lower", cmap=cmap, vmin=-vm, vmax=vm, interpolation="nearest")
        else:
            ax_slice.imshow(sl, origin="lower", cmap=cmap, vmin=np.percentile(q, 2), vmax=np.percentile(q, 99.7),
                            interpolation="nearest")
        ax_slice.set_title({"energy": "energy density, mid z-slice", "field": "field amplitude, mid z-slice",
                            "excess": "excess over vacuum, mid z-slice"}[args.quantity],
                           color="#c9d1d9", fontsize=9)
        ax_slice.set_xticks([]); ax_slice.set_yticks([])
        # ---- profile along z
        ax_prof.cla()
        prof_vac = np.array(sim.vacuum_profile(idx), dtype=float)
        prof_ens = np.array(sim.ensemble_profile(idx), dtype=float)
        h = sim.energy_density(idx)
        layer_mean = h.mean(axis=(0, 1))
        if sim.plates_on:                       # hide the plate layers so the axis zooms on the gap
            pl = sim.geom.plate_layers
            prof_vac[pl] = prof_ens[pl] = np.nan
            layer_mean = layer_mean.copy(); layer_mean[pl] = np.nan
        ax_prof.plot(np.arange(N), prof_vac, color="#8ec5ff", lw=1.5, label="exact vacuum <h>(z)")
        if np.nanmax(np.abs(prof_ens - prof_vac)) > 1e-9 * np.nanmax(np.abs(prof_vac)):
            ax_prof.plot(np.arange(N), prof_ens, color="#ff9f43", lw=1.2, label="exact <h>(z) with created particles")
        ax_prof.plot(np.arange(N), layer_mean, ".", color="#f8f8f2", ms=3, label="this sample, layer mean")
        ax_prof.set_xlabel("z layer", color="#c9d1d9", fontsize=8)
        ax_prof.set_title("energy density vs z  (M_s^4)", color="#c9d1d9", fontsize=9)
        ax_prof.legend(fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9", edgecolor="#3a4150", loc="lower right")
        ax_prof.grid(alpha=0.15)
        # ---- history
        ax_hist.cla()
        H = sim.history
        ax_hist.plot(H["t"], H["quanta"], color="#ff6b6b", lw=1.5)
        ax_hist.set_title("real particles created (exact ensemble mean)", color="#c9d1d9", fontsize=9)
        ax_hist.set_xlabel("t  (l_s / c)", color="#c9d1d9", fontsize=8)
        ax_hist.grid(alpha=0.15)
        ro = sim.readout()
        title.set_text(f"Lattice vacuum of {len(sim.fields)} string-tower fields   N={N}^3, a={sim.lat.a} l_s   "
                       f"t = {sim.t:6.2f} l_s/c    <h>_vac = {ro['zero_point']['sampled']:.3f} M_s^4   "
                       f"particles created = {ro['total_quanta']:.2f}")
        banner.set_text(state["banner"] if sim.t < state["banner_until"] else "")
        return []

    anim = animation.FuncAnimation(fig, draw, frames=args.frames, blit=False)
    out = args.out
    if out.lower().endswith(".mp4"):
        try:
            anim.save(out, writer=animation.FFMpegWriter(fps=args.fps, bitrate=2500))
        except Exception as e:  # noqa: BLE001
            print("ffmpeg not available (", e, ") -> writing GIF instead")
            out = os.path.splitext(out)[0] + ".gif"
            anim.save(out, writer=animation.PillowWriter(fps=args.fps))
    else:
        anim.save(out, writer=animation.PillowWriter(fps=args.fps))
    print("wrote", out)
    print("\n".join(sim.events[-8:]))


if __name__ == "__main__":
    main()
