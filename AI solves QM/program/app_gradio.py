#!/usr/bin/env python3
"""
app_gradio.py -- web demo of the potential builder + AI solver (for reviewers; runs on Hugging Face Spaces).

Local test:   pip install gradio   then   python app_gradio.py   -> opens http://127.0.0.1:7860
Deploy:       create a free Space at https://huggingface.co/new-space (SDK: Gradio), upload this folder
              (app_gradio.py, qm_model.py, run_me.py, pretrained_model.npz, requirements_space.txt
              renamed to requirements.txt).  The Space builds in ~2 min and gives a public URL you can link
              from a CV or embed in a web page with <iframe src="https://<user>-<space>.hf.space"></iframe>.

The browser cannot click on a matplotlib canvas, so points are entered in a table (x, y, segment type).
"""
from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from qm_model import (grid, fit, scattering, load_model, continuum_threshold, X_HALF, gaussian_packet,
                      SpectralEvolver, SplitOperator, fidelity, ThawedGaussian, ClassicalEnsemble, density_overlap)
from matplotlib import animation
from run_me import build_potential, SEGMENTS

NET, META = load_model()
BG = "#0b0d12"
EXAMPLES = {
    "double well": ([(-6, 4), (-3.5, -2), (-1.5, 0.5), (0.5, -3), (3, 4)], ["smooth step", "cubic", "cubic", "cubic"]),
    "asymmetric step well + bump": ([(-6, 3.0), (-3, -2.5), (0, -2.5), (1.5, 0.5), (3, -1.0), (5, 4.0)],
                                    ["smooth step", "step", "linear", "cubic", "exponential"]),
    "ramp (Airy-like)": ([(-7, 6), (-2, -4), (6, 2)], ["linear", "linear"]),
    "barrier in a box (resonances)": ([(-7, 6), (-5, 0), (-0.5, 0), (0.5, 2.5), (5, 0), (7, 6)],
                                      ["step", "step", "step", "step", "step"]),
}


def _parse(table):
    pts, kinds = [], []
    for row in table:
        try:
            x, y = float(row[0]), float(row[1])
        except (TypeError, ValueError):
            continue
        if pts and x <= pts[-1][0]:
            x = pts[-1][0] + 0.2
        if pts:
            kinds.append(row[2] if row[2] in SEGMENTS else "linear")
        pts.append((min(x, X_HALF - 0.1), y))
    return pts, kinds


def solve(table, levels_text, steps):
    pts, kinds = _parse(table)
    if len(pts) < 2:
        return None, "Add at least two points."
    x = grid()
    V = build_potential(x, pts, kinds)
    try:
        levels = sorted({int(s) for s in levels_text.replace(";", ",").split(",") if s.strip()})
    except ValueError:
        return None, "levels must be integers like 0,1,2"
    r = fit(x, V, levels, net=NET, steps=int(steps))
    fig, (axV, axP) = plt.subplots(2, 1, figsize=(9, 8), facecolor=BG)
    for ax in (axV, axP):
        ax.set_facecolor(BG); ax.tick_params(colors="#c9d1d9")
    axV.plot(x, V, color="#58a6ff", lw=2); axV.plot(*zip(*pts), "o", color="#ffa657")
    axV.axhline(r["threshold"], color="#8b949e", ls="--", lw=0.8)
    lines = [f"bound states: {r['n_bound_exact']} (E < {r['threshold']:.3f});  model: "
             + ("pre-trained + fine-tuned" if NET is not None else "trained from scratch"),
             "n | AI model | exact grid | WKB | AI error | WKB error"]
    offs = 0.0
    for k, lv in sorted(r["levels"].items()):
        if not lv["bound"]:
            lines.append(f"{k} | continuum (no bound state) | | | |"); continue
        axV.axhline(lv["E_model"], color="#3fb950", lw=1)
        axP.plot(x, lv["psi"] + offs, color="#ffa657", lw=1.8)
        axP.plot(x, r["psi_exact"][:, k] + offs, color="#8b949e", lw=3, alpha=0.5)
        axP.text(-X_HALF + 0.2, offs + 0.05, f"n={k}", color="#c9d1d9")
        offs += 1.2
        w = lv["E_wkb"]
        lines.append(f"{k} | {lv['E_model']:.5f} | {lv['E_exact']:.5f} | {w if w is None else round(w, 5)} | "
                     f"{abs(lv['E_model']-lv['E_exact']):.1e} | {'' if w is None else f'{abs(w-lv['E_exact']):.1e}'}")
    axV.set_title("potential and AI energies (green)", color="#c9d1d9")
    axP.set_title("eigenfunctions: AI model (orange) vs exact numerics (grey)", color="#c9d1d9")
    fig.tight_layout()
    return fig, "\n".join(lines)


def scatter(table, E_text):
    pts, kinds = _parse(table)
    if len(pts) < 2:
        return None, "Add at least two points."
    x = grid()
    V = build_potential(x, pts, kinds)
    Emin = max(V[0], V[-1]) + 1e-3
    Emax = Emin + max(4.0, V.max() - Emin + 3.0)
    Es = np.linspace(Emin, Emax, 300)
    Ts = np.array([scattering(x, V, E)["T"] for E in Es])
    try:
        E = float(E_text) if E_text.strip() else Es[np.argmax(Ts)]
    except ValueError:
        E = Es[np.argmax(Ts)]
    if E <= Emin:
        return None, f"scattering needs E > {Emin:.3f}"
    s = scattering(x, V, E)
    fig, (axV, axT, axP) = plt.subplots(3, 1, figsize=(9, 9), facecolor=BG)
    for ax in (axV, axT, axP):
        ax.set_facecolor(BG); ax.tick_params(colors="#c9d1d9")
    axV.plot(x, V, color="#58a6ff", lw=2); axV.axhline(E, color="#ffa657", lw=1)
    axV.set_title("potential and chosen energy", color="#c9d1d9")
    axT.plot(Es, Ts, color="#3fb950"); axT.set_ylim(0, 1.05); axT.set_xlabel("E", color="#c9d1d9")
    axT.set_title("transmission T(E)  (peaks = resonances)", color="#c9d1d9")
    axP.plot(x, np.abs(s["psi"]) ** 2, color="#ffa657", label="|psi|^2"); axP.plot(x, s["psi"].real, color="#58a6ff", lw=0.8, label="Re psi")
    axP.legend(facecolor="#161b22", labelcolor="#c9d1d9"); axP.set_title(f"scattering state, E = {E:.3f}, T = {s['T']:.4f}", color="#c9d1d9")
    fig.tight_layout()
    return fig, f"E = {E:.4f}: T = {s['T']:.4f}, R = {s['R']:.4f}"


def evolve(table, levels_text, init_text, duration):
    """Solve the levels, then animate psi(x,t): spectral (AI levels) vs full TDSE; returns a GIF path."""
    pts, kinds = _parse(table)
    if len(pts) < 2:
        return None, "Add at least two points."
    x = grid(); V = build_potential(x, pts, kinds)
    levels = sorted({int(s) for s in levels_text.replace(";", ",").split(",") if s.strip()})
    r = fit(x, V, levels, net=NET, steps=800)
    bound = [k for k in sorted(r["levels"]) if r["levels"][k]["bound"]]
    if not bound:
        return None, "no bound states among the requested levels"
    txt = init_text.strip()
    tga = cls = None
    try:
        if txt.startswith("gauss"):
            x0, sig, k0 = [float(v) for v in txt[5:].split(",")]
            psi0 = gaussian_packet(x, x0, sig, k0)
            tga = ThawedGaussian(x, V, x0, sig, k0); cls = ClassicalEnsemble(x, V, x0, sig, k0)
        else:
            psi0 = np.zeros(len(x), complex)
            for item in txt.split(","):
                k, c = item.split(":"); psi0 += complex(c) * r["levels"][int(k)]["psi"]
            psi0 /= np.sqrt(np.sum(np.abs(psi0) ** 2) * (x[1] - x[0]))
    except Exception as e:  # noqa: BLE001
        return None, f"init must be 'gauss x0,sigma,k0' or 'n:c,n:c' ({e})"
    spec = SpectralEvolver(x, [r["levels"][k]["E_model"] for k in bound], [r["levels"][k]["psi"] for k in bound], psi0)
    full = SplitOperator(x, V, psi0)
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=BG)
    ax.set_facecolor(BG); ax.tick_params(colors="#c9d1d9")
    ts = np.linspace(0, float(duration), 80)

    def frame(i):
        t = ts[i]; ps = spec.psi(t); pf = full.advance(t)
        ax.cla(); ax.set_facecolor(BG); ax.set_xlim(-X_HALF, X_HALF)
        ax.fill_between(x, 0, V / (abs(V).max() + 1e-9) * 0.6 * np.abs(psi0).max() ** 2, color="#3a4150", alpha=0.5)
        ax.plot(x, np.abs(pf) ** 2, color="white", lw=2.5, alpha=0.6, label="|psi|^2 full TDSE")
        ax.plot(x, np.abs(ps) ** 2, color="#ffa657", lw=1.5, label="|psi|^2 spectral (AI levels)")
        title = f"t = {t:.2f}   fidelity AI-levels = {fidelity(x, ps, pf):.4f}"
        if tga is not None:
            pt = tga.advance(t); rc = cls.advance(t)
            ax.plot(x, np.abs(pt) ** 2, "--", color="#3fb950", lw=1.2, label="thawed Gaussian (semiclassical)")
            ax.plot(x, rc, ":", color="#d2a8ff", lw=1.5, label="classical ensemble")
            title += f"  semiclassical = {fidelity(x, pt, pf):.3f}  classical = {density_overlap(x, rc, np.abs(pf) ** 2):.3f}"
        ax.set_ylim(0, 1.2 * np.abs(psi0).max() ** 2)
        ax.set_title(title, color="#c9d1d9", fontsize=9)
        ax.legend(facecolor="#161b22", labelcolor="#c9d1d9", loc="upper right")
        return []
    anim = animation.FuncAnimation(fig, frame, frames=len(ts), blit=False)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evolution.gif")
    anim.save(out, writer=animation.PillowWriter(fps=12)); plt.close(fig)
    msg = "probabilities |c_n|^2: " + ", ".join(f"n={k}: {p:.3f}" for k, p in zip(bound, spec.prob)) + \
          f"\ncaptured by solved levels: {spec.captured:.3f};  <E> = {spec.energy_mean():.4f}"
    return out, msg


def load_example(name):
    pts, kinds = EXAMPLES[name]
    return [[p[0], p[1], (kinds[i - 1] if i > 0 else "linear")] for i, p in enumerate(pts)]


with gr.Blocks(title="Draw a potential — AI finds the wave functions") as demo:
    gr.Markdown("# Draw a potential, let a neural network find the quantum states\n"
                "Enter anchor points from left to right (segment type = how the curve gets to that point from the previous one). "
                "Negative values are wells. The pre-trained model predicts each eigenfunction and is fine-tuned variationally; "
                "the table compares it with exact numerics and with WKB.  Units: ħ = m = 1, box −8 ≤ x ≤ 8.")
    with gr.Row():
        with gr.Column(scale=1):
            table = gr.Dataframe(headers=["x", "V(x)", "segment from previous"], datatype=["number", "number", "str"],
                                 value=load_example("asymmetric step well + bump"), row_count=(2, "dynamic"), col_count=(3, "fixed"))
            gr.Markdown("segment types: " + ", ".join(SEGMENTS))
            ex = gr.Dropdown(list(EXAMPLES), label="load an example", value="asymmetric step well + bump")
            ex.change(load_example, ex, table)
            levels = gr.Textbox("0,1,2", label="levels n to solve (bound states)")
            steps = gr.Slider(200, 3000, 1200, step=100, label="fine-tuning steps")
            b1 = gr.Button("Solve with AI", variant="primary")
            E_text = gr.Textbox("", label="scattering energy (blank = strongest resonance)")
            b2 = gr.Button("Scattering T(E)")
            init = gr.Textbox("gauss -1.5,0.8,0", label="initial state for time evolution ('gauss x0,sigma,k0' or 'n:c,n:c')")
            dur = gr.Slider(1, 20, 6, step=1, label="duration")
            b3 = gr.Button("Evolve in time (GIF)")
        with gr.Column(scale=2):
            plot = gr.Plot()
            out = gr.Textbox(label="results", lines=8)
            gif = gr.Image(label="time evolution", type="filepath")
    b1.click(solve, [table, levels, steps], [plot, out])
    b2.click(scatter, [table, E_text], [plot, out])
    b3.click(evolve, [table, levels, init, dur], [gif, out])

if __name__ == "__main__":
    demo.launch()
