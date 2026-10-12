#!/usr/bin/env python3
"""
train_pretrained.py -- build the stored model (pretrained_model.npz) from random potentials and exact numerics.

    python train_pretrained.py                 # ~2-4 minutes on a laptop, writes pretrained_model.npz
    python train_pretrained.py --n 1500 --epochs 4 --seed 1

Random potentials are assembled from the same primitives the potential builder offers (linear, polynomial,
exponential, step, smooth-step segments, plus smooth wells), their lowest bound states are computed by exact
sparse diagonalisation, and the network psi_k(x | V) is trained to reproduce them (mean-squared error on the
normalised eigenfunctions).  A held-out set reports the error of the raw prediction and of the energy obtained
from it, so you can see how much the variational fine-tuning later has to do.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qm_model import (grid, exact_states, continuum_threshold, descriptor, model_input, envelope, new_model,   # noqa: E402
                      Adam, MODEL_FILE, N_STATES, X_HALF, V_SCALE, hamiltonian)


def random_potential(rng, x):
    """Piecewise potential from 3-7 anchor points with random segment types, horizontal beyond the ends,
    occasionally with a smooth confining bowl added."""
    kind = rng.integers(0, 4)
    if kind == 0:                                   # smooth bowl / anharmonic
        a, b, c = rng.uniform(0.05, 0.6), rng.uniform(-0.3, 0.3), rng.uniform(0, 0.03)
        V = a * x ** 2 + b * x ** 3 * 0.1 + c * x ** 4
    elif kind == 1:                                 # double / triple well
        V = 0.05 * x ** 2
        for _ in range(rng.integers(1, 4)):
            V -= rng.uniform(1, 5) * np.exp(-((x - rng.uniform(-4, 4)) / rng.uniform(0.5, 1.5)) ** 2)
    else:                                           # piecewise, builder-like
        n_pts = rng.integers(3, 8)
        xs = np.sort(rng.uniform(-6, 6, n_pts))
        ys = rng.uniform(-4, 4, n_pts)
        ys[0] = abs(ys[0]) + 1.0 if rng.random() < 0.7 else ys[0]
        ys[-1] = abs(ys[-1]) + 1.0 if rng.random() < 0.7 else ys[-1]
        V = np.full_like(x, ys[0])
        for i in range(n_pts - 1):
            seg = (x >= xs[i]) & (x < xs[i + 1])
            t = (x[seg] - xs[i]) / max(xs[i + 1] - xs[i], 1e-9)
            s = rng.integers(0, 5)
            if s == 0:
                V[seg] = ys[i] + (ys[i + 1] - ys[i]) * t
            elif s == 1:
                V[seg] = ys[i] + (ys[i + 1] - ys[i]) * t ** rng.integers(2, 4)
            elif s == 2:
                r = rng.uniform(1, 4)
                V[seg] = ys[i] + (ys[i + 1] - ys[i]) * (np.exp(r * t) - 1) / (np.exp(r) - 1)
            elif s == 3:
                V[seg] = ys[i]
            else:
                V[seg] = ys[i] + (ys[i + 1] - ys[i]) * 0.5 * (1 + np.tanh(8 * (t - 0.5)))
        V[x >= xs[-1]] = ys[-1]
    V = np.clip(V, -3 * V_SCALE, 3 * V_SCALE)
    return V


def make_dataset(n, seed):
    rng = np.random.default_rng(seed)
    x = grid()
    data = []
    while len(data) < n:
        V = random_potential(rng, x)
        E, psi = exact_states(x, V, N_STATES)
        nb = int(np.sum(E < continuum_threshold(V)))
        if nb == 0:
            continue
        data.append((V, psi[:, :nb], E[:nb]))
    return x, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    t0 = time.time()
    x, data = make_dataset(args.n, args.seed)
    _, test = make_dataset(60, args.seed + 999)
    print(f"dataset: {len(data)} train / {len(test)} test potentials, {sum(p.shape[1] for _, p, _ in data)} eigenstates "
          f"({time.time()-t0:.0f} s)")
    env = envelope(x)
    net = new_model(args.seed)
    opt = Adam(net.params, args.lr)
    rng = np.random.default_rng(args.seed)
    step = 0
    for ep in range(args.epochs):
        order = rng.permutation(len(data))
        loss_acc = 0.0
        for idx in order:
            V, psis, _ = data[idx]
            for k in range(psis.shape[1]):
                X = model_input(x, V, k)
                out = net.forward(X) * env
                diff = out - psis[:, k]
                loss = np.mean(diff ** 2)
                opt.step(net.backward(2 * diff * env / len(x)))
                loss_acc += loss
                step += 1
        # evaluation on held-out: energy of the raw prediction vs exact
        errs, eerr = [], []
        for V, psis, Es in test:
            H = hamiltonian(x, V)
            for k in range(psis.shape[1]):
                out = net.forward(model_input(x, V, k)) * env
                nrm = out @ out * (x[1] - x[0])
                E_pred = (out @ (H @ out)) * (x[1] - x[0]) / nrm
                errs.append(np.mean((out / np.sqrt(nrm) - psis[:, k]) ** 2))
                eerr.append(abs(E_pred - Es[k]) / max(abs(Es[k]), 0.1))
        print(f"epoch {ep+1}: train mse {loss_acc/step*args.epochs:.3e}   held-out psi mse {np.mean(errs):.3e}   "
              f"held-out energy rel.err of raw prediction {np.median(eerr):.3e} (median)   [{time.time()-t0:.0f} s]")
        opt.lr *= 0.8
    net.save(MODEL_FILE, n_train=len(data), epochs=args.epochs, seed=args.seed, k_desc=32, n_states=N_STATES,
             x_half=X_HALF, v_scale=V_SCALE)
    print("saved", MODEL_FILE)


if __name__ == "__main__":
    main()
