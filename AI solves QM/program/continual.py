"""
continual.py -- the stored model keeps learning by itself, in the background, while the UI is idle.

Two sources of experience:
  * every potential the user solves (its exact grid eigenstates are kept in experience.npz), and
  * fresh random potentials of the same families as the pre-training set (train_pretrained.random_potential).
Each background tick does a few supervised steps (network output vs exact eigenfunction) on a mix of both
and, every `save_every` steps, writes the weights back to pretrained_model.npz.  Because the fine-tuning in
the UI starts from these weights, the start-up guess for the next drawn potential gets better with use.
A small held-out set (fixed seed) is re-evaluated at every save so the status line can show that the model
is actually improving (median energy error of the raw prediction) and not drifting.

MODIFY: `steps_per_tick` and the timer interval set how much CPU idle-learning takes; set `auto=False` in
the builder to disable.
"""
from __future__ import annotations
import os
import numpy as np

from qm_model import (grid, exact_states, continuum_threshold, model_input, envelope, hamiltonian, Adam,
                      MODEL_FILE, N_STATES, new_model)
from train_pretrained import random_potential

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIENCE_FILE = os.path.join(HERE, "experience.npz")


class ContinualTrainer:
    def __init__(self, net, x=None, lr=2e-4, steps_per_tick=12, save_every=300, seed=0):
        self.x = grid() if x is None else x
        self.env = envelope(self.x)
        self.net = net if net is not None else new_model(seed)
        self.opt = Adam(self.net.params, lr)
        self.rng = np.random.default_rng(seed)
        self.steps_per_tick, self.save_every = steps_per_tick, save_every
        self.steps = 0
        self.loss = np.nan
        self.heldout_err = np.nan
        self.experience = []          # list of (V, psi_matrix) from solved potentials
        self._load_experience()
        self._test = None
        self.pool = []                # random potentials with exact states, grown one per tick
        self.best_err = None
        self._snapshot = [p.copy() for p in self.net.params]

    # --- experience -----------------------------------------------------------------------------
    def _load_experience(self):
        if os.path.exists(EXPERIENCE_FILE):
            d = np.load(EXPERIENCE_FILE)
            n = int(d["n"])
            self.experience = [(d[f"V{i}"], d[f"P{i}"]) for i in range(n)]

    def remember(self, V, psi_exact, E_exact):
        """Store the solved potential (bound states only) and persist."""
        nb = int(np.sum(E_exact < continuum_threshold(V)))
        if nb == 0:
            return
        self.experience.append((np.asarray(V, float), np.asarray(psi_exact[:, :min(nb, N_STATES)], float)))
        self.experience = self.experience[-400:]
        np.savez_compressed(EXPERIENCE_FILE, n=len(self.experience),
                            **{f"V{i}": v for i, (v, _) in enumerate(self.experience)},
                            **{f"P{i}": p for i, (_, p) in enumerate(self.experience)})

    # --- training -------------------------------------------------------------------------------
    def _grow_pool(self):
        for _ in range(5):
            V = random_potential(self.rng, self.x)
            E, P = exact_states(self.x, V, N_STATES)
            nb = int(np.sum(E < continuum_threshold(V)))
            if nb:
                self.pool.append((V, P[:, :nb]))
                self.pool = self.pool[-300:]
                return

    def _sample(self):
        if self.experience and self.rng.random() < 0.5:
            return self.experience[self.rng.integers(len(self.experience))]
        if not self.pool:
            self._grow_pool()
        return self.pool[self.rng.integers(len(self.pool))]

    def tick(self):
        """One background slice of learning; cheap enough to run between UI frames."""
        acc = 0.0
        self._grow_pool()             # one exact diagonalisation per tick (~20 ms)
        for _ in range(self.steps_per_tick):
            V, P = self._sample()
            k = self.rng.integers(P.shape[1])
            out = self.net.forward(model_input(self.x, V, k)) * self.env
            diff = out - P[:, k]
            acc += np.mean(diff ** 2)
            self.opt.step(self.net.backward(2 * diff * self.env / len(self.x)))
            self.steps += 1
        self.loss = 0.9 * self.loss + 0.1 * acc / self.steps_per_tick if np.isfinite(self.loss) else acc / self.steps_per_tick
        if self.steps % self.save_every < self.steps_per_tick:
            self._checkpoint()
        return self.loss

    def _checkpoint(self):
        """Accept the new weights only if the held-out error did not get worse (no drift), else roll back."""
        err = self.evaluate()
        if self.best_err is None:
            self.best_err = err
        if err <= self.best_err * 1.02:
            self.best_err = min(self.best_err, err)
            self._snapshot = [p.copy() for p in self.net.params]
            self.save()
            self.accepted = True
        else:
            for p, q in zip(self.net.params, self._snapshot):
                p[...] = q
            self.accepted = False
        self.heldout_err = err

    def evaluate(self, n=25):
        if self._test is None:
            rng = np.random.default_rng(12345)
            self._test = []
            while len(self._test) < n:
                V = random_potential(rng, self.x)
                E, P = exact_states(self.x, V, N_STATES)
                nb = int(np.sum(E < continuum_threshold(V)))
                if nb:
                    self._test.append((V, E[:nb]))
        errs = []
        dx = self.x[1] - self.x[0]
        for V, Es in self._test:
            H = hamiltonian(self.x, V)
            for k in range(len(Es)):
                out = self.net.forward(model_input(self.x, V, k)) * self.env
                nrm = out @ out * dx
                errs.append(abs((out @ (H @ out)) * dx / nrm - Es[k]) / max(abs(Es[k]), 0.1))
        return float(np.median(errs))

    def save(self):
        self.net.save(MODEL_FILE, continual_steps=self.steps, k_desc=32, n_states=N_STATES)
