# Problems and solutions

A log of every problem met while building the AI Schrödinger solver, what caused it, and what fixed it.
Newest at the bottom. Dates are 2026-10-12 unless stated.

## 1. Training the wave function

### 1.1 "Solve" ran forever on level 0
**Problem.** Fine-tuning never finished; the energy was already correct to 1e-5 after 200 steps but the solver kept going for all 3,000 allowed steps.
**Cause.** The stop test required the energy to change by less than 1e-7 between checkpoints 100 steps apart. Adam's natural jitter at convergence is about 1e-6, so the test never fired.
**Fix.** Replaced by a stall detector (no improvement of the best energy over 150 steps), later superseded by the stability rule in 1.6.

### 1.2 Wrong energy for n = 1 in a three-well potential (error 0.23)
**Problem.** With a potential of one broad well and one narrow side well, the solver reported E₁ = 2.339 while the exact value was 2.114.
**Cause.** A genuine local minimum. The true n = 1 state lives in the narrow side well; the pre-trained network had all its amplitude in the broad well and converged to that well's second state, which is really n = 2. The gradient that would move amplitude through the barrier is exponentially small, so gradient descent cannot find the other well on its own.
**Fix.** See 1.3 and 1.4. Also, levels are now re-labelled by energy order at the end, so if a later level ends up below an earlier one the indices are corrected and the report says so.

### 1.3 Random noise on the weights did not escape the local minimum
**Problem.** Basin hopping with Gaussian noise added to all weights (15–45 % of their size), accept-if-lower, never reached the side well.
**Cause.** Weight noise mostly wrinkles the same wave function; it does not move amplitude across a barrier.
**Fix.** A physics-guided "bump" move: add a lump of amplitude where V < E but ψ is empty, teach the network that proposal for 150 supervised steps, then continue the variational descent; keep it only if the energy drops. This escaped (n = 1 error 1e-4). Both moves are still in `qm_model.py` (`explore=N`) but off by default, because 1.6 solved the case without them.

### 1.4 The exploration moves made every solve 3× slower
**Problem.** Four kicks per level were tried even when nothing was stuck.
**Fix.** Reduced to two, then switched off entirely when the stability rule (1.6) turned out to escape the wrong well by itself given enough time.

### 1.5 A 15 s hard cut-off per level
**Problem.** Requested as a guarantee against hanging: stop after 15 s and keep whatever was reached.
**Outcome.** Implemented, then replaced by 1.6 at the user's suggestion, because a fixed cut-off throws away a state that is still improving and accepts one that is still moving.

### 1.6 The stopping rule that works: wave-function stability (user's design)
**Rule.** Every 10 steps compute d = ∫(ψ − ψ̄)² dx by Simpson's rule, where ψ̄ is an exponential moving average of ψ. The level is *stable* when the median of d over the last 15 checks is below `psi tol` (default 1e-6). Once stable, a timer of `stable s` seconds (default 15) runs while training continues; if ψ becomes unstable the timer is cancelled and re-armed when stability returns. After the full hold the level is accepted and the lowest-energy state seen is kept. A hard limit (90 s or 4× `stable s`) per level is the backstop; at that point the best state is kept regardless of error and the report says so.
**Two refinements found in testing.** (a) Adam produces spike *bursts*, not single spikes, so the median window was widened from 7 to 15 checks. (b) The learning rate is lowered ×0.3 when stability is first reached and never raised again; restoring it on a reset re-triggered the swings.
**Result.** Converged levels accept at 16 s with zero timer resets and errors around 1e-6. The wrong-well case of 1.2 migrates to the correct well by itself after about 15,000 steps.

### 1.7 "Exact" is not exact
**Problem.** The reported AI error (1e-6) is relative to the grid solution, which is itself an approximation.
**Facts.** The grid has 320 points, dx = 0.05, three-point finite differences; its energies are off from the continuum by about 2e-5 (n = 0) to 2e-4 (n = 1) for piecewise-linear potentials and 4e-4 to 1.5e-3 with a hard step segment. The AI and the reference share the same discretised Hamiltonian, so the AI error measures how well the network solves the lattice problem.
**Status.** Documented in the README and THEORY; a Richardson-extrapolated "grid ±" column and Airy-function exact energies for all-linear potentials are open items.

## 2. The window

### 2.1 Status text overlapped the buttons
**Cause.** The status panel was placed above the Auto-learn button, and matplotlib's `wrap=True` misbehaves with monospace text.
**Fix.** Panel moved below the buttons with explicit wrapping at 58 columns; the report table narrowed to fit.

### 2.2 Training froze at the same step on several runs
**Problem.** The step counter stopped (at 210, 450, …) and the window stayed that way for minutes. A watchdog on a second timer never fired either.
**Cause.** matplotlib's animation timer (FuncAnimation on Tk, Windows) silently stopped firing.
**Fix.** The solve runs as a plain loop inside the button handler, redrawing every 0.25 s and pumping GUI events itself. Nothing external has to call it, so it cannot be starved.

### 2.3 The Solve button appeared dead
**Problem.** Clicks were logged but nothing happened.
**Causes.** (a) With no bound states in the requested range the loop finished before drawing anything. (b) The exact reference used ARPACK's iterative `eigsh`, which can throw a no-convergence error, and that line ran outside the error handling.
**Fix.** A direct tridiagonal eigensolver (`eigh_tridiagonal`, no iteration, agrees with ARPACK to 5e-14); an explicit "none of the requested levels is bound" message; the whole handler inside try/except with errors shown in the panel and written to `solver_log.txt`.

### 2.4 Nothing responded during time evolution; Clear did not stop it
**Cause.** Each animation frame rebuilt every artist with `cla()` and redrew the whole figure (~100 ms), more than the 40 ms interval, so the event loop never got a turn.
**Fix.** Persistent line artists updated with `set_data`, only the lower panel blitted (physics is ~7 ms per frame), the evolution runs as a loop that pumps events every frame, and every button sets a flag that ends it. The classical swarm was reduced from 20,000 to 6,000 trajectories.

### 2.5 Old eigenfunctions reappeared under the evolution
**Cause.** A splice error from an earlier edit left the tail of `draw()` inside `_clear_psi`, so every clear re-painted the last result.
**Fix.** Restored the correct function boundaries.

### 2.6 A green T(E) curve stayed on the training plot
**Cause.** `cla()` clears an axis but not the twin axis that Scattering creates.
**Fix.** Twin axes are removed explicitly before every redraw.

### 2.7 The window sometimes froze when placing points
**Cause.** Auto-learn trained inside the GUI thread every 0.35 s. Most ticks are short, but every 25th evaluates 25 held-out potentials and writes the model file, and after each solve up to 400 remembered potentials are compressed to disk; on a slower machine these take seconds and swallow any click that arrives meanwhile.
**Fix.** The learner runs in its own background thread, pauses itself during solving and evolution, and a lock guards the shared weights. The GUI thread only refreshes the statistics line.

### 2.8 Text boxes showed white text on white
**Cause.** matplotlib's TextBox turns its background white when focused.
**Fix.** Boxes stay dark in every state with light text and a visible cursor (earlier session).

### 2.9 Clear did not stop a running animation (earlier session)
**Fix.** Clear, Undo, placing a point, Solve and Scattering all stop any running animation first.

## 3. Background learning

### 3.1 The model got slightly worse while learning on its own
**Cause.** Single-sample updates drift.
**Fix.** Every 300 steps the model is scored on a fixed held-out set; the new weights are saved only if the error did not get worse (within 2 %), otherwise they are rolled back.

## 4. Packaging

### 4.1 Files for people versus files for the program
**Fix.** The repository folder has only readable or clickable files at the top level (README, THEORY, LICENSE, launchers, docs) and all code and data in `program/`. The script detects that layout and writes GIFs to a top-level `GIF/` folder.

### 4.2 Leftover duplicates after reorganising
**Cause.** Files on the user's computer could not be deleted from the build session (no shell on the device), so a reorganisation left old copies behind.
**Fix.** A fresh folder was created for each layout change (`github_release`, `ai_schrodinger_solver`, `…_v2`); the older ones can be deleted by hand.

## Open items
- Richardson "grid ±" column and Airy-function exact energies for all-linear potentials (1.7).
- The wrong-well case still needs more than 90 s to hold stable after migrating; raise the hard limit or re-enable the bump move when the energy variance stays high.
- Pre-train on multi-well potentials specifically so the first guess already puts amplitude in side wells.
