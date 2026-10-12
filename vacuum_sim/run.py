#!/usr/bin/env python3
"""
run.py -- start the interactive 3D vacuum-fluctuation simulator.

    python run.py                    # default: 32^3 lattice, a = 1 l_s, closed type II tower
    python run.py --N 48 --a 0.7     # bigger / finer lattice (slower)
    python run.py --seed 7
    python run.py --selftest         # run the physics validation tests (no GUI needed)
    python run.py --load state.npz   # resume a saved state

Install (once):   pip install -r requirements.txt
If the GUI cannot start (missing OpenGL / Qt), use   python run_headless.py   to render a movie.
"""
from __future__ import annotations
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=32, help="grid size per axis (24..64; cost ~ N^3)")
    ap.add_argument("--a", type=float, default=1.0, help="lattice spacing in string lengths")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--theory", default="closed type II superstring (D=10)")
    ap.add_argument("--load", default=None, help="state .npz to resume")
    ap.add_argument("--selftest", action="store_true", help="run tests/test_physics.py and exit")
    args = ap.parse_args()

    if args.selftest:
        import runpy
        sys.argv = [os.path.join(HERE, "tests", "test_physics.py")]
        runpy.run_path(sys.argv[0], run_name="__main__")
        return

    try:
        import pyqtgraph  # noqa: F401
        import pyqtgraph.opengl  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print("The interactive GUI needs PySide6, pyqtgraph and PyOpenGL:\n"
              "    pip install -r requirements.txt\n"
              f"(import error: {e})\n"
              "Alternatively render a movie without Qt:   python run_headless.py")
        sys.exit(1)

    from vacuumsim.gui import main as gui_main
    if args.load:
        from vacuumsim.simulation import Simulation
        sim = Simulation.load(args.load)
        sys.exit(gui_main(sim))
    sys.exit(gui_main(N=args.N, a=args.a, seed=args.seed, theory=args.theory))


if __name__ == "__main__":
    main()
