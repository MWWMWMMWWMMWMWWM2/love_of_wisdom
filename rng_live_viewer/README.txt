rng_live_viewer - real-time plots for the RNG 2 Click microPK rig
==================================================================
Install once (same terminal you use for rng_logger):
    python -m pip install pyserial pyqtgraph PyQt5 numpy

Run with the board (close rng_logger.py first - only one program can hold COM4):
    cd /d "D:\5 necto or microe other stuff\rng_live_viewer"
    python rng_live_viewer.py
Try it without the board (simulated chance data):
    python rng_live_viewer.py --sim

Top-left "Plot" selector:
    Live trials          ones per trial, 128 line, +/-2 sd band, smoothed trend
    Cumulative deviation the classic PEAR/GCP random-walk with the p=0.05 parabola
    By condition         HIGH / LOW / BASELINE cumulative walks overlaid ("tripolar")
    Running z            cumulative z-score, +/-1.96 lines
    Histogram            ones-per-trial distribution vs binomial expectation
    Variance deviation   cumulative sum(z^2-1) with envelope (GCP netvar style)
    Block z              z per block of N trials (Block spinner), |z|>2 in orange
    Bit bias             mean of each of the 256 bit positions - hardware sanity check
    Needle               feedback: swings with the recent z (Window spinner, default last 100)
    Tug of war           feedback: knot slides toward HIGH or LOW with the recent z

Condition buttons or keys 1 / 0 / b / r tag the trials, exactly like rng_logger.
Window = 0 shows everything; e.g. 500 shows only the last 500 trials.
Space pauses drawing (recording continues).

Every run writes rng_YYYYMMDD_HHMMSS.csv in this folder, same format as rng_logger,
so   python ..\rng2_clicker2_pic24\rng_logger.py --analyze <file>   works on it.
Add --no-record to skip the CSV.

Open an old session (no hardware needed):
    python rng_live_viewer.py --load rng_20261008_162430.csv
or click "Load CSV..." in the running viewer. All ten plots work on the file.

Unattended run (e.g. 30 min): start the viewer, leave it on BASELINE, walk
away, close it afterwards. Then --load the CSV and look at Running z,
Cumulative deviation and Block z to see how a pure-chance source wanders.

A caution, illustrated by the simulator: in a chance-only run of ~1000 trials the
HIGH condition came out at z = +2.04 (p = 0.04). With several conditions and
several plots, something will look "significant" by luck. The number that
matters is HIGH-LOW over many sessions, decided on in advance.
