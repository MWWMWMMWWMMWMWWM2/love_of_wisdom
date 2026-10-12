"""
rng_live_viewer.py - real-time plots for the RNG 2 Click microPK rig.

Install once:
    pip install pyserial pyqtgraph PyQt5 numpy

Run with the board:
    python rng_live_viewer.py            (auto-detects the FTDI / USB UART port)
    python rng_live_viewer.py COM5
Run without hardware (simulated chance data, for trying the plots):
    python rng_live_viewer.py --sim
Open an old session file (no hardware needed, nothing is recorded):
    python rng_live_viewer.py --load rng_20261008_162430.csv
    (or use the "Load CSV..." button; the "Window" spinner still applies)

Keys while the window has focus:
    1 = intention HIGH   0 = intention LOW   b = BASELINE   r = REST
    space = pause/resume drawing (recording continues)

Every sample from the board is 32 bytes = 256 bits. Under chance the number of
1-bits per sample is Binomial(256, 0.5): mean 128, sd 8. One sample = one "trial".

Plots (selector, top-left):
    Live trials            ones per trial with the 128 line and the +/-2 sd band
    Cumulative deviation   running sum of (ones - 128) vs trial, with the p = 0.05
                           parabola. The classic PEAR / GCP picture: chance data
                           wanders inside the parabola, an effect walks out of it.
    By condition           cumulative deviation of HIGH, LOW and BASELINE drawn
                           separately, each against its own trial count. The
                           "tripolar" plot: HIGH should drift up, LOW down,
                           BASELINE stay flat if intention does anything.
    Running z              cumulative z-score vs trial with +/-1.96 lines
    Histogram              distribution of ones per trial vs the binomial expectation
    Variance deviation     cumulative sum of (z^2 - 1) with its p = 0.05 envelope -
                           the GCP "network variance" style statistic; catches
                           effects that change spread rather than mean
    Block z                z-score of each block of N trials as bars; |z| > 2 shaded
    Bit bias               mean of each of the 256 bit positions (hardware check;
                           all should sit near 0.5)
    Needle                 feedback display: a needle that swings right when the
                           recent trials (Window, default last 100) run high and
                           left when they run low; the swing is the recent z-score
    Tug of war             feedback display: a rope between LOW and HIGH; the knot
                           slides with the recent z-score, flags mark z = +/-2

The CSV written when "Record" is on has the same columns as rng_logger.py
(time, condition, hex, ones_of_256), so  python rng_logger.py --analyze file.csv  works.
"""
import csv
import math
import sys
import time
from datetime import datetime

import numpy as np

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError:
    sys.exit("PyQt5 missing. Run:  pip install PyQt5 pyqtgraph pyserial numpy")
try:
    import pyqtgraph as pg
except ImportError:
    sys.exit("pyqtgraph missing. Run:  pip install pyqtgraph")

BITS = 256
MEAN = BITS / 2            # 128
SD = math.sqrt(BITS / 4)   # 8
BAUD = 115200

CONDITIONS = ["HIGH", "LOW", "BASELINE", "REST"]
KEYS = {"1": "HIGH", "0": "LOW", "b": "BASELINE", "r": "REST"}
# Validated categorical palette (dark surface): orange, blue, aqua, muted gray.
COLORS = {"HIGH": "#d95926", "LOW": "#3987e5", "BASELINE": "#199e70", "REST": "#8a8a84"}
SURFACE = "#1a1a19"
INK = "#ffffff"
INK2 = "#c3c2b7"
GRID = "#3a3a38"
ACCENT = "#3987e5"


# --------------------------------------------------------------------- sources
class SerialReader(QtCore.QThread):
    sample = QtCore.pyqtSignal(float, str, int)   # t, hex, ones
    message = QtCore.pyqtSignal(str)

    def __init__(self, port):
        super().__init__()
        self.port = port
        self._run = True

    def run(self):
        import serial
        try:
            ser = serial.Serial(self.port, BAUD, timeout=0.5)
            ser.reset_input_buffer()
        except Exception as e:
            self.message.emit(f"Cannot open {self.port}: {e}")
            return
        self.message.emit(f"Listening on {self.port}")
        while self._run:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if len(line) != 64:
                if line:
                    self.message.emit(f"board: {line}")
                continue
            try:
                data = bytes.fromhex(line)
            except ValueError:
                continue
            ones = sum(bin(b).count("1") for b in data)
            self.sample.emit(time.time(), line.upper(), ones)
        ser.close()

    def stop(self):
        self._run = False
        self.wait(2000)


class SimReader(QtCore.QThread):
    """Pure-chance simulator, ~8 trials/s, for trying the viewer without hardware."""
    sample = QtCore.pyqtSignal(float, str, int)
    message = QtCore.pyqtSignal(str)

    def __init__(self, rate=8.0):
        super().__init__()
        self.rate = rate
        self._run = True

    def run(self):
        rng = np.random.default_rng()
        self.message.emit("SIMULATION - chance data, no hardware")
        while self._run:
            data = rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
            ones = sum(bin(b).count("1") for b in data)
            self.sample.emit(time.time(), data.hex().upper(), ones)
            time.sleep(1.0 / self.rate)

    def stop(self):
        self._run = False
        self.wait(2000)


class NullReader(QtCore.QObject):
    """Stand-in reader for --load: no data source, nothing to start or stop."""
    sample = QtCore.pyqtSignal(float, str, int)
    message = QtCore.pyqtSignal(str)

    def start(self):
        pass

    def stop(self):
        pass


def find_port():
    import serial.tools.list_ports
    for p in serial.tools.list_ports.comports():
        text = f"{p.description} {p.manufacturer or ''} {p.hwid}".upper()
        if "FTDI" in text or "USB SERIAL" in text or "VID:PID=0403" in text:
            return p.device
    return None


# ----------------------------------------------------------------------- data
class Session:
    """Growing arrays of trials. Kept simple: numpy views are rebuilt per redraw."""

    def __init__(self):
        self.t = []
        self.ones = []
        self.cond = []          # index into CONDITIONS
        self.bitsum = np.zeros(BITS, dtype=np.int64)
        self.n_bits_trials = 0

    def add(self, t, hexstr, ones, cond_idx):
        self.t.append(t)
        self.ones.append(ones)
        self.cond.append(cond_idx)
        bits = np.unpackbits(np.frombuffer(bytes.fromhex(hexstr), dtype=np.uint8))
        self.bitsum += bits
        self.n_bits_trials += 1

    @property
    def n(self):
        return len(self.ones)

    def arrays(self, window=0):
        o = np.asarray(self.ones, dtype=float)
        c = np.asarray(self.cond, dtype=int)
        if window and len(o) > window:
            o, c = o[-window:], c[-window:]
        return o, c


def z_of(ones):
    n = len(ones)
    if n == 0:
        return float("nan")
    return (ones.sum() - n * MEAN) / (SD * math.sqrt(n))


def p_two_sided(z):
    return math.erfc(abs(z) / math.sqrt(2)) if not math.isnan(z) else float("nan")


# ------------------------------------------------------------------------ app
class Viewer(QtWidgets.QMainWindow):
    def __init__(self, reader, record=True, selftest=False):
        super().__init__()
        self.setWindowTitle("RNG 2 Click - live microPK viewer")
        self.resize(1200, 720)
        self.session = Session()
        self.cond_idx = 2            # BASELINE
        self.paused = False
        self.reader = reader
        self.csv_file = None
        self.csv_writer = None
        self.block_size = 50
        self.selftest = selftest

        pg.setConfigOptions(antialias=True, background=SURFACE, foreground=INK2)
        self._build_ui()
        if record:
            self._start_csv()

        reader.sample.connect(self.on_sample)
        reader.message.connect(self.on_message)
        reader.start()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.redraw)
        self.timer.start(150)         # ~7 redraws/s; data arrives independently

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(8, 8, 8, 8)

        bar = QtWidgets.QHBoxLayout()
        vbox.addLayout(bar)

        self.plot_select = QtWidgets.QComboBox()
        self.plot_select.addItems([
            "Live trials", "Cumulative deviation", "By condition", "Running z",
            "Histogram", "Variance deviation", "Block z", "Bit bias",
            "Needle", "Tug of war",
        ])
        self.plot_select.currentIndexChanged.connect(self.redraw)
        bar.addWidget(QtWidgets.QLabel("Plot:"))
        bar.addWidget(self.plot_select)

        bar.addSpacing(16)
        bar.addWidget(QtWidgets.QLabel("Condition:"))
        self.cond_buttons = []
        group = QtWidgets.QButtonGroup(self)
        for i, name in enumerate(CONDITIONS):
            key = [k for k, v in KEYS.items() if v == name][0]
            b = QtWidgets.QPushButton(f"{name} [{key}]")
            b.setCheckable(True)
            b.setMinimumWidth(110)
            b.setStyleSheet(
                f"QPushButton{{padding:4px 10px;border:1px solid {COLORS[name]};color:{INK};}}"
                f"QPushButton:checked{{background:{COLORS[name]};color:{SURFACE};font-weight:bold;}}")
            group.addButton(b, i)
            bar.addWidget(b)
            self.cond_buttons.append(b)
        group.idClicked.connect(self.set_condition)
        self.cond_buttons[self.cond_idx].setChecked(True)

        bar.addSpacing(16)
        bar.addWidget(QtWidgets.QLabel("Window (trials, 0 = all):"))
        self.window_spin = QtWidgets.QSpinBox()
        self.window_spin.setRange(0, 1_000_000)
        self.window_spin.setSingleStep(100)
        self.window_spin.setValue(0)
        bar.addWidget(self.window_spin)

        bar.addWidget(QtWidgets.QLabel("Block:"))
        self.block_spin = QtWidgets.QSpinBox()
        self.block_spin.setRange(5, 10000)
        self.block_spin.setValue(self.block_size)
        bar.addWidget(self.block_spin)

        self.pause_btn = QtWidgets.QPushButton("Pause [space]")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self.set_paused)
        bar.addWidget(self.pause_btn)

        self.load_btn = QtWidgets.QPushButton("Load CSV...")
        self.load_btn.clicked.connect(self.load_dialog)
        bar.addWidget(self.load_btn)
        bar.addStretch(1)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.getAxis("left").setTextPen(INK2)
        self.plot.getAxis("bottom").setTextPen(INK2)
        vbox.addWidget(self.plot, 1)

        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet(f"color:{INK}; font-family: Consolas, monospace;")
        vbox.addWidget(self.status)
        self.msg = QtWidgets.QLabel("")
        self.msg.setStyleSheet(f"color:{INK2};")
        vbox.addWidget(self.msg)

        self.setStyleSheet(f"QMainWindow, QWidget{{background:{SURFACE};color:{INK};}}"
                           f"QComboBox, QSpinBox{{padding:3px;}}")

        for key, name in KEYS.items():
            sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            sc.activated.connect(lambda n=name: self.set_condition(CONDITIONS.index(n)))
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self).activated.connect(
            lambda: self.pause_btn.toggle())

    # ---------------------------------------------------------- handlers
    def set_condition(self, idx):
        self.cond_idx = idx
        self.cond_buttons[idx].setChecked(True)
        self.on_message(f"Condition: {CONDITIONS[idx]}")

    def set_paused(self, on):
        self.paused = on

    def on_message(self, text):
        self.msg.setText(text)

    def on_sample(self, t, hexstr, ones):
        self.session.add(t, hexstr, ones, self.cond_idx)
        if self.csv_writer:
            self.csv_writer.writerow([f"{t:.3f}", CONDITIONS[self.cond_idx], hexstr, ones])
            if self.session.n % 50 == 0:
                self.csv_file.flush()

    def load_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open session CSV", "", "CSV (*.csv)")
        if path:
            self.load_csv(path)

    def load_csv(self, path):
        """Replace the live session with the contents of a recorded CSV."""
        self.reader.stop()
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = self.csv_writer = None
        self.session = Session()
        n = 0
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                cond = row.get("condition", "BASELINE")
                idx = CONDITIONS.index(cond) if cond in CONDITIONS else 2
                self.session.add(float(row["time"]), row["hex"], int(row["ones_of_256"]), idx)
                n += 1
        self.setWindowTitle(f"RNG 2 Click - viewing {path}  ({n} trials, not recording)")
        self.on_message(f"Loaded {n} trials from {path}. Live input stopped.")
        self.redraw()

    def _start_csv(self):
        name = datetime.now().strftime("rng_%Y%m%d_%H%M%S.csv")
        self.csv_file = open(name, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["time", "condition", "hex", "ones_of_256"])
        self.setWindowTitle(f"RNG 2 Click - live microPK viewer - recording {name}")

    def closeEvent(self, ev):
        self.timer.stop()
        self.reader.stop()
        if self.csv_file:
            self.csv_file.close()
        ev.accept()

    # ------------------------------------------------------------ drawing
    def redraw(self):
        if self.paused and not self.selftest:
            return
        s = self.session
        self.block_size = self.block_spin.value()
        o_all, c_all = s.arrays(0)
        o, c = s.arrays(self.window_spin.value())
        self._status(o_all, c_all)
        self.plot.clear()
        self.plot.showAxis("left"); self.plot.showAxis("bottom")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setAspectLocked(False)
        self.plot.enableAutoRange()
        if len(o) == 0:
            self.plot.setTitle("waiting for data...", color=INK2)
            return
        kind = self.plot_select.currentText()
        getattr(self, "draw_" + kind.lower().replace(" ", "_"))(o, c)

    def _status(self, o, c):
        parts = []
        for i, name in enumerate(CONDITIONS[:3]):
            sel = o[c == i]
            if len(sel):
                z = z_of(sel)
                parts.append(f"{name}: n={len(sel)} mean={sel.mean():.2f} z={z:+.2f} p={p_two_sided(z):.3f}")
        if len(o):
            z = z_of(o)
            parts.insert(0, f"ALL: n={len(o)} mean={o.mean():.2f} z={z:+.2f} p={p_two_sided(z):.3f}")
        h, l = o[c == 0], o[c == 1]
        if len(h) and len(l):
            zd = (h.mean() - l.mean()) / math.sqrt(SD**2 / len(h) + SD**2 / len(l))
            parts.append(f"HIGH-LOW: z={zd:+.2f} p={p_two_sided(zd):.3f}")
        self.status.setText("   |   ".join(parts))

    def _pen(self, color, width=2, style=None):
        pen = pg.mkPen(color, width=width)
        if style:
            pen.setStyle(style)
        return pen

    def _hline(self, y, color=INK2, dashed=True, label=None):
        line = pg.InfiniteLine(pos=y, angle=0,
                               pen=self._pen(color, 1, QtCore.Qt.DashLine if dashed else None))
        self.plot.addItem(line)
        if label:
            txt = pg.TextItem(label, color=INK2, anchor=(0, 1))
            txt.setPos(0, y)
            self.plot.addItem(txt)

    def _label(self, title, xlab, ylab):
        self.plot.setTitle(title, color=INK, size="11pt")
        self.plot.setLabel("bottom", xlab, color=INK2)
        self.plot.setLabel("left", ylab, color=INK2)

    def _cond_colors(self, c):
        return [COLORS[CONDITIONS[i]] for i in c]

    # 1 ---------------------------------------------------------------
    def draw_live_trials(self, o, c):
        self._label("Ones per trial (256 bits each)", "trial", "ones")
        x = np.arange(len(o))
        self._hline(MEAN, INK2, False, "128 = chance")
        self._hline(MEAN + 2 * SD, GRID, True, "+2 sd")
        self._hline(MEAN - 2 * SD, GRID, True, "-2 sd")
        brushes = [pg.mkBrush(col) for col in self._cond_colors(c)]
        self.plot.addItem(pg.ScatterPlotItem(x, o, size=5, pen=None, brush=brushes))
        if len(o) >= 20:
            k = max(5, len(o) // 50)
            sm = np.convolve(o, np.ones(k) / k, mode="valid")
            self.plot.plot(x[k - 1:], sm, pen=self._pen(INK, 2))
        self.plot.setYRange(MEAN - 5 * SD, MEAN + 5 * SD)

    # 2 ---------------------------------------------------------------
    def draw_cumulative_deviation(self, o, c):
        self._label("Cumulative deviation from chance (PEAR style)", "trial", "sum of (ones - 128)")
        n = np.arange(1, len(o) + 1)
        cum = np.cumsum(o - MEAN)
        env = 1.96 * SD * np.sqrt(n)
        self.plot.plot(n, env, pen=self._pen(GRID, 1, QtCore.Qt.DashLine))
        self.plot.plot(n, -env, pen=self._pen(GRID, 1, QtCore.Qt.DashLine))
        self._hline(0, INK2, False)
        self.plot.plot(n, cum, pen=self._pen(ACCENT, 2))
        txt = pg.TextItem("dashed = p 0.05 envelope", color=INK2, anchor=(1, 0))
        txt.setPos(n[-1], env[-1])
        self.plot.addItem(txt)

    # 3 ---------------------------------------------------------------
    def draw_by_condition(self, o, c):
        self._label("Cumulative deviation per condition (tripolar)", "trials in that condition",
                    "sum of (ones - 128)")
        self._hline(0, INK2, False)
        legend = self.plot.addLegend(offset=(10, 10))
        nmax = 1
        for i, name in enumerate(CONDITIONS[:3]):
            sel = o[c == i]
            if len(sel) == 0:
                continue
            n = np.arange(1, len(sel) + 1)
            nmax = max(nmax, len(sel))
            self.plot.plot(n, np.cumsum(sel - MEAN), pen=self._pen(COLORS[name], 2), name=name)
        n = np.arange(1, nmax + 1)
        env = 1.96 * SD * np.sqrt(n)
        self.plot.plot(n, env, pen=self._pen(GRID, 1, QtCore.Qt.DashLine))
        self.plot.plot(n, -env, pen=self._pen(GRID, 1, QtCore.Qt.DashLine))

    # 4 ---------------------------------------------------------------
    def draw_running_z(self, o, c):
        self._label("Running z-score", "trial", "z")
        n = np.arange(1, len(o) + 1)
        z = np.cumsum(o - MEAN) / (SD * np.sqrt(n))
        self._hline(0, INK2, False)
        self._hline(1.96, GRID, True, "+1.96")
        self._hline(-1.96, GRID, True, "-1.96")
        self.plot.plot(n, z, pen=self._pen(ACCENT, 2))
        self.plot.setYRange(min(-3, z.min() - 0.5), max(3, z.max() + 0.5))

    # 5 ---------------------------------------------------------------
    def draw_histogram(self, o, c):
        self._label("Distribution of ones per trial vs binomial(256, 0.5)", "ones", "trials")
        lo, hi = int(MEAN - 5 * SD), int(MEAN + 5 * SD)
        edges = np.arange(lo, hi + 2) - 0.5
        counts, _ = np.histogram(o, bins=edges)
        centers = np.arange(lo, hi + 1)
        bars = pg.BarGraphItem(x=centers, height=counts, width=0.8, brush=ACCENT, pen=SURFACE)
        self.plot.addItem(bars)
        # expected: binomial pmf * n, via normal approximation (n=256 is plenty)
        exp = len(o) * np.exp(-0.5 * ((centers - MEAN) / SD) ** 2) / (SD * math.sqrt(2 * math.pi))
        self.plot.plot(centers, exp, pen=self._pen(INK, 2))

    # 6 ---------------------------------------------------------------
    def draw_variance_deviation(self, o, c):
        self._label("Cumulative variance deviation  sum(z^2 - 1)  (GCP netvar style)",
                    "trial", "sum of (z^2 - 1)")
        n = np.arange(1, len(o) + 1)
        z2 = ((o - MEAN) / SD) ** 2
        cum = np.cumsum(z2 - 1)
        env = 1.96 * np.sqrt(2 * n)
        self.plot.plot(n, env, pen=self._pen(GRID, 1, QtCore.Qt.DashLine))
        self.plot.plot(n, -env, pen=self._pen(GRID, 1, QtCore.Qt.DashLine))
        self._hline(0, INK2, False)
        self.plot.plot(n, cum, pen=self._pen(ACCENT, 2))

    # 7 ---------------------------------------------------------------
    def draw_block_z(self, o, c):
        b = self.block_size
        self._label(f"z-score per block of {b} trials", "block", "z")
        nb = len(o) // b
        if nb == 0:
            self.plot.setTitle(f"need {b} trials for the first block ({len(o)} so far)", color=INK2)
            return
        blocks = o[: nb * b].reshape(nb, b)
        z = (blocks.sum(axis=1) - b * MEAN) / (SD * math.sqrt(b))
        x = np.arange(1, nb + 1)
        brushes = [pg.mkBrush(COLORS["HIGH"] if abs(v) > 1.96 else ACCENT) for v in z]
        self.plot.addItem(pg.BarGraphItem(x=x, height=z, width=0.8, brushes=brushes, pen=SURFACE))
        self._hline(0, INK2, False)
        self._hline(1.96, GRID, True)
        self._hline(-1.96, GRID, True)
        self.plot.setYRange(min(-3.5, z.min() - 0.5), max(3.5, z.max() + 0.5))

    # 8 ---------------------------------------------------------------
    def draw_bit_bias(self, o, c):
        s = self.session
        self._label("Mean of each bit position (hardware check: all near 0.5)", "bit", "mean")
        m = s.bitsum / max(1, s.n_bits_trials)
        se = 0.5 / math.sqrt(max(1, s.n_bits_trials))
        x = np.arange(BITS)
        self._hline(0.5, INK2, False)
        self._hline(0.5 + 1.96 * se, GRID, True, "+1.96 se")
        self._hline(0.5 - 1.96 * se, GRID, True)
        self.plot.addItem(pg.ScatterPlotItem(x, m, size=5, pen=None, brush=ACCENT))
        self.plot.setYRange(0.5 - 6 * se - 0.01, 0.5 + 6 * se + 0.01)

    # 9 ---------------------------------------------------------------
    def _recent_z(self, o):
        w = self.window_spin.value() or 100
        sel = o[-w:]
        return z_of(sel), len(sel)

    def draw_needle(self, o, c):
        z, n = self._recent_z(o)
        self._label(f"Needle   recent z = {z:+.2f}  (last {n} trials)", "", "")
        self.plot.hideAxis("left"); self.plot.hideAxis("bottom")
        self.plot.showGrid(x=False, y=False)
        self.plot.setAspectLocked(True)
        # dial
        ang = np.linspace(math.pi, 0, 200)
        self.plot.plot(np.cos(ang), np.sin(ang), pen=self._pen(GRID, 2))
        for zt in (-3, -2, -1, 0, 1, 2, 3):
            a = math.pi / 2 - zt * (math.pi / 2) / 3.5
            self.plot.plot([0.92 * math.cos(a), 1.0 * math.cos(a)],
                           [0.92 * math.sin(a), 1.0 * math.sin(a)],
                           pen=self._pen(INK2 if zt else INK, 2))
            t = pg.TextItem(f"{zt:+d}" if zt else "0", color=INK2, anchor=(0.5, 0.5))
            t.setPos(1.12 * math.cos(a), 1.12 * math.sin(a)); self.plot.addItem(t)
        for txt, x, col in (("LOW", -0.75, COLORS["LOW"]), ("HIGH", 0.75, COLORS["HIGH"])):
            t = pg.TextItem(txt, color=col, anchor=(0.5, 0.5)); t.setPos(x, -0.15); self.plot.addItem(t)
        # needle: z clamped to +/-3.5 spans the dial
        zc = max(-3.5, min(3.5, 0.0 if math.isnan(z) else z))
        a = math.pi / 2 - zc * (math.pi / 2) / 3.5
        col = COLORS["HIGH"] if zc > 0 else COLORS["LOW"] if zc < 0 else INK
        self.plot.plot([0, 0.85 * math.cos(a)], [0, 0.85 * math.sin(a)], pen=self._pen(col, 4))
        self.plot.addItem(pg.ScatterPlotItem([0], [0], size=14, brush=INK, pen=None))
        self.plot.setXRange(-1.3, 1.3); self.plot.setYRange(-0.3, 1.3)

    # 10 --------------------------------------------------------------
    def draw_tug_of_war(self, o, c):
        z, n = self._recent_z(o)
        self._label(f"Tug of war   recent z = {z:+.2f}  (last {n} trials)", "z", "")
        self.plot.hideAxis("left")
        self.plot.showGrid(x=False, y=False)
        self.plot.setAspectLocked(False)
        zc = max(-3.5, min(3.5, 0.0 if math.isnan(z) else z))
        # rope, with the two teams' halves coloured
        self.plot.plot([-3.5, zc], [0, 0], pen=self._pen(COLORS["LOW"], 6))
        self.plot.plot([zc, 3.5], [0, 0], pen=self._pen(COLORS["HIGH"], 6))
        for xf in (-2, 2):   # flags at p 0.05
            self.plot.plot([xf, xf], [-0.25, 0.25], pen=self._pen(GRID, 2, QtCore.Qt.DashLine))
            t = pg.TextItem("z = %+d" % xf, color=INK2, anchor=(0.5, 0)); t.setPos(xf, 0.28); self.plot.addItem(t)
        self.plot.plot([0, 0], [-0.15, 0.15], pen=self._pen(INK2, 1))
        # knot
        self.plot.addItem(pg.ScatterPlotItem([zc], [0], size=26, brush=INK, pen=pg.mkPen(SURFACE, width=3)))
        for txt, x, col in (("LOW pulls", -3.0, COLORS["LOW"]), ("HIGH pulls", 3.0, COLORS["HIGH"])):
            t = pg.TextItem(txt, color=col, anchor=(0.5, 1)); t.setPos(x, -0.3); self.plot.addItem(t)
        self.plot.setXRange(-3.8, 3.8); self.plot.setYRange(-1, 1)


# ----------------------------------------------------------------------- main
def main():
    args = sys.argv[1:]
    sim = "--sim" in args
    selftest = "--selftest" in args
    norec = "--no-record" in args
    load = args[args.index("--load") + 1] if "--load" in args and args.index("--load") + 1 < len(args) else None
    port_args = [a for a in args if not a.startswith("--") and a != load]

    app = QtWidgets.QApplication(sys.argv)
    if load:
        reader = NullReader()
    elif sim:
        reader = SimReader(rate=200 if selftest else 8)
    else:
        port = port_args[0] if port_args else find_port()
        if not port:
            QtWidgets.QMessageBox.critical(None, "No port", "No FTDI / USB UART port found.\n"
                                           "Plug in the USB UART Click or run:  rng_live_viewer.py COMx")
            sys.exit(1)
        reader = SerialReader(port)

    win = Viewer(reader, record=not norec and not selftest and not load, selftest=selftest)
    win.show()
    if load:
        win.load_csv(load)
        win.plot_select.setCurrentText("Running z")

    if selftest:
        # cycle through every plot a few times, then exit 0 if nothing raised
        def step(i=[0]):
            win.plot_select.setCurrentIndex(i[0] % win.plot_select.count())
            win.set_condition(i[0] % 3)
            win.redraw()
            i[0] += 1
            if i[0] >= win.plot_select.count() * 3:
                print(f"selftest OK: {win.session.n} trials, {win.plot_select.count()} plots")
                win.close()
                app.quit()
        t = QtCore.QTimer()
        t.timeout.connect(step)
        t.start(300)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
