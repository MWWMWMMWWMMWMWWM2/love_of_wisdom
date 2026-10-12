"""GUI entry point.  Requires: PySide6 (or PyQt6/PyQt5), pyqtgraph, PyOpenGL.  See run.py."""
from __future__ import annotations
import sys


def main(sim=None, **sim_kwargs):
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtGui, QtCore
    from ..simulation import Simulation
    from .main_window import MainWindow

    pg.setConfigOptions(antialias=False, background=(16, 18, 24), foreground=(200, 208, 220))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    # dark palette
    pal = QtGui.QPalette()
    dark = QtGui.QColor(24, 27, 34)
    mid = QtGui.QColor(36, 40, 49)
    text = QtGui.QColor(220, 226, 235)
    pal.setColor(QtGui.QPalette.ColorRole.Window, dark)
    pal.setColor(QtGui.QPalette.ColorRole.WindowText, text)
    pal.setColor(QtGui.QPalette.ColorRole.Base, mid)
    pal.setColor(QtGui.QPalette.ColorRole.AlternateBase, dark)
    pal.setColor(QtGui.QPalette.ColorRole.Text, text)
    pal.setColor(QtGui.QPalette.ColorRole.Button, mid)
    pal.setColor(QtGui.QPalette.ColorRole.ButtonText, text)
    pal.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(70, 120, 200))
    pal.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(255, 255, 255))
    pal.setColor(QtGui.QPalette.ColorRole.ToolTipBase, mid)
    pal.setColor(QtGui.QPalette.ColorRole.ToolTipText, text)
    app.setPalette(pal)

    if sim is None:
        sim = Simulation(**sim_kwargs)
    win = MainWindow(sim)
    win.resize(1500, 900)
    win.show()
    return app.exec() if hasattr(app, "exec") else app.exec_()
