"""vacuumsim -- exact lattice simulation of quantum-vacuum fluctuations of free (string-tower) fields.

Physics core (no GUI dependencies): geometry, field, perturbations, observables, simulation, spectrum, units.
GUI: vacuumsim.gui (PySide6/PyQt + pyqtgraph OpenGL).  Headless: run_headless.py (matplotlib).
"""
__version__ = "0.1.0"

from .simulation import Simulation, default_specs      # noqa: F401
from .field import FieldSpec                           # noqa: F401
from .units import Units                               # noqa: F401
