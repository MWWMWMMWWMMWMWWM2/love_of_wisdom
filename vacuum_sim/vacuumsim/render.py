"""
render.py -- numpy-only colour mapping shared by the OpenGL GUI and the headless renderer.

Volume rendering strategy: every voxel gets an RGBA colour; the alpha channel grows with the
quantity being shown so that the interior of the cube stays visible (low values are transparent).
GLVolumeItem (pyqtgraph) draws such an RGBA volume as a stack of textured slices.

MODIFY: add colour maps to CMAPS (list of RGB anchors 0-255, evenly spaced); change ALPHA_GAMMA
defaults in VolumeStyle for a denser / sparser look.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

CMAPS = {
    "inferno": [(0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99), (212, 72, 66),
                (245, 125, 21), (250, 193, 39), (252, 255, 164)],
    "viridis": [(68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142), (38, 130, 142),
                (31, 158, 137), (53, 183, 121), (109, 205, 89), (180, 222, 44), (253, 231, 37)],
    "plasma": [(13, 8, 135), (84, 2, 163), (139, 10, 165), (185, 50, 137), (219, 92, 104),
               (244, 136, 73), (254, 188, 43), (240, 249, 33)],
    "ice-fire": [(0, 20, 80), (0, 90, 200), (120, 200, 255), (245, 245, 245), (255, 170, 90),
                 (230, 60, 20), (120, 0, 0)],                   # diverging (negative -> positive)
    "coolwarm": [(59, 76, 192), (98, 130, 234), (141, 176, 254), (184, 208, 249), (221, 221, 221),
                 (245, 196, 173), (244, 154, 123), (222, 96, 77), (180, 4, 38)],
}
SEQUENTIAL = ("inferno", "viridis", "plasma")
DIVERGING = ("ice-fire", "coolwarm")


def lut(name: str, n: int = 256) -> np.ndarray:
    anchors = np.array(CMAPS[name], dtype=float)
    x = np.linspace(0, 1, len(anchors))
    xi = np.linspace(0, 1, n)
    return np.stack([np.interp(xi, x, anchors[:, c]) for c in range(3)], axis=1)  # (n, 3) floats 0-255


@dataclass
class VolumeStyle:
    cmap: str = "inferno"
    alpha_max: float = 0.9        # opacity of the strongest voxels
    alpha_gamma: float = 2.0      # alpha = alpha_max * v^gamma  (larger -> only peaks visible)
    lo_percentile: float = 5.0    # auto-scale range from these percentiles of the data
    hi_percentile: float = 99.5
    fixed_range: tuple | None = None   # (lo, hi) overrides percentiles


class RangeSmoother:
    """Exponential moving average of the (lo, hi) colour range to avoid flicker between frames."""
    def __init__(self, tau_frames: float = 15.0):
        self.k = 1.0 / max(tau_frames, 1.0)
        self.lo = None
        self.hi = None

    def update(self, lo, hi):
        if self.lo is None:
            self.lo, self.hi = lo, hi
        else:
            self.lo += self.k * (lo - self.lo)
            self.hi += self.k * (hi - self.hi)
        return self.lo, self.hi

    def reset(self):
        self.lo = self.hi = None


def scalar_to_rgba(data: np.ndarray, style: VolumeStyle, diverging: bool,
                   smoother: RangeSmoother | None = None, mask: np.ndarray | None = None):
    """Map a scalar volume to a uint8 RGBA volume.  Returns (rgba, (lo, hi)).
    diverging=True: data is signed, mapped symmetrically around zero."""
    d = np.asarray(data, dtype=float)
    if style.fixed_range is not None:
        lo, hi = style.fixed_range
    elif diverging:
        hi = float(np.percentile(np.abs(d), style.hi_percentile)) or 1e-30
        lo = -hi
    else:
        lo = float(np.percentile(d, style.lo_percentile))
        hi = float(np.percentile(d, style.hi_percentile))
        if hi <= lo:
            hi = lo + 1e-30
    if smoother is not None:
        lo, hi = smoother.update(lo, hi)
    table = lut(style.cmap)
    n = len(table)
    if diverging:
        v = np.clip(d / (hi if hi != 0 else 1e-30), -1.0, 1.0)
        idx = ((v + 1.0) * 0.5 * (n - 1)).astype(int)
        strength = np.abs(v)
    else:
        v = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
        idx = (v * (n - 1)).astype(int)
        strength = v
    rgba = np.empty(d.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = table[idx]
    alpha = style.alpha_max * strength ** style.alpha_gamma
    if mask is not None:
        alpha = np.where(mask, 0.0, alpha)
    rgba[..., 3] = (255 * np.clip(alpha, 0, 1)).astype(np.uint8)
    return rgba, (lo, hi)


def plate_rgba(shape, plate_layers, color=(180, 220, 255), alpha=70) -> np.ndarray:
    """Translucent slabs marking the mirror positions (for volume-only renderers)."""
    rgba = np.zeros(tuple(shape) + (4,), dtype=np.uint8)
    for z in plate_layers:
        rgba[:, :, z, :3] = color
        rgba[:, :, z, 3] = alpha
    return rgba
