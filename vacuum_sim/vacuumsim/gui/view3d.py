"""
view3d.py -- the OpenGL view of the vacuum volume (pyqtgraph.opengl).

Controls implemented here:
    left mouse drag   : orbit (rotate) around the look-at point          (pyqtgraph built-in)
    mouse wheel       : zoom (camera distance)                           (pyqtgraph built-in)
    middle drag /     : pan                                              (pyqtgraph built-in)
      Ctrl + left drag
    W / S             : fly forward / backward along the viewing direction
    A / D             : strafe left / right
    E / Q             : fly up / down (world z)
    Shift             : 4x faster,   Alt: 4x slower
    R                 : reset the camera
    arrow keys        : orbit in steps                                    (pyqtgraph built-in)

Rendering modes: 'volume' (RGBA slice-stack, GLVolumeItem), 'slice' (one plane of voxels shown as
a thin volume), 'isosurface' (marching cubes via pg.isosurface).

MODIFY: swap GLVolumeItem for a custom ray-marching shader (GLSL) if you want smoother volume
rendering -- the RGBA volume produced by render.scalar_to_rgba can be uploaded as a 3D texture.
"""
from __future__ import annotations
import math
import numpy as np

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui

Qt = QtCore.Qt


class VacuumGLView(gl.GLViewWidget):
    def __init__(self, N: int, parent=None):
        super().__init__(parent)
        self.N = N
        self.setBackgroundColor((11, 13, 18, 255))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._pressed = set()
        self._fly_speed = 0.35                      # lattice units per tick at 60 Hz
        self._items_static = []
        self._plate_items = []
        self._efield_items = []
        self._vol = None
        self._mesh = None
        self._mode = "volume"
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._fly_tick)
        self._timer.start(16)
        self._build_static()
        self.reset_camera()

    # ------------------------------------------------------------------------------------------
    def _build_static(self):
        for it in self._items_static:
            self.removeItem(it)
        self._items_static = []
        N = self.N
        box = gl.GLBoxItem(size=QtGui.QVector3D(N, N, N), color=(90, 100, 120, 160))
        box.translate(-N / 2, -N / 2, -N / 2)
        self.addItem(box)
        self._items_static.append(box)
        grid = gl.GLGridItem()
        grid.setSize(N * 1.6, N * 1.6, 1)
        grid.setSpacing(N / 8, N / 8, 1)
        grid.translate(0, 0, -N / 2 - 0.5)
        grid.setColor((60, 70, 90, 60))
        self.addItem(grid)
        self._items_static.append(grid)
        ax = gl.GLAxisItem(size=QtGui.QVector3D(N * 0.25, N * 0.25, N * 0.25))
        ax.translate(-N / 2, -N / 2, -N / 2)
        self.addItem(ax)
        self._items_static.append(ax)

    def set_lattice_size(self, N: int):
        """Called after a rebuild with a different grid size."""
        self.N = N
        self._build_static()
        if self._vol is not None:
            self.removeItem(self._vol)
            self._vol = None
        if self._mesh is not None:
            self.removeItem(self._mesh)
            self._mesh = None
        self.set_plates([])
        self.show_efield(False)
        self.reset_camera()

    def reset_camera(self):
        self.opts["center"] = pg.Vector(0, 0, 0)
        self.setCameraPosition(distance=self.N * 2.6, elevation=22, azimuth=-55)
        self.update()

    # ------------------------------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------------------------------
    def set_volume(self, rgba: np.ndarray):
        """rgba: uint8 array (N, N, N, 4)."""
        if self._mode != "volume" and self._mode != "slice":
            return
        if self._vol is None:
            self._vol = gl.GLVolumeItem(rgba, sliceDensity=1, smooth=True, glOptions="translucent")
            self._vol.translate(-self.N / 2, -self.N / 2, -self.N / 2)
            self.addItem(self._vol)
        else:
            self._vol.setData(rgba)
        self._vol.setVisible(True)
        if self._mesh is not None:
            self._mesh.setVisible(False)

    def set_isosurface(self, data: np.ndarray, level: float, color=(1.0, 0.6, 0.2, 0.75)):
        try:
            verts, faces = pg.isosurface(np.ascontiguousarray(data, dtype=np.float32), level)
        except Exception:      # noqa: BLE001  (degenerate data)
            return
        if len(verts) == 0 or len(faces) == 0:
            if self._mesh is not None:
                self._mesh.setVisible(False)
            return
        md = gl.MeshData(vertexes=verts, faces=faces)
        if self._mesh is None:
            self._mesh = gl.GLMeshItem(meshdata=md, smooth=True, color=color, shader="shaded",
                                       glOptions="translucent", drawEdges=False)
            self._mesh.translate(-self.N / 2, -self.N / 2, -self.N / 2)
            self.addItem(self._mesh)
        else:
            self._mesh.setMeshData(meshdata=md)
            self._mesh.setColor(color)
        self._mesh.setVisible(True)
        if self._vol is not None:
            self._vol.setVisible(False)

    def set_mode(self, mode: str):
        self._mode = mode

    def set_plates(self, z_layers, color=(0.55, 0.78, 1.0, 0.22)):
        for it in self._plate_items:
            self.removeItem(it)
        self._plate_items = []
        N = self.N
        for z in z_layers:
            zz = z + 0.5 - N / 2
            verts = np.array([[-N / 2, -N / 2, zz], [N / 2, -N / 2, zz], [N / 2, N / 2, zz], [-N / 2, N / 2, zz]])
            faces = np.array([[0, 1, 2], [0, 2, 3]])
            md = gl.MeshData(vertexes=verts, faces=faces)
            m = gl.GLMeshItem(meshdata=md, color=color, smooth=False, glOptions="translucent", drawEdges=True,
                              edgeColor=(0.7, 0.85, 1.0, 0.8))
            self.addItem(m)
            self._plate_items.append(m)

    def show_efield(self, on: bool, strength: float = 1.0):
        key = (bool(on), round(float(strength), 2))
        if getattr(self, "_efield_key", None) == key:
            return
        self._efield_key = key
        for it in self._efield_items:
            self.removeItem(it)
        self._efield_items = []
        if not on:
            return
        N = self.N
        L = 0.35 * N * min(1.0, 0.4 + 0.6 * strength)
        y0, z0 = 0.0, N / 2 + 3.0
        shaft = np.array([[-L / 2, y0, z0], [L / 2, y0, z0]])
        head = np.array([[L / 2, y0, z0], [L / 2 - 2.0, y0 + 1.5, z0], [L / 2, y0, z0], [L / 2 - 2.0, y0 - 1.5, z0]])
        for pts in (shaft, head):
            li = gl.GLLinePlotItem(pos=pts, color=(1.0, 0.82, 0.4, 1.0), width=3, antialias=True, mode="lines")
            self.addItem(li)
            self._efield_items.append(li)

    # ------------------------------------------------------------------------------------------
    # keyboard flying
    # ------------------------------------------------------------------------------------------
    FLY_KEYS = {Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D, Qt.Key.Key_E, Qt.Key.Key_Q}

    def keyPressEvent(self, ev):
        k = ev.key()
        if k in self.FLY_KEYS:
            self._pressed.add(k)
            ev.accept()
            return
        if k == Qt.Key.Key_R:
            self.reset_camera()
            ev.accept()
            return
        super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        k = ev.key()
        if k in self.FLY_KEYS:
            if not ev.isAutoRepeat():
                self._pressed.discard(k)
            ev.accept()
            return
        super().keyReleaseEvent(ev)

    def focusOutEvent(self, ev):
        self._pressed.clear()
        super().focusOutEvent(ev)

    def _camera_basis(self):
        elev = math.radians(self.opts["elevation"])
        azim = math.radians(self.opts["azimuth"])
        # camera sits at center + dist * (cos e cos a, cos e sin a, sin e)  -> forward is minus that
        fwd = np.array([-math.cos(elev) * math.cos(azim), -math.cos(elev) * math.sin(azim), -math.sin(elev)])
        up_world = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up_world)
        n = np.linalg.norm(right)
        right = right / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        return fwd, right, up_world

    def _fly_tick(self):
        if not self._pressed:
            return
        mods = QtGui.QGuiApplication.keyboardModifiers()
        speed = self._fly_speed * (self.N / 32.0)
        if mods & Qt.KeyboardModifier.ShiftModifier:
            speed *= 4.0
        if mods & Qt.KeyboardModifier.AltModifier:
            speed *= 0.25
        fwd, right, up = self._camera_basis()
        move = np.zeros(3)
        if Qt.Key.Key_W in self._pressed:
            move += fwd
        if Qt.Key.Key_S in self._pressed:
            move -= fwd
        if Qt.Key.Key_D in self._pressed:
            move += right
        if Qt.Key.Key_A in self._pressed:
            move -= right
        if Qt.Key.Key_E in self._pressed:
            move += up
        if Qt.Key.Key_Q in self._pressed:
            move -= up
        if not np.any(move):
            return
        move = move / np.linalg.norm(move) * speed
        c = self.opts["center"]
        self.opts["center"] = pg.Vector(c.x() + move[0], c.y() + move[1], c.z() + move[2])
        self.update()

    def screenshot(self, path: str):
        img = self.grabFramebuffer()
        img.save(path)
