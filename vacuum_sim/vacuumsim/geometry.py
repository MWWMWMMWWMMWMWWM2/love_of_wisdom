"""
geometry.py -- the lattice and the mode bases that diagonalise the free-field Hamiltonian.

We simulate a *lattice-regularised* free field.  The Hamiltonian per real scalar field is
    H = a^3 sum_x [ 1/2 pi_x^2 + 1/2 sum_i (phi_{x+i} - phi_x)^2 / a^2 + 1/2 m^2 phi_x^2 ]
On a periodic N^3 grid this is diagonalised by the discrete Fourier transform, with the *lattice
dispersion relation*
    omega_k^2 = m^2 + (2/a)^2 sum_i sin^2(k_i a / 2)                       (eq. G1)
which tends to the continuum k^2 + m^2 for k a << 1.  The lattice spacing a is the UV cutoff
(Lambda ~ pi/a) that every zero-point sum needs.  Nothing here is approximate: the lattice theory
is an exactly solvable quantum system (N^3 harmonic oscillators).

Perfectly reflecting "Casimir plates" are modelled as Dirichlet planes phi = 0 at two z-layers.
Between/outside them the z-direction is a Dirichlet chain of n sites, diagonalised by the
type-I discrete sine transform with eigenvalues (2/a)^2 sin^2(j pi / (2 (n+1))), j = 1..n.
The two chains (inside, outside-with-periodic-wrap) plus the FFT in x, y give the exact modes.

Each geometry exposes a list of *blocks*; a block knows how to go from the real-space grid slab
to its orthonormal mode coefficients and back, and how to compute its mode frequencies.
Using orthonormal transforms everywhere means: vacuum <|Q|^2> = 1/(2 omega), <|P|^2> = omega/2
for every coefficient (see field.py), regardless of geometry.

MODIFY:
 * Other boundary conditions (Neumann -> DCT type 1, mixed, absorbing) fit the same Block API.
 * A non-cubic box: change Lattice.shape handling (everything uses self.shape, not N alone).
 * Curved/expanding space: make `a` time dependent (scale factor) -> cosmological particle creation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy import fft as sfft

_FFT_KW = dict(norm="ortho", workers=-1)


@dataclass
class Lattice:
    N: int = 48
    a: float = 1.0                         # lattice spacing in string units (l_s)

    @property
    def shape(self):
        return (self.N, self.N, self.N)

    @property
    def L(self):
        return self.N * self.a

    @property
    def volume(self):
        return self.L ** 3

    @property
    def n_sites(self):
        return self.N ** 3

    def kgrid(self):
        """Periodic lattice momenta k_i = 2 pi n_i / (N a), n_i in FFT order."""
        k1 = 2 * np.pi * sfft.fftfreq(self.N, d=self.a)
        return np.meshgrid(k1, k1, k1, indexing="ij")

    def lattice_k2_1d(self, k1):
        """(2/a)^2 sin^2(k a/2): the eigenvalue of -Laplacian_1d for momentum k."""
        return (2.0 / self.a) ** 2 * np.sin(0.5 * k1 * self.a) ** 2

    @property
    def k_max(self):
        """Largest |k| component resolvable (Brillouin-zone edge)."""
        return np.pi / self.a

    @property
    def omega_max(self):
        """Largest lattice frequency for a massless field (corner of the zone)."""
        return np.sqrt(3.0) * 2.0 / self.a


# ----------------------------------------------------------------------------------------------
class Block:
    """A set of modes: a slab of the grid (all x, all y, a set of z-layers) with an orthonormal
    transform.  Subclasses: PeriodicBlock, DirichletBlock."""
    zindex: np.ndarray       # which z-layers of the full grid this block covers (in chain order)
    shape: tuple             # shape of the coefficient array

    def forward(self, slab: np.ndarray) -> np.ndarray:  # real-space slab -> coefficients
        raise NotImplementedError

    def inverse(self, coeff: np.ndarray) -> np.ndarray:  # coefficients -> real-space slab
        raise NotImplementedError

    def k2_spatial(self, kshift_x: float = 0.0) -> np.ndarray:
        """Lattice -Laplacian eigenvalue per mode, with the x-momentum shifted by kshift_x
        (used for the uniform electric field in the temporal gauge, see field.py)."""
        raise NotImplementedError

    def z_basis(self) -> np.ndarray:
        """Orthonormal z-basis functions as a matrix S[j, z_local] (complex for periodic)."""
        raise NotImplementedError


class PeriodicBlock(Block):
    def __init__(self, lat: Lattice):
        self.lat = lat
        self.zindex = np.arange(lat.N)
        self.shape = lat.shape
        kx, ky, kz = lat.kgrid()
        self.kx, self.ky, self.kz = kx, ky, kz
        self._k2_yz = lat.lattice_k2_1d(ky) + lat.lattice_k2_1d(kz)

    def forward(self, slab):
        return sfft.fftn(slab, axes=(0, 1, 2), **_FFT_KW)

    def inverse(self, coeff):
        return sfft.ifftn(coeff, axes=(0, 1, 2), **_FFT_KW)

    def k2_spatial(self, kshift_x=0.0):
        return self.lat.lattice_k2_1d(self.kx + kshift_x) + self._k2_yz

    def k2_transverse(self, kshift_x=0.0):
        """x,y part only (used for the ensemble energy-density profile along z)."""
        return self.lat.lattice_k2_1d(self.kx + kshift_x) + self.lat.lattice_k2_1d(self.ky)

    def z_basis(self):
        N = self.lat.N
        kz1 = 2 * np.pi * sfft.fftfreq(N, d=self.lat.a)
        z = np.arange(N) * self.lat.a
        # forward transform convention  Q_k = 1/sqrt(N) sum_z e^{-i k z} phi_z
        return np.exp(-1j * np.outer(kz1, z)) / np.sqrt(N)


class DirichletBlock(Block):
    """z-layers `zindex` (in chain order) form a Dirichlet chain of n = len(zindex) sites:
    phi = 0 on the two plates that bound it.  Transform: FFT in x,y and DST-I in z."""

    def __init__(self, lat: Lattice, zindex: np.ndarray):
        self.lat = lat
        self.zindex = np.asarray(zindex, dtype=int)
        n = len(self.zindex)
        self.n = n
        self.shape = (lat.N, lat.N, n)
        k1 = 2 * np.pi * sfft.fftfreq(lat.N, d=lat.a)
        kx, ky = np.meshgrid(k1, k1, indexing="ij")
        self.kx = kx[:, :, None]
        self._k2_y = lat.lattice_k2_1d(ky)[:, :, None]
        j = np.arange(1, n + 1)
        self._k2_z = ((2.0 / lat.a) ** 2 * np.sin(j * np.pi / (2 * (n + 1))) ** 2)[None, None, :]

    def forward(self, slab):
        c = sfft.fft2(slab, axes=(0, 1), **_FFT_KW)
        return sfft.dst(c, type=1, axis=2, norm="ortho")

    def inverse(self, coeff):
        c = sfft.dst(coeff, type=1, axis=2, norm="ortho")
        return sfft.ifft2(c, axes=(0, 1), **_FFT_KW)

    def k2_spatial(self, kshift_x=0.0):
        return self.lat.lattice_k2_1d(self.kx + kshift_x) + self._k2_y + self._k2_z

    def k2_transverse(self, kshift_x=0.0):
        return np.broadcast_to(self.lat.lattice_k2_1d(self.kx + kshift_x) + self._k2_y, self.shape)

    def z_basis(self):
        n = self.n
        j = np.arange(1, n + 1)[:, None]
        s = np.arange(1, n + 1)[None, :]
        return np.sqrt(2.0 / (n + 1)) * np.sin(np.pi * j * s / (n + 1))

    @property
    def length(self):
        """Distance between the two bounding Dirichlet planes (continuum-equivalent length)."""
        return (self.n + 1) * self.lat.a


# ----------------------------------------------------------------------------------------------
class Geometry:
    """Base: a list of blocks that together cover all non-plate sites."""
    def __init__(self, lat: Lattice):
        self.lat = lat
        self.blocks: list[Block] = []
        self.plate_layers: np.ndarray = np.array([], dtype=int)

    @property
    def kind(self):
        return "periodic"

    def n_modes(self):
        return sum(int(np.prod(b.shape)) for b in self.blocks)

    def slab(self, grid: np.ndarray, block: Block) -> np.ndarray:
        return grid[:, :, block.zindex]

    def assemble(self, slabs: list[np.ndarray], dtype=np.complex128) -> np.ndarray:
        out = np.zeros(self.lat.shape, dtype=dtype)
        for b, s in zip(self.blocks, slabs):
            out[:, :, b.zindex] = s
        return out

    def plate_mask(self) -> np.ndarray:
        m = np.zeros(self.lat.shape, dtype=bool)
        if len(self.plate_layers):
            m[:, :, self.plate_layers] = True
        return m


class PeriodicGeometry(Geometry):
    def __init__(self, lat: Lattice):
        super().__init__(lat)
        self.blocks = [PeriodicBlock(lat)]


class PlateGeometry(Geometry):
    """Two Dirichlet plates at z-layers z1 < z2 (0-based).  Inside chain: z1+1 .. z2-1.
    Outside chain (periodic wrap): z2+1 .. N-1, 0 .. z1-1."""
    def __init__(self, lat: Lattice, z1: int, z2: int):
        super().__init__(lat)
        N = lat.N
        z1, z2 = int(z1) % N, int(z2) % N
        if z1 > z2:
            z1, z2 = z2, z1
        if z2 - z1 < 2 or (N - (z2 - z1)) < 2:
            raise ValueError("plates need at least one free layer on each side")
        self.z1, self.z2 = z1, z2
        inside = np.arange(z1 + 1, z2)
        outside = np.concatenate([np.arange(z2 + 1, N), np.arange(0, z1)])
        self.blocks = [DirichletBlock(lat, inside), DirichletBlock(lat, outside)]
        self.plate_layers = np.array([z1, z2])

    @property
    def kind(self):
        return "plates"

    @property
    def separation(self):
        """Plate separation d (continuum-equivalent): (n_inside + 1) a."""
        return self.blocks[0].length

    @property
    def outside_length(self):
        return self.blocks[1].length


# ----------------------------------------------------------------------------------------------
def z_mixing_matrix(new_block: Block, old_block: Block) -> np.ndarray:
    """M[j, i] = sum_z S_new[j, z] * conj(S_old[i, z]) over the z-layers common to both blocks.
    Used to transfer the (mode-diagonal) covariance of a Gaussian state from one basis to another:
        <|Q_new_j|^2> = sum_i |M_ji|^2 <|Q_old_i|^2>     (exact if old modes are uncorrelated)
    The x,y part is the same FFT in both bases and drops out."""
    Snew = new_block.z_basis()           # (n_new, n_new_layers) over new_block.zindex
    Sold = old_block.z_basis()           # (n_old, n_old_layers) over old_block.zindex
    # map layers: which columns of Sold correspond to new_block.zindex
    pos_old = {int(z): i for i, z in enumerate(old_block.zindex)}
    cols = [pos_old.get(int(z), -1) for z in new_block.zindex]
    keep = [c for c in cols if c >= 0]
    rows_new = [r for r, c in enumerate(cols) if c >= 0]
    if not keep:
        return np.zeros((Snew.shape[0], Sold.shape[0]))
    return Snew[:, rows_new] @ np.conj(Sold[:, keep]).T
