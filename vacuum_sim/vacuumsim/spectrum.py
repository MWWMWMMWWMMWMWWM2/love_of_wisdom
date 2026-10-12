"""
spectrum.py -- the string mass tower ("which fields exist, with which masses and how many").

Source: Polchinski, String Theory Vol.1 sec. 1.3-1.4 (open/closed bosonic spectrum, the zero-point
constant a = -1 that produces the tachyon and the massless level), Vol.2 ch.10 (GSO projection ->
tachyon-free superstring spectrum; equal numbers of bosons and fermions at every level).

Mass formulae (string units, alpha' = 1):
    open  bosonic  : m^2 = (N - 1)          N = 0 tachyon, N = 1 massless (photon-like)
    closed bosonic : m^2 = 4 (N - 1)        N = 0 tachyon, N = 1 massless (graviton, B, dilaton)
    open  superstr.: m^2 = N                N = 0 massless
    closed type II : m^2 = 4 N              N = 0 massless
Degeneracies (number of physical states at level N) come from the light-cone partition functions:
    bosonic open        : prod_n (1 - q^n)^(-24)            -> 1, 24, 324, 3200, 25650, ...
    open superstring    : 8 prod_n ((1+q^n)/(1-q^n))^8      -> 8, 128, 1152, 7680, 42112, ... (bosons; fermions equal)
    closed = (left) x (right):   d_closed(N) = d_open(N)^2 (bosonic);  type II: bosons = fermions = 2 d(N)^2.
The bosonic-string tachyon is flagged and NEVER sampled: it is an instability of an inconsistent
toy theory (Polchinski Vol.1 p.219: "an artifact of a theory with a tachyon").

MODIFY: to add a new "theory" supply a function returning (mass^2(N), bosons(N), fermions(N)).
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np


@dataclass(frozen=True)
class Level:
    theory: str
    N: int
    mass2: float          # in units 1/alpha'
    bosons: int           # number of physical bosonic states
    fermions: int         # number of physical fermionic states
    tachyonic: bool

    @property
    def mass(self) -> float:
        return float(np.sqrt(self.mass2)) if self.mass2 >= 0 else float("nan")


def _series(prefactor: int, numer_pow: int, denom_pow: int, nmax: int) -> np.ndarray:
    """Coefficients of  prefactor * prod_{n>=1} (1+q^n)^numer_pow / (1-q^n)^denom_pow  up to q^nmax.
    Uses exact integer arithmetic (Python ints) because the numbers grow like exp(c sqrt(N))."""
    c = [0] * (nmax + 1)
    c[0] = 1
    for n in range(1, nmax + 1):
        for _ in range(numer_pow):            # multiply by (1 + q^n)
            for i in range(nmax, n - 1, -1):
                c[i] += c[i - n]
        for _ in range(denom_pow):            # multiply by 1/(1 - q^n) = sum_k q^{nk}
            for i in range(n, nmax + 1):
                c[i] += c[i - n]
    return np.array([prefactor * x for x in c], dtype=object)


@lru_cache(maxsize=None)
def open_bosonic_degeneracies(nmax: int):
    return _series(1, 0, 24, nmax)


@lru_cache(maxsize=None)
def open_superstring_degeneracies(nmax: int):
    return _series(8, 8, 8, nmax)


THEORIES = {
    "closed type II superstring (D=10)": "closed_typeII",
    "open superstring (D=10)": "open_superstring",
    "closed bosonic string (D=26)": "closed_bosonic",
    "open bosonic string (D=26)": "open_bosonic",
}


def tower(theory: str, n_levels: int = 6) -> list[Level]:
    """Return the first n_levels of the chosen theory."""
    key = THEORIES.get(theory, theory)
    levels = []
    if key == "open_bosonic":
        d = open_bosonic_degeneracies(n_levels)
        for N in range(n_levels):
            levels.append(Level(key, N, float(N - 1), int(d[N]), 0, N == 0))
    elif key == "closed_bosonic":
        d = open_bosonic_degeneracies(n_levels)
        for N in range(n_levels):
            levels.append(Level(key, N, 4.0 * (N - 1), int(d[N] ** 2), 0, N == 0))
    elif key == "open_superstring":
        d = open_superstring_degeneracies(n_levels)
        for N in range(n_levels):
            levels.append(Level(key, N, float(N), int(d[N]), int(d[N]), False))
    elif key == "closed_typeII":
        d = open_superstring_degeneracies(n_levels)
        for N in range(n_levels):
            # (8_v + 8_s) x (8_v + 8_s): bosons = vv + ss = 2 d^2, fermions = vs + sv = 2 d^2
            levels.append(Level(key, N, 4.0 * N, int(2 * d[N] ** 2), int(2 * d[N] ** 2), False))
    else:
        raise ValueError(f"unknown theory {theory!r}; choose from {list(THEORIES)}")
    return levels


def hagedorn_note(theory: str) -> str:
    """Growth of the level degeneracy: d(N) ~ exp(c sqrt(N)). This exponential growth is why a
    naive 'sum of zero-point energies over the whole tower' (Vol.1 eq. 7.3.12) diverges even worse
    than in field theory, and why string theory needs the modular-invariant torus integral
    (Vol.1 sec. 7.3, pp. 219-220) instead."""
    key = THEORIES.get(theory, theory)
    if "bosonic" in key:
        return "d(N) ~ N^(-27/4) exp(4 pi sqrt(N))  (open bosonic; Hagedorn growth)"
    return "d(N) ~ N^(-11/4) exp(2 pi sqrt(2N))  (open superstring; Hagedorn growth)"
