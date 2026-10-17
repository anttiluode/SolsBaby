"""A compact developmental recipe and its generated electrical tree.

All quantities are dimensionless. Drawing coordinates are deliberately absent
from the equations: cable lengths/diameters, not visual pose, set the couplings.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class Morphology:
    parent: np.ndarray
    axial: np.ndarray
    capacitance: np.ndarray
    leak: np.ndarray
    length: np.ndarray
    diameter: np.ndarray
    xy: np.ndarray
    tips: np.ndarray
    branch: np.ndarray
    depth: np.ndarray

    @property
    def n(self):
        return len(self.parent)

    @property
    def laplacian(self):
        out = np.zeros((self.n, self.n))
        for i in range(1, self.n):
            p, g = self.parent[i], self.axial[i]
            out[i, i] += g
            out[p, p] += g
            out[i, p] -= g
            out[p, i] -= g
        return out


@dataclass(frozen=True)
class Genome:
    """Illustrative growth rules, not a biological genome reconstruction."""
    seed: int = 3
    branches: int = 4
    segments: int = 5
    length_scale: float = 1.0
    diameter: float = .9
    taper: float = .86
    variation: float = .12

    def grow(self):
        for name, value, lower, upper in (
            ('seed', self.seed, 0, 2**32-1),
            ('branches', self.branches, 2, 8),
            ('segments', self.segments, 1, 16),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or not lower <= value <= upper:
                raise ValueError(f'{name} must be an integer in [{lower}, {upper}]')
        for name in ('length_scale', 'diameter', 'taper', 'variation'):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if self.length_scale <= 0 or self.diameter <= 0 or not 0 < self.taper <= 1:
            raise ValueError('length/diameter must be positive and taper in (0,1]')
        if not 0 <= self.variation <= .5:
            raise ValueError('variation must be in [0, .5]')
        rng = np.random.default_rng(self.seed)
        n = 1 + self.branches*self.segments
        parent = np.full(n, -1, dtype=int)
        branch = np.full(n, -1, dtype=int)
        depth = np.zeros(n, dtype=int)
        length, diameter = np.ones(n), np.ones(n)
        xy = np.zeros((n, 2))
        tips = []
        for b in range(self.branches):
            angle = 2*np.pi*b/self.branches + rng.uniform(-.12, .12)
            p = 0
            for k in range(self.segments):
                i = 1+b*self.segments+k
                parent[i], branch[i], depth[i] = p, b, k+1
                length[i] = .65*self.length_scale*rng.uniform(1-self.variation, 1+self.variation)
                diameter[i] = self.diameter*self.taper**k*rng.uniform(1-self.variation, 1+self.variation)
                angle += rng.uniform(-.16, .16)
                xy[i] = xy[p]+length[i]*np.array([np.cos(angle), np.sin(angle)])
                p = i
            tips.append(p)
        capacitance = .7*diameter*length
        capacitance[0] = 2.5
        leak = .07*capacitance
        axial = 1.4*diameter**2/length
        axial[0] = 0
        arrays = (parent, axial, capacitance, leak, length, diameter,
                  xy, np.array(tips, dtype=int), branch, depth)
        for a in arrays:
            a.setflags(write=False)
        return Morphology(*arrays)
