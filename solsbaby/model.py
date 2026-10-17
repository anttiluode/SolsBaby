"""Conductance cable, Hines tree solve, and a declared local memory rule.

The voltage-dependent active term is an illustrative NMDA-like boost. The
slow tag is an engineered activity-dependent shunt, not a biological finding.
"""
from dataclasses import dataclass, fields

import numpy as np

from .morphology import Morphology


@dataclass(frozen=True)
class Parameters:
    dt: float = .2
    reversal: float = 1.
    memory_strength: float = 1.2
    tag_rate: float = .35
    tag_threshold: float = .25
    tag_tau: float = 2000.
    active_strength: float = 0.
    active_threshold: float = .28
    active_width: float = .035

    def validate(self):
        if not all(np.isfinite(getattr(self, f.name)) for f in fields(self)):
            raise ValueError('all model parameters must be finite')
        if not 0 < self.dt <= 1:
            raise ValueError('dt must be in (0, 1]')
        if self.tag_tau <= 0 or self.active_width <= 0 or self.reversal <= 0:
            raise ValueError('time constant, active width and reversal must be positive')
        if min(self.memory_strength, self.tag_rate, self.active_strength, self.tag_threshold) < 0:
            raise ValueError('conductance strengths, tag rate and threshold must be nonnegative')


@dataclass
class State:
    v: np.ndarray
    tag: np.ndarray

    def copy(self):
        return State(self.v.copy(), self.tag.copy())

    def clear_fast(self):
        self.v.fill(0)

    def erase_memory(self):
        self.tag.fill(0)


def tree_solve(parent, axial, diagonal, rhs):
    """Solve a tree system with off-diagonals -axial[i], O(N) per batch.

Parents must precede children. Arbitrary leading batch axes are supported,
including complex diagonals for frequency response. Inputs are not mutated.
"""
    d, b = np.broadcast_arrays(diagonal, rhs)
    dtype = np.result_type(d.dtype, b.dtype, np.float64)
    d, b = d.astype(dtype, copy=True), b.astype(dtype, copy=True)
    n = len(parent)
    if d.shape[-1] != n or len(axial) != n:
        raise ValueError('tree system size mismatch')
    for i in range(n-1, 0, -1):
        p, g = parent[i], axial[i]
        factor = g/d[..., i]
        d[..., p] -= g*factor
        b[..., p] += factor*b[..., i]
    x = np.empty_like(b)
    x[..., 0] = b[..., 0]/d[..., 0]
    for i in range(1, n):
        x[..., i] = (b[..., i]+axial[i]*x[..., parent[i]])/d[..., i]
    return x


class Medium:
    def __init__(self, morphology: Morphology, parameters=None):
        self.morphology = morphology
        self.parameters = parameters or Parameters()
        self.parameters.validate()
        m = morphology
        if m.n < 2 or m.parent[0] != -1 or np.any(m.parent[1:] < 0) or np.any(m.parent[1:] >= np.arange(1, m.n)):
            raise ValueError('morphology must be a topologically ordered rooted tree')
        for a in (m.capacitance, m.leak, m.axial[1:]):
            if np.any(~np.isfinite(a)) or np.any(a <= 0):
                raise ValueError('electrical parameters must be finite and positive')
        self._axial_diagonal = np.diag(m.laplacian).copy()
        self._zero = np.zeros(m.n)

    def initial(self, batch=1):
        if isinstance(batch, bool) or not isinstance(batch, (int, np.integer)) or batch < 1:
            raise ValueError('batch must be a positive integer')
        shape = (batch, self.morphology.n)
        return State(np.zeros(shape), np.zeros(shape))

    def step(self, state, excitation=None, current=None, write=True):
        """Advance in place; excitation is a nonnegative conductance.

Optional current injection allows a genuinely linear passive control.
Writing is local to each compartment; no history/class labels enter here.
Even with writing disabled, slow memory undergoes its declared decay.
"""
        p, m = self.parameters, self.morphology
        g = self._zero if excitation is None else np.asarray(excitation, dtype=float)
        current = self._zero if current is None else np.asarray(current, dtype=float)
        if state.v.shape != state.tag.shape or state.v.ndim != 2 or state.v.shape[1] != m.n:
            raise ValueError('state must have matching (batch, compartments) arrays')
        if g.shape not in ((m.n,), state.v.shape) or current.shape not in ((m.n,), state.v.shape):
            raise ValueError('input shape must be (compartments,) or match state')
        if np.any(~np.isfinite(g)) or np.any(g < 0) or np.any(~np.isfinite(current)):
            raise ValueError('excitation must be finite/nonnegative; current must be finite')
        if np.any(~np.isfinite(state.v)) or np.any(~np.isfinite(state.tag)) or np.any((state.tag < 0) | (state.tag > 1)):
            raise ValueError('state must be finite and slow tags in [0, 1]')
        active = 0.
        if p.active_strength:
            logistic = lambda v: 1/(1+np.exp(np.clip(-(v-p.active_threshold)/p.active_width, -700, 700)))
            rest = logistic(0.)
            activation = np.maximum(0., (logistic(state.v)-rest)/(1-rest))
            active = p.active_strength*m.capacitance*activation*(m.depth >= 2)
        memory_g = p.memory_strength*m.capacitance*state.tag
        diagonal = m.capacitance/p.dt + self._axial_diagonal + m.leak + memory_g + g + active
        rhs = m.capacitance/p.dt*state.v + (g+active)*p.reversal + current
        state.v[:] = tree_solve(m.parent, m.axial, diagonal, rhs)
        rate = p.tag_rate*np.maximum(state.v-p.tag_threshold, 0.) if write else np.zeros_like(state.v)
        total = rate+1/p.tag_tau
        decay = np.exp(-p.dt*total)
        state.tag[:] = state.tag*decay + (rate/total)*(-np.expm1(-p.dt*total))
        return state

    def transfer(self, tag, frequencies):
        """Frozen PASSIVE tip-current -> soma-voltage transfer H(omega).

Active conductance is deliberately excluded. This diagnoses the learned
passive EQ; it is not the full nonlinear system's response function.
Frequencies are angular frequencies in inverse dimensionless time units.
"""
        m, p = self.morphology, self.parameters
        tag, omega = np.asarray(tag, float), np.atleast_1d(np.asarray(frequencies, float))
        if tag.shape != (m.n,) or np.any(~np.isfinite(tag)) or np.any((tag < 0) | (tag > 1)):
            raise ValueError('transfer requires one finite slow-state vector in [0,1]')
        if omega.ndim != 1 or np.any(~np.isfinite(omega)) or np.any(omega < 0):
            raise ValueError('frequencies must be a finite nonnegative vector')
        diagonal = (self._axial_diagonal+m.leak+p.memory_strength*m.capacitance*tag)[None, None, :] + 1j*omega[:, None, None]*m.capacitance
        rhs = np.zeros((len(omega), len(m.tips), m.n))
        rhs[:, np.arange(len(m.tips)), m.tips] = 1
        return tree_solve(m.parent, m.axial, diagonal, rhs)[..., 0]
