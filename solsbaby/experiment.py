"""Write/wait/clear/probe protocol and training-only somatic readers."""
from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral
import time

import numpy as np

from .model import Medium, Parameters, State
from .morphology import Genome


@dataclass(frozen=True)
class Protocol:
    name: str = 'SolsBaby v1 — somatic read of a learned shunt'
    seeds: tuple = (3, 11, 29, 47)
    branches: int = 4
    segments: int = 5
    train_per_class: int = 32
    test_per_class: int = 32
    write_steps: int = 120
    wait_steps: int = 200
    slot_steps: int = 60
    sample_stride: int = 3
    context_amplitude: float = 3.
    probe_amplitude: float = .35
    noise_std: float = .0002
    ridge: float = 1.
    active_strength: float = 0.
    gate_mean_accuracy: float = .80
    gate_min_seed_accuracy: float = .65
    gate_max_control_accuracy: float = .37

    def validate(self):
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError('seeds must be a nonempty sequence of distinct integers')
        for seed in self.seeds:
            Genome(seed=seed, branches=self.branches, segments=self.segments).grow()
        for name in ('train_per_class', 'test_per_class', 'write_steps', 'slot_steps', 'sample_stride', 'wait_steps'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < (0 if name == 'wait_steps' else 1):
                raise ValueError(f'{name} must be a positive integer (wait_steps may be zero)')
        if self.write_steps < 100 or self.slot_steps < 20 or self.sample_stride > self.branches*self.slot_steps:
            raise ValueError('write_steps >=100, slot_steps >=20 and sample_stride <= probe duration are required')
        for name in ('context_amplitude', 'probe_amplitude', 'noise_std', 'ridge', 'active_strength'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name in ('ridge', 'context_amplitude', 'probe_amplitude') and value == 0):
                raise ValueError(f'{name} must be finite and nonnegative (amplitudes/ridge strictly positive)')
        for name in ('gate_mean_accuracy', 'gate_min_seed_accuracy', 'gate_max_control_accuracy'):
            if not np.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0,1]')
        return self

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class EpisodeBatch:
    labels: np.ndarray
    amplitudes: np.ndarray
    jitter: np.ndarray


def generate_batch(protocol, seed, split):
    protocol.validate()
    if split not in ('train', 'test'):
        raise ValueError("split must be 'train' or 'test'")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xEA11, int(split == 'test')]))
    count = protocol.train_per_class if split == 'train' else protocol.test_per_class
    labels = rng.permutation(np.repeat(np.arange(protocol.branches), count))
    return EpisodeBatch(labels, rng.uniform(.8, 1.2, len(labels)), rng.integers(0, 6, len(labels)))


def expose(medium, batch, protocol, frames=None, initial_state=None):
    """Local writing during three pulses; decay-only washout; clear voltage.

The labels choose where an external experience arrives. They are never passed
to Medium.step or included in state. Optional frames record episode zero.
"""
    m = medium.morphology
    state = medium.initial(batch=len(batch.labels)) if initial_state is None else initial_state.copy()
    if state.v.shape != (len(batch.labels), m.n):
        raise ValueError('initial state must match the exposure batch')
    indices = np.arange(len(batch.labels))
    tips = m.tips[batch.labels]
    for t in range(protocol.write_steps):
        g = np.zeros_like(state.v)
        on = np.zeros(len(batch.labels), dtype=bool)
        for start in (10, 45, 80):
            on |= (t >= start+batch.jitter) & (t < start+batch.jitter+10)
        g[indices[on], tips[on]] = protocol.context_amplitude*batch.amplitudes[on]
        medium.step(state, g, write=True)
        if frames is not None:
            frames.append(state.v[0].copy())
    for _ in range(protocol.wait_steps):
        medium.step(state, write=False)
        if frames is not None:
            frames.append(state.v[0].copy())
    state.clear_fast()
    return state


def measure(medium, state, protocol, erase=False, retain_frames=False):
    """Identical branch-sweep probe minus a matched no-probe trajectory.

Measurement acts on copies, clears fast voltage, and freezes local writing.
Slow decay continues in both trajectories. Active and shunting dynamics are
preserved; only their response can reach the limited reader.
"""
    probe = state.copy()
    probe.clear_fast()
    if erase:
        probe.erase_memory()
    blank = probe.copy()
    steps = protocol.branches*protocol.slot_steps
    trace = np.empty((len(probe.v), steps))
    baseline = np.empty_like(trace)
    frames = []
    for t in range(steps):
        branch, local_t = divmod(t, protocol.slot_steps)
        g = np.zeros(medium.morphology.n)
        if 5 <= local_t < 15:
            g[medium.morphology.tips[branch]] = protocol.probe_amplitude
        medium.step(probe, g, write=False)
        medium.step(blank, write=False)
        trace[:, t] = probe.v[:, 0]-blank.v[:, 0]
        baseline[:, t] = blank.v[:, 0]
        if retain_frames:
            frames.append(probe.v[0].copy())
    return {'trace': trace, 'baseline': baseline, 'tag_after': probe.tag.copy(),
            'frames': np.asarray(frames),
            'time': medium.parameters.dt*np.arange(1, steps+1)}


class RidgeReader:
    """Linear classifier; centering/scaling are fitted on training samples only."""
    def __init__(self, ridge=1.):
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError('ridge must be finite and positive')
        self.ridge = ridge
        self.mean = None

    def fit(self, x, labels):
        x, labels = np.asarray(x, float), np.asarray(labels)
        if x.ndim != 2 or labels.shape != (len(x),) or not len(x) or np.any(~np.isfinite(x)):
            raise ValueError('reader requires finite (episodes, features) and one label per episode')
        self.classes, indices = np.unique(labels, return_inverse=True)
        if len(self.classes) < 2:
            raise ValueError('reader training requires at least two classes')
        self.mean, self.scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8)
        z = (x-self.mean)/self.scale
        targets = np.eye(len(self.classes))[indices]
        self.prior = targets.mean(axis=0)
        self.weights = np.linalg.solve(z.T@z+self.ridge*np.eye(x.shape[1]),
                                       z.T@(targets-self.prior))
        return self

    def predict(self, x):
        x = np.asarray(x, float)
        if self.mean is None:
            raise ValueError('fit the reader before prediction')
        if x.ndim != 2 or x.shape[1] != len(self.mean) or np.any(~np.isfinite(x)):
            raise ValueError('prediction features must be finite and match training width')
        score = ((x-self.mean)/self.scale)@self.weights+self.prior
        return self.classes[np.argmax(score, axis=1)]


def _score(train, test, train_labels, test_labels, protocol):
    prediction = RidgeReader(protocol.ridge).fit(train, train_labels).predict(test)
    confusion = np.zeros((protocol.branches, protocol.branches), dtype=int)
    np.add.at(confusion, (test_labels, prediction), 1)
    return float(np.mean(prediction == test_labels)), prediction.tolist(), confusion.tolist()


def run_benchmark(protocol=None, progress=None):
    protocol = (protocol or Protocol()).validate()
    started = time.perf_counter()
    trials = []
    for seed in protocol.seeds:
        start = time.perf_counter()
        morphology = Genome(seed=seed, branches=protocol.branches, segments=protocol.segments).grow()
        medium = Medium(morphology, Parameters(active_strength=protocol.active_strength))
        frozen = Medium(morphology, Parameters(tag_rate=0, active_strength=protocol.active_strength))
        batches = [generate_batch(protocol, seed, split) for split in ('train', 'test')]
        states = [expose(medium, b, protocol) for b in batches]
        frozen_states = [expose(frozen, b, protocol) for b in batches]
        clean = [measure(medium, s, protocol)['trace'] for s in states]
        erased = [measure(medium, s, protocol, erase=True)['trace'] for s in states]
        frozen_traces = [measure(frozen, s, protocol)['trace'] for s in frozen_states]
        conditions = {'learned_soma': clean, 'frozen_writer': frozen_traces, 'erased_memory': erased}
        accuracy, predictions, confusion = {}, {}, {}
        observed = None
        for index, (name, traces) in enumerate(conditions.items()):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 0x1234, index]))
            features = [a[:, ::protocol.sample_stride]+rng.normal(0, protocol.noise_std, a[:, ::protocol.sample_stride].shape)
                        for a in traces]
            accuracy[name], predictions[name], confusion[name] = _score(*features, batches[0].labels, batches[1].labels, protocol)
            if name == 'learned_soma':
                observed = features
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5678]))
        accuracy['shuffled_labels'], predictions['shuffled_labels'], confusion['shuffled_labels'] = _score(
            *observed, rng.permutation(batches[0].labels), batches[1].labels, protocol)
        accuracy['slow_state_oracle'], predictions['slow_state_oracle'], confusion['slow_state_oracle'] = _score(
            states[0].tag, states[1].tag, batches[0].labels, batches[1].labels, protocol)
        difference = np.linalg.norm(clean[1]-erased[1], axis=1)
        reference = np.maximum(np.linalg.norm(erased[1], axis=1), 1e-15)
        trials.append({'seed': int(seed), 'accuracy': accuracy, 'predictions': predictions,
                       'confusion': confusion, 'test_labels': batches[1].labels.tolist(),
                       'relative_response_change_mean': float(np.mean(difference/reference)),
                       'clean_soma_peak': float(np.max(np.abs(clean[1]))),
                       'memory_effect_rms': float(np.sqrt(np.mean((clean[1]-erased[1])**2))),
                       'slow_tag_max': float(states[1].tag.max()),
                       'fast_voltage_before_probe_max': float(np.max(np.abs(states[1].v))),
                       'seconds': time.perf_counter()-start})
        if progress:
            progress(trials[-1])
    names = trials[0]['accuracy']
    summary = {name: {'mean': float(np.mean([t['accuracy'][name] for t in trials])),
                      'min': float(np.min([t['accuracy'][name] for t in trials])),
                      'max': float(np.max([t['accuracy'][name] for t in trials]))}
               for name in names}
    checks = {
        'learned_mean': summary['learned_soma']['mean'] >= protocol.gate_mean_accuracy,
        'learned_min_seed': summary['learned_soma']['min'] >= protocol.gate_min_seed_accuracy,
        'frozen_control': summary['frozen_writer']['mean'] <= protocol.gate_max_control_accuracy,
        'erased_control': summary['erased_memory']['mean'] <= protocol.gate_max_control_accuracy,
    }
    return {'protocol': asdict(protocol), 'protocol_sha256': protocol.fingerprint,
            'summary': summary, 'trials': trials,
            'gate': {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks},
            'budget': {'compartments': morphology.n, 'persistent_tag_bytes': morphology.n*8,
                       'fast_voltage_bytes': morphology.n*8, 'somatic_ports': 1,
                       'somatic_samples_per_read': clean[0][:, ::protocol.sample_stride].shape[1],
                       'probe_steps': protocol.branches*protocol.slot_steps,
                       'integration_steps_per_read_with_blank': 2*protocol.branches*protocol.slot_steps,
                       'class_label_information_bits': float(np.log2(protocol.branches))},
            'seconds': time.perf_counter()-started}


def walkthrough(seed=3, branch=0, active=False):
    protocol = Protocol(seeds=(seed,), train_per_class=1, test_per_class=1,
                        active_strength=1.2 if active else 0.)
    protocol.validate()
    if not isinstance(branch, Integral) or not 0 <= branch < protocol.branches:
        raise ValueError('branch must be an integer in [0,3]')
    medium = Medium(Genome(seed=seed).grow(), Parameters(active_strength=protocol.active_strength))
    batch = EpisodeBatch(np.array([branch]), np.ones(1), np.zeros(1, dtype=int))
    frames = []
    state = expose(medium, batch, protocol, frames)
    return {'medium': medium, 'state': state, 'protocol': protocol, 'branch': branch,
            'experience_frames': np.asarray(frames),
            'learned': measure(medium, state, protocol, retain_frames=True),
            'erased': measure(medium, state, protocol, erase=True, retain_frames=True)}
