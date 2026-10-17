import unittest
from dataclasses import replace

import numpy as np

from solsbaby.experiment import (
    Protocol, RidgeReader, generate_batch, expose, measure, run_benchmark, walkthrough,
)
from solsbaby.model import Medium, Parameters
from solsbaby.morphology import Genome


class ProtocolTests(unittest.TestCase):
    def test_episode_splits_are_balanced_reproducible_and_independent(self):
        p = Protocol(train_per_class=3, test_per_class=3)
        a, b, c = generate_batch(p, 7, 'train'), generate_batch(p, 7, 'train'), generate_batch(p, 7, 'test')
        np.testing.assert_array_equal(a.labels, b.labels)
        np.testing.assert_array_equal(a.amplitudes, b.amplitudes)
        np.testing.assert_array_equal(np.bincount(a.labels), [3]*4)
        self.assertFalse(np.array_equal(a.amplitudes, c.amplitudes))
        self.assertFalse(np.array_equal(a.labels, c.labels) and np.array_equal(a.jitter, c.jitter))

    def test_reader_does_not_refit_preprocessing_on_test_data(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(100, 4))
        labels = (x[:, 0] > 0).astype(int)
        reader = RidgeReader(.1).fit(x, labels)
        mean, scale = reader.mean.copy(), reader.scale.copy()
        self.assertGreater(np.mean(reader.predict(x) == labels), .9)
        reader.predict(np.full((4, 4), 1000.))
        np.testing.assert_array_equal(reader.mean, mean)
        np.testing.assert_array_equal(reader.scale, scale)

    def test_probe_copies_state_and_writing_is_disabled(self):
        p = Protocol(train_per_class=1, test_per_class=1)
        medium = Medium(Genome().grow())
        state = expose(medium, generate_batch(p, 3, 'train'), p)
        before = state.copy()
        measured = measure(medium, state, p)
        np.testing.assert_array_equal(state.tag, before.tag)
        np.testing.assert_array_equal(state.v, before.v)
        expected = before.tag*np.exp(-p.branches*p.slot_steps*medium.parameters.dt/medium.parameters.tag_tau)
        np.testing.assert_allclose(measured['tag_after'], expected, atol=1e-14)
        self.assertEqual(measured['trace'].shape, (4, 240))

    def test_erasing_memory_removes_context_from_clean_probe(self):
        p = Protocol(train_per_class=1, test_per_class=1)
        medium = Medium(Genome().grow())
        state = expose(medium, generate_batch(p, 3, 'train'), p)
        learned = measure(medium, state, p)['trace']
        erased = measure(medium, state, p, erase=True)['trace']
        self.assertGreater(np.linalg.norm(learned-erased), .001)
        np.testing.assert_allclose(erased, np.broadcast_to(erased[0], erased.shape), atol=1e-14)
        frozen = Medium(medium.morphology, Parameters(tag_rate=0))
        frozen_state = expose(frozen, generate_batch(p, 3, 'train'), p)
        np.testing.assert_array_equal(frozen_state.tag, 0)

    def test_bad_protocols_and_readers_fail_early(self):
        for kwargs in ({'train_per_class': 0}, {'noise_std': -1}, {'seeds': ()},
                       {'slot_steps': 0}, {'sample_stride': 0}, {'noise_std': np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Protocol(**kwargs).validate()
        with self.assertRaises(ValueError):
            generate_batch(Protocol(), 3, 'invalid')
        with self.assertRaises(ValueError):
            RidgeReader(1).fit(np.ones((2, 3)), np.array([0]))
        with self.assertRaises(ValueError):
            RidgeReader(1).predict(np.ones((2, 3)))

    def test_small_benchmark_reports_controls_and_counts(self):
        result = run_benchmark(Protocol(seeds=(3,), train_per_class=2, test_per_class=2))
        trial = result['trials'][0]
        for condition in ('learned_soma', 'frozen_writer', 'erased_memory', 'shuffled_labels', 'slow_state_oracle'):
            self.assertIn(condition, trial['accuracy'])
            self.assertEqual(len(trial['predictions'][condition]), 8)
            self.assertTrue(np.isfinite(trial['accuracy'][condition]))
        self.assertEqual(result['budget']['persistent_tag_bytes'], 21*8)
        self.assertEqual(result['budget']['somatic_samples_per_read'], 80)
        self.assertIn(result['gate']['status'], ('PASS', 'FAIL'))

    def test_walkthrough_uses_real_exposure_and_same_probe(self):
        d = walkthrough(seed=3)
        self.assertGreater(d['state'].tag.max(), .2)
        self.assertEqual(d['learned']['trace'].shape, d['erased']['trace'].shape)
        self.assertEqual(d['learned']['frames'].shape[1], d['medium'].morphology.n)


if __name__ == '__main__':
    unittest.main()
