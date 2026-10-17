import unittest
from dataclasses import replace

import numpy as np

from solsbaby.model import Medium, Parameters, tree_solve
from solsbaby.morphology import Genome


class MorphologyTests(unittest.TestCase):
    def test_recipe_is_reproducible_and_electrically_positive(self):
        a, b = Genome(seed=7).grow(), Genome(seed=7).grow()
        np.testing.assert_array_equal(a.parent, b.parent)
        np.testing.assert_array_equal(a.axial, b.axial)
        self.assertEqual(a.n, 21)
        self.assertEqual(a.parent[0], -1)
        self.assertTrue(np.all(a.parent[1:] < np.arange(1, a.n)))
        for field in (a.capacitance, a.leak, a.axial[1:], a.length, a.diameter):
            self.assertTrue(np.all(field > 0))
        self.assertFalse(np.allclose(a.axial, Genome(seed=8).grow().axial))

    def test_axial_coupling_conserves_current(self):
        m = Genome().grow()
        np.testing.assert_allclose(m.laplacian @ np.ones(m.n), 0, atol=1e-14)
        np.testing.assert_allclose(m.laplacian, m.laplacian.T)

    def test_bad_recipes_are_rejected(self):
        for kwargs in ({'branches': 0}, {'segments': 0}, {'seed': -1},
                       {'length_scale': np.nan}, {'diameter': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Genome(**kwargs).grow()


class MediumTests(unittest.TestCase):
    def setUp(self):
        self.morph = Genome(seed=4).grow()
        self.medium = Medium(self.morph)

    def test_tree_solve_matches_independent_dense_linear_algebra(self):
        m = self.morph
        rng = np.random.default_rng(9)
        diagonal = np.diag(m.laplacian)[None, :] + rng.uniform(.1, 2, (3, m.n))
        rhs = rng.normal(size=(3, m.n))
        got = tree_solve(m.parent, m.axial, diagonal, rhs)
        off = m.laplacian - np.diag(np.diag(m.laplacian))
        want = np.stack([np.linalg.solve(off + np.diag(d), b)
                         for d, b in zip(diagonal, rhs)])
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_implicit_step_solves_the_passive_cable_equation(self):
        m, p = self.morph, self.medium.parameters
        state = self.medium.initial()
        state.v[0] = np.linspace(0, 1, m.n)
        state.tag[0] = np.linspace(0, .7, m.n)
        old_v, old_p = state.v[0].copy(), state.tag[0].copy()
        g = np.zeros(m.n)
        g[m.tips[0]] = .4
        G = m.laplacian + np.diag(m.leak + p.memory_strength*m.capacitance*old_p + g)
        expected = np.linalg.solve(np.diag(m.capacitance/p.dt) + G,
                                   m.capacitance/p.dt*old_v + g*p.reversal)
        self.medium.step(state, g, write=False)
        np.testing.assert_allclose(state.v[0], expected, atol=1e-12)

    def test_unforced_passive_cable_loses_energy(self):
        state = self.medium.initial(batch=5)
        state.v[:] = np.random.default_rng(2).normal(size=state.v.shape)
        before = .5*np.sum(self.morph.capacitance*state.v**2, axis=-1)
        self.medium.step(state, write=False)
        after = .5*np.sum(self.morph.capacitance*state.v**2, axis=-1)
        self.assertTrue(np.all(after < before))

    def test_passive_incremental_response_ignores_previous_voltage(self):
        rng = np.random.default_rng(8)
        initial = self.medium.initial(batch=2)
        initial.v[0] = rng.normal(size=self.morph.n)
        initial.v[1] = rng.normal(size=self.morph.n)
        probe, blank = initial.copy(), initial.copy()
        current = np.zeros(self.morph.n)
        current[self.morph.tips[0]] = .3
        for _ in range(50):
            self.medium.step(probe, current=current, write=False)
            self.medium.step(blank, write=False)
        delta = probe.v - blank.v
        np.testing.assert_allclose(delta[0], delta[1], atol=2e-14)

    def test_local_writer_is_bounded_and_keeps_branch_information(self):
        state = self.medium.initial()
        g = np.zeros(self.morph.n)
        g[self.morph.tips[0]] = 3
        for _ in range(140):
            self.medium.step(state, g)
        self.assertGreater(state.tag[0, self.morph.tips[0]], .3)
        self.assertGreater(state.tag[0, self.morph.tips[0]],
                           np.max(state.tag[0, self.morph.tips[1:]]) + .2)
        self.assertTrue(np.all((state.tag >= 0) & (state.tag <= 1)))

    def test_frozen_writer_only_decays_and_fast_reset_preserves_memory(self):
        state = self.medium.initial()
        state.tag[:] = .6
        state.v[:] = .7
        before = state.tag.copy()
        state.clear_fast()
        np.testing.assert_array_equal(state.v, 0)
        np.testing.assert_array_equal(state.tag, before)
        g = np.full(self.morph.n, 4.)
        self.medium.step(state, g, write=False)
        expected = before*np.exp(-self.medium.parameters.dt/self.medium.parameters.tag_tau)
        np.testing.assert_allclose(state.tag, expected)
        state.erase_memory()
        np.testing.assert_array_equal(state.tag, 0)

    def test_state_copies_do_not_share_arrays(self):
        a = self.medium.initial()
        b = a.copy()
        b.tag[:] = 1
        b.v[:] = 2
        np.testing.assert_array_equal(a.tag, 0)
        np.testing.assert_array_equal(a.v, 0)

    def test_shunting_reduces_the_frozen_passive_transfer(self):
        tag = np.zeros(self.morph.n)
        original = self.medium.transfer(tag, np.array([0., 1., 10.]))
        tag[self.morph.branch == 0] = .8
        changed = self.medium.transfer(tag, np.array([0., 1., 10.]))
        self.assertLess(abs(changed[0, 0]), abs(original[0, 0]))
        self.assertTrue(np.all(abs(original[2]) < abs(original[0])))

    def test_electrical_geometry_matters_and_display_pose_does_not(self):
        frequencies = np.array([0., .1, 1.])
        zero = np.zeros(self.morph.n)
        a = self.medium.transfer(zero, frequencies)
        moved = Medium(replace(self.morph, xy=self.morph.xy*900 + 5))
        np.testing.assert_array_equal(a, moved.transfer(zero, frequencies))
        stretched = Medium(Genome(seed=4, length_scale=1.4).grow())
        self.assertGreater(np.linalg.norm(a-stretched.transfer(zero, frequencies)), .01)

    def test_active_conductance_changes_responses_without_voltage_blowup(self):
        active = Medium(self.morph, Parameters(active_strength=1.2))
        a, b = self.medium.initial(), active.initial()
        g = np.zeros(self.morph.n)
        g[self.morph.tips[0]] = 2
        for _ in range(60):
            self.medium.step(a, g, write=False)
            active.step(b, g, write=False)
        self.assertGreater(np.linalg.norm(a.v-b.v), .01)
        self.assertTrue(np.all((b.v >= 0) & (b.v <= 1)))

    def test_nonfinite_and_negative_inputs_fail_early(self):
        for kwargs in ({'dt': 0}, {'dt': float('nan')}, {'tag_tau': -2},
                       {'memory_strength': -1}, {'active_strength': float('inf')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Medium(self.morph, Parameters(**kwargs))
        for g in (np.full(self.morph.n, -1.), np.full(self.morph.n, np.nan)):
            with self.assertRaises(ValueError):
                self.medium.step(self.medium.initial(), g)

    def test_negative_active_threshold_is_rejected_before_it_can_make_nans(self):
        with self.assertRaisesRegex(ValueError, 'threshold'):
            Medium(self.morph, Parameters(active_strength=1.2, active_threshold=-1.3))

    def test_malformed_morphology_is_rejected_before_broadcast_or_indexing(self):
        bad = [replace(self.morph, capacitance=np.array([2.5])),
               replace(self.morph, parent=self.morph.parent.astype(float)),
               replace(self.morph, tips=self.morph.tips.astype(float)),
               replace(self.morph, xy=np.zeros((2, 3))),
               replace(self.morph, branch=np.full(self.morph.n, 99)),
               replace(self.morph, leak=np.ones((self.morph.n, 1)))]
        for case, morphology in enumerate(bad):
            with self.subTest(case=case), self.assertRaises(ValueError):
                Medium(morphology)


if __name__ == '__main__':
    unittest.main()
