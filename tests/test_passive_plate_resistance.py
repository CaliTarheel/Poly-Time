"""Passivity, semismooth derivatives, historical state and force-solve budgets."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import plate_balance as balance


def fixture(shape='negative', version=1, size=3):
    model = balance.Balance.__new__(balance.Balance)
    model.s = SimpleNamespace()  # no numerical accuracy policy: the historical force tolerance
    model.resistance_version = version
    model.size = size
    model.stiffness = np.eye(size)
    model.torque = np.zeros(size)
    model.hinge = np.zeros((0, size))
    model.hinge_coefficient = np.zeros(0)
    normal, tangent = np.eye(size)[:2]
    rows = (normal[None], tangent[None]) if shape == 'cone' else (normal[None],)
    model.elements = [dict(kind='megathrust' if shape == 'cone' else 'compression',
                          shape=shape, rows=rows, coefficient=np.array([5.]), scale=np.ones(1))]
    return model


class PassiveConstitutiveTests(unittest.TestCase):
    def test_missing_and_explicit_legacy_state_keep_historical_law(self):
        self.assertEqual(balance.resistance_version(SimpleNamespace()), 0)
        self.assertEqual(balance.resistance_version(SimpleNamespace(plate_resistance_version=0)), 0)
        old = fixture(version=0)
        absent = deepcopy(old)
        del absent.resistance_version
        x = np.array([.3, .1, 0.])
        for left, right in zip(old._evaluate(x, 1.)[:3], absent._evaluate(x, 1.)[:3]):
            np.testing.assert_array_equal(left, right)
        work, checks = old._resistance_work(x, 1.)
        self.assertLess(work['compression'], 0.)
        self.assertEqual(checks['negative_resisting_work_elements'], 1)

    def test_unknown_or_ambiguous_version_is_rejected(self):
        for version in (True, False, '1', 1., None, -1, 2):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'constitutive version'):
                balance.resistance_version(SimpleNamespace(plate_resistance_version=version))
        self.assertEqual(balance.resistance_version(SimpleNamespace(plate_resistance_version=np.int64(1))), 1)

    def test_compression_is_passive_over_many_rate_and_width_scales(self):
        values = np.r_[-np.logspace(-14, 8, 90), 0., np.logspace(-14, 8, 90)]
        for width in (1.e-6, .01, 2.):
            potential, gradient, hessian = balance._passive_negative_terms(values, width)
            self.assertTrue(np.all(potential >= 0.))
            self.assertTrue(np.all(gradient*values >= 0.))
            self.assertTrue(np.all(hessian >= 0.))
            np.testing.assert_array_equal(gradient[values >= 0.], 0.)
            np.testing.assert_array_equal(potential[values >= 0.], 0.)

    def test_compression_derivatives_in_elastic_yielded_and_open_branches(self):
        values = np.array([-4., -.8, -.2, .2, 4.])
        step = 1.e-6
        value, gradient, hessian = balance._passive_negative_terms(values, 1.)
        plus = balance._passive_negative_terms(values+step, 1.)
        minus = balance._passive_negative_terms(values-step, 1.)
        np.testing.assert_allclose((plus[0]-minus[0])/(2.*step), gradient, atol=2.e-10)
        np.testing.assert_allclose((plus[1]-minus[1])/(2.*step), hessian, atol=2.e-10)

    def test_compression_c1_joins_have_valid_zero_sided_curvature(self):
        for join in (-1., 0.):
            _, force, curvature = balance._passive_negative_terms(np.array([join]), 1.)
            self.assertEqual(curvature[0], 0.)
            for sign in (-1., 1.):
                near = balance._passive_negative_terms(np.array([join+sign*1.e-8]), 1.)
                self.assertLessEqual(abs(near[1][0]-force[0]), 1.1e-8)
            # The chosen curvature belongs to the generalized Hessian [0,1].
            self.assertLessEqual(curvature[0], 1.)
            self.assertGreaterEqual(curvature[0], 0.)

    def test_cone_derivatives_and_curvature_away_from_c1_join(self):
        model = fixture('cone')
        model.stiffness[:] = 0.
        for opening in (-5., -.1, .1, 1., 5.):
            for tangential in (-.3, 0., .3):
                x = np.array([opening, tangential, 0.])
                step = 1.e-5
                value, gradient, hessian, _, _ = model._evaluate(x, 1.)
                numeric_gradient = np.array([(model._evaluate(x+axis*step, 1.)[0]
                    -model._evaluate(x-axis*step, 1.)[0])/(2.*step) for axis in np.eye(3)])
                numeric_hessian = np.column_stack([(model._evaluate(x+axis*step, 1.)[1]
                    -model._evaluate(x-axis*step, 1.)[1])/(2.*step) for axis in np.eye(3)])
                np.testing.assert_allclose(gradient, numeric_gradient, atol=1.e-8, rtol=1.e-7)
                np.testing.assert_allclose(hessian, numeric_hessian, atol=1.e-8, rtol=1.e-7)
                self.assertGreaterEqual(np.linalg.eigvalsh(hessian).min(), -1.e-12)
                self.assertGreaterEqual(value, 0.)
                self.assertGreaterEqual(x@gradient, 0.)
                if opening > 0.:
                    self.assertEqual(gradient[0], 0.)
                    self.assertEqual(hessian[0, 0], 0.)

    def test_cone_zero_normal_join_is_c1_with_open_side_hessian(self):
        model = fixture('cone')
        model.stiffness[:] = 0.
        for tangential in (0., .5, -2.):
            x = np.array([0., tangential, 0.])
            _, gradient, hessian, _, _ = model._evaluate(x, 1.)
            self.assertEqual(gradient[0], 0.)
            self.assertEqual(hessian[0, 0], 0.)
            for sign in (-1., 1.):
                near = model._evaluate(x+np.array([sign*1.e-8, 0., 0.]), 1.)
                np.testing.assert_allclose(near[1], gradient, atol=5.1e-8, rtol=0.)
                self.assertGreaterEqual(np.linalg.eigvalsh(near[2]).min(), -1.e-12)

    def test_megathrust_has_no_hidden_offset_or_opening_work(self):
        model = fixture('cone')
        model.stiffness[:] = 0.
        for opening in (0., 1.e-14, .5, 5.):
            for width in (1.e-6, .1, 10.):
                x = np.array([opening, 0., 0.])
                potential, gradient, _, _, _ = model._evaluate(x, width)
                self.assertEqual(potential, 0.)
                np.testing.assert_array_equal(gradient, 0.)
                self.assertEqual(model._resistance_work(x, width)[0]['megathrust'], 0.)

    def test_tiny_cone_potential_is_resolved_without_subtractive_cancellation(self):
        model = fixture('cone')
        model.stiffness[:] = 0.
        potential = model._evaluate(np.array([-1.e-12, 1.e-12, 0.]), 1.)[0]
        self.assertGreater(potential, 0.)
        self.assertAlmostEqual(potential/5.e-24, 1., places=14)

    def test_all_resistance_work_matches_gradient_including_local_weld_and_hinge(self):
        model = fixture('cone')
        model.hinge = np.array([[1., 0., 0.]])
        model.hinge_coefficient = np.array([2.])
        model.elements.extend([
            dict(kind='compression', shape='negative', rows=(np.array([[1., 0., 0.]]),),
                 coefficient=np.array([3.]), scale=np.ones(1)),
            dict(kind='strike_slip', shape='abs', rows=(np.array([[0., 1., 0.]]),),
                 coefficient=np.array([4.]), scale=np.ones(1)),
            dict(kind='suture_weld', shape='arc_opening', rows=(np.array([[[1., 0., 0.], [0., 1., 0.]]]),),
                 coefficient=np.array([6.]), scale=np.ones(1), bounds=np.array([[0., .8]]))])
        for x in (np.array([-2., .3, .1]), np.array([2., -.5, -.1]), np.zeros(3)):
            _, gradient, _, potential, _ = model._evaluate(x, .5)
            work, checks = model._resistance_work(x, .5)
            self.assertTrue(all(value >= 0. for value in work.values()))
            self.assertEqual(checks['negative_resisting_work_elements'], 0)
            self.assertAlmostEqual(sum(work.values())+x@model.stiffness@x, x@gradient, places=12)
            if x[0] < 0.:
                self.assertAlmostEqual(work['hinge'], 2.*potential['hinge'])
                self.assertGreater(work['compression'], potential['compression'])


class PassiveForceSolveTests(unittest.TestCase):
    def test_single_normal_force_matches_piecewise_exact_solution(self):
        for driving in (-10., -.2, 2.):
            model = fixture()
            model.stiffness = np.diag([2., 3., 4.])
            model.torque[0] = driving
            model.solve()
            delta = balance.HUBER_CONTINUATION_KM_MYR[-1]*balance.KM_MYR_CM_YR
            expected = ((driving+5.)/2. if driving < -5.-2.*delta else
                        driving/(2.+5./delta) if driving < 0. else driving/2.)
            np.testing.assert_allclose(model.x, [expected, 0., 0.], atol=1.e-10, rtol=1.e-7)
            gradient = model._evaluate(model.x, delta)[1]
            self.assertLessEqual(model._relative_residual(gradient), balance.FORCE_RELATIVE_TOLERANCE)

    def test_subyield_two_plate_relative_creep_shrinks_with_smoothing(self):
        motions = []
        for continuation in ((100., 10., 1., .1), (100., 10., 1., .1, .01, .001)):
            model = fixture(size=6)
            normal = np.array([[-1., 0., 0., 1., 0., 0.]])
            model.elements[0]['rows'] = (normal,)
            model.elements[0]['coefficient'] = np.array([10.])
            model.torque = np.array([.2, 0., 0., -.2, 0., 0.])
            with patch.object(balance, 'HUBER_CONTINUATION_KM_MYR', continuation):
                model.solve()
            delta = continuation[-1]*balance.KM_MYR_CM_YR
            relative = float((normal@model.x)[0])
            self.assertAlmostEqual(relative/(-.4/(1.+20./delta)), 1., places=9)
            work, checks = model._resistance_work(model.x, delta)
            self.assertEqual(checks['negative_resisting_work_elements'], 0)
            self.assertAlmostEqual(model.torque@model.x, model.x@model.stiffness@model.x+sum(work.values()), places=12)
            motions.append(abs(relative))
        self.assertGreater(motions[0]/motions[1], 99.)

    def test_native_world_force_and_checkpoint_preserve_passive_version(self):
        import checkpoint
        import native_engine
        world = native_engine.Simulation(dict(width=48, height=24, mesh_level=2,
            coast_geometry_level=2, plate_count=4, mechanics_nodes=128, seed=37,
            physics_profile='reviewed_v1', primordial_subduction={'enabled': True}))
        world.plate_resistance_version = 1
        model = balance.Balance(world, 2.)
        model.solve()
        diag = model.diagnostics(model.rotation())
        self.assertEqual(diag['plate_resistance_version'], 1)
        self.assertEqual(diag['negative_resisting_work_elements'], 0)
        self.assertTrue(all(work >= 0. for work in diag['dissipation_w'].values()))
        self.assertLessEqual(diag['scaled_force_residual'], balance.FORCE_RELATIVE_TOLERANCE)
        self.assertLess(abs(diag['power_balance_error_w']), 1.e-7*max(diag['driver_work_w'], 1.))
        self.assertIn('Finite smoothing', diag['regularization_note'])
        self.assertNotEqual(diag['dissipation_w'], diag['dissipation_potential_w'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.npz'
            checkpoint.write_checkpoint(path, world, dict(config=world.config), {})
            restored, _ = checkpoint.read_checkpoint(path, {}, native_engine.Simulation)
        self.assertEqual(restored.plate_resistance_version, 1)
        replay = balance.Balance(restored, 2.)
        replay.solve()
        np.testing.assert_array_equal(replay.x, model.x)


if __name__ == '__main__':
    unittest.main()
