"""Convex opening solves must not alternate across narrow contact transitions."""
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

import breakup_mode as opening
import plate_balance as pb
from balance_force_ledger import CompiledMode


def _model(size):
    model = object.__new__(CompiledMode)
    model.size = size
    model.common_size = size-1
    model.resistance_version = 1
    model.stiffness = np.eye(size)
    model.torque = np.zeros(size)
    model.hinge = np.empty((0, size))
    model.hinge_coefficient = np.empty(0)
    model.elements = []
    return model


def _contact_fixture():
    """Only constitutive arrays and solve inputs; no world or capture metadata."""
    path = Path(__file__).with_name('fixtures')/'breakup_contact_overshoot.npz'
    with np.load(path, allow_pickle=False) as arrays:
        model = _model(len(arrays['initial']))
        for name in ('stiffness', 'torque', 'hinge', 'hinge_coefficient'):
            setattr(model, name, arrays[name])
        for name in ('common_size', 'resistance_version'):
            setattr(model, name, int(arrays[name]))
        for index in range(int(arrays['element_count'])):
            prefix = 'element_%d_' % index
            element = {name: str(arrays[prefix+name]) for name in ('kind', 'shape')}
            element.update({name: arrays[prefix+name] for name in ('coefficient', 'scale')})
            element['rows'] = tuple(arrays[prefix+'row_%d' % i]
                                   for i in range(int(arrays[prefix+'row_count'])))
            if element['shape'] == 'arc_opening':
                element['bounds'] = arrays[prefix+'bounds']
            model.elements.append(element)
        return model, arrays['initial'], float(arrays['cost'])


class BreakupGlobalizationTests(unittest.TestCase):
    def test_contact_transition_fixture_converges_with_strict_force_and_power_checks(self):
        # Armijo alone exhausted all 60 final-width iterations, alternating a
        # common plate across saturated friction with backward error 0.421754.
        model, initial, cost = _contact_fixture()
        before = deepcopy((model.stiffness, model.torque, initial, model.elements))
        motion, receipt = opening._minimize(model, initial, cost)
        gradient = model._evaluate(motion, opening._FINAL_WIDTH)[1].copy()
        gradient[-1] += cost
        self.assertLessEqual(opening._residual(model, motion, gradient, cost), pb.FORCE_RELATIVE_TOLERANCE)
        self.assertLessEqual(receipt['force_relative_residual'], pb.FORCE_RELATIVE_TOLERANCE)
        self.assertGreater(motion[-1], 0.)
        work, passivity = model._resistance_work(motion, opening._FINAL_WIDTH)
        drive = float(model.torque@motion)
        resistance = float(motion@model.stiffness@motion)+sum(work.values())
        cut_power = cost*motion[-1]
        closure = abs(drive-resistance-cut_power)/max(abs(drive), abs(resistance)+cut_power, 1.)
        self.assertLessEqual(closure, 1e-8)
        self.assertEqual(passivity['negative_resisting_work_elements'], 0)
        for expected, actual in zip(before[:3], (model.stiffness, model.torque, initial)):
            np.testing.assert_array_equal(actual, expected)
        for expected, actual in zip(before[3], model.elements):
            for name in ('coefficient', 'scale', 'bounds'):
                if name in expected:
                    np.testing.assert_array_equal(actual[name], expected[name])
            for a, b in zip(actual['rows'], expected['rows']):
                np.testing.assert_array_equal(a, b)

    def test_independent_equilibrium_energy_cannot_hide_nonlinear_overshoot(self):
        roots = []
        for reference in (1., 1e15):
            model = _model(7)
            model.torque = np.array([.1, 0., 0., reference, 0., 0., 0.])
            model.elements = [dict(kind='friction', shape='abs',
                rows=(np.array([[1., 0., 0., 0., 0., 0., 0.]]),),
                coefficient=np.array([1e6]), scale=np.ones(1))]
            initial = np.array([1000., 0., 0., reference, 0., 0., 0.])
            motion, receipt = opening._minimize(model, initial, 1.)
            self.assertLessEqual(receipt['force_relative_residual'], pb.FORCE_RELATIVE_TOLERANCE)
            self.assertEqual(motion[-1], 0.)
            roots.append(motion[0])
        np.testing.assert_allclose(roots[0], roots[1], rtol=1e-8, atol=0.)

    def test_continuation_can_reach_active_bound_with_negative_directional_slope(self):
        model = _model(4)
        model.torque = np.array([1., 0., 0., 2.])
        model.elements = [dict(kind='friction', shape='abs',
            rows=(np.array([[1., 0., 0., 1.]]),),
            coefficient=np.array([10.]), scale=np.ones(1))]
        observed = []
        evaluate = model._evaluate
        def record(x, delta):
            observed.append(float(x[-1]))
            return evaluate(x, delta)
        model._evaluate = record
        initial = np.array([1., 0., 0., 0.])
        motion, receipt = opening._minimize(model, initial, 1., relax_common=False)
        self.assertGreater(max(observed), 0.)
        self.assertEqual(motion[-1], 0.)
        self.assertGreater(receipt['opening_bound_gradient_w'], 0.)
        np.testing.assert_array_equal(motion[:-1], initial[:-1])
        self.assertEqual(receipt['force_relative_residual'], 0.)


if __name__ == '__main__':
    unittest.main()
