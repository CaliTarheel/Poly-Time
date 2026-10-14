"""Uniform port projection must preserve unilateral hinge activation exactly."""
from dataclasses import replace
from copy import copy
import unittest

import numpy as np

import balance_force_ledger as ledger
import breakup_mode as opening
import plate_balance as pb
from tests.test_breakup_mode import fixture, run_case


def rounded_port_case(error=8*np.finfo(float).eps):
    data = list(fixture())
    state, balance, cells, piece, axis = data[:5]
    # A common Euler motion lies exactly on the relative hinge's activation
    # surface. An independently accumulated port sum can differ by a few ulps.
    balance.x = np.tile([.1, -.4, .7], balance.count)
    report = ledger.export(balance, balance.x)
    term = next(term for term in report['local_terms'] if term.shape == 'hinge')
    port = replace(term.ports[0], matrices=term.ports[0].matrices.copy())
    port.matrices[0, 2, 2] += error
    term.ports = (port,)+term.ports[1:]
    basis, _ = opening._basis(state, balance, report, 0, cells, piece, axis)
    return data, report, basis, term, port


class CommonModeProjectionTests(unittest.TestCase):
    def test_roundoff_scale_port_error_cannot_activate_common_hinge(self):
        data, report, basis, term, port = rounded_port_case()
        balance = data[1]
        identity_error = np.linalg.norm(port.matrices.sum(axis=0)-np.eye(3))/np.sqrt(3.)
        self.assertGreater(identity_error, 0.)
        self.assertLess(identity_error, opening._COMMON_RESTRICTION_TOLERANCE)
        # Demonstrate the original raw projection's activation error without
        # depending on the compiled implementation used by the assertion.
        raw_blocks = []
        for p in term.ports:
            index = np.full(len(data[0].xyz), -1, int)
            index[report['plates'][p.owner]['cells']] = np.arange(len(basis[p.owner]))
            raw_blocks.append(np.einsum('cij,cik->jk', p.matrices, basis[p.owner][index[p.cells]]))
        raw_projection = np.vstack(raw_blocks)
        raw_rate = float((term.rows[0]@raw_projection)@np.r_[balance.x, 0.])
        self.assertLess(raw_rate, 0.)
        self.assertEqual(float(balance.hinge[0]@balance.x), 0.)
        compiled = ledger.compile_mode(report, basis)
        actual_rate = compiled.hinge@np.r_[balance.x, 0.]
        np.testing.assert_array_equal(actual_rate, balance.hinge@balance.x)
        for offset in (-1e-12, 0., 1e-12):
            motion = balance.x.copy()
            motion[5] += offset
            got = compiled._evaluate(np.r_[motion, 0.], opening._FINAL_WIDTH)
            expected = balance._evaluate(motion, opening._FINAL_WIDTH)
            for candidate, oracle in ((got[0], expected[0]), (got[1][:-1], expected[1]),
                                      (got[2][:-1, :-1], expected[2])):
                error = np.linalg.norm(candidate-oracle)/max(np.linalg.norm(oracle), 1.)
                self.assertLess(error, 2e-12)
            np.testing.assert_array_equal(compiled.hinge@np.r_[motion, 0.] < 0.,
                                          balance.hinge@motion < 0.)

    def test_full_daughter_solve_accepts_roundoff_projection_and_closes_work(self):
        data, report, _, _, _ = rounded_port_case()
        before = data[1].x.copy()
        result = run_case(data, .5, ledger=report, allow_zero_fast_path=False)
        self.assertTrue(result['opening_mode_supported'], result.get('opening_mode_reason'))
        self.assertFalse(result['zero_rate_fast_path'])
        self.assertLess(result['common_restriction_relative_error'], 1e-9)
        self.assertLess(result['differential_virtual_work_relative_error'], 1e-9)
        self.assertLess(result['mode_work']['closure_relative'], 1e-8)
        self.assertGreater(result['relative_rotation_rad_myr'], 0.)
        np.testing.assert_array_equal(data[1].x, before)

    def test_nonuniform_port_field_keeps_original_adjoint(self):
        _, _, _, _, port = rounded_port_case()
        rng = np.random.default_rng(161)
        field = rng.normal(size=(int(port.cells.max())+1, 3, 5))
        expected = np.einsum('cij,cik->jk', port.matrices, field[port.cells])
        matrices = port.matrices.copy()
        # Removing the computed uniform residual changes only its product
        # with the anchor, plus bounded summation roundoff for this stencil.
        anchor = field[port.cells[0]]
        residual = np.eye(3)-port.matrices.sum(axis=0).T
        magnitude = np.einsum('cij,cik->jk', np.abs(port.matrices), np.abs(field[port.cells]))
        allowance = np.linalg.norm(residual@anchor)+32*np.finfo(float).eps*np.linalg.norm(magnitude+np.abs(anchor))
        self.assertLessEqual(np.linalg.norm(port.basis(field)-expected), allowance)
        force = np.array([.3, -.7, 1.1])
        velocity = field[:, :, 2]
        np.testing.assert_array_equal(port.basis(field)[:, 2], port.motion(velocity))
        lifted = port.lift(force)
        gross = float(np.sum(np.abs(lifted)))
        np.testing.assert_allclose(lifted.sum(axis=0), force, rtol=0.,
                                   atol=8*np.finfo(float).eps*max(gross, 1.))
        self.assertAlmostEqual(float(force@port.basis(field)[:, 2]),
                               float(np.sum(lifted*velocity[port.cells])),
                               delta=16*np.finfo(float).eps*max(gross*np.max(np.abs(velocity)), 1.))
        constant = np.broadcast_to(force, velocity.shape)
        np.testing.assert_array_equal(port.motion(constant), force)
        np.testing.assert_array_equal(port.matrices, matrices)

    def test_material_port_corruption_is_rejected_in_full_response(self):
        data, report, _, _, port = rounded_port_case(error=1e-5)
        identity_error = np.linalg.norm(port.matrices.sum(axis=0)-np.eye(3))/np.sqrt(3.)
        self.assertGreater(identity_error, opening._COMMON_RESTRICTION_TOLERANCE)
        result = run_case(data, .5, ledger=report, allow_zero_fast_path=False)
        self.assertFalse(result['opening_mode_supported'])
        self.assertRegex(result['opening_mode_reason'], 'uniform|restriction')
        field = np.zeros((int(port.cells.max())+1, 3))
        for operation in (lambda: port.motion(field), lambda: port.basis(field[:, :, None]),
                          lambda: port.lift(np.ones(3))):
            with self.assertRaisesRegex(ledger.UnsupportedForceLedger, 'uniform motion'):
                operation()

    def test_fractional_only_support_accepts_global_and_local_common_bases(self):
        import plate_balance
        from tests.test_balance_force_ledger import fractional_only_octants, uniform_basis
        state, _ = fractional_only_octants()
        balance = plate_balance.Balance(state, 1., slab_tethers=True)
        motion = np.array([.1, -.4, .7, -.2, .8, -.6])
        report = ledger.export(balance, motion)
        self.assertFalse(np.any(state.plate[report['plates'][1]['cells']] == 1))
        basis = uniform_basis(balance, extra=1)
        local = {p: basis[p][entry['cells']] for p, entry in report['plates'].items()}
        expected = balance._evaluate(motion, opening._FINAL_WIDTH)
        for field in (basis, local):
            compiled = ledger.compile_mode(report, field)
            got = compiled._evaluate(np.r_[motion, 0.], opening._FINAL_WIDTH)
            for candidate, oracle in ((got[0], expected[0]), (got[1][:-1], expected[1]),
                                      (got[2][:-1, :-1], expected[2])):
                error = np.linalg.norm(candidate-oracle)/max(np.linalg.norm(oracle), 1.)
                self.assertLess(error, 2e-12)

    def test_native_width_common_rates_preserve_passive_boundary_laws(self):
        direction = np.array([.7227245365932603, .467554294409754, .5089815576079886])
        rows = np.array([np.r_[-direction, direction, np.zeros(3)],
                         np.r_[direction, np.zeros(3), -direction]])
        motion = np.tile([1.408980908812071, .7643936147541728, .5523841174311587], 3)
        for shape in ('negative', 'cone', 'arc_opening'):
            with self.subTest(shape=shape):
                source = object.__new__(pb.Balance)
                source.size = 9; source.stiffness = np.eye(9); source.torque = np.zeros(9)
                source.hinge = np.empty((0, 9)); source.hinge_coefficient = np.empty(0)
                source.resistance_version = 1
                if shape == 'arc_opening':
                    operators = (rows[None],)
                    extra = np.array([[[.37], [-.23]]])
                else:
                    operators = (rows,) if shape == 'negative' else (rows, np.zeros_like(rows))
                    extra = np.array([[.37], [-.23]])
                count = len(operators[0])
                element = dict(kind=shape, shape=shape, rows=operators, coefficient=np.full(count, 1e4),
                               scale=np.ones(count), bounds=np.tile([-.4, .7], (count, 1)))
                source.elements = [element]
                reduced = object.__new__(ledger.CompiledMode)
                reduced.size = 10; reduced.common_size = 9
                reduced.stiffness = np.eye(10); reduced.torque = np.zeros(10)
                reduced.hinge = np.empty((0, 10)); reduced.hinge_coefficient = np.empty(0)
                reduced.resistance_version = 1
                widened = tuple(np.concatenate((operator, extra), axis=-1) for operator in operators)
                reduced.elements = [dict(element, rows=widened)]
                for operator, extended in zip(operators, widened):
                    np.testing.assert_array_equal(operator, extended[..., :9])
                    np.testing.assert_array_equal(source._row_rates(operator, motion),
                                                  reduced._row_rates(extended, np.r_[motion, 0.]))
                expected = source._evaluate(motion, opening._FINAL_WIDTH)
                actual = reduced._evaluate(np.r_[motion, 0.], opening._FINAL_WIDTH)
                for got, want in ((actual[0], expected[0]), (actual[1][:-1], expected[1]),
                                  (actual[2][:-1, :-1], expected[2])):
                    self.assertLess(np.linalg.norm(got-want)/max(np.linalg.norm(want), 1.), 2e-12)
                actual_work = reduced._resistance_work(np.r_[motion, 0.], opening._FINAL_WIDTH)[0][shape]
                expected_work = source._resistance_work(motion, opening._FINAL_WIDTH)[0][shape]
                self.assertAlmostEqual(actual_work, expected_work,
                                       delta=2e-12*max(abs(expected_work), 1e-100))

    def test_appended_daughter_column_preserves_native_width_hinge_evaluation(self):
        data = list(fixture())
        state, balance, cells, piece, axis = data[:5]
        direction = np.array([.7227245365932603, .467554294409754, .5089815576079886])
        balance.hinge[0, :3], balance.hinge[0, 3:6] = -direction, direction
        balance.hinge[1, :3], balance.hinge[1, 6:9] = direction, -direction
        motion = np.tile([1.408980908812071, .7643936147541728, .5523841174311587], balance.count)
        report = ledger.export(balance, balance.x)
        # Exact direct ports isolate dot-product width from spatial lift error.
        # The same common rows can change their zero-rate sign when BLAS gains
        # one differential column, even though that coordinate is zero.
        for term in report['local_terms']:
            if term.shape == 'hinge':
                term.ports = tuple(replace(port, cells=port.cells[:1].copy(), matrices=np.eye(3)[None])
                                   for port in term.ports)
        basis, _ = opening._basis(state, balance, report, 0, cells, piece, axis)
        compiled = ledger.compile_mode(report, basis)
        np.testing.assert_array_equal(compiled.hinge[:, :-1], balance.hinge)
        original = balance._evaluate(motion, opening._FINAL_WIDTH)
        restricted = compiled._evaluate(np.r_[motion, 0.], opening._FINAL_WIDTH)
        for candidate, oracle in ((restricted[0], original[0]), (restricted[1][:-1], original[1]),
                                  (restricted[2][:-1, :-1], original[2])):
            error = np.linalg.norm(candidate-oracle)/max(np.linalg.norm(oracle), 1.)
            self.assertLess(error, 2e-12)

        # The same split evaluation must hold at positive and negative daughter
        # rates, with the original one-sided law; a special q==0 bypass is not
        # a consistent nonlinear operator.
        bare = copy(compiled)
        bare.hinge = np.empty((0, compiled.size)); bare.hinge_coefficient = np.empty(0)
        for differential in (-1e-12, 0., 1e-12):
            z = np.r_[motion, differential]
            rate = balance.hinge@motion+compiled.hinge[:, -1]*differential
            closing = np.minimum(rate, 0.)
            weight = balance.hinge_coefficient
            base = bare._evaluate(z, opening._FINAL_WIDTH)
            expected = (base[0]+.5*np.sum(weight*closing**2),
                        base[1]+(weight*closing)@compiled.hinge,
                        base[2]+np.einsum('e,ei,ej->ij', weight[rate < 0.],
                                         compiled.hinge[rate < 0.], compiled.hinge[rate < 0.]))
            got = compiled._evaluate(z, opening._FINAL_WIDTH)
            for candidate, oracle in zip(got[:3], expected):
                error = np.linalg.norm(candidate-oracle)/max(np.linalg.norm(oracle), 1.)
                self.assertLess(error, 2e-12)


if __name__ == '__main__':
    unittest.main()
