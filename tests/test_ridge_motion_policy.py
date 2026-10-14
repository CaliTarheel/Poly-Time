"""Oblique opening must agree across labels, production and review geometry."""
import unittest
from unittest.mock import patch

import numpy as np

from native_engine import Simulation
import native_spreading
import ridge_spreading
from tests.test_native_spreading_pairing import scene, RADIUS


def boundary_scene(opening, sliding, *, spreading_version=0):
    """One prescribed meridional ocean front, speeds in km/Myr at its centre."""
    s, axis = scene()
    s.omega[0] = [0., sliding/(2*RADIUS), -opening/(2*RADIUS)]
    s.omega[1] = -s.omega[0]
    s.edge_a, s.edge_b = s.ba.copy(), s.bb.copy()
    s.crust = np.zeros(s.n, np.uint8)
    s.age = np.full(s.n, 80.)
    s.polarity = np.full((3, 3), -1, np.int16)
    s.native_spreading_version = 1
    s.normal_spreading_version = spreading_version
    s.native_boundary_geometry.update(
        normals=np.array([[0., 1., 0.]]), midpoints=np.array([[1., 0., 0.]]),
        lengths=np.array([np.arccos(np.clip(axis[0]@axis[1], -1., 1.))*RADIUS]),
        diagnostics={})
    # Prescribe geometry, then exercise the real engine classification and
    # real material clipping/production. Trench history is outside this fixture.
    with patch('native_boundary_geometry.reconstruct', return_value=s.native_boundary_geometry), \
            patch('trench_history.prepare'):
        Simulation._boundaries(s)
    return s


class RidgeMotionPolicyTests(unittest.TestCase):
    def test_slow_opening_matches_independent_spherical_strip_area(self):
        from tests._native_spreading_oracle import polygon_area
        opening, dt = .1, 2.
        s = boundary_scene(opening, 0., spreading_version=1)
        np.testing.assert_array_equal(s.bcode, [1])
        np.testing.assert_array_equal(ridge_spreading._active(s), [0])
        self.assertEqual([r['code'] for r in native_spreading.boundary_segments(s)], [1])
        axis = np.array([s.native_boundary_geometry['segments_start'][0],
                         s.native_boundary_geometry['segments_end'][-1]])
        angle = opening*dt/(2*RADIUS)
        matrix = np.array([[np.cos(angle), -np.sin(angle), 0.],
                           [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        shore = axis@matrix.T
        expected = 2.*polygon_area(np.array([axis[0],shore[0],shore[1],axis[1]]))
        transported = s.support.copy()
        birth = native_spreading.advance(s, transported, dt)
        self.assertGreater(expected, 0.)
        self.assertAlmostEqual(float(birth@s.cell_area), expected, delta=1e-6)
        self.assertAlmostEqual(s.process_totals['ocean_created_km2'], expected, delta=1e-6)
        self.assertAlmostEqual(s.spreading_diagnostics['side_p_area_km2'], expected/2., delta=1e-6)
        self.assertAlmostEqual(s.spreading_diagnostics['side_q_area_km2'], expected/2., delta=1e-6)
        np.testing.assert_allclose(transported.sum(axis=0),s.support.sum(axis=0),atol=1e-12)

    def test_slow_oblique_opening_and_pure_sliding_are_distinct(self):
        for opening, expected in ((.1, True), (0., False), (-.1, False)):
            s = boundary_scene(opening, 40., spreading_version=1)
            birth = native_spreading.advance(s,s.support.copy(),2.)
            self.assertEqual(float(birth@s.cell_area)>0., expected)

    def test_common_rotation_does_not_create_crust_with_stale_ridge_labels(self):
        s = boundary_scene(.1,0.,spreading_version=1)
        native_spreading.prepare(s)
        s.omega[:] = [.012,-.03,.002]
        self.assertEqual(len(ridge_spreading._active(s)),0)
        np.testing.assert_array_equal(native_spreading.advance(s,s.support.copy(),2.),0.)

    def test_missing_policy_keeps_historical_gate_and_invalid_versions_fail(self):
        from ridge_geometry import normal_motion_threshold
        s = boundary_scene(.1,0.,spreading_version=1)
        del s.normal_spreading_version
        self.assertEqual(len(ridge_spreading._active(s)),0)
        np.testing.assert_array_equal(native_spreading.advance(s,s.support.copy(),2.),0.)
        for version in (True,1.,-1,2,'1'):
            with self.subTest(version=version),self.assertRaises(ValueError):
                normal_motion_threshold(0.,spreading_version=version)

    def test_checkpoint_preserves_new_and_missing_policy(self):
        from pathlib import Path
        from types import SimpleNamespace
        import tempfile
        import checkpoint
        from ridge_geometry import normal_motion_threshold
        for version in (None,1):
            s=SimpleNamespace(t=0.,config={},rng=np.random.default_rng(42))
            if version is not None:
                s.normal_spreading_version=version
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'state.npz'
                checkpoint.write_checkpoint(path,s,{'config':{}},{})
                restored,_=checkpoint.read_checkpoint(path,None,SimpleNamespace)
            self.assertEqual(hasattr(restored,'normal_spreading_version'),version is not None)
            self.assertEqual(float(normal_motion_threshold(0.,spreading_version=
                getattr(restored,'normal_spreading_version',0))),0. if version==1 else 2.)

    def test_oblique_opening_creates_paired_crust_and_stays_a_ridge(self):
        # 8/40=.2 is between the former inconsistent .15 and .35 thresholds.
        s = boundary_scene(8., 40.)
        np.testing.assert_array_equal(s.bcode, [1])
        rows = native_spreading.boundary_segments(s)
        self.assertEqual([row['code'] for row in rows], [1])
        transported = s.support.copy()
        birth = native_spreading.advance(s, transported, 2.)
        area = float(birth@s.cell_area)
        self.assertGreater(area, 0.)
        report = s.spreading_diagnostics
        self.assertAlmostEqual(report['side_p_area_km2'], report['side_q_area_km2'], delta=1e-6)
        self.assertAlmostEqual(area, report['side_p_area_km2']+report['side_q_area_km2'], delta=1e-6)
        self.assertAlmostEqual(area, s.process_totals['ocean_created_km2'], delta=1e-6)
        np.testing.assert_allclose(transported.sum(axis=0), s.support.sum(axis=0), atol=1e-12)

    def test_sliding_stopped_closing_and_threshold_contacts_create_no_crust(self):
        for opening, sliding, code in ((0., 40., 3), (0., 0., 3), (-8., 40., 2),
                                       (6., 40., 3), (2., 0., 3), (1., 0., 3)):
            with self.subTest(opening=opening, sliding=sliding):
                s = boundary_scene(opening, sliding)
                np.testing.assert_array_equal(s.bcode, [code])
                self.assertEqual([r['code'] for r in native_spreading.boundary_segments(s)], [code])
                transported = s.support.copy()
                birth = native_spreading.advance(s, transported, 2.)
                np.testing.assert_array_equal(birth, 0.)
                np.testing.assert_array_equal(transported, s.support)
                self.assertEqual(s.process_totals['ocean_created_km2'], 0.)

    def test_native_front_cache_uses_native_policy_without_changing_raster_policy(self):
        s = boundary_scene(8., 40.)
        np.testing.assert_array_equal(ridge_spreading._active(s), [0])
        # The retained raster edition still uses its historical .35 policy.
        del s.native_mesh
        self.assertEqual(len(ridge_spreading._active(s)), 0)

    def test_motion_change_expires_spreading_without_reusing_old_labels(self):
        s = boundary_scene(8., 40.)
        native_spreading.prepare(s)
        s.omega[:, 2] = 0.  # Keep sliding, remove opening; bcode is deliberately stale.
        self.assertEqual([r['code'] for r in native_spreading.boundary_segments(s)], [3])
        birth = native_spreading.advance(s, s.support.copy(), 2.)
        np.testing.assert_array_equal(birth, 0.)


if __name__ == '__main__':
    unittest.main()
