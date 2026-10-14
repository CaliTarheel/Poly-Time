"""A contact is measured on the contour pieces its length came from (contact share version 1)."""
from types import SimpleNamespace
import unittest

import numpy as np

import effective_subduction as effective
import effective_subduction_initiation as initiation
import mesh_geometry
import native_boundary_geometry as boundary
import native_subduction
from tests._native_spreading_oracle import material_rectangles
from tests.test_effective_subduction_initiation import motion, set_pieces, tick, world
from tests.test_native_boundary_geometry import fixture
from tests.test_native_subduction import history_scene


def piece_km(geometry):
    a, b = geometry['segments_start'], geometry['segments_end']
    return 6371.*np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a*b, axis=1))


class ContactShareTests(unittest.TestCase):
    def test_the_shares_are_the_division_that_gave_each_contact_its_length(self):
        for level, hard, axis in ((4, False, (.37, .72, -.48)), (4, True, (.37, .72, -.48)), (3, False, (.2, .5, .84))):
            with self.subTest(level=level, hard=hard):
                mesh, owner, support, edges, _ = fixture(level=level, axis=axis, hard=hard)
                plain = boundary.reconstruct(mesh, owner, support, edges)
                shared = boundary.reconstruct(mesh, owner, support, edges, with_shares=True)
                self.assertFalse(boundary.has_shares(plain))
                self.assertTrue(boundary.has_shares(shared))
                # Nothing the reconstruction returned before is different.
                for key in plain:
                    if key != 'diagnostics':
                        np.testing.assert_array_equal(plain[key], shared[key])
                self.assertEqual(plain['diagnostics'], shared['diagnostics'])
                lengths = piece_km(shared)
                np.testing.assert_allclose(boundary.shares(shared, lengths, len(edges)), shared['lengths'], rtol=0, atol=1e-9)
                # Every piece is given away whole.
                given = np.bincount(shared['share_segment'], weights=shared['share_weight'], minlength=len(lengths))
                np.testing.assert_allclose(given, 1., rtol=0, atol=1e-12)
                self.assertTrue(np.all(shared['share_weight'] > 0.))
                # The case the shares exist for: contacts that hold length and are listed with no piece.
                listed = np.bincount(shared['contact_index'], minlength=len(edges))
                orphan = (listed == 0) & (shared['lengths'] > 1e-10)
                self.assertGreater(np.count_nonzero(orphan), 5)
                self.assertTrue(np.all(np.isin(np.flatnonzero(orphan), shared['share_contact'])))

    def test_a_contact_that_keeps_its_raw_edge_takes_that_edge_whole(self):
        mesh = mesh_geometry.icosphere(3)
        owner = np.zeros(len(mesh['faces']), np.int32); owner[200] = 1
        support = np.eye(2)[owner].T
        shared = boundary.reconstruct(mesh, owner, support, with_shares=True)
        count = len(shared['lengths'])
        self.assertEqual(shared['diagnostics']['fallback_edges'], count)
        np.testing.assert_array_equal(shared['share_contact'], shared['contact_index'])
        np.testing.assert_array_equal(shared['share_segment'], np.arange(count))
        np.testing.assert_array_equal(shared['share_weight'], 1.)
        np.testing.assert_allclose(boundary.shares(shared, piece_km(shared), count), shared['lengths'], rtol=1e-12)
        empty = boundary.reconstruct(mesh, np.zeros(len(mesh['faces']), np.int32), np.ones((1, len(mesh['faces']))),
                                     with_shares=True)
        self.assertEqual(len(empty['lengths']), 0)

    def test_incoming_ocean_is_measured_on_the_ground_the_length_came_from(self):
        # One contour piece, half of it against the incoming plate's continent. Its length is
        # divided between two contacts; only the first is listed with the piece.
        def scene():
            s = history_scene(); s.crust[:] = 1
            vertices, faces = material_rectangles([(-200., 200., 0., 900.)])
            s.material_surface = dict(vertices=vertices, faces=faces, face_owner=np.zeros(len(faces), int))
            for key in ('ba', 'bb', 'bp', 'bq', 'bcode', 'normal_speed', 'bl'):
                setattr(s, key, np.repeat(getattr(s, key), 2))
            s.bl = s.bl*np.array([.6, .4])
            s.down = np.array([0, 0])
            return s
        s = scene()
        np.testing.assert_allclose(native_subduction.edge_ocean_fraction(s, 0), [.5, 0.], atol=1e-10)
        s = scene()
        s.native_boundary_geometry.update(share_segment=np.array([0, 0], np.int32), share_contact=np.array([0, 1], np.int32),
                                          share_weight=np.array([.6, .4]))
        np.testing.assert_allclose(native_subduction.edge_ocean_fraction(s, 0), [.5, .5], atol=1e-10)
        # The cached answer is not reused when only the division changes.
        s.native_boundary_geometry['share_contact'] = np.array([0, 0], np.int32)
        np.testing.assert_allclose(native_subduction.edge_ocean_fraction(s, 0), [.5, 0.], atol=1e-10)

    def test_an_induced_trench_pulls_through_a_contact_listed_with_no_piece(self):
        # Three pieces born as one trench. The boundary is then cut into two pieces and three
        # contacts: the third is listed with no piece and holds a third of each.
        def redrawn(shared):
            s = world()
            initiation.upgrade(s, dict(version=2, enabled=True))
            tick(s)
            for _ in range(10):
                tick(s)
            self.assertTrue(initiation.activate_pending(s))
            set_pieces(s, np.array([-.03, 0., .06]), parents=[0, 1])
            motion(s, 12., 12.)
            full = s.bl.copy()
            for key in ('ba', 'bb', 'bp', 'bq', 'bcode', 'down', 'normal_speed', 'trench_id', 'trench_maturity'):
                setattr(s, key, np.append(getattr(s, key), getattr(s, key)[:1]))
            s.bmid = np.vstack((s.bmid, s.bmid[:1])); s.bn = np.vstack((s.bn, s.bn[:1]))
            s.bl = np.array([full[0]*2/3, full[1]*2/3, (full[0]+full[1])/3])
            if shared:
                s.native_boundary_geometry.update(share_segment=np.array([0, 0, 1, 1], np.int32),
                                                  share_contact=np.array([0, 2, 1, 2], np.int32),
                                                  share_weight=np.array([2/3, 1/3, 2/3, 1/3]))
            effective.prepare(s)
            return s, full
        # The trace runs from -0.03 to 0.03: all of the first piece, half of the second.
        s, full = redrawn(False)
        pull = s.trench_systems[0]['effective_subduction']['force_n_per_m']
        forces = effective.line_state(s)[1]
        np.testing.assert_allclose(forces, pull*np.array([1., .5, 0.]), rtol=1e-9)
        s, full = redrawn(True)
        forces = effective.line_state(s)[1]
        third = (full[0]*1.+full[1]*.5)/(full[0]+full[1])
        np.testing.assert_allclose(forces, pull*np.array([1., .5, third]), rtol=1e-9)
        # With the shares the trench pulls on exactly the length of its trace that is still boundary.
        self.assertAlmostEqual(float(forces@s.bl)/pull, 6371.*.06, delta=1e-6)

    def test_the_version_is_explicit_and_takes_effect_at_the_next_reconstruction(self):
        s = SimpleNamespace(t=300.)
        self.assertEqual(boundary.share_version(s), 0)
        report = boundary.upgrade_shares(s)
        self.assertEqual((boundary.share_version(s), report['version'], report['time_myr']), (1, 1, 300.))
        self.assertEqual(boundary.upgrade_shares(s), report)
        for bad in (True, 2, '1', 1.):
            s.contact_share_version = bad
            with self.assertRaises(ValueError):
                boundary.share_version(s)


if __name__ == '__main__':
    unittest.main()
