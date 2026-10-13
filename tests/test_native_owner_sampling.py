"""Saved compact owner scores reproduce contours without display-grid lookup."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

import mesh_history
import native_frame_sampling as sampling
from orientation import rotation_matrix
from server import SimulationManager
try:
    from .test_native_gospl_sampling import frame
except ImportError:
    from test_native_gospl_sampling import frame


def owner_frame(*, overlap=False, complete=False):
    source = frame(centers=[[0, 0, 1]]*2 if overlap else [], heights=[500., 1200.] if overlap else [])
    source['plates'] = [dict(id=2, uid=10, angular_velocity=[0., 0., 0.]),
                        dict(id=7, uid=20, angular_velocity=[0., 0., 0.])]
    source['mesh_plate'][:] = 2
    source['plate'][:] = 2
    source['age'] = np.full(source['width']*source['height'], 60.)
    if overlap:
        source['material_owner'][:] = [2, 7]
    score = source['mesh_vertices'][:, 1]
    source.update(owner_reconstruction_version=1,
                  mesh_owner_slots=np.array([2, 7], np.int32),
                  mesh_vertex_support=np.array([1.+.7*score, 1.-.7*score]))
    if complete:
        for name in list(source):
            if name.startswith('trace_'):
                source.pop(name)
        for name, dtype in mesh_history.DTYPES.items():
            if name in source:
                continue
            # This fixture intentionally represents the pre-transport,
            # pre-arc, pre-contact epoch schema. New optional arrays must not
            # be fabricated without their explicit version metadata.
            if name in (set(mesh_history.TRANSPORT_DTYPES) | set(mesh_history.DEFORMATION_DTYPES)
                        | set(mesh_history.ARC_DTYPES) | set(mesh_history.collision_contacts.ARRAY_FIELDS)
                        | set(mesh_history.collision_surface.ARRAY_FIELDS)
                        | set(mesh_history.COAST_DTYPES) | set(mesh_history.localized_accretion.ARRAY_FIELDS)
                        | set(mesh_history.lip_events.ARRAY_FIELDS) | set(mesh_history.eclogite_sink.ARRAY_FIELDS)
                        | set(mesh_history.ENHANCED_RIFT_DTYPES)
                        | set(mesh_history.numerical_accuracy.ARRAY_FIELDS)):
                continue
            if name == 'native_boundary_edges':
                source[name] = np.empty((0, 2), dtype)
            elif name.startswith('native_boundary_'):
                source[name] = np.empty(0, dtype)
            else:
                prefix = 'mesh' if name.startswith('mesh_') else 'material'
                source[name] = np.zeros(len(source[prefix+'_faces']), dtype)
    return source


class NativeOwnerSamplingTests(unittest.TestCase):
    def test_sparse_saved_slots_select_continuous_owner_inside_raw_control_cells(self):
        source = owner_frame()
        points = np.array([[1., .1, 0.], [1., -.1, 0.], [1., 0., 0.]])
        result = sampling.sample_frame(source, points)
        np.testing.assert_array_equal(result['plate'], [2, 7, 2])
        # Slot 7 has no hard control-cell label in this fixture; using those
        # labels or compact score row indices would give the wrong answer.
        np.testing.assert_array_equal(source['mesh_plate'], 2)
        np.testing.assert_array_equal(result['crust'], 0)

    def test_continuous_owner_controls_overlap_preference_before_material_identity(self):
        source = owner_frame(overlap=True)
        source['mesh_vertex_support'][:] = np.array([.1, .9])[:, None]
        actual = sampling.sample_frame(source, np.array([[0., 0., 1.]]))
        np.testing.assert_array_equal(actual['plate'], [7])
        np.testing.assert_array_equal(actual['material_face'], [1])
        np.testing.assert_array_equal(actual['elevation'], [1200.])
        # Actual material wins where the preferred owner has no sheet.
        source['material_owner'][:] = 2
        actual = sampling.sample_frame(source, np.array([[0., 0., 1.]]))
        np.testing.assert_array_equal(actual['plate'], [2])
        np.testing.assert_array_equal(actual['material_face'], [0])

    def test_unflagged_native_history_keeps_original_cell_owner_and_overlap_rule(self):
        source = owner_frame(overlap=True)
        source.pop('owner_reconstruction_version')
        source.pop('mesh_owner_slots'); source.pop('mesh_vertex_support')
        actual = sampling.sample_frame(source, np.array([[0., 0., 1.]]))
        np.testing.assert_array_equal(actual['plate'], [2])
        np.testing.assert_array_equal(actual['material_face'], [0])
        np.testing.assert_array_equal(actual['elevation'], [500.])

    def test_saved_scores_and_rotated_native_reader_preserve_owner_sampling(self):
        source = owner_frame(complete=True)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]/'tmp') as directory:
            manager = SimulationManager(directory)
            manager.current = dict(run_id='owner-sampling', frame_count=1)
            manager.path().mkdir()
            manager.save_frame(0, source)
            native = manager.native_frame(0)
            self.assertEqual(native['owner_reconstruction_version'], 1)
            self.assertEqual(native['mesh_owner_slots'].dtype, np.dtype(np.int32))
            np.testing.assert_array_equal(native['mesh_owner_slots'], [2, 7])
            np.testing.assert_array_equal(native['mesh_vertex_support'], source['mesh_vertex_support'])
            self.assertNotIn('mesh_vertex_support', manager.frame(0))
            angles = dict(yaw=42., pitch=65., roll=-17.)
            rotated = manager.native_frame(0, angles)
            np.testing.assert_array_equal(rotated['mesh_vertex_support'], native['mesh_vertex_support'])
            points = np.array([[1., .1, 0.], [1., -.1, 0.], [1., 0., 0.]])
            before = sampling.sample_frame(native, points)
            after = sampling.sample_frame(rotated, points@rotation_matrix(angles))
            for key in ('plate', 'crust', 'elevation'):
                np.testing.assert_allclose(after[key], before[key], atol=1e-9)

    def test_owner_sampling_ignores_the_display_resolution_and_labels(self):
        source = owner_frame()
        points = np.array([[1., .1, 0.], [1., -.1, 0.]])
        before = sampling.sample_frame(source, points)
        source.update(width=800, height=400, plate=np.full(320000, 99),
                      crust=np.full(320000, 3), elevation=np.full(320000, 9000.))
        after = sampling.sample_frame(source, points)
        for key in before:
            np.testing.assert_array_equal(after[key], before[key], err_msg=key)

    def test_native_schema_accepts_legacy_and_rejects_incomplete_or_invalid_scores(self):
        source = owner_frame(complete=True)
        mesh_history.arrays(source)
        legacy = deepcopy(source)
        for key in ('owner_reconstruction_version', 'mesh_owner_slots', 'mesh_vertex_support'):
            legacy.pop(key)
        result = mesh_history.arrays(legacy)
        self.assertNotIn('mesh_owner_slots', result)
        mutations = [dict(mesh_owner_slots=np.array([7, 2])),
                     dict(mesh_owner_slots=np.array([2, 2])),
                     dict(mesh_owner_slots=np.array([2, 99])),
                     dict(mesh_vertex_support=np.zeros((2, 1))),
                     dict(mesh_vertex_support=np.full_like(source['mesh_vertex_support'], np.nan)),
                     dict(mesh_vertex_support=-source['mesh_vertex_support']),
                     dict(owner_reconstruction_version=2)]
        for update in mutations:
            with self.subTest(update=list(update)):
                with self.assertRaises(ValueError):
                    mesh_history.arrays(dict(source, **update))
                with self.assertRaises(ValueError):
                    sampling.prepare(dict(source, **update))
        missing = deepcopy(source); missing.pop('mesh_vertex_support')
        with self.assertRaisesRegex(ValueError, 'owner'):
            mesh_history.arrays(missing)
        with self.assertRaisesRegex(ValueError, 'owner'):
            sampling.prepare(missing)


if __name__ == '__main__':
    unittest.main()
