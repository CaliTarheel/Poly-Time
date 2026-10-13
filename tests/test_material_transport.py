"""Material correspondence and physical budgets survive nonrigid remeshing."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import adaptive_material
import gospl_export as export
import material_transport as transport
import mesh_geometry
import mesh_history
import native_frame_sampling as sampling
from orientation import orient_frame, rotation_matrix
from validate_gospl import validate_package
try:
    from .test_material_reconstruction import continuous_frame
    from .test_native_gospl_sampling import frame, write_frame
except ImportError:
    from test_material_reconstruction import continuous_frame
    from test_native_gospl_sampling import frame, write_frame


def unit(x):
    return x/np.linalg.norm(x, axis=-1, keepdims=True)


def tracked(source=None):
    source = continuous_frame() if source is None else deepcopy(source)
    count = len(source['material_faces'])
    source.update(material_transport_version=1,
                  material_root_id=source['material_face_id'].copy(),
                  material_reference_corners=np.tile(np.eye(3), (count, 1, 1)),
                  material_erosion_total_m=np.zeros(count))
    return source


def refine(source):
    result = adaptive_material.refine(source['material_vertices'], source['material_faces'],
        source['material_owner'], source['material_kind'], desired_edge_km=100.,
        face_ids=source['material_face_id'])
    after = deepcopy(source)
    for field in ('material_owner', 'material_kind', 'material_height_m',
                  'material_erosion_rate_m_myr', 'material_root_id', 'material_erosion_total_m'):
        after[field] = source[field][result['source_face']]
    after['material_vertices'], after['material_faces'] = result['vertices'], result['faces']
    after['material_face_id'] = np.arange(len(result['faces']), dtype=np.int64)+1000
    inherited = source['material_reference_corners'][result['corner_source_face']]
    after['material_reference_corners'] = np.einsum('mij,mijk->mik', result['corner_barycentric'], inherited)
    after['material_reference_corners'] /= result['corner_radial_scale'][:, :, None]
    return after, result


def query(source, face_indices, weights):
    return unit(np.einsum('ni,nij->nj', weights,
                         source['material_vertices'][source['material_faces'][face_indices]]))


def complete_schema(source):
    result = deepcopy(source)
    for name, dtype in mesh_history.DTYPES.items():
        if (name in result or name in mesh_history.OWNER_DTYPES or name in mesh_history.TRANSPORT_DTYPES or name in mesh_history.ARC_DTYPES
                or name in mesh_history.DEFORMATION_DTYPES
                or name in mesh_history.collision_contacts.ARRAY_FIELDS
                or name in mesh_history.collision_surface.ARRAY_FIELDS
                # This chart-only fixture does not enable optional source histories.
                or name in mesh_history.lip_events.ARRAY_FIELDS
                or name in mesh_history.eclogite_sink.ARRAY_FIELDS
                or name in mesh_history.ENHANCED_RIFT_DTYPES
                # This chart-only fixture does not enable local welding.
                or name in mesh_history.localized_accretion.ARRAY_FIELDS
                # Nor does it declare a coastal reference or a numerical accuracy policy.
                or name in mesh_history.COAST_DTYPES
                or name in mesh_history.numerical_accuracy.ARRAY_FIELDS):
            continue
        n = len(result['material_faces']) if name.startswith('material_') else len(result['mesh_faces'])
        if name.startswith('native_boundary_'):
            result[name] = np.zeros((0, 2) if name == 'native_boundary_edges' else (0,), dtype)
        else:
            result[name] = np.zeros(n, dtype)
    result['material_reference_area_km2'][:] = 1.
    return result


class MaterialTransportTests(unittest.TestCase):
    def test_nonrigid_vertices_move_material_by_actual_triangle_map(self):
        first = tracked(); second = deepcopy(first)
        second['material_vertices'][1] = unit(second['material_vertices'][1]+[0., .11, .04])
        faces = np.array([0, 0, 1]); weights = np.array([[.2, .3, .5], [.4, .5, .1], [.2, .6, .2]])
        moved = transport.map_hits(transport.prepare(first), transport.prepare(second), faces, weights)
        np.testing.assert_allclose(moved['xyz'], query(second, faces, weights), atol=3e-16)
        self.assertGreater(np.linalg.norm(moved['xyz'][0]-query(first, faces, weights)[0]), .02)

    def test_spherical_refinement_preserves_points_not_just_face_centroids(self):
        first = tracked(); second, _ = refine(first)
        faces = np.repeat([0, 1], 50)
        weights = np.random.default_rng(6).dirichlet([1., 1., 1.], len(faces))
        points = query(first, faces, weights)
        moved = transport.map_hits(transport.prepare(first), transport.prepare(second), faces, weights)
        np.testing.assert_allclose(moved['xyz'], points, atol=4e-16)
        # Removing the radial scale creates the subtle remesh-only drift this
        # contract prevents, even though all original vertices stay fixed.
        wrong = deepcopy(second)
        wrong['material_reference_corners'] /= wrong['material_reference_corners'].sum(axis=2)[:, :, None]
        bad = transport.map_hits(transport.prepare(first), transport.prepare(wrong), faces, weights)
        self.assertGreater(np.linalg.norm(bad['xyz']-points, axis=1).max(), .0001)

    def test_repeated_refinement_and_registered_parent_return_preserve_chart(self):
        first = tracked(); second, _ = refine(first); third, _ = refine(second)
        faces = np.array([0, 1]); weights = np.array([[.19, .3, .51], [.2, .6, .2]])
        step = transport.map_hits(transport.prepare(first), transport.prepare(third), faces, weights)
        back = transport.map_hits(transport.prepare(third), transport.prepare(first), step['face'], step['weights'])
        np.testing.assert_allclose(back['xyz'], query(first, faces, weights), atol=5e-16)
        np.testing.assert_allclose(back['weights'], weights, atol=5e-16)

    def test_polar_rotation_equivariance_of_deformation_and_refinement(self):
        first = tracked(); second, _ = refine(first)
        second['material_vertices'][5] = unit(second['material_vertices'][5]+[.01, .015, .03])
        faces = np.array([0, 1]); weights = np.array([[.2, .3, .5], [.4, .2, .4]])
        angles = dict(yaw=179., pitch=90., roll=-17.); matrix = rotation_matrix(angles)
        a = transport.map_hits(transport.prepare(first), transport.prepare(second), faces, weights)
        b = transport.map_hits(transport.prepare(orient_frame(first, angles)),
                               transport.prepare(orient_frame(second, angles)), faces, weights)
        np.testing.assert_allclose(b['xyz'], a['xyz']@matrix, atol=4e-16)
        np.testing.assert_array_equal(a['face'], b['face'])

    def test_overlapping_disconnected_roots_never_substitute_for_missing_material(self):
        first = tracked(frame(centers=[(0., 0., 1.), (0., 0., 1.)]))
        second = deepcopy(first)
        second['material_vertices'][3:] = unit(second['material_vertices'][3:]+[.03, 0., 0.])
        faces = np.array([0, 1]); weights = np.full((2, 3), 1/3)
        result = transport.map_hits(transport.prepare(first), transport.prepare(second), faces, weights)
        np.testing.assert_array_equal(result['face'], faces)
        self.assertGreater(np.linalg.norm(np.diff(result['xyz'], axis=0)), .02)
        second['material_root_id'][:] = 100
        with self.assertRaisesRegex(ValueError, 'missing material root 101'):
            transport.map_hits(transport.prepare(first), transport.prepare(second), faces, weights)

    def test_missing_reference_region_fails_and_shared_edge_tie_is_stable(self):
        first = tracked(); second, _ = refine(first)
        source_face = np.array([0]); weights = np.array([[.5, .5, 0.]])
        result = transport.map_hits(transport.prepare(first), transport.prepare(second), source_face, weights)
        inverse_order = np.arange(len(second['material_faces']))[::-1]
        reordered = deepcopy(second)
        for key in ('material_faces', 'material_root_id', 'material_reference_corners',
                    'material_face_id', 'material_erosion_total_m'):
            reordered[key] = reordered[key][inverse_order]
        other = transport.map_hits(transport.prepare(first), transport.prepare(reordered), source_face, weights)
        self.assertEqual(second['material_face_id'][result['face'][0]], reordered['material_face_id'][other['face'][0]])
        removed = deepcopy(first)
        removed['material_reference_corners'][0] = [[1., 0., 0.], [.5, .5, 0.], [.5, 0., .5]]
        with self.assertRaisesRegex(ValueError, 'missing reference-chart region'):
            transport.map_hits(transport.prepare(first), transport.prepare(removed), source_face, [[.1, .8, .1]])

    def test_empty_surface_and_signed_cumulative_net_erosion_are_valid(self):
        empty = tracked(frame(centers=[]))
        actual = transport.map_hits(transport.prepare(empty), transport.prepare(empty), np.empty(0, int), np.empty((0, 3)))
        self.assertEqual(actual['xyz'].shape, (0, 3))
        first = tracked(); first['material_erosion_total_m'] = np.array([-10., 90.])
        np.testing.assert_array_equal(transport.prepare(first)['erosion_total_m'], [-10., 90.])

    def test_schema_requires_version_complete_finite_invertible_homogeneous_charts(self):
        good, _ = refine(tracked())
        source = complete_schema(good)
        source['material_parent_id'] = np.full(len(source['material_faces']), -1, np.int64)
        saved = mesh_history.arrays(source)
        np.testing.assert_array_equal(saved['material_reference_corners'], good['material_reference_corners'])
        for kind in ('missing', 'unflagged', 'singular', 'negative', 'nonfinite', 'wrongshape'):
            with self.subTest(kind=kind):
                bad = deepcopy(source)
                if kind == 'missing': del bad['material_erosion_total_m']
                if kind == 'unflagged': del bad['material_transport_version']
                if kind == 'singular': bad['material_reference_corners'][0, 0] = bad['material_reference_corners'][0, 1]
                if kind == 'negative': bad['material_root_id'][0] = -1
                if kind == 'nonfinite': bad['material_reference_corners'][0, 0, 0] = np.nan
                if kind == 'wrongshape': bad['material_reference_corners'] = bad['material_reference_corners'][:, 0]
                with self.assertRaises(ValueError): mesh_history.arrays(bad)
                with self.assertRaises(ValueError): sampling.prepare(bad)
        old = complete_schema(continuous_frame())
        self.assertFalse(set(mesh_history.TRANSPORT_DTYPES).intersection(mesh_history.arrays(old)))

    def test_pure_remesh_has_zero_physical_forcing_despite_changed_display_reconstruction(self):
        first = tracked(); second, _ = refine(first); second['time_myr'] = 2.
        faces = np.array([0, 0, 1, 1]); weights = np.array([[.2, .2, .6], [.6, .3, .1], [.7, .2, .1], [.1, .3, .6]])
        points = query(first, faces, weights)
        fields, info = export.interval_forcing(points, first, second, {'dt_myr': 2.})
        np.testing.assert_allclose(fields['hdisp'], 0., atol=1e-14)
        np.testing.assert_array_equal(fields['upsub'], 0.)
        np.testing.assert_array_equal(fields['relaxation_correction'], 0.)
        self.assertGreater(np.abs(fields['display_geometric_rate']).max()*2e6, 10.)
        np.testing.assert_array_equal(fields['display_geometric_rate'], fields['reconstruction_difference_rate'])
        self.assertEqual(info['material_transport_version'], 1)

    def test_erosion_correction_follows_new_face_ids_without_markers_or_rate_fallback(self):
        first = tracked(); first['material_erosion_total_m'] = np.array([80., 110.])
        second, mesh = refine(first); second['time_myr'] = 2.
        loss = np.array([60., 90.])[mesh['source_face']]
        second['material_height_m'] -= loss
        second['material_erosion_total_m'] += loss
        for source in (first, second):
            for field in list(source):
                if field.startswith('trace_'): del source[field]
        points = query(first, np.array([0, 1]), np.full((2, 3), 1/3))
        fields, info = export.interval_forcing(points, first, second, {'dt_myr': 2.})
        np.testing.assert_array_equal(fields['upsub'], 0.)
        np.testing.assert_allclose(fields['geometric_rate']*2e6, [-60., -90.], atol=1e-5)
        np.testing.assert_allclose(fields['relaxation_correction']*2e6, [60., 90.], atol=1e-5)
        np.testing.assert_array_equal(fields['correction_supported'], 1)
        self.assertEqual(info['relaxation_fallback_nodes'], 0)

    def test_nonrigid_transport_and_real_column_gain_survive_scalar_reconstruction(self):
        first = tracked(); second, _ = refine(first); second['time_myr'] = 2.
        second['material_vertices'] = unit(second['material_vertices']+[0., .03, .02])
        second['material_height_m'] += 250.
        faces = np.array([0, 1]); weights = np.array([[.2, .3, .5], [.4, .3, .3]])
        points = query(first, faces, weights)
        mapped = transport.map_hits(transport.prepare(first), transport.prepare(second), faces, weights)
        fields, _ = export.interval_forcing(points, first, second, {})
        np.testing.assert_allclose(fields['hdisp'], (mapped['xyz']-points)*export.RADIUS_M/2e6, atol=1e-8)
        np.testing.assert_allclose(fields['upsub'], 250/2e6, atol=1e-11)

    def test_instantaneous_thermal_change_is_counted_once_on_the_corresponding_material(self):
        first = tracked(frame(time=2.)); first['ridge_episodes'] = [dict(
            started_myr=0., peak_myr=2., end_myr=12., center=[0., 0., 1.],
            radius_km=1000., thermal_peak_m=500., volcanic_budget_m=200., overriding_plate_uid=20)]
        second = deepcopy(first); second['time_myr'] = 4.
        points = np.array([[0., 0., 1.]])
        before = sampling.sample_frame(first, points)['thermal_support_m']
        after = sampling.sample_frame(second, points)['thermal_support_m']
        fields, _ = export.interval_forcing(points, first, second, {})
        self.assertLess(after[0], before[0])
        np.testing.assert_allclose(fields['upsub']*2e6, after-before, atol=1e-5)
        np.testing.assert_array_equal(fields['relaxation_correction'], 0.)

    def test_new_ocean_motion_uses_recorded_euler_even_when_markers_are_nonrigid(self):
        first = tracked(); second = deepcopy(first); second['time_myr'] = 2.
        first['trace_plate_uid'][:] = 10; second['trace_plate_uid'][:] = 10
        second['trace_xyz'] = unit(second['trace_xyz']+[0., .25, .1])
        second['plates'][0]['angular_velocity'] = [0., .02, 0.]
        points = np.array([[0., 0., -1.]])
        fields, info = export.interval_forcing(points, first, second, {'dt_myr': 2.})
        moved = points@export.rotation([0., .02, 0.], 2.)
        np.testing.assert_allclose(fields['hdisp'], (moved-points)*export.RADIUS_M/2e6, atol=5e-9)
        self.assertTrue(all(row['method'] == 'recorded_euler_approximation' for row in info['motion']))

    def test_mixed_transport_versions_fail_explicitly(self):
        first = tracked(); second = deepcopy(first); second['time_myr'] = 2.
        for key in ('material_transport_version', *transport.FIELDS): del second[key]
        with self.assertRaisesRegex(ValueError, 'mix material transport versions'):
            export.interval_forcing(np.array([[0., 0., 1.]]), first, second, {})

    def test_actual_package_keeps_continuous_initial_final_terrain_and_declares_budget(self):
        first = tracked(); second, _ = refine(first); second['time_myr'] = 2.
        angles = dict(yaw=42., pitch=65., roll=-17.)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); run = root/'run'; run.mkdir()
            for index, source in enumerate((first, second)): write_frame(run, index, source)
            manifest = dict(run_id='deforming-contract', config={'dt_myr': 2.},
                            frames=[{'time_myr': 0.}, {'time_myr': 2.}])
            with patch.object(export, 'sample', side_effect=AssertionError('No review-grid lookup')):
                meta = export.build_history(run, root/'out', manifest, subdivisions=2,
                                             dt_years=100000, orientation=angles)
            self.assertTrue(validate_package(root/'out')['passed'])
            vertices, _ = export.icosphere(2); points = vertices@rotation_matrix(angles).T
            with np.load(root/'out'/'input'/'mesh.npz') as data:
                np.testing.assert_array_equal(data['z'], sampling.sample_frame(first, points)['elevation'].astype(np.float32))
            with np.load(root/'out'/'input'/'forcing_0000.npz') as data:
                np.testing.assert_array_equal(data['reference_z_end'], sampling.sample_frame(second, points)['elevation'].astype(np.float32))
                np.testing.assert_allclose(data['upsub'], data['geometric_rate']+data['relaxation_correction'], atol=1e-10)
            self.assertEqual(meta['material_transport_version'], 1)
            self.assertIn('material_transport.py', meta['native_sampling_sources_sha256'])
            self.assertIn('refinement alone contributes zero', (root/'out'/'README.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
