"""Conserved face-local magma additions after rejected juvenile lateral growth."""
from copy import deepcopy
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT)]

import arc_birth_profile as profile
import arc_emplacement_geometry as geometry
import arc_local_deposition as deposition
import checkpoint
import crust_inventory as inventory
import crustal_structure as columns
import dense_crust
import native_arc_material as arcs
import raster_engine
from test_arc_birth_footprint import world
from test_arc_source_geometry import add, positions_in_cell


def source_point(state, face):
    return arcs._unit(state.material_surface['vertices'][state.material_surface['faces'][face]].sum(axis=0))[None]


def traced_face(state):
    return int(np.flatnonzero(state.parcel_patch == state.trace_patch[0])[0])


def component(state):
    return np.arange(len(state.mass), dtype=int)


def grades(state):
    surface = state.material_surface
    return profile.measure(surface['vertices'], surface['faces'], surface['area_km2'],
                           columns.elevation(state.structure), state.parcel_arc_basal_m)['face_grade']


def material_arrays(state):
    """Physical material state, excluding additive diagnostic/source records."""
    result = {}
    for name, value in vars(state).items():
        if isinstance(value, np.ndarray):
            result[name] = value.copy()
    for name in ('structure', 'trace_structure', 'material_surface', 'material_lineage'):
        for key, value in getattr(state, name, {}).items():
            if isinstance(value, np.ndarray):
                result[name + '.' + key] = value.copy()
    return result


def fallback(state, face, amount, *, owners=None):
    # Force only the old lateral mesh proposal to fail. Admission, source routing,
    # deposition capacity, source bookkeeping and material commit remain real.
    with patch.object(arcs, '_grow_geometry', return_value=None):
        return add(state, source_point(state, face), [amount], owners=owners)


class ArcLocalDepositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.initial = world()
        add(cls.initial, positions_in_cell(cls.initial)[:1], [10000.])
        cls.initial.native_arc_deposition_version = 1
        assert len(cls.initial.mass) == 24
        cls.tight = world()
        add(cls.tight, positions_in_cell(cls.tight)[:1], [1000.])
        cls.tight.native_arc_deposition_version = 1

    def setUp(self):
        self.state = deepcopy(self.initial)

    def assert_fixed_geometry_and_reference(self, before, state):
        for name in ('mass', 'parcel_patch', 'parcel_plate', 'parcel_arc_id',
                     'trace_patch', 'trace_xyz', 'trace_rift_tangent'):
            np.testing.assert_array_equal(getattr(state, name), getattr(before, name), err_msg=name)
        for name in ('vertices', 'faces', 'area_km2', 'reference_area_km2',
                     'face_id', 'face_owner', 'face_kind'):
            np.testing.assert_array_equal(state.material_surface[name],
                                          before.material_surface[name], err_msg=name)
        for source in ('structure', 'trace_structure'):
            for name in ('area_factor', 'reference_thickness_km', 'reference_elevation_m',
                         'denudation_m', 'rebound_m', 'thermal_uplift_m',
                         'thermal_subsidence_m'):
                np.testing.assert_array_equal(getattr(state, source)[name],
                                              getattr(before, source)[name], err_msg=source + '.' + name)

    def test_positive_deposition_keeps_geometry_reference_and_other_faces_exact(self):
        s = self.state
        face = traced_face(s)
        before = deepcopy(s)
        plan = deposition.propose(s, component(s), np.array([face]), np.array([1.]))
        self.assertGreater(plan['accepted_area_km2'], 0.)
        deposition.commit(s, plan)
        self.assert_fixed_geometry_and_reference(before, s)
        delta = s.structure['thickness_km'] - before.structure['thickness_km']
        untouched = np.arange(len(s.mass)) != face
        np.testing.assert_array_equal(delta[untouched], 0.)
        actual = float(s.material_surface['area_km2'] @ delta)
        self.assertAlmostEqual(actual, 25. * plan['accepted_area_km2'], places=7)
        self.assertAlmostEqual(plan['added_volume_km3'], actual, places=7)
        expected_uplift = columns.elevation(s.structure) - columns.elevation(before.structure)
        np.testing.assert_allclose(s.structure['magmatic_uplift_m']
                                   - before.structure['magmatic_uplift_m'],
                                   expected_uplift, rtol=0., atol=1e-10)
        added = s.structure['added_volume_km_per_reference_km2'] - before.structure['added_volume_km_per_reference_km2']
        self.assertAlmostEqual(float(s.mass @ added), actual, places=7)
        self.assertTrue(np.all((s.structure['thickness_km'] >= 8.) & (s.structure['thickness_km'] <= 75.)))
        self.assertTrue(np.all(grades(s) <= np.maximum(profile.MAX_GRADE, grades(before)) + 1e-10))

    def test_trace_uses_its_own_reference_area_and_preserves_old_inventory(self):
        s = self.state
        face = traced_face(s)
        # Trace and parcel reference charts need not have the same area factor.
        s.trace_structure['area_factor'] *= .7
        for fields in (s.structure, s.trace_structure):
            inventory.initialize(fields)
            dense_crust.initialize(fields, 650.)
            fields['denudation_m'][:] = 23.
            fields['rebound_m'][:] = 11.
        before = deepcopy(s)
        plan = deposition.propose(s, component(s), np.array([face]), np.array([1.]))
        self.assertGreater(plan['accepted_area_km2'], 0.)
        deposition.commit(s, plan)
        self.assert_fixed_geometry_and_reference(before, s)
        host_delta = s.structure['thickness_km'][face] - before.structure['thickness_km'][face]
        trace = np.flatnonzero(s.trace_patch == s.parcel_patch[face])
        np.testing.assert_allclose(s.trace_structure['thickness_km'][trace]
                                   - before.trace_structure['thickness_km'][trace],
                                   host_delta, rtol=0., atol=1e-12)
        for name, ids in (('structure', np.array([face])), ('trace_structure', trace)):
            fields, old = getattr(s, name), getattr(before, name)
            increment = host_delta * old['area_factor'][ids]
            np.testing.assert_allclose(fields[inventory.ADDED][ids] - old[inventory.ADDED][ids],
                                       .6 * increment, rtol=2e-10, atol=1e-12)
            np.testing.assert_allclose(fields[inventory.REMAINING][ids] - old[inventory.REMAINING][ids],
                                       .6 * increment, rtol=2e-10, atol=1e-12)
            for key in (dense_crust.HEAT, dense_crust.HEAT_ADDED):
                np.testing.assert_allclose(fields[key][ids] - old[key][ids],
                                           1200. * increment, rtol=2e-10, atol=1e-9)
            for key in (inventory.BASELINE, inventory.ERODED, inventory.RETURNED,
                        dense_crust.DENSE, dense_crust.CONVERTED, dense_crust.ERODED,
                        dense_crust.RETURNED, dense_crust.HEAT_BASELINE,
                        dense_crust.HEAT_ERODED, dense_crust.HEAT_RETURNED, dense_crust.HEAT_BATH):
                np.testing.assert_array_equal(fields[key], old[key], err_msg=name + '.' + key)
            inventory.validate(fields)
            dense_crust.validate(fields)

    def test_plan_is_pure_and_stale_column_plan_cannot_partially_commit(self):
        s = self.state
        face = traced_face(s)
        before = pickle.dumps(s, protocol=5)
        plan = deposition.propose(s, component(s), np.array([face]), np.array([1.]))
        self.assertEqual(pickle.dumps(s, protocol=5), before)
        s.structure['thickness_km'][face] += .01
        before_commit = material_arrays(s)
        with self.assertRaises(ValueError):
            deposition.commit(s, plan)
        for key, value in material_arrays(s).items():
            np.testing.assert_array_equal(value, before_commit[key], err_msg=key)

    def test_trace_column_limit_rejects_before_parcel_or_geometry_changes(self):
        s = self.state
        face = traced_face(s)
        trace = s.trace_patch == s.parcel_patch[face]
        s.trace_structure['thickness_km'][trace] = 75.
        before = material_arrays(s)
        plan = deposition.propose(s, component(s), np.array([face]), np.array([100.]))
        self.assertEqual(plan['accepted_area_km2'], 0.)
        deposition.commit(s, plan)
        for key, value in material_arrays(s).items():
            np.testing.assert_array_equal(value, before[key], err_msg=key)

    def test_one_origin_cannot_spend_another_faces_available_capacity(self):
        s = self.state
        full = traced_face(s)
        available = int(np.argmin(s.structure['thickness_km']))
        self.assertNotEqual(full, available)
        s.structure['thickness_km'][full] = 75.
        plan = deposition.propose(s, component(s), np.array([full, available]), np.array([1., 1.]))
        spent = np.asarray(plan['spent_area_km2'])
        self.assertEqual(spent[0], 0.)
        self.assertGreater(spent[1], 0.)
        before = deepcopy(s)
        deposition.commit(s, plan)
        self.assertEqual(s.structure['thickness_km'][full], before.structure['thickness_km'][full])
        affected = np.flatnonzero(s.structure['thickness_km'] != before.structure['thickness_km'])
        np.testing.assert_array_equal(affected, [available])

    def test_oversteep_source_face_cannot_report_roundoff_only_progress(self):
        s = deepcopy(self.tight)
        face = int(np.argmax(s.structure['thickness_km']))
        s.structure['thickness_km'][face] = 74.
        self.assertGreater(float(grades(s).max()), profile.MAX_GRADE)
        before = material_arrays(s)
        plan = deposition.propose(s, component(s), np.array([face]), np.array([100.]))
        self.assertEqual(plan['accepted_area_km2'], 0.)
        self.assertEqual(plan['added_volume_km3'], 0.)
        deposition.commit(s, plan)
        for key, value in material_arrays(s).items():
            np.testing.assert_array_equal(value, before[key], err_msg=key)

    def test_public_failed_growth_deposits_with_conserved_pending_origins(self):
        s = self.state
        face = traced_face(s)
        before = deepcopy(s)
        report = fallback(s, face, 1e5)
        self.assertGreater(report['added_volume_km3'], 0.)
        self.assertGreater(report['pending_magma_volume_km3'], 0.)
        self.assert_fixed_geometry_and_reference(before, s)
        row = next(row for row in report['emplacement_geometry']['sources'] if row['mode'] == 'deposition')
        placed = float(np.sum(row['source_placed_area_km2']))
        pending = float(np.sum(row['source_pending_area_km2']))
        self.assertAlmostEqual(placed + pending, 1e5, places=7)
        actual = float(s.material_surface['area_km2'] @
                       (s.structure['thickness_km'] - before.structure['thickness_km']))
        self.assertAlmostEqual(actual, 25. * placed, places=6)
        self.assertAlmostEqual(report['added_volume_km3'], actual, places=6)
        np.testing.assert_array_equal(s.native_arc_pending['xyz'], source_point(s, face))
        origin = row['source_origin_ids'][0]
        self.assertEqual(s.native_arc_pending['source_provenance'][0]['origin_id'], origin)
        placed_rows = s.native_arc_source_placements[len(before.native_arc_source_placements):]
        self.assertTrue(placed_rows)
        self.assertTrue(all(row['mode'] == 'deposition' for row in placed_rows))
        self.assertAlmostEqual(sum(row['volume_km3'] for row in placed_rows), actual, places=6)
        frame = dict(arcs.snapshot_fields(s), arc_material_diagnostics=report, time_myr=float(s.t))
        profile.validate_frame(frame)
        geometry.validate_frame(frame)

    def test_absent_version_preserves_failed_growth_pending_behavior(self):
        s = self.state
        del s.native_arc_deposition_version
        before = deepcopy(s)
        report = fallback(s, traced_face(s), 1.)
        self.assertEqual(report['added_volume_km3'], 0.)
        self.assertEqual(report['pending_magma_volume_km3'], 25.)
        self.assert_fixed_geometry_and_reference(before, s)
        np.testing.assert_array_equal(s.structure['thickness_km'], before.structure['thickness_km'])
        self.assertNotIn('arc_deposition_version', arcs.snapshot_fields(s))

    def test_foreign_or_continental_containing_material_never_receives_deposition(self):
        for obstruction in ('foreign', 'continental'):
            with self.subTest(obstruction=obstruction):
                s = deepcopy(self.initial)
                face = traced_face(s)
                if obstruction == 'foreign':
                    s.parcel_plate[:] = 1
                else:
                    s.kind[:] = 1
                before = deepcopy(s)
                report = fallback(s, face, 1., owners=[0])
                self.assertEqual(report['added_volume_km3'], 0.)
                self.assertEqual(report['rejected_area_km2'], 1.)
                np.testing.assert_array_equal(s.structure['thickness_km'], before.structure['thickness_km'])
                np.testing.assert_array_equal(s.mass, before.mass)
                np.testing.assert_array_equal(s.material_surface['vertices'], before.material_surface['vertices'])



    def test_source_permutation_keeps_physical_transaction_identical(self):
        faces = np.array([traced_face(self.state), int(np.argmin(self.state.structure['thickness_km']))])
        amounts = np.array([1., 2.])
        first, second = deepcopy(self.state), deepcopy(self.state)
        a = deposition.propose(first, component(first), faces, amounts)
        b = deposition.propose(second, component(second), faces[::-1], amounts[::-1])
        np.testing.assert_array_equal(a['spent_area_km2'], np.asarray(b['spent_area_km2'])[::-1])
        deposition.commit(first, a)
        deposition.commit(second, b)
        other = material_arrays(second)
        for key, value in material_arrays(first).items():
            np.testing.assert_array_equal(value, other[key], err_msg=key)

    def test_incomplete_slope_component_is_rejected_without_mutation(self):
        s = self.state
        face = traced_face(s)
        before = pickle.dumps(s, protocol=5)
        with self.assertRaises(ValueError):
            deposition.propose(s, np.array([face]), np.array([face]), np.array([1.]))
        self.assertEqual(pickle.dumps(s, protocol=5), before)

    def test_same_identity_disconnected_component_keeps_its_columns_and_traces(self):
        s = world()
        positions = s.xyz[[17, 180]].copy()
        add(s, positions, [10000., 10000.])
        self.assertEqual(len(s.mass), 48)
        s.native_arc_deposition_version = 1
        s.parcel_arc_id[:] = 1
        selected = np.arange(24)
        before = deepcopy(s)
        plan = deposition.propose(s, selected, np.array([0]), np.array([1.]))
        self.assertGreater(plan['accepted_area_km2'], 0.)
        deposition.commit(s, plan)
        self.assert_fixed_geometry_and_reference(before, s)
        for key in s.structure:
            np.testing.assert_array_equal(s.structure[key][24:], before.structure[key][24:], err_msg=key)
        remote_trace = np.isin(s.trace_patch, s.parcel_patch[24:])
        for key in s.trace_structure:
            np.testing.assert_array_equal(s.trace_structure[key][remote_trace],
                                          before.trace_structure[key][remote_trace], err_msg=key)

    def test_saved_receipts_reject_nonlocal_or_unfunded_deposition(self):
        s = self.state
        report = fallback(s, traced_face(s), 1.)
        self.assertGreater(report['added_volume_km3'], 0.)
        frame = dict(arcs.snapshot_fields(s), arc_material_diagnostics=report, time_myr=float(s.t))
        geometry.validate_frame(frame)
        def receipt(value):
            return next(row for row in value['arc_material_diagnostics']['emplacement_geometry']['sources']
                        if row['mode'] == 'deposition')['column_deposition']
        def decision(value):
            return next(row for row in value['arc_material_diagnostics']['emplacement_geometry']['sources']
                        if row['mode'] == 'deposition')
        changes = [
            ('missing version', lambda f: f.pop('arc_deposition_version')),
            ('foreign receiving face', lambda f: receipt(f)['receiving_faces'].__setitem__(0, 99999)),
            ('forged receiving ID', lambda f: receipt(f)['receiving_face_ids'].__setitem__(0, 99999)),
            ('changed footprint', lambda f: receipt(f).update(footprint_area_change_km2=1.)),
            ('changed reference', lambda f: receipt(f).update(reference_area_change_km2=1.)),
            ('unfunded volume', lambda f: receipt(f)['face_volume_added_km3'].__setitem__(0, 1e9)),
            ('column above cap', lambda f: receipt(f)['face_thickness_after_km'].__setitem__(0, 75.01)),
            ('legacy mode bypass', lambda f: decision(f).update(mode='growth')),
            ('geometry changed', lambda f: receipt(f).update(geometry_unchanged=False)),
            ('deposited area aggregate', lambda f: f['arc_material_diagnostics'].update(deposited_source_area_km2=0.)),
            ('deposited volume aggregate', lambda f: f['arc_material_diagnostics'].update(deposited_volume_km3=0.)),
            ('deposited patch aggregate', lambda f: f['arc_material_diagnostics'].update(deposited_patches=0)),
            ('invented reference mass', lambda f: f['arc_material_diagnostics'].update(physical_mass_added_km2=1.)),
            ('unclosed reference mass', lambda f: f['arc_material_diagnostics'].update(physical_mass_residual_km2=1.)),
            ('relaxed frozen slope', lambda f: decision(f)['profile_capacity'].update(maximum_allowed_grade=9.)),

        ]
        for reason, mutate in changes:
            with self.subTest(reason=reason):
                altered = deepcopy(frame)
                mutate(altered)
                with self.assertRaises(ValueError):
                    geometry.validate_frame(altered)
                    profile.validate_frame(altered)
        altered = deepcopy(frame)
        added = next(row for row in reversed(altered['arc_source_placements']) if row['mode'] == 'deposition')
        added['receiving_face_id'] = -1
        with self.assertRaises(ValueError):
            profile.validate_frame(altered)


    def test_deposition_preserves_cumulative_reference_closure_and_continental_retention(self):
        s = self.state
        # Synthetic ledger baseline: retained reference consists of 4000 units
        # of original material and the rest of prior juvenile reference growth.
        # This tests the inherited snapshot statistic without advancing a world.
        s.original_mass = 4000.
        s.process_totals['arc_added_km2'] = float(s.mass.sum()) - s.original_mass
        s.process_totals['arc_deposited_source_area_km2'] = 0.
        s.process_totals['arc_deposited_volume_km3'] = 0.
        before = raster_engine.Simulation.snapshot(s)['stats']
        self.assertAlmostEqual(before['continental_mass_retained_fraction'], 1.)
        original_reference = s.mass.copy()
        accepted = 0.
        for _ in range(2):
            report = fallback(s, traced_face(s), 1.)
            accepted += report['deposited_source_area_km2']
            self.assertGreater(report['deposited_volume_km3'], 0.)
        np.testing.assert_array_equal(s.mass, original_reference)
        self.assertAlmostEqual(s.process_totals['arc_deposited_source_area_km2'], accepted, places=9)
        self.assertAlmostEqual(s.process_totals['arc_deposited_volume_km3'], 25. * accepted, places=7)
        created_reference = (s.process_totals['arc_added_km2']
                             - s.process_totals['arc_deposited_source_area_km2'])
        self.assertAlmostEqual(float(s.mass.sum()), s.original_mass + created_reference, places=8)
        after = raster_engine.Simulation.snapshot(s)['stats']
        self.assertEqual(after['continental_mass_retained_fraction'],
                         before['continental_mass_retained_fraction'])

    def test_cumulative_frame_counters_must_match_enduring_deposition_origins(self):
        s = self.state
        report = fallback(s, traced_face(s), 1.)
        self.assertGreater(report['deposited_volume_km3'], 0.)
        frame = dict(arcs.snapshot_fields(s), arc_material_diagnostics=report,
                     stats=deepcopy(s.process_totals), time_myr=float(s.t))
        profile.validate_frame(frame)
        for field in ('arc_deposited_source_area_km2', 'arc_deposited_volume_km3'):
            for corruption in ('missing', 'zero', 'too large'):
                with self.subTest(field=field, corruption=corruption):
                    altered = deepcopy(frame)
                    if corruption == 'missing':
                        altered['stats'].pop(field)
                    else:
                        altered['stats'][field] = 0. if corruption == 'zero' else 1e9
                    with self.assertRaises(ValueError):
                        profile.validate_frame(altered)

    def test_unversioned_positive_deposition_totals_are_rejected_but_zero_is_compatible(self):
        s = self.state
        del s.native_arc_deposition_version
        report = deepcopy(s.arc_material_diagnostics)
        report.update(deposited_source_area_km2=0., deposited_volume_km3=0., deposited_patches=0)
        frame = dict(arcs.snapshot_fields(s), arc_material_diagnostics=report, time_myr=float(s.t))
        geometry.validate_frame(frame)
        for field, value in (('deposited_source_area_km2', .001),
                             ('deposited_volume_km3', .025), ('deposited_patches', 1)):
            with self.subTest(field=field):
                altered = deepcopy(frame)
                altered['arc_material_diagnostics'][field] = value
                with self.assertRaises(ValueError):
                    geometry.validate_frame(altered)
        frame['stats'] = {'arc_deposited_source_area_km2': 0., 'arc_deposited_volume_km3': 0.}
        profile.validate_frame(frame)
        for field in frame['stats']:
            with self.subTest(field=field):
                altered = deepcopy(frame)
                altered['stats'][field] = 1.
                with self.assertRaises(ValueError):
                    profile.validate_frame(altered)

    def test_checkpoint_continuation_repeats_deposition_and_source_receipts(self):
        s = self.state
        face = traced_face(s)
        fallback(s, face, 1.)
        compatibility = dict(engine_sha256='local-deposition-fixture',
                             auxiliary_sources_sha256={}, numpy_version=np.__version__)
        manifest = dict(run_id='local-deposition-fixture', config=s.config,
                        frames=[], frame_count=0, state='paused')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.npz'
            checkpoint.write_checkpoint(path, s, manifest, compatibility)
            restored, _ = checkpoint.read_checkpoint(path, compatibility, type(s))
        self.assertEqual(restored.native_arc_deposition_version, 1)
        for state in (s, restored):
            fallback(state, face, 1.)
        for key, value in material_arrays(s).items():
            np.testing.assert_array_equal(value, material_arrays(restored)[key], err_msg=key)
        self.assertEqual(s.native_arc_source_placements, restored.native_arc_source_placements)
        self.assertEqual(s.arc_material_diagnostics, restored.arc_material_diagnostics)


if __name__ == '__main__':
    unittest.main(verbosity=2)
