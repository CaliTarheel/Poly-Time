"""Analytic local-stack oracles, including nonuniform cover and triple overlap."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

import burial_depth
import eclogite_sink as sink
import mesh_coverage
import checkpoint
import structure_engine
from deforming_regions import IncompleteContactStepError
from tests.test_collision_surface_acceptance import world, triangle, triangle_area, unit


def halves():
    a, b, c = triangle()
    midpoint = unit(a+b)
    return np.array([a, midpoint, c]), np.array([midpoint, b, c])


def prepare(specs, relations):
    s = world(specs, relations=relations)
    s.foundering_version = 1
    s.trace_kind = np.empty(0, np.uint8)
    structure_engine.initialize_traces(s, np.empty(0, int))
    sink.ensure_fields(s)
    s.parcel_burial_myr[:] = 20.
    s.parcel_root_age_myr[:] = 20.
    s._collision_overlap = mesh_coverage.material_overlaps(
        s.material_surface['vertices'], s.material_surface['faces'], s.parcel_collision_sheet)
    return s


def split_cover(refined=(False, False, False)):
    left, right = halves()
    return prepare([(left, 10., 0., refined[0]), (right, 50., 0., refined[1]),
                    (triangle(), 20., 0., refined[2])], [(1, 3), (2, 3)])


def eligible(s):
    return sink.eligible_thickness_km(s, s.structure['thickness_km'])


def narrow_production_overlap():
    # Exact faces 65, 260 and 262 from the detached four-ocean 0->2 Myr trial.
    # The first two share sheet 1 and cover disjoint parts of the third face.
    triangles = np.array([
        [[.31729941023188474, .797382014419688, -.513325439995427],
         [.309000591474564, .8090590571346356, -.49994207318126804],
         [.3449631258650391, .7984584889767653, -.4934212025991202]],
        [[.34548604677179245, .7987688550140368, -.49255223859562713],
         [.3449631258650391, .7984584889767653, -.4934212025991202],
         [.309000591474564, .8090590571346356, -.49994207318126804]],
        [[.3449558395698185, .7984544942386708, -.4934327607453221],
         [.34547454117327475, .798766156480543, -.4925646847495424],
         [.34608622466087885, .7981002446600431, -.49321427855807154]]])
    surface = dict(vertices=triangles.reshape(-1, 3), faces=np.arange(9).reshape(3, 3),
                   area_km2=np.array([13531.813806820142, 697.3342685901349, 22.06204433736971]),
                   radius_km=6371.)
    s = SimpleNamespace(material_surface=surface, parcel_collision_sheet=np.array([1, 1, 6]),
                        parcel_burial_myr=np.full(3, 20.), parcel_root_age_myr=np.full(3, 20.))
    # Independent 72-digit Decimal clipping and solid-angle evaluation of the
    # normalized represented input vertices, not values from the float kernel.
    expected = np.array([3.9164432631787335e-6, .09543252101206291])
    return s, triangles, expected


class BurialIntegrationTests(unittest.TestCase):
    def test_self_overlap_rejects_the_complete_coupled_timestep(self):
        # Real coincident covers exercise the current batched geometry path.
        # Patching partition_face no longer intercepts integrate's session API.
        s = prepare([(triangle(), 20., 0., False), (triangle(), 20., 0., False),
                     (triangle(), 20., 0., False)], [(1, 3), (2, 3)])
        s.parcel_collision_sheet[1] = s.parcel_collision_sheet[0]
        upper = np.array([0, 1]); lower = np.array([2, 2])
        weight = float(s.material_surface['area_km2'][2])
        pairs = (upper, lower, np.full(2, weight), np.ones(2, int))
        with self.assertRaisesRegex(IncompleteContactStepError,
                                    'complete coupled timestep'):
            burial_depth.integrate(s, np.full(3, 20.), pairs,
                                   depth_km=50., heating_delay_myr=5.)

    def test_narrow_production_overlaps_match_independent_high_precision_areas(self):
        s, triangles, expected = narrow_production_overlap()
        overlap = mesh_coverage.material_overlaps(s.material_surface['vertices'],
            s.material_surface['faces'], s.parcel_collision_sheet)
        np.testing.assert_array_equal(overlap['first'], [0, 1])
        np.testing.assert_array_equal(overlap['second'], [2, 2])
        np.testing.assert_allclose(overlap['area_km2'], expected, rtol=1e-11, atol=1e-15)
        # Forward and reverse clipping exercise different arithmetic paths.
        # The old planes alone missed the second area by more than 1e-9 km2.
        for upper, oracle in zip(triangles[:2], expected):
            for first, second in ((upper, triangles[2]), (triangles[2], upper)):
                actual = mesh_coverage._polygon_area(mesh_coverage.clip_triangle(first, second), 6371.)
                self.assertAlmostEqual(actual, oracle, delta=1e-11*max(oracle, 1.))

    def test_narrow_burial_partition_closes_without_relaxing_or_bypassing_cache_check(self):
        s, _, expected = narrow_production_overlap()
        overlap = mesh_coverage.material_overlaps(s.material_surface['vertices'],
            s.material_surface['faces'], s.parcel_collision_sheet)
        thickness = np.array([35.07062586788924, 35.01286942477479, 35.04187003142728])
        pairs = (overlap['first'], overlap['second'], overlap['area_km2'], np.ones(2, int))
        value, scope, attribution = burial_depth.integrate(s, thickness, pairs,
            depth_km=50., heating_delay_myr=5.)
        expected_volume = expected@(thickness[:2]+thickness[2]-50.)
        self.assertAlmostEqual(float(s.material_surface['area_km2']@value), expected_volume, delta=3e-10)
        self.assertAlmostEqual(scope['covered_union_area_km2'], float(expected.sum()), delta=1e-11)
        self.assertLess(scope['maximum_region_area_relative_error'], 2e-10)
        self.assertAlmostEqual(float(attribution['eligible_pair_volume_km3'].sum()), expected_volume, delta=3e-10)
        corrupted = list(pairs)
        corrupted[2] = pairs[2].copy()
        corrupted[2][1] += 1e-8
        with self.assertRaisesRegex(ValueError, 'cached overlap'):
            burial_depth.integrate(s, thickness, corrupted, depth_km=50., heating_delay_myr=5.)

    def test_reported_average_threshold_case_is_10_km_instead_of_zero(self):
        s = split_cover()
        self.assertAlmostEqual(eligible(s)[0][-1], 0., places=10)
        sink.upgrade_depth_integration(s)
        value, scope, attribution = eligible(s)
        self.assertAlmostEqual(value[-1], 10., places=10)
        self.assertAlmostEqual(scope['eligible_km3']/triangle_area(triangle()), 10., places=10)
        self.assertEqual(scope['depth_integration_version'], 1)
        shares = sink._contact_shares(s, np.array([0., 0., 100.]), attribution)
        self.assertEqual(shares.get(1, 0.), 0.)
        self.assertAlmostEqual(shares[2], 100.)

    def test_refining_upper_lower_or_both_keeps_integrated_eligibility(self):
        total_area = triangle_area(triangle())
        for refined in ((True, False, False), (False, True, False),
                        (False, False, True), (True, True, True)):
            with self.subTest(refined=refined):
                s = split_cover(refined); sink.upgrade_depth_integration(s)
                value, scope, _ = eligible(s)
                lower = s.parcel_collision_sheet == 3
                actual = s.material_surface['area_km2'][lower]@value[lower]
                self.assertAlmostEqual(actual/total_area, 10., places=9)
                self.assertLess(scope['maximum_region_area_relative_error'], 1e-12)

    def test_triple_overlap_adds_local_depth_without_counting_area_twice(self):
        left, _ = halves()
        s = prepare([(left, 20., 0., False), (triangle(), 20., 0., False),
                     (triangle(), 20., 0., False)], [(1, 2), (2, 3)])
        sink.upgrade_depth_integration(s)
        value, scope, attribution = eligible(s)
        np.testing.assert_allclose(value, [0., 0., 5.], atol=2e-11)
        self.assertLessEqual(value[-1], 20.)
        self.assertEqual(scope['faces_with_cover_beyond_own_area'], 1)
        shares = sink._contact_shares(s, np.array([0., 0., 10.]), attribution)
        self.assertAlmostEqual(sum(shares.values()), 10.)

    def test_full_triple_cover_saturates_once_and_is_order_independent(self):
        s = prepare([(triangle(), 30., 0., False), (triangle(), 30., 0., False),
                     (triangle(), 20., 0., False)], [(1, 2), (2, 3)])
        sink.upgrade_depth_integration(s)
        value, _, _ = eligible(s)
        np.testing.assert_allclose(value, [0., 10., 20.], atol=1e-12)
        for name in ('first', 'second', 'area_km2'):
            s._collision_overlap[name] = s._collision_overlap[name][::-1].copy()
        np.testing.assert_allclose(eligible(s)[0], value, atol=1e-12)

    def test_uncovered_root_and_covered_heating_are_distinct(self):
        left, _ = halves()
        s = prepare([(left, 30., 0., False), (triangle(), 60., 0., False)], [(1, 2)])
        sink.upgrade_depth_integration(s)
        self.assertAlmostEqual(eligible(s)[0][-1], .5*40.+.5*10., places=10)
        s.parcel_burial_myr[:] = 0.
        self.assertAlmostEqual(eligible(s)[0][-1], 5., places=10)
        s.parcel_root_age_myr[:] = 0.
        np.testing.assert_array_equal(eligible(s)[0], 0.)
        s.parcel_burial_myr[:] = 20.
        self.assertAlmostEqual(eligible(s)[0][-1], 20., places=10)

    def test_common_rotation_preserves_local_stack_integrals(self):
        s = split_cover(); sink.upgrade_depth_integration(s)
        original = eligible(s)[0]
        axis = unit([.3, -.2, .8]); angle = .73
        cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
        rotation = np.eye(3)*np.cos(angle)+(1.-np.cos(angle))*np.outer(axis, axis)+np.sin(angle)*cross
        s.material_surface['vertices'] = s.material_surface['vertices']@rotation.T
        s._collision_overlap = mesh_coverage.material_overlaps(
            s.material_surface['vertices'], s.material_surface['faces'], s.parcel_collision_sheet)
        np.testing.assert_allclose(eligible(s)[0], original, atol=1e-10)

    def test_stale_pair_area_and_unordered_stack_are_rejected(self):
        s = split_cover(); s._collision_overlap['area_km2'] *= .5
        with self.assertRaisesRegex(ValueError, 'cached overlap'):
            sink.upgrade_depth_integration(s)
        self.assertFalse(hasattr(s, 'foundering_depth_version'))
        s = split_cover(); s.collision_contacts = []
        with self.assertRaisesRegex(ValueError, 'vertical order'):
            sink.upgrade_depth_integration(s)

    def test_explicit_migration_is_idempotent_and_changes_no_material(self):
        s = split_cover(); sink.upgrade_inventory(s)
        before = deepcopy(s.structure); clock = s.parcel_root_age_myr.copy()
        del s._collision_overlap  # migration computes missing geometry read-only
        report = sink.upgrade_depth_integration(s)
        self.assertEqual(sink.upgrade_depth_integration(s), report)
        self.assertFalse(report['heating_model_changed'])
        self.assertFalse(hasattr(s, '_collision_overlap'))
        for name in before: np.testing.assert_array_equal(s.structure[name], before[name])
        np.testing.assert_array_equal(s.parcel_root_age_myr, clock)

    def test_actual_deformation_uses_local_depth_and_closes_mantle_ledger(self):
        from tests.test_eclogite_sink import stack
        import crust_inventory
        s = stack(); sink.upgrade_inventory(s); sink.upgrade_depth_integration(s)
        value = eligible(s)[0]
        expected = np.minimum(value*(1.-np.exp(-2./20.)), s.structure['thickness_km']-10.)
        area = s.material_surface['area_km2'].copy()
        s.t = 2.
        structure_engine.deform(s, np.zeros(s.n), np.zeros(s.n), np.zeros(s.n), 2.)
        self.assertAlmostEqual(s.mantle_return_km3['total']/float(area@expected), 1., places=11)
        self.assertEqual(s.foundering_diagnostics['depth_integration_version'], 1)
        crust_inventory.validate(s.structure)

    def test_checkpoint_resume_and_frame_keep_explicit_depth_version(self):
        s = split_cover(); sink.upgrade_inventory(s); sink.upgrade_depth_integration(s)
        s.config = {}; s.rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.npz'
            checkpoint.write_checkpoint(path, s, dict(config=s.config), {})
            restored, _ = checkpoint.read_checkpoint(path, {}, SimpleNamespace)
        np.testing.assert_array_equal(eligible(restored)[0], eligible(s)[0])
        frame = dict(material_faces=s.material_surface['faces'], **sink.snapshot_fields(s))
        sink.validate_frame(frame)
        self.assertEqual(frame['foundering_depth_version'], 1)
        for bad in (True, 2, -1):
            with self.assertRaises(ValueError): sink.validate_frame(dict(frame, foundering_depth_version=bad))
        with self.assertRaises(ValueError): sink.validate_frame(dict(frame, foundering_version=0))


if __name__ == '__main__':
    unittest.main()
