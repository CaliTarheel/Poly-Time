"""Arc magma thickens existing arc crust under ridge and summit caps and founds new ground."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT)]

import arc_ribbon as ribbon
import checkpoint
import crustal_structure as columns
import mesh_history
import native_arc_material as arcs
from test_arc_birth_footprint import world
from test_arc_source_geometry import add, positions_in_cell


def unit(value):
    value = np.asarray(value, float)
    return value/np.linalg.norm(value, axis=-1, keepdims=True)


def offer(state, point, area, dt=1.):
    cells = state._indices(point)
    return arcs.add_arc_crust(state, cells, np.asarray([area], float), positions=point,
                              owners=state.plate[cells], dt=dt)


def volume(state):
    return float(state.material_surface['area_km2']@state.structure['thickness_km'])


def frame(state):
    state._rasterize(); state._boundaries(); state._update_surface_domains()
    result = state.snapshot()
    mesh_history.arrays(result)
    return result


class ArcRibbonGeometryTests(unittest.TestCase):
    def test_binned_nearest_distance_equals_the_exhaustive_search_within_its_cutoff(self):
        rng = np.random.default_rng(7)
        centre = unit([.3, -.5, .8])
        start = unit(centre+rng.normal(scale=.03, size=(400, 3)))
        end = unit(start+rng.normal(scale=.004, size=(400, 3)))
        points = unit(centre+rng.normal(scale=.035, size=(900, 3)))
        exhaustive = ribbon.Arcs(start, end).nearest(points)
        binned = ribbon.Arcs(start, end, 30.).nearest(points)
        near = exhaustive <= 30.
        self.assertTrue(near.any() and (~near).any())
        np.testing.assert_array_equal(binned[near], exhaustive[near])
        self.assertTrue(np.isinf(binned[~near]).all())

    def test_distance_to_an_arc_is_perpendicular_inside_it_and_to_an_end_beyond_it(self):
        a, b = unit([1., 0., 0.]), unit([np.cos(.1), np.sin(.1), 0.])
        over = unit([np.cos(.05), np.sin(.05), .01])
        beyond = unit([np.cos(.13), np.sin(.13), 0.])
        distance = ribbon.nearest_arc_km([over, beyond, a], [a], [b])
        np.testing.assert_allclose(distance, [ribbon.RADIUS_KM*np.arcsin(over[2]), ribbon.RADIUS_KM*.03, 0.],
                                   rtol=1e-9, atol=1e-6)
        # An arc of no length is its point.
        np.testing.assert_allclose(ribbon.nearest_arc_km([b], [a], [a]), [ribbon.RADIUS_KM*.1], rtol=1e-9)

    def test_cap_holds_the_rim_at_the_ridge_top_then_climbs_to_the_summit(self):
        policy = ribbon.POLICY
        grade = 1000.*np.tan(np.deg2rad(policy['constructive_slope_deg']))
        rise = (policy['summit_m']-policy['ridge_top_m'])/grade
        depth = np.array([0., policy['rim_km'], policy['rim_km']+1., policy['rim_km']+rise, 500., np.inf])
        np.testing.assert_allclose(ribbon.cap_m(depth), [policy['ridge_top_m'], policy['ridge_top_m'],
            policy['ridge_top_m']+grade, policy['summit_m'], policy['summit_m'], policy['summit_m']])

    def test_span_is_the_recorded_trench_trace_carried_to_the_source_point(self):
        trench = unit([[1., -.03, 0.], [1., 0., 0.], [1., 0., 0.], [1., .03, .002]]).reshape(2, 2, 3)
        middle = unit(trench.reshape(-1, 3).sum(axis=0))
        point = unit(middle+[0., 0., 180./ribbon.RADIUS_KM])
        (start, end), (lone_start, lone_end) = ribbon.spans(
            [point, point], [dict(trace_xyz=trench.tolist()), dict(trace_xyz=[])])
        np.testing.assert_allclose(unit(np.vstack((start, end)).sum(axis=0)), point, atol=1e-12)
        np.testing.assert_allclose(np.sum(start*end, axis=1), np.sum(trench[:, 0]*trench[:, 1], axis=1), atol=1e-14)
        np.testing.assert_array_equal(lone_start, point[None])
        np.testing.assert_array_equal(lone_end, point[None])

    def test_level_fill_raises_the_lowest_columns_first_and_never_exceeds_its_supply_or_limits(self):
        base = np.array([0., 10., 30., 5.])
        limit = np.array([40., 40., 40., 8.])
        weight = np.array([2., 1., 1., 4.])
        added = ribbon.level_fill(base, limit, weight, 20.)
        self.assertAlmostEqual(float(added@weight), 20., places=12)
        # The two lowest columns share one level below the next column's base.
        np.testing.assert_allclose(base+added, [20/3, 10., 30., 20/3], atol=1e-12)
        added = ribbon.level_fill(base, limit, weight, 34.)
        self.assertAlmostEqual(float(added@weight), 34., places=12)
        # The fourth column stops at its own limit; the first two rise together below the third.
        np.testing.assert_allclose(base+added, [32/3, 32/3, 30., 8.], atol=1e-12)
        full = ribbon.level_fill(base, limit, weight, 1e9)
        np.testing.assert_array_equal(full, limit-base)
        np.testing.assert_array_equal(ribbon.level_fill(base, limit, weight, 0.), np.zeros(4))
        self.assertLessEqual(float(ribbon.level_fill(base, limit, weight, 3.)@weight), 3.)


class ArcRibbonTransactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.initial = world()
        cls.point = positions_in_cell(cls.initial)[:1]
        add(cls.initial, cls.point, [10000.])
        assert len(cls.initial.mass) == 24 and not cls.initial.native_arc_pending['area'].sum()
        cls.far = positions_in_cell(cls.initial, 60)[:1]

    def setUp(self):
        self.state = deepcopy(self.initial)

    def test_an_unmarked_state_keeps_the_previous_emplacement(self):
        report = offer(self.state, self.point, 400.)
        self.assertNotIn('ribbon', report)
        self.assertEqual(ribbon.snapshot_fields(self.state), {})
        self.assertNotIn('arc_ribbon_version', frame(self.state))

    def test_thickening_adds_exactly_the_spent_magma_within_rate_and_caps_on_unchanged_geometry(self):
        state = self.state
        ribbon.activate(state)
        ground = ribbon.belt(state, 1.)
        before = dict(volume=volume(state), thickness=state.structure['thickness_km'].copy(),
                      elevation=columns.elevation(state.structure), mass=state.mass.copy(),
                      vertices=state.material_surface['vertices'].copy(), faces=state.material_surface['faces'].copy())
        report = offer(state, self.point, 400.)
        account = report['ribbon']
        gain = state.structure['thickness_km']-before['thickness']
        self.assertGreater(account['thickened_source_area_km2'], 0.)
        self.assertAlmostEqual(volume(state)-before['volume'], 25.*account['thickened_source_area_km2'], places=6)
        self.assertTrue(np.all(gain >= 0.) and gain.max() <= ribbon.POLICY['maximum_thickening_km_per_myr']+1e-12)
        self.assertAlmostEqual(account['maximum_thickening_km'], float(gain.max()), places=12)
        after = columns.elevation(state.structure)
        self.assertTrue(np.all(after[ground['faces']] <= np.maximum(before['elevation'][ground['faces']],
                                                                  ground['cap_m'])+1e-6))
        self.assertTrue(np.all(after[gain > 0.] > before['elevation'][gain > 0.]))
        np.testing.assert_array_equal(state.material_surface['vertices'], before['vertices'])
        np.testing.assert_array_equal(state.material_surface['faces'], before['faces'])
        np.testing.assert_array_equal(state.mass, before['mass'])
        self.assertEqual(account['foundations_born'], 0)
        self.assertAlmostEqual(account['pending_area_km2'], 400.-account['thickened_source_area_km2'], places=9)
        self.assertAlmostEqual(float(state.native_arc_pending['area'].sum()), account['pending_area_km2'], places=9)

    def test_the_lowest_surface_is_built_first(self):
        state = self.state
        ribbon.activate(state)
        ground = ribbon.belt(state, 1.)
        before = state.structure['thickness_km'].copy()
        offer(state, self.point, 5.)
        raised = np.flatnonzero(state.structure['thickness_km'] > before)
        self.assertTrue(len(raised))
        local = np.flatnonzero(np.isin(ground['faces'], raised))
        untouched = np.setdiff1d(np.flatnonzero(ground['step_km'] > 0.), local)
        self.assertTrue(len(untouched))
        self.assertLessEqual(ground['air'][local].max(), ground['air'][untouched].min()+1e-9)

    def test_half_an_interval_thickens_half_as_much(self):
        whole, half = deepcopy(self.state), self.state
        for state in (whole, half): ribbon.activate(state)
        full = offer(whole, self.point, 4000.)['ribbon']
        part = offer(half, self.point, 4000., dt=.5)['ribbon']
        self.assertAlmostEqual(full['maximum_thickening_km'], 1., places=12)
        self.assertAlmostEqual(part['maximum_thickening_km'], .5, places=12)
        with self.assertRaisesRegex(ValueError, 'length of its interval'):
            offer(self.state, self.point, 1., dt=None)

    def test_magma_far_from_any_arc_crust_founds_a_flat_foundation_in_open_water(self):
        state = self.state
        ribbon.activate(state)
        before = dict(faces=len(state.mass), volume=volume(state), arcs=int(state.parcel_arc_id.max()))
        account = offer(state, self.far, 600.)['ribbon']
        self.assertEqual((account['foundations_born'], account['foundation_attempts']), (1, 1))
        born = np.arange(before['faces'], len(state.mass))
        self.assertEqual(len(born), 24)
        self.assertTrue(np.all(state.parcel_arc_id[born] == before['arcs']+1) and np.all(state.kind[born] == 3))
        thickness = state.structure['thickness_km'][born].copy()
        floor = ribbon.POLICY['foundation_column_km']
        self.assertTrue(np.all(thickness >= floor) and thickness.max() <= floor*1.01)
        row = account['foundations'][0]
        radius = ribbon.POLICY['foundation_maximum_radius_km']
        self.assertLessEqual(row['radius_km'], radius)
        self.assertAlmostEqual(row['source_area_km2'], ribbon.FOOTPRINT_AREA_PER_RADIUS2*radius*radius*floor/25., places=9)
        self.assertAlmostEqual(float(state.material_surface['area_km2'][born].sum()),
                               row['source_area_km2']*25./floor*ribbon.FOOTPRINT_FILL, delta=1e-6)
        self.assertAlmostEqual(volume(state)-before['volume'], 25.*account['placed_area_km2'], places=5)
        self.assertAlmostEqual(account['pending_area_km2'], 600.-row['source_area_km2'], places=9)
        # The same origin later thickens the ground it founded and founds nothing more.
        later = offer(state, self.far, 0.)['ribbon']
        self.assertEqual(later['foundations_born'], 0)
        self.assertGreater(later['thickened_source_area_km2'], 0.)
        self.assertTrue(np.all(state.structure['thickness_km'][born] <= thickness+1.+1e-9))

    def test_magma_the_belt_will_hold_founds_nothing(self):
        state = self.state
        ribbon.activate(state)
        account = offer(state, self.point, 400.)['ribbon']
        self.assertEqual(account['overflow_area_km2'], 0.)
        self.assertEqual((account['foundations_born'], account['foundation_attempts']), (0, 0))

    def test_every_origin_stays_fully_placed_or_pending_and_the_saved_frame_validates(self):
        state = self.state
        ribbon.activate(state)
        offered = 0.
        for area in (400., 300.):
            offer(state, self.point, area); offered += area
        offer(state, self.far, 600.); offered += 600.
        ledger = [entry for entry in state.native_arc_source_placements if entry.get('ribbon') == ribbon.VERSION]
        placed = sum(entry['area_km2'] for entry in ledger)
        self.assertAlmostEqual(placed+float(state.native_arc_pending['area'].sum()), offered, places=8)
        self.assertEqual({entry['mode'] for entry in ledger}, {'deposition', 'birth'})
        origins = [entry['source_provenance']['origin_id'] for entry in ledger if entry['mode'] == 'deposition']
        self.assertEqual(len(origins), len(set(origins)))
        saved = frame(state)
        self.assertEqual(saved['arc_ribbon_version'], 1)
        self.assertEqual(saved['arc_ribbon_state']['transactions'], 3)
        self.assertEqual(saved['arc_ribbon_state']['foundations'], 1)
        self.assertAlmostEqual(saved['stats']['arc_added_km2'], 10000.+placed, places=7)
        deposited = sum(entry['area_km2'] for entry in ledger if entry['mode'] == 'deposition')
        self.assertAlmostEqual(saved['stats']['arc_deposited_source_area_km2'], deposited, places=8)
        self.assertEqual(len(saved['volcanic_point_features']), len(state.native_arc_pending['area']))

    def test_a_tampered_saved_policy_or_ledger_is_refused(self):
        state = self.state
        ribbon.activate(state)
        offer(state, self.point, 400.)
        saved = frame(state)
        changed = deepcopy(saved); changed['arc_ribbon_policy']['ridge_top_m'] = -500.
        with self.assertRaisesRegex(ValueError, 'ribbon policy'):
            ribbon.validate_frame(changed)
        changed = deepcopy(saved); changed['arc_material_diagnostics']['ribbon']['placed_area_km2'] += 1.
        with self.assertRaisesRegex(ValueError, 'does not close'):
            ribbon.validate_frame(changed)
        changed = deepcopy(saved); changed['arc_material_diagnostics']['ribbon']['maximum_thickening_km'] = 3.
        with self.assertRaisesRegex(ValueError, 'rate limit'):
            ribbon.validate_frame(changed)
        changed = deepcopy(saved); del changed['arc_ribbon_version']
        with self.assertRaisesRegex(ValueError, 'explicit policy version'):
            ribbon.validate_frame(changed)

    def test_a_typed_checkpoint_keeps_the_policy_and_continues_identically(self):
        state = self.state
        ribbon.activate(state)
        offer(state, self.point, 400.)
        compatibility = dict(engine_sha256='ribbon-fixture', auxiliary_sources_sha256={}, numpy_version=np.__version__)
        manifest = dict(run_id='ribbon-fixture', config=state.config, frames=[dict(time_myr=0.)], frame_count=1,
                        state='paused')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.npz'
            checkpoint.write_checkpoint(path, state, manifest, compatibility)
            restored, _ = checkpoint.read_checkpoint(path, compatibility, type(state))
        self.assertEqual(restored.native_arc_ribbon_version, 1)
        self.assertEqual(restored.arc_ribbon_state, state.arc_ribbon_state)
        for each in (state, restored):
            offer(each, self.point, 250.); offer(each, self.far, 600.)
        np.testing.assert_array_equal(restored.structure['thickness_km'], state.structure['thickness_km'])
        np.testing.assert_array_equal(restored.material_surface['vertices'], state.material_surface['vertices'])
        np.testing.assert_array_equal(restored.native_arc_pending['area'], state.native_arc_pending['area'])
        self.assertEqual(restored.native_arc_source_placements, state.native_arc_source_placements)

    def test_activation_is_explicit_single_and_requires_the_source_policies(self):
        state = self.state
        ribbon.activate(state)
        self.assertEqual(state.arc_ribbon_state['activated_myr'], float(state.t))
        with self.assertRaisesRegex(ValueError, 'already active'):
            ribbon.activate(state)
        other = deepcopy(self.initial); other.native_arc_point_version = 0
        with self.assertRaisesRegex(ValueError, 'requires geographic sources'):
            ribbon.activate(other)


if __name__ == '__main__':
    unittest.main()
