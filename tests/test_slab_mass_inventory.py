"""Conservation oracles for slab identity, trace geometry, and accepted feeding."""
from copy import deepcopy
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import checkpoint
import native_subduction
import plate_balance
import slab_memory as slab
import trench_history as history
from tests.test_trench_history import fixture, step
from tests.test_native_subduction_oracle import fixture as capture_fixture


def legacy_row(depth=100., length=1000., ident=1):
    # Independent cooling/isostasy expression, not the production mass helper.
    mass_per_area = (3300.-1030.)*(5651.-2473.*math.exp(-.0278*100.)-2600.)
    return dict(id=ident, phase='mature', maturity=1., length_km=length,
        downgoing_plate_uid=11, overriding_plate_uid=22,
        slab_fed_area_km2=length*depth, slab_retained_area_km2=length*depth,
        slab_retained_buoyancy_area_km2=length*depth,
        slab_retired_area_km2=0., slab_attachment=1.,
        slab_line_load_kg_per_m=mass_per_area*depth*1000.)


def convert(*rows):
    s = SimpleNamespace(slab_memory_version=1, trench_systems=list(rows))
    slab.upgrade_mass_inventory(s)
    return s


def represented_mass(row):
    return row['length_km']*1000.*row[slab.LINE_LOAD_FIELD]


def seed_mass(row, mass=4e18):
    row['slab_initial_excess_mass_kg'] = row[slab.RETAINED_MASS_FIELD] = mass
    slab.refresh_line_load(row)


class SlabMassInventoryTests(unittest.TestCase):
    def assert_close(self, actual, expected):
        self.assertAlmostEqual(actual/max(abs(expected), 1.), expected/max(abs(expected), 1.), places=12)

    def test_reported_join_reproducer_conserves_excess_mass(self):
        s = convert(legacy_row(100.), legacy_row(500., ident=2))
        source, target = s.trench_systems
        before = sum(represented_mass(r) for r in s.trench_systems)
        slab.join(source, target)
        target['length_km'] = 2000.
        slab.refresh_line_load(target)
        self.assert_close(represented_mass(target), before)
        self.assertEqual(target['slab_retained_area_km2'], 600000.)
        for key in slab.FIELDS+slab.MASS_FIELDS:
            self.assertEqual(source[key], 0.)
        slab.validate_row(target, require_mass=True)

    def test_split_join_conserves_every_inventory_with_unequal_pieces(self):
        parent = convert(legacy_row()).trench_systems[0]
        original = deepcopy(parent)
        children = []
        remaining = 1.
        for fraction in (.1, .2, .4):
            child = dict(length_km=original['length_km']*fraction)
            slab.partition(parent, child, fraction/remaining)
            remaining -= fraction
            parent['length_km'] = original['length_km']*remaining
            slab.refresh_line_load(parent)
            children.append(child)
        for key in slab.FIELDS+slab.MASS_FIELDS:
            self.assert_close(parent[key]+sum(c[key] for c in children), original[key])
        for child in children:
            slab.join(child, parent)
        parent['length_km'] = original['length_km']
        slab.refresh_line_load(parent)
        for key in slab.FIELDS+slab.MASS_FIELDS+(slab.LINE_LOAD_FIELD,):
            self.assert_close(parent[key], original[key])

    def test_trace_shortening_growth_and_zero_length_do_not_change_mass(self):
        row = convert(legacy_row()).trench_systems[0]
        mass = row[slab.RETAINED_MASS_FIELD]
        for length in (10., 10000., 0., 1000.):
            row['length_km'] = length
            slab.refresh_line_load(row)
            slab.validate_row(row, require_mass=True)
            self.assertEqual(row[slab.RETAINED_MASS_FIELD], mass)
            self.assert_close(represented_mass(row), mass if length else 0.)

    def test_force_allocation_conserves_mass_under_rematching_and_refinement(self):
        s = convert(legacy_row())
        mass = s.trench_systems[0][slab.RETAINED_MASS_FIELD]
        s.active = np.array([True, True]); s.plate_uid = np.array([11,22])
        for edges in ([1000.], [50.,100.,150.], [25.]*12):
            n = len(edges)
            s.ba = np.arange(n); s.bl = np.asarray(edges)
            s.bp = np.zeros(n,int); s.bq = np.ones(n,int); s.trench_id = np.ones(n,int)
            before = deepcopy(s.trench_systems)
            owner, load, _ = slab.line_load(s)
            np.testing.assert_array_equal(owner, 0)
            self.assert_close(float(load@(s.bl*1000.)), mass)
            self.assertEqual(before, s.trench_systems, 'Force reads must not modify stored inventory.')
        s.trench_id[:] = 0
        np.testing.assert_array_equal(slab.line_load(s)[1], 0.)
        self.assertEqual(s.trench_systems[0][slab.RETAINED_MASS_FIELD], mass)

    def test_actual_slab_driver_is_invariant_to_collinear_edge_subdivision(self):
        s = convert(legacy_row())
        s.active = np.array([True, True]); s.plate_uid = np.array([11,22])
        expected = s.trench_systems[0][slab.RETAINED_MASS_FIELD]*9.81*math.sin(math.radians(50.))
        for pieces in (1, 5, 20):
            s.ba = np.arange(pieces); s.bl = np.full(pieces,300./pieces)
            s.bp = np.zeros(pieces,int); s.bq = np.ones(pieces,int); s.trench_id = np.ones(pieces,int)
            model = object.__new__(plate_balance.Balance)
            model.s = s; model.size = 6; model.slot = {0:0,1:1}
            model.effective_subduction = False  # the retained-mass law this test is about
            model.slab_owner, model.slab_load, model.slab_length = slab.line_load(s)
            model.trench_edges = np.arange(pieces); model.bp = s.bp
            model.normal = np.tile([0.,1.,0.],(pieces,1))
            model.radial = np.tile([1.,0.,0.],(pieces,1)); model.length_m = s.bl*1000.
            model.drivers = {}; model.notes = {}
            model._ridge_push = lambda: np.zeros(6)
            model._collision_torque = lambda: np.zeros(6)
            model._drive()
            self.assert_close(model.drivers['slab'][2]/plate_balance.CM_YR_M_S, expected)
            np.testing.assert_array_equal(model.drivers['slab'][[0,1,3,4,5]], 0.)

    def feed(self, s, dt, amount):
        s.t = getattr(s, 't', 0.)+dt
        s.native_subduction_diagnostics = dict(step_end_myr=s.t, step_duration_myr=dt,
            removed_area_km2=amount, removed_area_by_trench=[dict(trench_id=1,
                downgoing_plate_uid=11, area_km2=amount, buoyancy_area_km2=amount)])
        slab.advance(s, dt)

    def test_feed_retirement_and_subdivision_match_analytic_mass_solution(self):
        whole = convert(legacy_row()); halves = deepcopy(whole)
        initial = whole.trench_systems[0][slab.RETAINED_MASS_FIELD]
        area = 2000.; dt = 10.; tau = 50.
        # buoyancy 1 corresponds to cooling age 80 Myr.
        per_area = (3300.-1030.)*(5651.-2473.*math.exp(-.0278*80.)-2600.)
        fed = area*1e6*per_area
        retained = initial*math.exp(-dt/tau)+fed*tau/dt*(1.-math.exp(-dt/tau))
        self.feed(whole, dt, area)
        self.feed(halves, dt/2., area/2.)
        halves.trench_systems[0]['length_km'] = 300.
        slab.refresh_line_load(halves.trench_systems[0])
        self.feed(halves, dt/2., area/2.)
        for s in (whole, halves):
            row = s.trench_systems[0]
            self.assert_close(row[slab.RETAINED_MASS_FIELD], retained)
            self.assert_close(row['slab_fed_excess_mass_kg'], fed)
            self.assert_close(row['slab_retired_excess_mass_kg'], initial+fed-retained)
            before = deepcopy(row)
            slab.advance(s, dt/2. if s is halves else dt)
            self.assertEqual(before, s.trench_systems[0], 'A repeated commit must not feed twice.')

    def test_retirement_without_feed_closes_inventory(self):
        s = convert(legacy_row())
        original = s.trench_systems[0][slab.RETAINED_MASS_FIELD]
        self.feed(s, 50., 0.)
        row = s.trench_systems[0]
        self.assert_close(row[slab.RETAINED_MASS_FIELD], original/math.e)
        self.assert_close(row['slab_retired_excess_mass_kg'], original*(1.-1./math.e))

    def test_migration_is_explicit_atomic_idempotent_and_preserves_boundary_load(self):
        rows = [legacy_row(), legacy_row(500., ident=2)]
        before = deepcopy(rows)
        s = SimpleNamespace(slab_memory_version=1, trench_systems=rows)
        report = slab.upgrade_mass_inventory(s)
        self.assertFalse(report['already_converted'])
        self.assertEqual(rows, before, 'Migration must stage before replacing rows.')
        for old, new in zip(before, s.trench_systems):
            self.assertEqual(new[slab.RETAINED_MASS_FIELD], represented_mass(old))
            for key in slab.FIELDS+(slab.LINE_LOAD_FIELD,):
                self.assertEqual(new[key], old[key])
        after = deepcopy(vars(s))
        self.assertTrue(slab.upgrade_mass_inventory(s)['already_converted'])
        self.assertEqual(vars(s), after)
        bad = SimpleNamespace(slab_memory_version=1, trench_systems=[legacy_row(), legacy_row(length=0.)])
        before = deepcopy(vars(bad))
        with self.assertRaisesRegex(ValueError, 'positive recorded trace'):
            slab.upgrade_mass_inventory(bad)
        self.assertEqual(vars(bad), before)

    def test_missing_corrupt_or_unbalanced_mass_state_is_rejected(self):
        good = convert(legacy_row()).trench_systems[0]
        for mutate in (lambda r:r.pop(slab.RETAINED_MASS_FIELD),
                       lambda r:r.update(slab_retained_excess_mass_kg=-1.),
                       lambda r:r.update(slab_fed_excess_mass_kg=np.nan),
                       lambda r:r.update(slab_retired_excess_mass_kg=1e15),
                       lambda r:r.update(length_km=r['length_km']*2.)):
            row = deepcopy(good); mutate(row)
            with self.assertRaises(ValueError): slab.validate_row(row, require_mass=True)
        with self.assertRaisesRegex(ValueError, 'explicit'):
            slab.ensure(legacy_row(), conservative=True)

    def test_invalid_partition_is_rejected_without_inventory_mutation(self):
        for fraction in (-.1, 1.1, np.nan, np.inf):
            parent = convert(legacy_row()).trench_systems[0]; before = deepcopy(parent)
            with self.assertRaises(ValueError): slab.partition(parent, {}, fraction)
            self.assertEqual(parent, before)

    def test_legacy_does_not_silently_claim_conservative_behavior(self):
        s = SimpleNamespace(slab_memory_version=1, trench_systems=[legacy_row()])
        before = deepcopy(s.trench_systems)
        self.assertEqual(slab.snapshot(s)['slab_memory_version'], 1)
        self.assertEqual(s.trench_systems, before)
        s.slab_memory_version = 2
        with self.assertRaisesRegex(ValueError, 'complete excess-mass'):
            slab.snapshot(s)

    def test_checkpoint_round_trip_preserves_mass_and_provenance(self):
        s = convert(legacy_row())
        s.t = 536.; s.config = {}; s.rng = np.random.default_rng(123)
        expected = slab.snapshot(s)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.npz'
            checkpoint.write_checkpoint(path, s, {'config':{}}, {})
            restored, _ = checkpoint.read_checkpoint(path, None, SimpleNamespace)
        self.assertEqual(restored.trench_systems, s.trench_systems)
        self.assertEqual(slab.snapshot(restored), expected)
        self.assertEqual(restored.rng.bit_generator.state, s.rng.bit_generator.state)


class SlabMassIntegrationTests(unittest.TestCase):
    def make_history(self, split=False):
        s = fixture(((20,42,63),))
        s.slab_memory_version = slab.VERSION
        if split: s.bcode[8:14], s.normal_speed[8:14] = 1, 20.
        history.initialize(s)
        for i,row in enumerate(s.trench_systems): seed_mass(row, (i+1)*1e18)
        return s

    def mass(self,s):
        for row in s.trench_systems: slab.validate_row(row, require_mass=True)
        return sum(r[slab.RETAINED_MASS_FIELD] for r in s.trench_systems)

    def test_production_history_join_and_trace_length_change_conserve_mass(self):
        s = self.make_history(split=True); before = self.mass(s)
        s.bcode[:], s.normal_speed[:] = 2, -20.
        step(s)
        self.assertEqual(sum(r['phase']=='joined' for r in s.trench_systems), 1)
        self.assertAlmostEqual(self.mass(s)/before, 1., places=14)
        s.bl *= .6
        step(s)
        self.assertAlmostEqual(self.mass(s)/before, 1., places=14)

    def test_production_history_separation_conserves_mass(self):
        s = self.make_history(); before = self.mass(s)
        s.bcode[8:14], s.normal_speed[8:14] = 1, 20.
        for _ in range(10): step(s)
        self.assertGreater(len(s.trench_systems), 1)
        self.assertAlmostEqual(self.mass(s)/before, 1., places=14)

    def test_overriding_transfer_partitions_mass_without_inventing_load(self):
        s = self.make_history(); before = self.mass(s)
        s.active = np.array([True,True,True]); s.plate_uid = np.array([11,22,33])
        s.omega = np.zeros((3,3))
        moved = np.arange(len(s.bb)) >= len(s.bb)//2
        s.plate[s.bb[moved]] = 2
        history.transfer_overriding(s,1,2)
        self.assertEqual(len(s.trench_systems),2)
        self.assertAlmostEqual(self.mass(s)/before,1.,places=14)
        self.assertAlmostEqual(sum(represented_mass(r) for r in s.trench_systems)/before,1.,places=14)

    def test_native_capture_keeps_accepted_feed_provenance_in_version_two(self):
        s, _, transported = capture_fixture(level=2)
        s.slab_memory_version = slab.VERSION
        s.trench_id = np.ones(len(s.ba),int)
        s.age = np.full(s.n,80.)
        s.plate_uid = np.arange(len(s.support))+11
        s.trench_systems = [dict(id=1, phase='mature', length_km=1000.,
            downgoing_plate_uid=11, overriding_plate_uid=12)]
        slab.ensure(s.trench_systems[0], conservative=True)
        native_subduction.removal(s, transported, 2.)
        record = s.native_subduction_diagnostics
        self.assertGreater(record['removed_area_km2'], 0.)
        self.assertAlmostEqual(sum(r['area_km2'] for r in record['removed_area_by_trench']),
                               record['removed_area_km2'], places=8)
        s.t = 2.
        slab.advance(s, 2.)
        self.assertGreater(s.trench_systems[0][slab.RETAINED_MASS_FIELD], 0.)
        slab.validate_row(s.trench_systems[0], require_mass=True)


if __name__ == '__main__': unittest.main()
