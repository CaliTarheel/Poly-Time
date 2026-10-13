"""Independent geometry, work, saturation and inventory oracles for slab pull."""
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import plate_balance as balance
import slab_memory as slab
from tests.test_slab_mass_inventory import convert, legacy_row


def inventory(extent):
    # Start directly in v2: very deep inventories cannot be represented by the
    # historical v1 validator, so converting an invalid v1 fixture is not valid.
    row = legacy_row(extent)
    mass = row[slab.LINE_LOAD_FIELD]*row['length_km']*1000.
    row.update(dict.fromkeys(slab.MASS_FIELDS, 0.))
    row['slab_initial_excess_mass_kg'] = row[slab.RETAINED_MASS_FIELD] = mass
    return SimpleNamespace(slab_memory_version=2, trench_systems=[row])


def attached(s, edges=(1000.,)):
    count = len(edges)
    s.active = np.array([True, True])
    s.plate_uid = np.array([11, 22])
    s.ba = np.zeros(count, int)
    s.bb = np.ones(count, int)
    s.bp = np.zeros(count, int)
    s.bq = np.ones(count, int)
    s.bl = np.array(edges, float)
    s.trench_id = np.ones(count, int)
    return s


def model_for(s):
    """Production slab/basal assembly with an analytically simple edge geometry."""
    s.xyz = np.vstack((np.eye(3), -np.eye(3)))
    s.cell_area = np.full(6, 1000.)
    s.plate = np.array([0, 0, 0, 1, 1, 1])
    s.crust = np.zeros(6, int)
    s.age = np.full(6, 150.)
    model = object.__new__(balance.Balance)
    model.s = s
    model.effective_subduction = False  # the depth-window law, not the effective-force one
    model.plates = [0, 1]
    model.slot = {0: 0, 1: 1}
    model.size = 6
    model.bp, model.bq = s.bp, s.bq
    model.live = np.arange(len(s.ba))
    model.slab_owner, model.slab_load, model.slab_length = slab.line_load(s)
    model.trench = model.slab_owner >= 0
    model.normal = np.tile([0., 1., 0.], (len(s.ba), 1))
    model.radial = np.tile([1., 0., 0.], (len(s.ba), 1))
    model.an = np.cross(model.radial, model.normal)
    model.at = -model.normal
    model.length_m = s.bl*1000.
    model.notes, model.drivers = {}, {}
    model._ridge_push = lambda: np.zeros(6)
    model._collision_torque = lambda: np.zeros(6)
    model._viscous()
    model._drive()
    return model


class SlabDepthWindowTests(unittest.TestCase):
    def assert_relative(self, actual, expected):
        self.assertAlmostEqual(actual/max(abs(expected), 1.), expected/max(abs(expected), 1.), places=12)

    def test_depth_conversion_at_shallow_default_and_vertical_dips(self):
        for dip in (30., 50., 90.):
            self.assert_relative(slab.upper_mantle_length_km(dip)*math.sin(math.radians(dip)), 660.)
        self.assertAlmostEqual(slab.upper_mantle_length_km(50.), 861.568811, places=6)
        for invalid in (0., -1., 91., math.nan, math.inf):
            with self.assertRaises(ValueError): slab.upper_mantle_length_km(invalid)

    def test_depth_window_bounds_force_and_drag_without_mutating_mass(self):
        cap = 660./math.sin(math.radians(50.))
        for extent in (100., 660., cap, 4000., 10000.):
            s = attached(inventory(extent))
            before = deepcopy(s.trench_systems)
            row = s.trench_systems[0]
            owner, load, length = slab.line_load(s)
            expected_fraction = min(1., cap/extent)
            self.assert_relative(float(load@(s.bl*1000.)), row[slab.RETAINED_MASS_FIELD]*expected_fraction)
            np.testing.assert_array_equal(owner, 0)
            np.testing.assert_allclose(length, min(extent, cap))
            self.assertEqual(s.trench_systems, before)
            slab.validate_row(row, require_mass=True)

    def test_force_and_drag_use_current_trace_and_ignore_edge_subdivision(self):
        # The saved 1,000-km trace is stale. Its 600,000 km2 slab extends 2,000
        # km down dip on the currently matched 300-km trace and must be windowed.
        s = convert(legacy_row(600.))
        baseline = model_for(attached(deepcopy(s), (300.,)))
        for edges in ((100., 200.), (30.,)*10):
            refined = model_for(attached(deepcopy(s), edges))
            np.testing.assert_allclose(refined.stiffness, baseline.stiffness, rtol=2e-14)
            np.testing.assert_allclose(refined.torque, baseline.torque, rtol=2e-14)
        cap = 660./math.sin(math.radians(50.))
        expected_mass = s.trench_systems[0][slab.RETAINED_MASS_FIELD]*cap/2000.
        self.assert_relative(baseline.torque[2]/balance.CM_YR_M_S,
                             expected_mass*9.81*math.sin(math.radians(50.)))

    def test_more_deep_inventory_cannot_raise_terminal_slab_speed(self):
        at_cap = model_for(attached(inventory(660./math.sin(math.radians(50.)))))
        far_deeper = model_for(attached(inventory(10000.)))
        np.testing.assert_allclose(at_cap.stiffness, far_deeper.stiffness, rtol=2e-14)
        np.testing.assert_allclose(at_cap.torque, far_deeper.torque, rtol=2e-14)
        np.testing.assert_allclose(np.linalg.solve(at_cap.stiffness, at_cap.torque),
                                   np.linalg.solve(far_deeper.stiffness, far_deeper.torque), rtol=2e-14)

    def test_slab_force_matches_gravitational_work_at_dip_limits(self):
        # Short enough that all three dips keep all of the mass above 660 km.
        s = attached(convert(legacy_row(100.)))
        mass = s.trench_systems[0][slab.RETAINED_MASS_FIELD]
        for dip in (.01, 50., 90.):
            with patch.object(slab, 'SUBDUCTION_DIP_DEG', dip):
                model = model_for(deepcopy(s))
            horizontal_force = model.torque[2]/balance.CM_YR_M_S
            feed_speed = .037
            vertical_speed = feed_speed*math.sin(math.radians(dip))
            self.assert_relative(horizontal_force*feed_speed, mass*9.81*vertical_speed)

    def test_fast_old_ocean_feed_is_bounded_for_200_myr_and_closes_ledger(self):
        for speed in (30., 80., 100.):
            s = attached(convert(legacy_row(0.)))
            buoyancy = math.sqrt(150./80.)
            mass_per_area = (3300.-1030.)*(3051.-2473.*math.exp(-.0278*150.))
            cap = 660./math.sin(math.radians(50.))
            for year in range(1, 201):
                amount = speed*1000.
                s.t = float(year)
                s.native_subduction_diagnostics = dict(step_end_myr=s.t, step_duration_myr=1.,
                    removed_area_km2=amount, removed_area_by_trench=[dict(trench_id=1,
                    downgoing_plate_uid=11, area_km2=amount, buoyancy_area_km2=amount*buoyancy)])
                slab.advance(s, 1.)
                _, load, length = slab.line_load(s)
                self.assertLessEqual(float(load[0]), mass_per_area*cap*1000.*(1.+1e-12))
                self.assertLessEqual(float(length[0]), cap*(1.+1e-12))
                slab.validate_row(s.trench_systems[0], require_mass=True)
            row = s.trench_systems[0]
            self.assert_relative(row['slab_fed_area_km2'], speed*1000.*200.)
            self.assertGreater(row['slab_retained_area_km2']/1000., cap)
            self.assert_relative(load[0], mass_per_area*cap*1000.)

    def test_bootstrap_matches_native_inventory_before_and_after_depth_saturation(self):
        for extent in (100., 1500.):
            native = attached(convert(legacy_row(extent)))
            row = native.trench_systems[0]
            # Match the source age encoded in legacy_row's independently computed mass.
            row['slab_retained_buoyancy_area_km2'] = row['slab_retained_area_km2']*math.sqrt(100./80.)
            migrated = deepcopy(native)
            old = migrated.trench_systems[0]
            old['shortening_km'] = 2.*extent  # Includes already-retired consumption.
            old[slab.LINE_LOAD_FIELD] = 0.
            old.update(dict.fromkeys(slab.MASS_FIELDS, 0.))
            slab.bootstrap_slab_state(migrated)
            self.assert_relative(old[slab.RETAINED_MASS_FIELD], row[slab.RETAINED_MASS_FIELD])
            np.testing.assert_allclose(slab.line_load(migrated)[1], slab.line_load(native)[1], rtol=1e-13)
            before = deepcopy(migrated.trench_systems)
            self.assertEqual(slab.bootstrap_slab_state(migrated)['bootstrapped'], 0)
            self.assertEqual(migrated.trench_systems, before)

    def test_absent_trace_shutdown_and_empty_area_have_no_geometric_pull(self):
        for change in (lambda s:s.trench_id.fill(0),
                       lambda s:s.trench_systems[0].update(phase='shutdown'),
                       lambda s:s.trench_systems[0].update(slab_retained_area_km2=0.,
                           slab_retained_buoyancy_area_km2=0., slab_retired_area_km2=100000.)):
            s = attached(convert(legacy_row()))
            change(s)
            before = deepcopy(s.trench_systems)
            np.testing.assert_array_equal(slab.line_load(s)[1], 0.)
            self.assertEqual(s.trench_systems, before)


if __name__ == '__main__': unittest.main()
