"""Trench-driven admission begins with zero work and preserves material mass."""
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

import continental_entry
import checkpoint
import entry_regions
import material_surface
import native_engine
import plate_balance
import slab_tether_history as slab_history
from slab_tether import Neck
from native_engine import Simulation
from ridge_geometry import rotate
from tests.test_continental_entry import native_fixture, unit
from tests.test_slab_tether_native import world as slab_world, advance as native_advance


def approaching_face(*, crossed=False):
    state = native_fixture()
    state.t = 0.
    state.plate_balance_version = 1
    state.material_mechanics_version = 1
    state.config['deforming_regions'] = 1
    state.automatic_entry_version = 1
    state.parcel_patch = state.material_surface['face_id'].copy()
    state.parcel_collision_sheet = np.ones(1, np.int64)
    state.kind = np.ones(1, np.uint8)
    if not crossed:
        vertices = unit([[1., -.04, -.02], [1., -.02, -.01], [1., -1e-7, .04]])
        area = material_surface.spherical_face_areas(vertices, state.material_surface['faces'])
        state.material_surface['vertices'] = vertices
        state.material_surface['area_km2'] = area
        state.material_surface['reference_area_km2'] = area.copy()
        state.mass = area.copy()
    return state


def automatic_native_world(*,near_front=True):
    state = slab_world()
    state._add_arc_crust(np.array([34]), np.array([1000.]))
    rotation = [0., 0., -.05]
    state.material_surface['vertices'] = rotate(state.material_surface['vertices'], rotation)
    state.trace_xyz = rotate(state.trace_xyz, rotation)
    material_surface.refresh_geometry(state.material_surface)
    state._rasterize()
    state._boundaries()
    state.automatic_entry_version = 1
    state.material_mechanics_version = 1
    state.subduction_response_version = 1
    state.config['deforming_regions'] = 1
    if near_front:
        # The source fixture's arc is otherwise over 600 km from the trench.
        # Place it just short of first contact, with no initial entry work.
        def candidate(angle):
            probe=deepcopy(state)
            probe.material_surface['vertices']=rotate(probe.material_surface['vertices'],[0.,0.,angle])
            probe.trace_xyz=rotate(probe.trace_xyz,[0.,0.,angle])
            material_surface.refresh_geometry(probe.material_surface)
            probe._rasterize();probe._boundaries()
            return probe
        low,high=.1,.12
        for _ in range(25):
            middle=(low+high)/2.
            try:entry_regions.admit_approaching(candidate(middle),0.)
            except ValueError as error:
                if 'before zero-energy admission' not in str(error):raise
                high=middle
            else:low=middle
        state=candidate(low-1e-7)
    return state


class AutomaticEntryTests(unittest.TestCase):
    def test_detached_coupled_trace_cannot_admit_new_continental_faces(self):
        state=automatic_native_world()
        model=plate_balance.Balance(state,1.,slab_tethers=True)
        model.solve();forecast=model.rotation()
        state.trench_shutdown_version=1
        row=state.trench_systems[0]
        for index,channel in enumerate(row[slab_history.FIELD]):
            slab_history.rupture(row,index,replace(Neck(**channel['neck']),damage=1.))
        report=entry_regions.admit_approaching(state,.000005,forecast_omega=forecast)
        self.assertEqual(report['admitted_face_ids'],[])
        self.assertFalse(hasattr(state,'continental_entry_regions'))

    def test_registered_coupled_entry_requires_attached_source_slab(self):
        state=automatic_native_world()
        model=plate_balance.Balance(state,1.,slab_tethers=True)
        model.solve()
        admitted=entry_regions.admit_approaching(state,.000005,forecast_omega=model.rotation())
        self.assertTrue(admitted['admitted_face_ids'])
        state.trench_shutdown_version=1
        self.assertEqual(entry_regions.energy_j(state),0.)
        row=state.trench_systems[0]
        for index,channel in enumerate(row[slab_history.FIELD]):
            slab_history.rupture(row,index,replace(Neck(**channel['neck']),damage=1.))
        with self.assertRaisesRegex(ValueError,'lost its attached source slab without a breakoff handoff'):
            entry_regions.energy_j(state)

    def test_native_rupture_with_registered_entry_rolls_back_without_handoff(self):
        state=automatic_native_world()
        model=plate_balance.Balance(state,1.,slab_tethers=True)
        model.solve()
        admitted=entry_regions.admit_approaching(state,.000005,forecast_omega=model.rotation())
        self.assertEqual(len(admitted['admitted_face_ids']),3)
        for channel in state.trench_systems[0][slab_history.FIELD]:
            channel['neck']['failure_opening_m']=1e-6
            channel['neck']['damage']=.99
        slab_history.validate(state.trench_systems[0])
        before=deepcopy(state)
        with self.assertRaisesRegex(ValueError,'lost its attached source slab without a breakoff handoff'):
            native_advance(state,.00005,max_source_step_myr=.000005)
        self.assertEqual(state.t,before.t)
        self.assertEqual(state.trench_systems,before.trench_systems)
        np.testing.assert_array_equal(state.parcel_entry_region,before.parcel_entry_region)
        np.testing.assert_array_equal(state.material_surface['vertices'],
                                      before.material_surface['vertices'])
        self.assertEqual(state.rng.bit_generator.state,before.rng.bit_generator.state)

    def test_failed_native_source_rolls_back_automatic_registration(self):
        state = automatic_native_world()
        before = deepcopy(state)
        original = native_engine.Simulation._advance_step
        def fail_after_source(staged,dt,**kwargs):
            original(staged,dt,**kwargs)
            raise ValueError('injected failure after automatic entry source')
        with patch.object(native_engine.Simulation,'_advance_step',fail_after_source), \
             self.assertRaisesRegex(ValueError,'injected failure'):
            native_advance(state,.000005,max_source_step_myr=.000005)
        self.assertFalse(hasattr(state,'continental_entry_regions'))
        self.assertEqual(state.t,before.t)
        np.testing.assert_array_equal(state.material_surface['vertices'],
                                      before.material_surface['vertices'])
        self.assertEqual(state.rng.bit_generator.state,
                         before.rng.bit_generator.state)

    def test_entry_potential_vanishes_beyond_finite_trench_arc(self):
        state = approaching_face()
        entry_regions.admit_approaching(state, .01)
        potential = entry_regions.frozen(state)
        entered = rotate(state.material_surface['vertices'], [0., 0., .03])
        inside = potential.evaluate(entered, state.material_surface['faces'],
                                    potential.volumes, potential.sheets,
                                    radius=potential.radius)
        self.assertGreater(inside['energy_j'], 0.)
        escaped = rotate(entered, [0., .3, 0.])
        outside = potential.evaluate(escaped, state.material_surface['faces'],
                                     potential.volumes, potential.sheets,
                                     radius=potential.radius)
        self.assertEqual(outside['energy_j'], 0.)
        np.testing.assert_array_equal(outside['vertex_gradient_j'], 0.)

    def test_solved_admission_and_checkpoint_preserve_local_regions(self):
        state = automatic_native_world()
        model=plate_balance.Balance(state,1.,slab_tethers=True)
        model.solve()
        first=entry_regions.admit_approaching(state,.000005,forecast_omega=model.rotation())
        count=len(first['admitted_face_ids'])
        self.assertGreater(count,0)
        self.assertLess(count,24)
        self.assertEqual(entry_regions.energy_j(state),0.)
        admitted = np.isin(state.parcel_patch,
                          first['admitted_face_ids'])
        self.assertEqual(int(admitted.sum()), count)
        self.assertTrue(np.all(state.parcel_entry_region[admitted] > 0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'automatic-entry.npz'
            checkpoint.write_checkpoint(path, state, dict(config=state.config), {})
            resumed, _ = checkpoint.read_checkpoint(path, None, Simulation)
        np.testing.assert_array_equal(state.parcel_entry_region, resumed.parcel_entry_region)
        np.testing.assert_array_equal(state.material_surface['vertices'],
                                      resumed.material_surface['vertices'])
        self.assertEqual(entry_regions.energy_j(resumed),0.)
        left=plate_balance.Balance(state,1.,slab_tethers=True)
        right=plate_balance.Balance(resumed,1.,slab_tethers=True)
        left.solve();right.solve()
        np.testing.assert_allclose(left.rotation(),right.rotation(),rtol=0.,atol=1e-14)

    def test_native_source_and_checkpoint_replay_preserve_local_entry(self):
        state=automatic_native_world()
        initial_trench_length=sum(row['length_km'] for row in state.trench_systems)
        first=native_advance(state,.000005,max_source_step_myr=.000005)
        admitted=first['automatic_entry_admissions'][0]['admitted_face_ids']
        self.assertEqual(len(admitted),3)
        self.assertTrue(first['persistent_entry_regions'])
        self.assertEqual(first['final_zero_work_admission']['admitted_face_ids'],[])
        self.assertEqual(state.continental_entry_regions['epoch_myr'],state.t)
        self.assertLessEqual(state.process_totals['trench_swept_km2'],
                             8.*.000005*initial_trench_length)
        budget=state.material_column_budget
        self.assertLess(abs(budget['residual_km3'])/budget['after_columns_volume_km3'],1e-11)
        interval=first['source_intervals'][0]
        self.assertLess(abs(interval['entry_energy_after_j']-
                            interval['entry_energy_after_motion_j']-
                            interval['entry_source_energy_change_j']-
                            interval['entry_remesh_energy_change_j'])/
                        max(abs(interval['entry_energy_after_j']),1.),1e-12)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'native-entry.npz'
            checkpoint.write_checkpoint(path,state,dict(config=state.config),{})
            resumed,_=checkpoint.read_checkpoint(path,None,Simulation)
        left=native_advance(state,.000005,max_source_step_myr=.000005)
        right=native_advance(resumed,.000005,max_source_step_myr=.000005)
        self.assertEqual(left,right)
        np.testing.assert_array_equal(state.parcel_entry_region,resumed.parcel_entry_region)
        np.testing.assert_array_equal(state.material_surface['vertices'],
                                      resumed.material_surface['vertices'])

    def test_distant_material_is_not_admitted_during_native_source(self):
        state=automatic_native_world(near_front=False)
        report=native_advance(state,.000005,max_source_step_myr=.000005)
        self.assertEqual(report['automatic_entry_admissions'][0]['admitted_face_ids'],[])
        self.assertFalse(report['persistent_entry_regions'])
        self.assertFalse(hasattr(state,'continental_entry_regions'))

    def test_new_material_birth_appends_region_without_resetting_old_identity(self):
        state = approaching_face()
        entry_regions.admit_approaching(state, .01)
        original = state.parcel_entry_region.copy()
        new_vertices = unit([[1., -.04, .045], [1., -1e-7, .046], [1., -.02, .075]])
        new_area = material_surface.spherical_face_areas(new_vertices, [[0, 1, 2]])
        surface = state.material_surface
        offset = len(surface['vertices'])
        surface['vertices'] = np.vstack((surface['vertices'], new_vertices))
        surface['faces'] = np.vstack((surface['faces'], [[offset, offset+1, offset+2]]))
        surface['face_id'] = np.r_[surface['face_id'], 918]
        surface['face_owner'] = np.r_[surface['face_owner'], 0]
        surface['vertex_owner'] = np.r_[surface['vertex_owner'], [0, 0, 0]]
        surface['area_km2'] = np.r_[surface['area_km2'], new_area]
        surface['reference_area_km2'] = np.r_[surface['reference_area_km2'], new_area]
        state.parcel_patch = np.r_[state.parcel_patch, 918]
        state.parcel_plate = np.r_[state.parcel_plate, 0]
        state.parcel_collision_sheet = np.r_[state.parcel_collision_sheet, 2]
        state.kind = np.r_[state.kind, 1]
        state.mass = np.r_[state.mass, new_area]
        for key in ('thickness_km', 'area_factor'):
            state.structure[key] = np.r_[state.structure[key], state.structure[key][0]]
        result = entry_regions.admit_approaching(state, .01)
        self.assertEqual(result['admitted_face_ids'], [918])
        np.testing.assert_array_equal(state.parcel_entry_region[:1], original)
        self.assertEqual(len(np.unique(state.parcel_entry_region)), 2)

    def test_zero_work_registration_precedes_partial_face_entry(self):
        state = approaching_face()
        volume, mass = continental_entry.native_inventory(state)
        result = entry_regions.admit_approaching(state, .01)
        self.assertEqual(result['admitted_face_ids'], [917])
        self.assertEqual(state.continental_entry_regions['assignment'],
                         'automatic finite trench front')
        self.assertEqual(entry_regions.energy_j(state), 0.)
        self.assertEqual(entry_regions.admit_approaching(state, .01)['admitted_face_ids'], [])
        state.material_surface['vertices'] = rotate(state.material_surface['vertices'],
                                                    [0., 0., .03])
        state.omega[0] = [0., 0., .03]
        state.omega[1] = 0.
        entry_regions.advance_hinges(state, 1.)
        state.t = 1.
        potential = entry_regions.frozen(state)
        measured = potential.evaluate(state.material_surface['vertices'],
                                      state.material_surface['faces'],
                                      potential.volumes, potential.sheets,
                                      radius=potential.radius)
        self.assertGreater(measured['energy_j'], 0.)
        self.assertGreater(measured['entered_reference_fraction'][0], 0.)
        self.assertLess(measured['entered_reference_fraction'][0], 1.)
        after_volume, after_mass = continental_entry.native_inventory(state)
        np.testing.assert_array_equal(after_volume, volume)
        np.testing.assert_array_equal(after_mass, mass)

    def test_finite_front_admits_closing_motion_but_not_opening_motion(self):
        state = approaching_face()
        state.material_surface['vertices'] = rotate(
            state.material_surface['vertices'], [0., 0., -.0005])
        toward = np.array([[0., 0., .01], [0., 0., 0.]])
        away = -toward
        common = np.array([[0., 0., .01], [0., 0., .01]])
        self.assertEqual(entry_regions.admit_approaching(state, .1,
                         forecast_omega=away)['admitted_face_ids'], [])
        self.assertEqual(entry_regions.admit_approaching(state, .1,
                         forecast_omega=common)['admitted_face_ids'], [])
        self.assertFalse(hasattr(state, 'continental_entry_regions'))
        self.assertEqual(entry_regions.admit_approaching(state, .1,
                         forecast_omega=toward)['admitted_face_ids'], [917])
        self.assertEqual(entry_regions.energy_j(state), 0.)

    def test_finite_edge_predictor_is_pure_and_uses_relative_future_geometry(self):
        state = approaching_face()
        vertices = rotate(state.material_surface['vertices'], [0., 0., -.0005])
        triangles = vertices[state.material_surface['faces']]
        before = triangles.copy()
        center = unit(triangles.sum(axis=1))
        radius = np.max(np.arccos(np.clip(np.einsum('fi,fji->fj', center, triangles), -1., 1.)), axis=1)*6371.
        row = state.trench_systems[0]
        down = int(np.flatnonzero(state.plate_uid == row['downgoing_plate_uid'])[0])
        over = int(np.flatnonzero(state.plate_uid == row['overriding_plate_uid'])[0])
        # This fixture has two co-located finite edges; the longer one reaches
        # the face during the lookahead.
        edge = int(np.argmax(np.where(state.trench_id == row['id'], state.bl, -1.)))
        midpoint = unit(state.bmid[edge])
        normal = state.bn[edge]*(1. if state.bp[edge] == down else -1.)
        normal = unit(normal-midpoint*np.dot(midpoint, normal))
        def predict(down_omega, over_omega):
            return entry_regions._edge_approach_candidates(triangles, center, radius,
                np.array([0]), midpoint, normal, float(state.bl[edge])*.5,
                np.asarray(down_omega), np.asarray(over_omega), .1)
        self.assertEqual(predict([0., 0., -.01], [0., 0., 0.]), [])
        self.assertEqual(predict([0., 0., .01], [0., 0., .01]), [])
        self.assertEqual([index for index, _ in predict([0., 0., .01], [0., 0., 0.])], [0])
        np.testing.assert_array_equal(triangles, before)

    def test_already_crossed_unregistered_face_is_rejected_without_mutation(self):
        state = approaching_face(crossed=True)
        before = state.material_surface['vertices'].copy()
        with self.assertRaisesRegex(ValueError, 'before zero-energy admission'):
            entry_regions.admit_approaching(state, .01)
        self.assertFalse(hasattr(state, 'continental_entry_regions'))
        np.testing.assert_array_equal(state.material_surface['vertices'], before)

    def test_empty_native_world_keeps_original_trajectory(self):
        state = slab_world()
        state.automatic_entry_version = 1
        report = native_advance(state, .000001)
        self.assertEqual(report['automatic_entry_admissions'][0]['admitted_face_ids'], [])
        self.assertFalse(report['persistent_entry_regions'])

    def test_finite_front_native_source_path_consumes_the_endpoint_without_a_sliver(self):
        from experiments.native_entry_approach import run

        report = run(.5, .05, automatic_finite_front=True)
        self.assertEqual(report['accepted_source_intervals'], 10)
        self.assertGreater(report['actual_source_step_min_myr'], .049)
        self.assertEqual(report['automatic_admitted_faces'], 3)
        self.assertGreater(report['entered_reference_area_km2'], 0.)
        self.assertGreater(report['entry_energy_j'], 0.)
        self.assertEqual(report['active_contact_count'], 0)
        self.assertEqual(report['rupture_count'], 0)


if __name__ == '__main__':
    unittest.main()
