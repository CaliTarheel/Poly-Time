"""Induced initiation from the history of each fault stretch (policy version 2)."""
from copy import deepcopy
import unittest

import numpy as np

import effective_subduction as effective
import effective_subduction_initiation as initiation
from tests.test_effective_subduction_initiation import motion, set_pieces, tick, world

LOCAL = dict(version=2, enabled=True)


def local_world(**controls):
    s = world()
    initiation.upgrade(s, dict(LOCAL, **controls))
    return s


def graded(s, closing, shear, gradient):
    """Closing that grows along the fault: -closing - gradient*phi at angle phi."""
    s.omega[:] = 0.
    s.omega[0] = -np.array([gradient, shear, -closing])/6371.


def tracks(s):
    return s.effective_subduction_initiation['tracks']


class LocalInitiationTests(unittest.TestCase):
    def test_first_observation_receives_nothing_and_the_old_policy_is_untouched(self):
        s = local_world()
        state = s.effective_subduction_initiation
        self.assertEqual((state['version'], state['tracks'], state['candidates']), (2, [], []))
        self.assertTrue(initiation.local(s))
        tick(s)
        self.assertEqual(len(tracks(s)), 3)
        for row in tracks(s):
            self.assertEqual((row['incoming_plate_uid'], row['overriding_plate_uid']), (11, 22))
            self.assertEqual((row['loading_myr'], row['shortening_km'], row['shear_slip_km'], row['lapse_myr']), (0., 0., 0., 0.))
        old = world()
        initiation.upgrade(old)
        self.assertFalse(initiation.local(old))
        self.assertEqual(old.effective_subduction_initiation['version'], 1)

    def test_closing_with_shear_is_born_after_ten_loading_myr(self):
        s = local_world()
        tick(s)
        for _ in range(9):
            tick(s)
        self.assertFalse(s.trench_systems)
        self.assertTrue(all(row['loading_myr'] == 9. for row in tracks(s)))
        tick(s)
        self.assertEqual(len(s.trench_systems), 1)
        trench = s.trench_systems[0]
        self.assertTrue(trench['effective_subduction']['activation_pending'])
        self.assertEqual((trench['downgoing_plate_uid'], trench['overriding_plate_uid']), (11, 22))
        self.assertAlmostEqual(trench['length_km'], 3*127.42, delta=1e-6)
        self.assertFalse(np.any(effective.line_state(s)[1]))
        state = s.effective_subduction_initiation
        event = state['events'][-1]
        self.assertEqual((event['kind'], event['route']), ('eligible_pending', 'shortening and shear'))
        self.assertEqual(event['consecutive_myr'], 10.)
        self.assertGreaterEqual(event['shortening_km'], 100.)
        self.assertEqual([row['phase'] for row in state['candidates']], ['pending'])
        self.assertEqual(tracks(s), [])
        self.assertTrue(initiation.activate_pending(s))
        self.assertEqual(state['candidates'][0]['phase'], 'born')
        self.assertTrue(np.all(effective.line_state(s)[1] > 0.))
        tick(s)
        self.assertEqual(len(s.trench_systems), 1)

    def test_a_stretch_that_closes_is_born_though_the_rest_of_the_fault_does_not(self):
        # Six pieces; closing runs from opening at one end to 16 km/Myr at the other.
        s = local_world()
        set_pieces(s, np.linspace(-.06, .06, 7))
        graded(s, 6., 12., 200.)
        whole = world()
        set_pieces(whole, np.linspace(-.06, .06, 7))
        graded(whole, 6., 12., 200.)
        initiation.upgrade(whole)
        for _ in range(40):
            tick(s)
            tick(whole)
        # The whole-fault policy takes the slowest piece and can never qualify.
        self.assertFalse(whole.trench_systems)
        self.assertEqual(whole.effective_subduction_initiation['candidates'][0]['shortening_km'], 0.)
        self.assertEqual(len(s.trench_systems), 1)
        trench = s.trench_systems[0]
        self.assertAlmostEqual(trench['length_km'], 3*127.42, delta=1e-6)
        ends = np.asarray(trench['geometry_xyz'])
        self.assertGreater(np.arcsin(ends[:, 2]).min(), -1e-9)       # only the half that closes
        # The other half is still tracked, and is not loading.
        rest = tracks(s)
        self.assertEqual(len(rest), 3)
        self.assertTrue(all(row['loading_myr'] == 0. and row['shortening_km'] == 0. for row in rest))
        self.assertTrue(all(row['shear_slip_km'] > 100. for row in rest))

    def test_a_short_lapse_pauses_and_a_long_one_starts_again(self):
        for lapse, born_after in ((4, 4), (6, 10)):
            with self.subTest(lapse=lapse):
                s = local_world()
                tick(s)
                for _ in range(6):
                    tick(s)
                held = [row['shortening_km'] for row in tracks(s)]
                motion(s, 0., 12.)
                for _ in range(lapse):
                    tick(s)
                kept = [row['shortening_km'] for row in tracks(s)]
                self.assertEqual(kept, held if lapse <= 5 else [0., 0., 0.])
                self.assertEqual([row['lapse_myr'] for row in tracks(s)], [float(lapse)]*3)
                motion(s, 12., 12.)
                for _ in range(born_after-1):
                    tick(s)
                self.assertFalse(s.trench_systems)
                tick(s)
                self.assertEqual(len(s.trench_systems), 1)

    def test_a_stretch_that_has_qualified_bears_nothing_while_it_is_paused(self):
        # Too short to be born, so the fault qualifies and waits.
        s = local_world(minimum_length_km=1000.)
        tick(s)
        for _ in range(11):
            tick(s)
        ready = lambda: initiation.snapshot(s)['effective_subduction_initiation_diagnostics']['ready_length_km']
        self.assertAlmostEqual(ready(), 3*127.42, delta=1e-6)
        # The length rule is relaxed while the fault only slides: still nothing is born.
        for controls in (s.config['effective_subduction']['initiation'], s.effective_subduction_initiation['parameters']):
            controls['minimum_length_km'] = 300.
        motion(s, 0., 12.)
        tick(s)
        self.assertFalse(s.trench_systems)
        self.assertEqual(ready(), 0.)
        self.assertTrue(all(row['loading_myr'] == 11. and row['lapse_myr'] == 1. for row in tracks(s)))
        motion(s, 12., 12.)
        tick(s)
        self.assertEqual(len(s.trench_systems), 1)

    def test_head_on_shortening_alone_qualifies_at_the_higher_threshold(self):
        s = local_world()
        motion(s, 12., 0.)
        tick(s)
        for _ in range(10):
            tick(s)
        self.assertFalse(s.trench_systems)                       # ten Myr and 118 km, no shear
        self.assertTrue(all(100. < row['shortening_km'] < 200. and row['shear_slip_km'] == 0. for row in tracks(s)))
        ticks = 10
        while not s.trench_systems and ticks < 40:
            tick(s)
            ticks += 1
        event = s.effective_subduction_initiation['events'][-1]
        self.assertEqual(event['route'], 'shortening alone')
        self.assertGreaterEqual(event['shortening_km'], 200.)
        self.assertLess(event['shortening_km'], 212.)
        # The whole-fault policy asks every fault for shear slip and never accepts this one.
        old = world()
        initiation.upgrade(old)
        motion(old, 12., 0.)
        for _ in range(40):
            tick(old)
        self.assertFalse(old.trench_systems)

    def test_young_ocean_is_not_tracked(self):
        s = local_world()
        s.age[:] = 10.
        for _ in range(15):
            tick(s)
        self.assertEqual(tracks(s), [])
        self.assertFalse(s.trench_systems)

    def test_ground_added_to_a_trained_fault_receives_no_history(self):
        s = local_world(minimum_length_km=100.)
        tick(s)
        for _ in range(9):
            tick(s)
        set_pieces(s, np.array([-.03, -.01, .01, .03, .05]))      # one more piece at the far end
        motion(s, 12., 12.)
        tick(s)
        self.assertEqual(len(s.trench_systems), 1)
        self.assertAlmostEqual(s.trench_systems[0]['length_km'], 3*127.42, delta=1e-6)
        left = tracks(s)
        self.assertEqual(len(left), 1)
        self.assertEqual((left[0]['loading_myr'], left[0]['shortening_km']), (0., 0.))

    def test_a_parent_contact_is_not_born_while_one_of_its_pieces_is_untrained(self):
        # Two parent contacts of two pieces each; the second parent's far piece is new ground.
        s = local_world(minimum_length_km=100.)
        set_pieces(s, np.array([-.03, -.01, .01]), parents=[0, 0])
        motion(s, 12., 12.)
        tick(s)
        for _ in range(9):
            tick(s)
        set_pieces(s, np.array([-.03, -.01, .01, .03, .05]), parents=[0, 0, 1, 1])
        motion(s, 12., 12.)
        tick(s)
        self.assertEqual(len(s.trench_systems), 1)
        self.assertAlmostEqual(s.trench_systems[0]['length_km'], 2*127.42, delta=1e-6)
        self.assertEqual(len(tracks(s)), 2)

    def test_replacing_the_whole_fault_policy_keeps_its_lower_bounds_and_named_candidates(self):
        s = world()
        initiation.upgrade(s)
        tick(s)
        for _ in range(5):
            tick(s)
        before = deepcopy(s.effective_subduction_initiation)
        configured = deepcopy(s.config)
        row = before['candidates'][0]
        self.assertEqual(row['phase'], 'weakening')
        state = initiation.upgrade_local(s)
        self.assertEqual((state['version'], state['parameters']['version']), (2, 2))
        # The configuration still names the policy the world was made with; the state names
        # the one that replaced it, and what it replaced.
        self.assertEqual(s.config, configured)
        self.assertEqual(state['predecessor']['parameters'], before['parameters'])
        self.assertEqual(state['parameters']['minimum_shortening_alone_km'], 200.)
        self.assertEqual(state['predecessor']['candidates'], dict(weakening=1, pending=0, born=0, retired=0))
        self.assertEqual(len(tracks(s)), 3)
        for track in tracks(s):
            self.assertEqual(track['loading_myr'], row['consecutive_lower_bound_myr'])
            self.assertEqual(track['shortening_km'], row['shortening_lower_bound_km'])
            self.assertEqual(track['shear_slip_km'], row['shear_lower_bound_km'])
        self.assertEqual(tracks(s)[0]['loading_myr'], 5.)
        self.assertEqual(initiation.upgrade_local(s), state)       # already local: nothing changes
        for _ in range(4):
            tick(s)
        self.assertFalse(s.trench_systems)
        tick(s)
        self.assertEqual(len(s.trench_systems), 1)
        # A trench the old policy bore keeps the candidate its law names.
        born = world()
        initiation.upgrade(born)
        for _ in range(11):
            tick(born)
        named = born.trench_systems[0]['effective_subduction']['initiation_candidate_id']
        initiation.upgrade_local(born)
        self.assertEqual([row['id'] for row in born.effective_subduction_initiation['candidates']], [named])
        self.assertTrue(initiation.activate_pending(born))

    def test_controls_and_saved_state_are_validated(self):
        for raw in (dict(LOCAL, minimum_shortening_alone_km=50.), dict(LOCAL, minimum_consecutive_myr=10.),
                    dict(LOCAL, maximum_lapse_myr=0.), dict(LOCAL, unknown=1.), dict(version=3, enabled=True)):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                initiation.normalize(raw)
        self.assertEqual(initiation.normalize(LOCAL)['minimum_shortening_alone_km'], 200.)
        s = local_world()
        for _ in range(3):
            tick(s)
        self.assertTrue(initiation.enabled(s))
        diagnostics = initiation.snapshot(s)['effective_subduction_initiation_diagnostics']
        self.assertEqual(initiation.snapshot(s)['effective_subduction_initiation_version'], 2)
        self.assertEqual(diagnostics['fault_tracks'], 3)
        self.assertAlmostEqual(diagnostics['loading_length_km'], 3*127.42, delta=1e-6)
        self.assertEqual(diagnostics['ready_length_km'], 0.)
        for change in (dict(loading_myr=-1.), dict(start=[1., 0., 0.], end=[1., 0., 0.]),
                       dict(incoming_plate_uid=22), dict(last_seen_myr=1e9)):
            broken = deepcopy(s)
            broken.effective_subduction_initiation['tracks'][0].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'fault track'):
                initiation.enabled(broken)
        broken = deepcopy(s)
        broken.effective_subduction_initiation['policy'] = initiation.POLICY
        with self.assertRaisesRegex(ValueError, 'metadata'):
            initiation.enabled(broken)
        # A state that replaced the configured policy must name valid controls of its own and
        # the configured ones as its predecessor.
        replaced = world()
        initiation.upgrade(replaced)
        initiation.upgrade_local(replaced, dict(minimum_shortening_alone_km=250.))
        self.assertTrue(initiation.local(replaced))
        self.assertEqual(replaced.effective_subduction_initiation['parameters']['minimum_shortening_alone_km'], 250.)
        for change in (lambda state: state['parameters'].update(minimum_shortening_alone_km=50.),
                       lambda state: state['parameters'].update(enabled=False),
                       lambda state: state['parameters'].pop('maximum_lapse_myr'),
                       lambda state: state['predecessor']['parameters'].update(minimum_length_km=1.),
                       lambda state: state.pop('predecessor')):
            broken = deepcopy(replaced)
            change(broken.effective_subduction_initiation)
            with self.assertRaisesRegex(ValueError, 'metadata'):
                initiation.enabled(broken)
        s.t += .5
        with self.assertRaisesRegex(ValueError, 'clock'):
            initiation.update(s, 1.)


if __name__ == '__main__':
    unittest.main()
