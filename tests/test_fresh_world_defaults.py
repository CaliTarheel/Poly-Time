"""New reviewed worlds select Lite; saved configs keep their recorded laws.

The Lite law choices (effective force, fixed trench and enhanced rifting)
are made once, where a world is created,
and written explicitly into config.json. The engine's missing-key defaults stay
the historical laws, because branch re-validates saved configs and
enhanced_rifting.enabled reads config each step. The selection lives in the
server-only fresh_world module so the hashed engine helper sources, and with
them every saved checkpoint's compatibility, are untouched.
"""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
from http.server import ThreadingHTTPServer
from urllib.request import urlopen
import unittest
from unittest.mock import patch

import fresh_world
import budget
import native_engine
import physics_profile
import progressive_rifting
import run_simulation
import server
from server import SimulationManager


# SEP21T (run 20260921-011257-b2f85b) config.json values for these laws.
SEP21T_RIFTING = dict(version=1, enabled=True, seed=37, amplitude=.25,
                      correlation_km=350., craton_margin_km=150.)
SMALL = dict(width=48, height=24, mesh_level=2, coast_geometry_level=2, mechanics_nodes=128,
             adaptive_refinement=0, seed=37, duration_myr=4, dt_myr=2, snapshot_myr=2, plate_count=4)


def fresh(config):
    return native_engine.validate_config(fresh_world.fresh_world_request(config))


class FreshWorldRequestTests(unittest.TestCase):
    def test_reviewed_request_selects_lite_laws(self):
        self.assertEqual(fresh_world.REVIEWED_PROFILE, physics_profile.NAME)
        config = fresh({'physics_profile': 'reviewed_v1'})
        self.assertEqual(config['retained_phases'], 'disabled')
        self.assertEqual(config['enhanced_rifting'], SEP21T_RIFTING)
        self.assertEqual(config['subduction_response'], 'fixed_trench')
        self.assertEqual(fresh_world.REVIEWED_FRESH_LAWS['subduction_response'], 'fixed_trench')
        self.assertEqual(config['effective_subduction'], {'enabled': True, 'force_n_per_m': 3e13})
        self.assertTrue(config['primordial_subduction']['enabled'])
        self.assertTrue(config['primordial_ocean']['enabled'])
        self.assertFalse(config['continental_lifecycle']['enabled'])
        self.assertFalse(config['rift_traction']['enabled'])
        self.assertFalse(config['force_limit_rifting']['enabled'])
        fixed = fresh({'physics_profile': 'reviewed_v1', 'subduction_response': 'fixed_trench'})
        self.assertEqual(fixed['subduction_response'], 'fixed_trench')

    def test_explicit_detailed_mode_selects_prior_fresh_laws(self):
        config = fresh({'physics_profile': 'reviewed_v1', 'effective_subduction': {'enabled': False}})
        self.assertFalse(config['effective_subduction']['enabled'])
        self.assertEqual(config['subduction_response'], 'moving_hinge_v1')
        self.assertEqual(config['retained_phases'], 'thermal_v1')
        self.assertEqual(config['enhanced_rifting'], SEP21T_RIFTING)
        self.assertFalse(config['primordial_subduction']['enabled'])
        self.assertFalse(config['primordial_ocean']['enabled'])

    def test_fresh_partial_force_uses_new_default_and_explicit_force_survives(self):
        for settings in ({}, {'enabled': True}):
            request = dict(physics_profile='reviewed_v1', effective_subduction=settings)
            before = deepcopy(request)
            self.assertEqual(fresh(request)['effective_subduction'],
                             dict(enabled=True, force_n_per_m=3e13))
            self.assertEqual(request, before)
        for force in (5e12, 9e12, 3e13):
            request = dict(physics_profile='reviewed_v1',
                           effective_subduction=dict(enabled=True, force_n_per_m=force))
            self.assertEqual(fresh(request)['effective_subduction']['force_n_per_m'], force)
        # Revalidating an older enabled law without a recorded force still uses
        # the engine's historical 5e12 default, without fresh-world selection.
        saved = native_engine.validate_config(dict(physics_profile='reviewed_v1',
            effective_subduction=dict(enabled=True), primordial_subduction=dict(enabled=True)))
        self.assertEqual(saved['effective_subduction']['force_n_per_m'], 5e12)

    def test_ocean_defaults_preserve_parameters_opt_out_and_alternative_initializer(self):
        ocean = dict(plate_count=6, half_spreading_rate_cm_yr=3., maximum_age_myr=150.,
                     continental_attachments=[dict(region_index=1, continental_plate_uid=3)])
        request = dict(physics_profile='reviewed_v1', primordial_ocean=ocean)
        before = deepcopy(request)
        self.assertEqual(fresh(request)['primordial_ocean'], dict(ocean, enabled=True))
        self.assertEqual(request, before)
        off = fresh(dict(physics_profile='reviewed_v1', primordial_ocean={'enabled': False}))
        self.assertFalse(off['primordial_ocean']['enabled'])
        ring = dict(physics_profile='reviewed_v1', supercontinent_ring={'enabled': True})
        self.assertFalse(fresh(ring)['primordial_ocean']['enabled'])
        with self.assertRaisesRegex(ValueError, 'alternative ocean initializations'):
            fresh(dict(ring, primordial_ocean={'enabled': True}))

    def test_lite_keeps_the_full_detail_contract(self):
        config = server.fresh_world_defaults()
        expected = dict(width=512, height=256, mesh_level=4,
                        coast_geometry_level=4, adaptive_refinement=1,
                        mechanics_nodes=1024, deforming_regions=1, snapshot_myr=2.)
        self.assertEqual({key: config[key] for key in expected}, expected)
        higher_detail = dict(width=2048, height=1024, mesh_level=5,
                             coast_geometry_level=5, adaptive_refinement=2,
                             mechanics_nodes=4096, dt_myr=1., snapshot_myr=1.)
        explicit = fresh(dict(higher_detail, physics_profile='reviewed_v1'))
        self.assertEqual({key: explicit[key] for key in higher_detail}, higher_detail)

    def test_map_declaration_is_adopted_without_overriding_an_explicit_selection(self):
        initial = dict(initial_subduction=dict(enabled=True, target_margin_fraction=.35, selection_seed=41),
                       continental_lifecycle={'enabled': True}, rift_traction={'enabled': True})
        before = deepcopy(initial)
        request = fresh_world.fresh_world_request({'physics_profile': 'reviewed_v1'}, initial)
        self.assertEqual(request['primordial_subduction'], initial['initial_subduction'])
        self.assertFalse(request['continental_lifecycle']['enabled'])
        self.assertFalse(request['rift_traction']['enabled'])
        explicit = dict(enabled=True, target_margin_fraction=.7, selection_seed=9)
        request = fresh_world.fresh_world_request(dict(physics_profile='reviewed_v1',
            primordial_subduction=explicit), initial)
        self.assertEqual(request['primordial_subduction'], explicit)
        self.assertEqual(initial, before)
        partial = fresh(dict(physics_profile='reviewed_v1', primordial_subduction={'selection_seed': 17},
                             effective_subduction={'force_n_per_m': 8e12}))
        self.assertEqual(partial['primordial_subduction']['selection_seed'], 17)
        self.assertTrue(partial['primordial_subduction']['enabled'])
        self.assertEqual(partial['effective_subduction'], dict(enabled=True, force_n_per_m=8e12))

    def test_explicit_incompatible_lite_laws_are_not_silently_overridden(self):
        for override in ({'subduction_response': 'moving_hinge_v1'}, {'retained_phases': 'thermal_v1'},
                         {'primordial_subduction': {'enabled': False}},
                         {'continental_lifecycle': {'enabled': True}}, {'rift_traction': {'enabled': True}}):
            request = dict(physics_profile='reviewed_v1', **override)
            chosen = fresh_world.fresh_world_request(request)
            for key, value in override.items():
                self.assertEqual(chosen[key], value)
            with self.subTest(override=override), self.assertRaises(ValueError):
                native_engine.validate_config(chosen)

    def test_explicit_choices_survive(self):
        self.assertEqual(fresh({'physics_profile': 'reviewed_v1',
                                'subduction_response': 'fixed_trench'})['subduction_response'], 'fixed_trench')
        self.assertEqual(fresh({'physics_profile': 'reviewed_v1',
                                'retained_phases': 'disabled'})['retained_phases'], 'disabled')
        off = fresh({'physics_profile': 'reviewed_v1', 'enhanced_rifting': {'enabled': False}})
        self.assertFalse(off['enhanced_rifting']['enabled'])
        off = fresh({'physics_profile': 'reviewed_v1', 'enhanced_rifting': {'enabled': False, 'seed': 5}})
        self.assertEqual((off['enhanced_rifting']['enabled'], off['enhanced_rifting']['seed']), (False, 5))
        partial = fresh({'physics_profile': 'reviewed_v1', 'enhanced_rifting': {'seed': 5}})
        self.assertEqual(partial['enhanced_rifting'], dict(SEP21T_RIFTING, seed=5))

    def test_request_is_not_mutated(self):
        request = {'physics_profile': 'reviewed_v1', 'enhanced_rifting': {'seed': 5}}
        before = deepcopy(request)
        fresh_world.fresh_world_request(request)
        self.assertEqual(request, before)

    def test_legacy_and_missing_profiles_are_unchanged(self):
        for request in ({}, {'physics_profile': 'legacy'}, {'seed': 3, 'width': 128, 'height': 64},
                        {'physics_profile': 'legacy', 'enhanced_rifting': {'seed': 9}}):
            with self.subTest(request=request):
                self.assertEqual(fresh_world.fresh_world_request(request), request)
                self.assertEqual(fresh(request), native_engine.validate_config(request))
        self.assertEqual(fresh_world.fresh_world_request(None), {})

    def test_engine_missing_key_contract_is_unchanged(self):
        config = native_engine.validate_config({'physics_profile': 'reviewed_v1'})
        self.assertEqual(config['subduction_response'], 'fixed_trench')
        self.assertEqual(config['retained_phases'], 'disabled')
        self.assertFalse(config['enhanced_rifting']['enabled'])
        # Shape of the key-less 2026-09-15 reviewed runs (e.g. 20260915-190512):
        # re-validation on branch must keep their recorded laws.
        saved = native_engine.validate_config(dict(SMALL, physics_profile='reviewed_v1'))
        for key in ('subduction_response', 'retained_phases'):
            saved.pop(key)
        rebranched = native_engine.validate_config(saved)
        self.assertEqual(rebranched['subduction_response'], 'fixed_trench')
        self.assertEqual(rebranched['retained_phases'], 'disabled')
        self.assertEqual(native_engine.NATIVE_DEFAULTS['physics_profile'], 'legacy')
        self.assertEqual(native_engine.NATIVE_DEFAULTS['subduction_response'], 'fixed_trench')
        self.assertEqual(native_engine.NATIVE_DEFAULTS['retained_phases'], 'disabled')
        self.assertIsNone(native_engine.NATIVE_DEFAULTS['enhanced_rifting'])
        self.assertFalse(config['effective_subduction']['enabled'])
        self.assertFalse(config['primordial_ocean']['enabled'])

    def test_selection_is_outside_the_hashed_engine_sources(self):
        # Checkpoints record a hash of every engine helper; resume and branch
        # refuse a mismatch. Choosing fresh-world settings must not touch them.
        self.assertIn('physics_profile.py', server.AUXILIARY_SOURCES)
        self.assertNotIn('fresh_world.py', server.AUXILIARY_SOURCES)
        for name, payload in server.AUXILIARY_SOURCES.items():
            self.assertNotIn(b'fresh_world', payload, name)
        self.assertNotIn(b'fresh_world', server.ENGINE_SOURCE)


class ServerFreshWorldTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1]/'tmp'
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.manager = SimulationManager(self.temporary.name)
        self.addCleanup(self.manager.parallel.close)

    def test_form_defaults_are_reviewed_and_engine_defaults_untouched(self):
        before = deepcopy(server.DEFAULT_CONFIG)
        defaults = server.fresh_world_defaults()
        self.assertEqual(server.DEFAULT_CONFIG, before)
        self.assertEqual(server.DEFAULT_CONFIG['physics_profile'], 'legacy')
        self.assertEqual(defaults['physics_profile'], 'reviewed_v1')
        self.assertEqual(defaults['subduction_response'], 'fixed_trench')
        self.assertEqual(defaults['retained_phases'], 'disabled')
        self.assertTrue(defaults['effective_subduction']['enabled'])
        self.assertEqual(defaults['effective_subduction']['force_n_per_m'], 3e13)
        self.assertEqual(defaults['enhanced_rifting'], SEP21T_RIFTING)
        self.assertEqual(set(defaults), set(server.DEFAULT_CONFIG))
        normalized_legacy = server.validate_config(deepcopy(server.DEFAULT_CONFIG))
        for key in set(server.DEFAULT_CONFIG) - {'physics_profile', 'subduction_response',
                                                 'retained_phases', 'enhanced_rifting',
                                                 'world_design', 'lip_events',
                                                 'primordial_ocean', 'primordial_subduction',
                                                 'force_limit_rifting', 'supercontinent_ring',
                                                 'effective_subduction', 'continental_lifecycle', 'rift_traction'}:
            self.assertEqual(defaults[key], normalized_legacy[key], key)
        self.assertTrue(defaults['primordial_ocean']['enabled'])
        self.assertTrue(defaults['primordial_subduction']['enabled'])
        self.assertFalse(defaults['lip_events']['enabled'])
        self.assertFalse(defaults['force_limit_rifting']['enabled'])
        self.assertFalse(defaults['supercontinent_ring']['enabled'])

    def test_http_config_route_serves_fresh_defaults(self):
        class LocalHandler(server.Handler):
            manager = self.manager
            def log_message(self, *args):
                pass
        http = ThreadingHTTPServer(('127.0.0.1', 0), LocalHandler)
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(http.server_close)
        self.addCleanup(http.shutdown)
        with urlopen(f'http://127.0.0.1:{http.server_port}/api/config', timeout=10) as response:
            body = json.load(response)
        self.assertEqual(body['defaults'], json.loads(json.dumps(server.fresh_world_defaults())))

    def start(self, config, initial=None):
        with patch.object(SimulationManager, '_run', lambda self, *args: None):
            run = self.manager.start(config, initial)
        self.manager.worker.join(timeout=30)
        return json.loads((Path(self.temporary.name)/run['run_id']/'config.json').read_text(encoding='utf-8'))

    def test_start_writes_reviewed_laws_explicitly(self):
        saved = self.start(dict(SMALL, physics_profile='reviewed_v1'))
        self.assertEqual(saved['subduction_response'], 'fixed_trench')
        self.assertEqual(saved['retained_phases'], 'disabled')
        self.assertTrue(saved['effective_subduction']['enabled'])
        self.assertEqual(saved['effective_subduction']['force_n_per_m'], 3e13)
        self.assertTrue(saved['primordial_subduction']['enabled'])
        self.assertTrue(saved['primordial_ocean']['enabled'])
        self.assertEqual(saved['enhanced_rifting'], SEP21T_RIFTING)
        # Re-validating the saved file (branch) reproduces it exactly.
        self.assertEqual(native_engine.validate_config(saved), saved)

    def test_start_adopts_imported_map_declaration_and_keeps_explicit_override(self):
        initial = native_engine.make_initial(SMALL)
        initial['initial_subduction'] = dict(enabled=True, target_margin_fraction=.35, selection_seed=41)
        initial['continental_lifecycle'] = {'enabled': True}
        initial['rift_traction'] = {'enabled': True}
        saved = self.start(dict(SMALL, physics_profile='reviewed_v1'), initial)
        self.assertEqual(saved['primordial_subduction']['target_margin_fraction'], .35)
        self.assertEqual(saved['primordial_subduction']['selection_seed'], 41)
        self.assertFalse(saved['continental_lifecycle']['enabled'])
        self.assertFalse(saved['rift_traction']['enabled'])
        saved = self.start(dict(SMALL, physics_profile='reviewed_v1',
            primordial_subduction=dict(enabled=True, target_margin_fraction=.7, selection_seed=9)), initial)
        self.assertEqual(saved['primordial_subduction']['target_margin_fraction'], .7)
        self.assertEqual(saved['primordial_subduction']['selection_seed'], 9)

    def test_start_explicit_detailed_mode_preserves_prior_fresh_selection(self):
        saved = self.start(dict(SMALL, physics_profile='reviewed_v1', effective_subduction={'enabled': False}))
        self.assertEqual(saved['subduction_response'], 'moving_hinge_v1')
        self.assertEqual(saved['retained_phases'], 'thermal_v1')
        self.assertFalse(saved['effective_subduction']['enabled'])

    def test_start_without_profile_is_legacy_as_before(self):
        saved = self.start(dict(SMALL))
        self.assertEqual(saved, json.loads(json.dumps(native_engine.validate_config(dict(SMALL)))))
        self.assertEqual(saved['physics_profile'], 'legacy')
        self.assertFalse(saved['enhanced_rifting']['enabled'])

    def test_branch_does_not_apply_fresh_selection(self):
        # branch() re-validates the saved simulation config with validate_config
        # only; this pins that fresh_world_request is not reached from it.
        source = Path(server.__file__).read_text(encoding='utf-8')
        branch = source[source.index('    def branch('):source.index('\n    def ', source.index('    def branch(')+1)]
        self.assertNotIn('fresh_world_request', branch)
        self.assertIn('validate_config(dict(simulation.config', branch)


class CommandLineTests(unittest.TestCase):
    def args(self, **values):
        base = dict(config=None, initial=None, seed=None, width=None, duration=None,
                    physics_profile=None, workers=None)
        return Namespace(**dict(base, **values))

    def test_default_profile_matches_the_app(self):
        self.assertEqual(run_simulation.build_config(self.args())['physics_profile'], 'reviewed_v1')
        self.assertTrue(run_simulation.build_config(self.args())['primordial_ocean']['enabled'])
        parsed = run_simulation.parser().parse_args(['--physics-profile', 'legacy', '--seed', '4'])
        config = run_simulation.build_config(parsed)
        self.assertEqual((config['physics_profile'], config['seed']), ('legacy', 4))

    def rerun(self, saved, **flags):
        """build_config then start()'s selection and validation, from a file."""
        scratch = Path(__file__).resolve().parents[1]/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            path = Path(folder)/'config.json'
            path.write_text(json.dumps(saved), encoding='utf-8')
            return fresh(run_simulation.build_config(self.args(config=path, **flags)))

    def test_config_file_profile_and_laws_are_respected(self):
        self.assertEqual(self.rerun({'physics_profile': 'legacy', 'seed': 8}),
                         native_engine.validate_config({'physics_profile': 'legacy', 'seed': 8}))
        explicit = {'physics_profile': 'reviewed_v1', 'subduction_response': 'moving_hinge_v1',
                    'retained_phases': 'thermal_v1', 'enhanced_rifting': {'enabled': True}}
        self.assertEqual(self.rerun(explicit), native_engine.validate_config(explicit))

    def test_saved_partial_ocean_config_keeps_historical_disabled_meaning(self):
        for ocean in ({}, {'plate_count': 6}, {'enabled': False}, {'enabled': True, 'plate_count': 3}):
            saved = dict(physics_profile='reviewed_v1', primordial_ocean=ocean)
            self.assertEqual(self.rerun(saved), native_engine.validate_config(saved))

    def test_saved_partial_force_keeps_historical_value(self):
        for settings in ({}, {'force_n_per_m': 9e12}, {'enabled': True},
                         *[dict(enabled=True, force_n_per_m=force) for force in (5e12, 9e12, 3e13)]):
            saved = dict(physics_profile='reviewed_v1', effective_subduction=settings,
                         primordial_subduction=dict(enabled=True))
            self.assertEqual(self.rerun(saved), native_engine.validate_config(saved))

    def test_saved_config_without_profile_reproduces_as_legacy(self):
        # Runs saved before physics_profile existed (e.g. 20260915-180344).
        original = native_engine.validate_config(dict(SMALL))
        saved = json.loads(json.dumps(original))
        saved.pop('physics_profile')
        self.assertEqual(self.rerun(saved), original)

    def test_keyless_saved_reviewed_config_keeps_its_recorded_laws(self):
        # Shape of the four 2026-09-15 reviewed runs: no law keys, rifting explicit off.
        original = native_engine.validate_config(dict(SMALL, physics_profile='reviewed_v1'))
        saved = json.loads(json.dumps(original))
        for key in ('subduction_response', 'retained_phases'):
            saved.pop(key)
        self.assertEqual(self.rerun(saved), original)
        # Even with no enhanced_rifting key at all, the file keeps rifting off.
        saved.pop('enhanced_rifting')
        self.assertEqual(self.rerun(saved), original)

    def test_profile_flag_on_a_saved_file_changes_only_the_profile(self):
        original = native_engine.validate_config(dict(SMALL))
        saved = json.loads(json.dumps(original))
        saved.pop('physics_profile')
        upgraded = self.rerun(saved, physics_profile='reviewed_v1')
        self.assertEqual(upgraded, native_engine.validate_config(dict(original, physics_profile='reviewed_v1')))
        self.assertEqual((upgraded['retained_phases'], upgraded['enhanced_rifting']['enabled']),
                         ('disabled', False))


class FreshReviewedWorldSmokeTest(unittest.TestCase):
    def test_fresh_reviewed_world_records_lite_laws_and_steps(self):
        # Resolve the distinct ocean-ridge and coastal contacts; level 2 was
        # sufficient only for the previous single-ocean default.
        config = fresh(dict(SMALL, mesh_level=4, physics_profile='reviewed_v1'))
        s = native_engine.Simulation(config)
        self.assertEqual(s.primordial_ocean_version, 1)
        self.assertEqual(s.normal_spreading_version, 1)
        self.assertEqual(s.primordial_ocean_diagnostics['water_region_count'], 4)
        self.assertGreater(s.primordial_ocean_diagnostics['ridge_length_km'], 0.)
        self.assertEqual(s.subduction_response_version, 0)
        self.assertEqual(s.backarc_driver_version, 2)
        self.assertEqual(getattr(s, 'retained_dense_crust_version', 0), 0)
        self.assertEqual(s.effective_subduction_version, 1)
        self.assertEqual(s.slab_memory_version, 0)
        self.assertEqual(s.enhanced_rifting_version, 1)
        # Fresh reviewed worlds gate breakup on kilometres of realized extension.
        self.assertEqual(s.rupture_criterion_version, progressive_rifting.RUPTURE_CRITERION_VERSION)
        calibration = progressive_rifting._rupture_calibration(s)
        self.assertEqual(calibration['damage_strain_measure'],
                         'realized native material-link logarithmic extension')
        self.assertEqual(calibration['breakup_extension_km'], progressive_rifting.BREAKUP_EXTENSION_KM)
        self.assertNotIn('rupture_strain_threshold', calibration)
        # This is a physical-law smoke test, not a wall-time benchmark. The
        # refined control mesh cannot reliably finish within the production
        # budget's nine seconds for a .005 Myr interval on CI hosts.
        with budget.step_budget(.005, seconds_per_myr=120000):
            s.step(.005)
        self.assertEqual(s.t, .005)


if __name__ == '__main__':
    unittest.main()
