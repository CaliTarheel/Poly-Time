"""New command-line worlds use Lite while saved worlds retain their laws."""
import json
from pathlib import Path
import tempfile
import unittest

import run_simulation
import native_engine


class EffectiveSubductionCommandLineTests(unittest.TestCase):
    def parse(self, *arguments):
        return run_simulation.build_config(run_simulation.parser().parse_args(list(arguments)))

    def test_explicit_opt_in_selects_compatible_laws(self):
        config = self.parse('--effective-subduction')
        self.assertTrue(config['effective_subduction']['enabled'])
        self.assertEqual(config['effective_subduction']['force_n_per_m'], 3e13)
        self.assertTrue(config['primordial_subduction']['enabled'])
        self.assertEqual(config['physics_profile'], 'reviewed_v1')
        self.assertEqual(config['subduction_response'], 'fixed_trench')
        self.assertEqual(config['retained_phases'], 'disabled')
        self.assertEqual(config['trench_persistence'], 'kinematic')
        self.assertEqual(config['slab_allocation'], 'uniform')
        self.assertEqual(config['continental_lifecycle'], {'enabled': False})
        self.assertEqual(config['rift_traction'], {'enabled': False})

    def test_force_flag_uses_si_units_and_enables_the_law(self):
        self.assertEqual(self.parse('--subduction-force-n-per-m', '7.25e12')['effective_subduction'],
                         {'enabled': True, 'force_n_per_m': 7.25e12})

    def test_normal_new_world_default_uses_lite_and_detailed_can_be_selected(self):
        default = self.parse()
        self.assertTrue(default['effective_subduction']['enabled'])
        self.assertEqual(default['effective_subduction']['force_n_per_m'], 3e13)
        self.assertTrue(default['primordial_subduction']['enabled'])
        self.assertEqual(default['subduction_response'],'fixed_trench')
        self.assertEqual(default['retained_phases'],'disabled')
        detailed = self.parse('--no-effective-subduction')
        self.assertFalse(detailed['effective_subduction']['enabled'])
        self.assertEqual(detailed['subduction_response'],'moving_hinge_v1')
        self.assertEqual(detailed['retained_phases'],'thermal_v1')
        self.assertNotIn('effective_subduction', self.parse('--physics-profile','legacy'))

    def test_conflicting_flags_reject(self):
        with self.assertRaisesRegex(ValueError, 'requires reviewed_v1'):
            self.parse('--effective-subduction', '--physics-profile', 'legacy')
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            self.parse('--no-effective-subduction', '--subduction-force-n-per-m', '5e12')

    def test_saved_experiment_is_preserved_and_can_be_explicitly_disabled(self):
        saved = dict(seed=3, physics_profile='reviewed_v1',
                     effective_subduction=dict(enabled=True, force_n_per_m=9e12),
                     continental_lifecycle=dict(enabled=True, neck_thickness_km=85.),
                     rift_traction=dict(enabled=True, reach_km=1000.),
                     trench_persistence='attached_slab_v1', slab_allocation='fed_v1',
                     primordial_subduction=dict(enabled=True, target_margin_fraction=.4, selection_seed=7))
        scratch = Path(__file__).resolve().parents[1] / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(saved), encoding='utf-8')
            config = self.parse('--config', str(path))
            self.assertEqual(config['effective_subduction'], saved['effective_subduction'])
            config = self.parse('--config', str(path), '--effective-subduction')
            self.assertEqual(config['primordial_subduction'], saved['primordial_subduction'])
            self.assertEqual(config['effective_subduction']['force_n_per_m'], 9e12)
            self.assertFalse(config['continental_lifecycle']['enabled'])
            self.assertEqual(config['continental_lifecycle']['neck_thickness_km'],85.)
            self.assertFalse(config['rift_traction']['enabled'])
            self.assertEqual(config['rift_traction']['reach_km'],1000.)
            self.assertEqual(config['trench_persistence'],'kinematic')
            self.assertEqual(config['slab_allocation'],'uniform')
            self.assertTrue(native_engine.validate_config(config)['effective_subduction']['enabled'])
            disabled = self.parse('--config', str(path), '--no-effective-subduction')
            self.assertFalse(disabled['effective_subduction']['enabled'])

    def test_highland_initial_preset_uses_compatible_law_and_retains_its_declaration(self):
        base = dict(seed=41,width=96,height=48,physics_profile='reviewed_v1',
                    mesh_level=2,coast_geometry_level=3,plate_count=8,mechanics_nodes=128)
        initial = native_engine.make_initial(base,preset='highland65')
        self.assertTrue(initial['continental_lifecycle']['enabled'])
        self.assertTrue(initial['rift_traction']['enabled'])
        scratch = Path(__file__).resolve().parents[1] / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            initial_path = Path(folder) / 'initial.json'
            initial_path.write_text(json.dumps(initial,default=lambda value:value.tolist()),encoding='utf-8')
            config_path = Path(folder) / 'config.json'
            config_path.write_text(json.dumps(base),encoding='utf-8')
            config = self.parse('--config',str(config_path),'--initial',str(initial_path),'--effective-subduction')
            new_default = self.parse('--initial',str(initial_path),'--width','96','--seed','41')
            self.assertTrue(new_default['effective_subduction']['enabled'])
            self.assertEqual(new_default['primordial_subduction'],initial['initial_subduction'])
        self.assertEqual(config['primordial_subduction'],initial['initial_subduction'])
        simulation = native_engine.Simulation(config,initial)
        self.assertEqual(simulation.effective_subduction_version,1)
        self.assertEqual(simulation.slab_memory_version,0)
        self.assertFalse(simulation.config['continental_lifecycle']['enabled'])
        self.assertFalse(simulation.config['rift_traction']['enabled'])
        self.assertEqual(simulation.initial_plate_topology_diagnostics['mode'],'passive_margin_aprons')
        self.assertEqual(simulation.config['primordial_subduction']['target_margin_fraction'],.45)
        self.assertEqual(simulation.config['primordial_subduction']['selection_seed'],41)
        self.assertGreater(len(simulation.trench_systems),0)
        self.assertTrue(initial['continental_lifecycle']['enabled'])
        self.assertTrue(initial['rift_traction']['enabled'])


if __name__ == '__main__':
    unittest.main()
