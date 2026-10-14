"""Progressive rift records survive persistence and globe reorientation."""
from copy import deepcopy
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

from history import history_schema
from orientation import orient_frame, rotation_matrix
from checkpoint import read_checkpoint, write_checkpoint
from server import Handler, SimulationManager, capture_auxiliary_sources, json_bytes
from tectonics import Simulation


class RiftReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]/'tmp')
        self.addCleanup(self.temp.cleanup)
        self.manager = SimulationManager(self.temp.name)
        self.manager.current = dict(run_id='rift-review', frame_count=1)
        self.manager.path().mkdir()
        self.frame = dict(width=8, height=4, time_myr=40.,
            elevation=np.zeros(32), plate=np.zeros(32, np.int32),
            crust=np.ones(32, np.uint8), age=np.zeros(32), boundary=np.zeros(32, np.uint8),
            rift_damage=np.linspace(0, 1, 32), rift_strength_relative=np.linspace(.2, 4, 32),
            rift_systems=[dict(id=1, plate_uid=7, phase='active', started_myr=20.,
                last_active_myr=40., extension_km=38.2, peak_damage=.8, mean_strength=.7,
                geometry_xyz=[[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]],
                history=[dict(time_myr=20., phase='incipient', reason='local_extension'),
                         dict(time_myr=34., phase='active', reason='sustained_stretching')])],
            rift_mechanics=dict(mesh_nodes=128, mesh_edges=480, target_nodes=128,
                solver_iterations=10, residual=1e-7, converged=True, model='fixture'))

    def test_optional_grids_and_system_records_round_trip_separately(self):
        original = deepcopy(self.frame)
        self.manager.save_frame(0, self.frame)
        result = self.manager.frame(0)
        for key in ('rift_damage', 'rift_strength_relative'):
            self.assertEqual(result[key].dtype, np.float32)
            np.testing.assert_allclose(result[key], original[key], rtol=1e-6)
        self.assertEqual(result['rift_systems'], original['rift_systems'])
        self.assertEqual(result['rift_mechanics'], original['rift_mechanics'])
        metadata = json.loads((self.manager.path()/'frame_0000.json').read_text())
        self.assertNotIn('rift_damage', metadata)
        with np.load(self.manager.path()/'frame_0000.npz') as archive:
            self.assertNotIn('rift_systems', archive.files)
        self.assertEqual(self.frame['rift_systems'], original['rift_systems'])

    def test_bad_grid_sizes_nonfinite_damage_and_nonpositive_strength_rejected(self):
        for field, values in [('rift_damage', [0.]*31), ('rift_damage', [1.01]*32),
                ('rift_damage', [-.01]*32), ('rift_damage', [np.nan]*32),
                ('rift_strength_relative', [0.]*32), ('rift_strength_relative', [-1.]*32),
                ('rift_strength_relative', [np.inf]*32)]:
            with self.subTest(field=field, value=values[0]):
                frame = {**self.frame, field: values}
                with self.assertRaisesRegex(ValueError, field):
                    self.manager.save_frame(0, frame)

    def test_legacy_frame_does_not_invent_mechanical_records(self):
        frame = {key: value for key, value in self.frame.items()
                 if key not in ('rift_damage', 'rift_strength_relative', 'rift_systems', 'rift_mechanics')}
        self.manager.save_frame(0, frame)
        result = self.manager.frame(0)
        for field in ('rift_damage', 'rift_strength_relative', 'rift_systems', 'rift_mechanics'):
            self.assertNotIn(field, result)

    def test_globe_rotation_moves_unordered_anchors_and_preserves_scalar_history(self):
        pose = dict(yaw=33., pitch=71., roll=-12.)
        original = deepcopy(self.frame)
        result = orient_frame(self.frame, pose)
        np.testing.assert_allclose(result['rift_systems'][0]['geometry_xyz'],
            np.asarray(original['rift_systems'][0]['geometry_xyz']) @ rotation_matrix(pose))
        for key, value in original['rift_systems'][0].items():
            if key != 'geometry_xyz':
                self.assertEqual(result['rift_systems'][0][key], value)
        self.assertEqual(result['rift_mechanics'], original['rift_mechanics'])
        self.assertTrue(np.all((result['rift_damage'] >= 0) & (result['rift_damage'] <= 1)))
        self.assertTrue(np.all(result['rift_strength_relative'] > 0))
        np.testing.assert_array_equal(self.frame['rift_damage'], original['rift_damage'])
        self.assertEqual(self.frame['rift_systems'], original['rift_systems'])

    def test_rotated_persistence_and_schema_keep_mechanics_distinct_from_map_resolution(self):
        self.manager.save_frame(0, self.frame)
        result = self.manager.frame(0, dict(yaw=90.))
        self.assertEqual(result['rift_mechanics']['mesh_nodes'], 128)
        self.assertEqual(result['width']*result['height'], 32)
        schema = history_schema()
        self.assertIn('rift_damage', schema['optional_grid_fields'])
        self.assertIn('rift_strength_relative', schema['optional_grid_fields'])
        self.assertIn('unordered', schema['rift_systems'])
        self.assertIn('rift_system_id', schema['rift_system_events'])

    def test_engine_snapshot_http_contract_excludes_internal_bonds_and_preserves_orientation(self):
        simulation = Simulation(dict(width=48, height=24, mesh_level=2, mechanics_nodes=128, duration_myr=6))
        simulation.step(2)
        # Explicit lifecycle fixture exercises serialization even when a short
        # smoke integration has not developed enough damage to create a rift.
        row = deepcopy(self.frame['rift_systems'][0])
        row['_bonds'] = {(4, 5), (5, 9)}
        simulation.rift_systems.append(row)
        frame = simulation.snapshot()
        self.manager.save_frame(0, frame)
        manager = self.manager

        class LocalHandler(Handler):
            def log_message(self, *args):
                pass
        LocalHandler.manager = manager
        http = ThreadingHTTPServer(('127.0.0.1', 0), LocalHandler)
        worker = threading.Thread(target=lambda: http.serve_forever(poll_interval=.01), daemon=True)
        worker.start()
        try:
            query = urlencode(dict(frame=0, orientation=json.dumps(dict(yaw=90.))))
            with urlopen(f'http://127.0.0.1:{http.server_port}/api/frame?{query}', timeout=10) as response:
                result = json.load(response)
            self.assertEqual(result['rift_mechanics']['target_nodes'], 128)
            self.assertGreater(result['rift_mechanics']['mesh_nodes'], 0)
            self.assertNotIn('_bonds', result['rift_systems'][-1])
            self.assertEqual(result['rift_systems'][-1]['history'], row['history'])
            np.testing.assert_allclose(result['rift_systems'][-1]['geometry_xyz'],
                np.asarray(row['geometry_xyz']) @ rotation_matrix(dict(yaw=90.)), atol=1e-14)
            self.assertEqual(len(result['rift_damage']), 48*24)
            self.assertTrue(all(0 <= value <= 1 for value in result['rift_damage']))
            self.assertTrue(all(value > 0 for value in result['rift_strength_relative']))
        finally:
            http.shutdown();http.server_close();worker.join(5)
        self.assertEqual(simulation.rift_systems[-1]['_bonds'], {(4, 5), (5, 9)})
        # Public snapshots can be JSON-encoded without custom set/object types.
        json_bytes(frame)

    def test_new_mechanical_state_checkpoint_resumes_exactly_and_sources_are_captured(self):
        root = Path(__file__).parents[1]
        sources = capture_auxiliary_sources((root/'tectonics.py').read_bytes())
        for name in ('progressive_rifting.py', 'rift_material.py', 'rift_mesh.py', 'rift_mechanics.py'):
            self.assertIn(name, sources)
            self.assertEqual(sources[name], (root/name).read_bytes())
        simulation = Simulation(dict(width=48, height=24, mesh_level=2, mechanics_nodes=128, duration_myr=6))
        simulation.step(2)
        row = deepcopy(self.frame['rift_systems'][0]);row['_bonds'] = {(4, 5), (5, 9)}
        simulation.rift_systems.append(row)
        path = Path(self.temp.name)/'rift-checkpoint.npz'
        manifest = dict(time_myr=simulation.t, next_output_myr=4., config=deepcopy(simulation.config))
        write_checkpoint(path, simulation, manifest, self.manager.compatibility())
        restored, returned = read_checkpoint(path, self.manager.compatibility(), Simulation)
        self.assertEqual(returned, manifest)
        self.assertEqual(restored.rift_bonds, simulation.rift_bonds)
        self.assertEqual(restored.rift_systems, simulation.rift_systems)
        self.assertEqual(restored.rift_mechanics, simulation.rift_mechanics)
        # No rift is pending in this world: the engine holds None, not an empty mapping.
        self.assertEqual(restored.rift_pending is None, simulation.rift_pending is None)
        self.assertEqual(set(restored.rift_pending or ()), set(simulation.rift_pending or ()))
        simulation.step(2);restored.step(2)
        actual, expected = restored.snapshot(), simulation.snapshot()
        self.assertEqual(set(actual), set(expected))
        for key, value in expected.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(actual[key], value, err_msg=key)
            else:
                self.assertEqual(actual[key], value, key)


if __name__ == '__main__':
    unittest.main()
