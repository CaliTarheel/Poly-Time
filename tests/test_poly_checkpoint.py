"""Restart preserves pending material and the driver's substep/rifting phase."""
import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import numpy as np
import spherely

from poly import dynamics, sphere
from poly.build import cap_world
from poly.checkpoint import CheckpointError, load_checkpoint, save_checkpoint, world_to_data
from poly.mesh import ContinentMesh
from poly.regions import from_network
from poly.run import advance, diagnostics
from poly.world import Crust


def fixture():
    network, rotations = cap_world(spacing_km=500., cap_omega=(0., 0., .003))
    world = from_network(network)
    shape = sphere.polygon(sphere.disc(sphere.from_lonlat(0., 0.), 400., count=20))
    world.crust[0].shape = sphere.difference(world.crust[0].shape, shape)
    mesh = ContinentMesh.build(shape, 45.)
    mesh.lost_nodes = mesh.nodes[:2].copy()
    c = Crust(1, "continent", shape, thickness_km=45., mesh=mesh)
    c.escape_km2 = .125
    world.crust.append(c)
    world.sutures = [{"plate": 1, "points": mesh.nodes[:3].copy(), "time_myr": 0.}]
    world.parents = {8: 7, 7: 1}
    for b in world.boundaries:
        b.arc_km3 = np.ones((len(b.kinds), 2)) * [10., .25]
    world.simplified_myr = 0.
    world.simplified_count = 1000
    return world, rotations


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "checkpoint.json"

    def test_full_state_and_large_polygon_round_trip(self):
        world, _ = fixture()
        world.ledger = [{"time_myr": 0., "plates_km2": {1: 42.}, "sample": (np.array([1., 2.]), 4)}]
        save_checkpoint(world, self.path, {"progress": {"remaining": .5}, "rotations": {1: np.zeros(3)}})
        restored, meta = load_checkpoint(self.path)
        self.assertEqual(restored.parents, world.parents)
        self.assertEqual(restored.simplified_count, 1000)
        self.assertEqual(restored.simplified_myr, 0.)
        self.assertEqual(restored.next_ident, world.next_ident)
        self.assertEqual(restored.crust[-1].escape_km2, .125)
        np.testing.assert_array_equal(restored.crust[-1].mesh.lost_nodes, world.crust[-1].mesh.lost_nodes)
        np.testing.assert_array_equal(restored.crust[-1].mesh.triangles, world.crust[-1].mesh.triangles)
        np.testing.assert_array_equal(restored.crust[-1].mesh.thickness, world.crust[-1].mesh.thickness)
        np.testing.assert_array_equal(restored.sutures[0]["points"], world.sutures[0]["points"])
        self.assertIsInstance(restored.ledger[0]["sample"], tuple)
        self.assertEqual(restored.ledger[0]["plates_km2"], {1: 42.})
        np.testing.assert_array_equal(meta["rotations"][1], np.zeros(3))
        self.assertGreater(sphere.area_km2(restored.regions[2]), sphere.SPHERE_KM2 / 2.)
        for plate in world.regions:
            self.assertEqual(world.regions[plate].__getstate__(), restored.regions[plate].__getstate__())
            self.assertLess(sphere.area_km2(spherely.symmetric_difference(world.regions[plate], restored.regions[plate])), 1e-5)
        for before, after in zip(world.boundaries, restored.boundaries):
            np.testing.assert_array_equal(before.idents, after.idents)
            np.testing.assert_array_equal(before.arc_km3, after.arc_km3)

    def test_unknown_state_rejected_without_replacing_last_good_file(self):
        world, _ = fixture()
        save_checkpoint(world, self.path)
        before = self.path.read_bytes()
        world.unhandled_budget = 100.
        with self.assertRaisesRegex(CheckpointError, "unsupported state"):
            save_checkpoint(world, self.path)
        self.assertEqual(self.path.read_bytes(), before)

    def test_polygon_collection_with_zero_area_debris_is_preserved_exactly(self):
        world, _ = fixture()
        world.crust[0].shape = spherely.union(world.crust[0].shape, spherely.from_wkt("LINESTRING (120 0, 121 1)"))
        self.assertEqual(world.crust[0].shape.__getstate__()[0], 6)
        save_checkpoint(world, self.path)
        restored, _ = load_checkpoint(self.path)
        self.assertEqual(world.crust[0].shape.__getstate__(), restored.crust[0].shape.__getstate__())

    def test_disconnected_multipolygon_is_preserved_exactly(self):
        world, _ = fixture()
        island = sphere.polygon(sphere.disc(sphere.from_lonlat(120., 0.), 50.))
        world.crust[-1].shape = spherely.union(world.crust[-1].shape, island)
        self.assertEqual(world.crust[-1].shape.__getstate__()[0], 5)
        save_checkpoint(world, self.path)
        restored, _ = load_checkpoint(self.path)
        self.assertEqual(world.crust[-1].shape.__getstate__(), restored.crust[-1].shape.__getstate__())

    def test_bad_schema_values_and_version_rejected(self):
        world, _ = fixture()
        save_checkpoint(world, self.path)
        good = json.loads(self.path.read_text())
        mutations = [lambda v: v.update(version=42),
                     lambda v: v["geometry_codec"].update(spherely_version="unsupported"),
                     lambda v: v["world"]["boundaries"][0].pop("arc_km3"),
                     lambda v: v["world"]["frames"][0][1][0].__setitem__(0, 2.),
                     lambda v: v["world"]["crust"][-1]["mesh"]["triangles"][0].__setitem__(0, 10**8),
                     lambda v: v["world"]["crust"][-1].update(escape_km2=-1.),
                     lambda v: v["world"]["regions"][0][1].update(area_km2=1.),
                     lambda v: v["world"]["regions"][0][1].update(sha256="0" * 64)]
        for mutate in mutations:
            bad = copy.deepcopy(good)
            mutate(bad)
            self.path.write_text(json.dumps(bad))
            with self.assertRaises(CheckpointError):
                load_checkpoint(self.path)
        self.path.write_text('{"format":"a","format":"b"}')
        with self.assertRaisesRegex(CheckpointError, "duplicate"):
            load_checkpoint(self.path)

    def test_resume_matches_continuation_with_history_and_pending_budgets(self):
        world, rotations = fixture()
        world.step(rotations, .05)
        save_checkpoint(world, self.path)
        resumed, _ = load_checkpoint(self.path)
        self.assertEqual(world_to_data(world), world_to_data(resumed))
        world.step(rotations, .05)
        resumed.step(rotations, .05)
        self.assertEqual(world_to_data(world), world_to_data(resumed))
        self.assertAlmostEqual(world.time_myr, resumed.time_myr)
        self.assertEqual(len(world.ledger), len(resumed.ledger))
        self.assertEqual(world.plates(), resumed.plates())
        self.assertEqual(world.parents, resumed.parents)
        self.assertEqual(world.next_ident, resumed.next_ident)
        self.assertEqual(world.simplified_myr, resumed.simplified_myr)
        self.assertEqual(world.simplified_count, resumed.simplified_count)
        left, right = diagnostics(world), diagnostics(resumed)
        for key in ("continent_volume_km3", "pending_escape_km2", "pending_arc_km3"):
            np.testing.assert_allclose(left[key], right[key], rtol=1e-8, atol=1e-5)
        for e1, e2 in zip(world.ledger, resumed.ledger):
            self.assertEqual(e1["events"], e2["events"])
            for side in ("gained", "lost"):
                for key in e1[side]:
                    self.assertAlmostEqual(e1[side][key], e2[side][key], delta=.05)
        for plate in world.regions:
            self.assertLess(sphere.area_km2(spherely.symmetric_difference(world.regions[plate], resumed.regions[plate])), .1)

    def test_resume_inside_output_interval_keeps_phase_and_committed_state(self):
        world, rotations = fixture()
        config = {"dt_myr": .2, "motion": "prescribed", "rotations_per_myr": rotations,
                  "parameters": asdict(dynamics.Parameters(step_limit_km=1.)), "rifting": None}
        phase = {"completed_intervals": 0, "remaining_myr": 0.}
        continuous = copy.deepcopy(world)
        advance(continuous, config, dict(phase), 2, lambda w, p: None)
        calls = []

        def interrupt(w, p):
            save_checkpoint(w, self.path, {"phase": dict(p)})
            calls.append(w.time_myr)
            raise RuntimeError("simulated process interruption after committed substep")

        with self.assertRaisesRegex(RuntimeError, "simulated process"):
            advance(world, config, phase, 2, interrupt)
        restored, metadata = load_checkpoint(self.path)
        self.assertGreater(metadata["phase"]["remaining_myr"], 0.)
        advance(restored, config, metadata["phase"], 2, lambda w, p: None)
        self.assertEqual(metadata["phase"], {"completed_intervals": 2, "remaining_myr": 0.})
        self.assertAlmostEqual(restored.time_myr, continuous.time_myr)
        self.assertEqual(len(restored.ledger), len(continuous.ledger))
        np.testing.assert_allclose(diagnostics(restored)["pending_arc_km3"], diagnostics(continuous)["pending_arc_km3"], rtol=1e-8)

    def test_force_resume_does_not_repeat_rift_decision_inside_interval(self):
        world, rotations = fixture()
        config = {"dt_myr": .2, "motion": "forces", "rotations_per_myr": rotations,
                  "parameters": asdict(dynamics.Parameters(step_limit_km=1., sample_points=1000)),
                  "rifting": asdict(dynamics.Rifting())}
        phase = {"completed_intervals": 0, "remaining_myr": 0.}
        continuous = copy.deepcopy(world)
        with mock.patch("poly.dynamics.cuts", return_value=[]) as cuts:
            advance(continuous, config, dict(phase), 2, lambda w, p: None)
            expected_decisions = cuts.call_count

        def interrupt(w, p):
            save_checkpoint(w, self.path, {"phase": dict(p)})
            raise RuntimeError("interrupted")

        with mock.patch("poly.dynamics.cuts", return_value=[]) as cuts:
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                advance(world, config, phase, 2, interrupt)
            restored, metadata = load_checkpoint(self.path)
            self.assertGreater(metadata["phase"]["remaining_myr"], 0.)
            advance(restored, config, metadata["phase"], 2, lambda w, p: None)
            self.assertEqual(cuts.call_count, expected_decisions)
        self.assertEqual(world_to_data(restored), world_to_data(continuous))

    def test_failed_step_never_replaces_last_accepted_checkpoint(self):
        world, rotations = fixture()
        config = {"dt_myr": .1, "motion": "prescribed", "rotations_per_myr": rotations,
                  "parameters": asdict(dynamics.Parameters()), "rifting": None}
        phase = {"completed_intervals": 0, "remaining_myr": 0.}
        save_checkpoint(world, self.path, {"phase": phase})
        before = self.path.read_bytes()
        with mock.patch.object(world, "step", side_effect=ValueError("rejected step")):
            with self.assertRaisesRegex(ValueError, "rejected step"):
                advance(world, config, phase, 1, lambda w, p: save_checkpoint(w, self.path))
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
