"""Seeded pending points and funded compact promotion, without a history run."""
from copy import deepcopy
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT)]

import arc_birth_footprint as footprint
import arc_birth_profile as profile
import arc_emplacement_geometry as geometry
import arc_point_nucleation as points
import arc_source_cohorts as cohorts
import checkpoint
import crustal_structure as columns
import native_arc_material as arcs
from ridge_geometry import rotate
from test_arc_birth_footprint import world
from test_arc_source_cohorts import fixture, observe, unit
from test_arc_source_geometry import positions_in_cell


def pending_fixture():
    state, xyz = fixture()
    state.config = {"seed": 17}
    state.rng = np.random.default_rng(17)
    state.native_arc_point_version = 1
    state.native_arc_birth_profile_version = 1
    state.native_arc_footprint_version = 1
    state.native_arc_emplacement_version = 1
    rows = observe(state, xyz, [50.])
    state.native_arc_pending = dict(xyz=xyz.copy(), owner=np.array([0]),
        area=np.array([50.]), source_provenance=rows,
        emplacement_version=1, source_column_km=25.)
    return state


def point_frame(state):
    return dict(points.snapshot_fields(state), time_myr=float(state.t),
        arc_birth_profile_version=1, arc_footprint_version=1,
        arc_emplacement_version=1,
        arc_pending_source=geometry.pending_snapshot(state.native_arc_pending))


def arrays(state):
    result = {}
    for name in ("mass", "kind", "parcel_plate", "parcel_patch", "parcel_arc_id",
                 "relief", "trace_patch", "trace_xyz", "trace_relief_m"):
        result[name] = np.asarray(getattr(state, name)).copy()
    for name in ("material_surface", "structure", "trace_structure"):
        for key, value in getattr(state, name).items():
            if isinstance(value, np.ndarray):
                result[name + "." + str(key)] = value.copy()
    return result


def qualified_source(state, xyz, amount, *, links=(), component=1, episode=1):
    owner = int(state.plate[state._indices(xyz[None])[0]])
    row = cohorts.normalize(state, None, 1, owners=[owner], areas=[amount])[0]
    host = int(state.plate_uid[owner])
    tangent = np.cross(xyz, [0., 0., 1.])
    if np.linalg.norm(tangent) < .1:
        tangent = np.cross(xyz, [0., 1., 0.])
    tangent /= np.linalg.norm(tangent)
    row.update(trench_id=1, episode=episode, downgoing_plate_uid=host + 10000,
        overriding_plate_uid=host, component_id=component,
        advection_host_plate_uid=host, valid=True, reason="qualified_test_source",
        trace_xyz=[[unit(xyz - .001 * tangent).tolist(),
                    unit(xyz + .001 * tangent).tolist()]], links=list(links))
    cohorts.validate([row], 1)
    return row, owner


def supply(state, xyz, amounts, provenance):
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    cells = state._indices(xyz)
    return arcs.add_arc_crust(state, cells, np.asarray(amounts, float), positions=xyz,
        owners=np.asarray(state.plate[cells], np.int64), source_provenance=provenance)


def retry(state):
    return arcs.add_arc_crust(state, np.empty(0, np.int64), np.empty(0),
        positions=np.empty((0, 3)), owners=np.empty(0, np.int64), source_provenance=[])


class PointGeometryTests(unittest.TestCase):
    def test_compact_profile_succeeds_where_same_funded_oval_fails(self):
        centre = np.array([1., 0., 0.])
        direction = np.array([0., 1., 0.])
        old = footprint.propose(lambda area: arcs._patch(centre, direction, area),
                                100., -6000.)
        new = points.propose(centre, direction, 100., -6000., points.rank(17, [1]))
        self.assertIsNone(old["plan"])
        self.assertGreater(old["profile"]["diagnostics"]["maximum_slope_deg"], 26.)
        self.assertIsNotNone(new["plan"])
        patch, fields = new["plan"], new["profile"]
        self.assertEqual(patch["faces"].shape, (24, 3))
        np.testing.assert_array_equal(patch["vertices"][0], centre)
        self.assertAlmostEqual(float(fields["area_km2"] @ fields["thickness_km"]), 2500., places=7)
        self.assertTrue(np.all((fields["thickness_km"] >= 8.) & (fields["thickness_km"] <= 75.)))
        self.assertLessEqual(fields["diagnostics"]["maximum_slope_deg"], 20.)
        self.assertLess(fields["height_m"].max(), 0.)
        self.assertEqual(new["point_nucleation"]["relocation_km"], 0.)
        self.assertAlmostEqual(new["point_nucleation"]["mean_funded_column_km"],
                               2500. / float(patch["area_km2"].sum()), places=11)

    def test_insufficient_magma_cannot_be_promoted_by_a_seed(self):
        for seed in (1, 17, 12345):
            with self.subTest(seed=seed):
                result = points.propose([1., 0., 0.], [0., 1., 0.], 50., -6000.,
                                        points.rank(seed, [1, 2, 3]))
                self.assertIsNone(result["plan"])
                self.assertFalse(result["point_nucleation"]["admissible"])
                self.assertEqual(result["point_nucleation"]["source_volume_km3"], 1250.)
                self.assertEqual(result["point_nucleation"]["physical_footprint_area_km2"], 0.)

    def test_seed_is_order_independent_stateless_and_never_jitters_the_source(self):
        state = pending_fixture()
        before = pickle.dumps(state, protocol=5)
        key = points.seed_key(state, [{"origin_id": 7}, {"origin_id": 3}])
        self.assertEqual(key, points.seed_key(state, [{"origin_id": 3}, {"origin_id": 7}]))
        first = points.propose([1., 0., 0.], [0., 1., 0.], 150., -6000., key)
        again = points.propose([1., 0., 0.], [0., 1., 0.], 150., -6000., key)
        other = points.propose([1., 0., 0.], [0., 1., 0.], 150., -6000.,
                               points.rank(18, [3, 7]))
        self.assertEqual(pickle.dumps(state, protocol=5), before)
        np.testing.assert_array_equal(first["plan"]["vertices"], again["plan"]["vertices"])
        np.testing.assert_array_equal(first["profile"]["thickness_km"], again["profile"]["thickness_km"])
        self.assertGreater(np.max(np.abs(first["plan"]["vertices"] - other["plan"]["vertices"])), 1e-6)
        for result in (first, again, other):
            np.testing.assert_array_equal(result["plan"]["vertices"][0], [1., 0., 0.])
            self.assertAlmostEqual(result["profile"]["diagnostics"]["actual_volume_km3"], 3750., places=7)

    def test_compact_proposal_is_equivariant_through_poles_and_longitude_seam(self):
        centre, direction = np.array([1., 0., 0.]), np.array([0., 1., 0.])
        key = points.rank(17, [1])
        baseline = points.propose(centre, direction, 150., -6000., key)
        for rotation in ([0., np.pi / 2., 0.], [0., 0., np.pi], [.2, -.7, .4]):
            with self.subTest(rotation=rotation):
                result = points.propose(rotate(centre, rotation), rotate(direction, rotation),
                                        150., -6000., key)
                np.testing.assert_allclose(result["plan"]["vertices"],
                    rotate(baseline["plan"]["vertices"], rotation), rtol=0., atol=2e-10)
                np.testing.assert_allclose(result["profile"]["thickness_km"],
                    baseline["profile"]["thickness_km"], rtol=0., atol=1e-7)
                self.assertAlmostEqual(result["profile"]["diagnostics"]["actual_volume_km3"], 3750., places=6)


class PendingPointTests(unittest.TestCase):
    def test_points_are_exact_readonly_views_of_pending_inventory(self):
        state = pending_fixture()
        before = pickle.dumps(state, protocol=5)
        frame = point_frame(state)
        self.assertEqual(points.validate_frame(frame), 1)
        self.assertEqual(pickle.dumps(state, protocol=5), before)
        feature = frame["volcanic_point_features"][0]
        self.assertEqual(feature["id"], state.native_arc_pending["source_provenance"][0]["origin_id"])
        self.assertEqual(feature["owner_uid"], 11)
        self.assertEqual(feature["volume_km3"], 1250.)
        self.assertEqual(feature["status"], "pending")
        np.testing.assert_array_equal(feature["geometry_xyz"], state.native_arc_pending["xyz"])
        for _ in range(3):
            self.assertEqual(points.snapshot_fields(state), points.snapshot_fields(state))
        self.assertEqual(pickle.dumps(state, protocol=5), before)

    def test_pending_feature_advects_once_and_never_rebinds_a_reused_owner_slot(self):
        state = pending_fixture()
        original = state.native_arc_pending["xyz"].copy()
        cohorts.advect(state, 2.)
        state.t = 2.
        feature = point_frame(state)["volcanic_point_features"][0]
        np.testing.assert_allclose(feature["geometry_xyz"],
            unit(rotate(original, state.omega[0] * 2.)), rtol=0., atol=2e-15)
        state.plate_uid[0] = 999
        before = state.native_arc_pending["xyz"].copy()
        cohorts.advect(state, 2.)
        state.t = 4.
        feature = point_frame(state)["volcanic_point_features"][0]
        np.testing.assert_array_equal(feature["geometry_xyz"], before)
        self.assertEqual(feature["owner_uid"], 11)
        self.assertEqual(feature["reason"], "unresolved_advection_host")
        self.assertEqual(feature["volume_km3"], 1250.)

    def test_frame_rejects_unfunded_moved_duplicated_or_relabelled_points(self):
        frame = point_frame(pending_fixture())
        variants = [
            ("missing tag", lambda f: f.pop("arc_point_version")),
            ("invented volume", lambda f: f["volcanic_point_features"][0].update(volume_km3=1251.)),
            ("changed source", lambda f: f["volcanic_point_features"][0].update(id=999)),
            ("changed owner", lambda f: f["volcanic_point_features"][0].update(owner=1)),
            ("rebound host", lambda f: f["volcanic_point_features"][0].update(owner_uid=999)),
            ("moved position", lambda f: f["volcanic_point_features"][0].update(geometry_xyz=[[0., 1., 0.]])),
            ("retained status", lambda f: f["volcanic_point_features"][0].update(status="retained")),
            ("duplicate point", lambda f: f["volcanic_point_features"].append(deepcopy(f["volcanic_point_features"][0]))),
        ]
        for name, mutate in variants:
            with self.subTest(name=name):
                altered = deepcopy(frame)
                mutate(altered)
                with self.assertRaises(ValueError):
                    points.validate_frame(altered)

    def test_oriented_snapshot_rotates_pending_points_with_canonical_source_geometry(self):
        from orientation import orient_frame, rotation_matrix

        state = pending_fixture()
        frame = point_frame(state)
        frame.update(width=8, height=4, crust=np.zeros(32, np.uint8))
        frozen = pickle.dumps(frame, protocol=5)
        source_xyz = np.asarray(frame["arc_pending_source"]["geometry_xyz"])
        original = frame["volcanic_point_features"][0]
        for angles in (dict(yaw=180.), dict(pitch=90.),
                       dict(yaw=177., pitch=89., roll=31.)):
            with self.subTest(angles=angles):
                oriented = orient_frame(frame, angles)
                self.assertEqual(points.validate_frame(oriented), 1)
                expected = source_xyz @ rotation_matrix(angles)
                feature = oriented["volcanic_point_features"][0]
                np.testing.assert_allclose(feature["geometry_xyz"], expected,
                                           rtol=0., atol=2e-15)
                np.testing.assert_array_equal(feature["geometry_xyz"],
                    oriented["arc_pending_source"]["geometry_xyz"])
                for field in ("id", "owner", "owner_uid", "volume_km3",
                              "source_time_myr", "status", "reason"):
                    self.assertEqual(feature[field], original[field], msg=field)
                self.assertEqual(pickle.dumps(frame, protocol=5), frozen)

    def test_absent_version_keeps_legacy_state_and_explicit_old_profile_disables_points(self):
        state = pending_fixture()
        del state.native_arc_point_version
        before = pickle.dumps(state, protocol=5)
        self.assertEqual(points.version(state), 0)
        self.assertEqual(points.snapshot_fields(state), {})
        self.assertEqual(points.validate_frame({}), 0)
        self.assertEqual(pickle.dumps(state, protocol=5), before)
        state.native_arc_point_version = 1
        state.native_arc_birth_profile_version = 0
        self.assertEqual(points.version(state), 0)


class PointPromotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.initial = world()
        cls.initial.native_arc_point_version = 1
        cls.initial.config["seed"] = 17

    def setUp(self):
        self.state = deepcopy(self.initial)

    def pending_source(self):
        state = self.state
        xyz = positions_in_cell(state)[0]
        row, _ = qualified_source(state, xyz, 50.)
        report = supply(state, [xyz], [50.], [row])
        self.assertEqual(report["added_volume_km3"], 0.)
        self.assertAlmostEqual(report["pending_magma_volume_km3"], 1250.)
        return xyz, row

    def test_compatible_supply_promotes_only_the_funded_volume_once(self):
        state = self.state
        before = arrays(state)
        xyz, first = self.pending_source()
        for name, value in arrays(state).items():
            np.testing.assert_array_equal(value, before[name], err_msg=name)
        self.assertEqual(points.snapshot_fields(state)["volcanic_point_features"][0]["volume_km3"], 1250.)
        state.t = 1.
        second, _ = qualified_source(state, xyz, 50., links=[first["origin_id"]])
        report = supply(state, [xyz], [50.], [second])
        self.assertEqual(report["new_faces"], 24)
        self.assertAlmostEqual(report["added_volume_km3"], 2500., places=7)
        self.assertAlmostEqual(report["pending_magma_volume_km3"], 0., places=7)
        self.assertEqual(points.snapshot_fields(state)["volcanic_point_features"], [])
        actual = float(state.material_surface["area_km2"] @ state.structure["thickness_km"])
        self.assertAlmostEqual(actual, 2500., places=7)
        self.assertAlmostEqual(float(state.mass.sum()), 100., places=7)
        self.assertAlmostEqual(sum(row["volume_km3"] for row in state.native_arc_source_placements), actual, places=7)
        self.assertEqual({row["source_provenance"]["origin_id"] for row in state.native_arc_source_placements},
                         {first["origin_id"], second["origin_id"]})
        self.assertTrue(np.all((state.structure["thickness_km"] >= 8.) &
                               (state.structure["thickness_km"] <= 75.)))
        frame = dict(arcs.snapshot_fields(state), arc_material_diagnostics=report, time_myr=state.t)
        points.validate_frame(frame)
        profile.validate_frame(frame)
        geometry.validate_frame(frame)
        physical = arrays(state)
        result = retry(state)
        self.assertEqual(result["added_volume_km3"], 0.)
        for name, value in arrays(state).items():
            np.testing.assert_array_equal(value, physical[name], err_msg=name)

    def test_post_admission_empty_containment_rejects_compact_and_legacy_receipts(self):
        from unittest.mock import patch
        import arc_cohort_footprint

        state = self.state
        xyz = positions_in_cell(state)[0]
        source, _ = qualified_source(state, xyz, 1000.)
        before = arrays(state)
        real_admit = geometry.admit
        real_contained = arc_cohort_footprint.contained
        positive_admissions = []

        def admit_then_arm(*args, **kwargs):
            admitted = real_admit(*args, **kwargs)
            if admitted["accepted_area_km2"] > 0.:
                positive_admissions.append(admitted["accepted_area_km2"])
            return admitted

        def lose_containment_after_admission(candidate, queries):
            if positive_admissions:
                return np.zeros(len(queries), bool)
            return real_contained(candidate, queries)

        # Keep initial cohort construction and both geographic admissions real.
        # Only the final contributor-containment check loses its anchor.
        with patch.object(geometry, "admit", side_effect=admit_then_arm), \
             patch.object(arc_cohort_footprint, "contained",
                          side_effect=lose_containment_after_admission):
            report = supply(state, [xyz], [1000.], [source])

        self.assertEqual(len(positive_admissions), 2)
        self.assertEqual(report["new_faces"], 0)
        self.assertEqual(report["added_volume_km3"], 0.)
        self.assertEqual(report["pending_magma_volume_km3"], 25000.)
        np.testing.assert_array_equal(state.native_arc_pending["xyz"], [xyz])
        np.testing.assert_array_equal(state.native_arc_pending["area"], [1000.])
        decisions = report["emplacement_geometry"]["sources"]
        self.assertEqual(len(decisions), 1)
        for row in decisions:
            self.assertEqual(row["accepted_area_km2"], 0.)
            self.assertEqual(row["pending_area_km2"], 1000.)
            self.assertIsNone(row["accepted_footprint"])
            self.assertNotIn("point_promotion", row)
            self.assertEqual([attempt["strategy"] for attempt in row["point_attempts"]],
                             ["point", "legacy"])
            self.assertTrue(all(attempt["accepted_area_km2"] == 0.
                                for attempt in row["point_attempts"]))
        for name, value in arrays(state).items():
            np.testing.assert_array_equal(value, before[name], err_msg=name)

    def test_failed_promotion_retry_retains_exact_sources_and_no_material(self):
        state = self.state
        xyz, _ = self.pending_source()
        before = arrays(state)
        provenance = deepcopy(state.native_arc_pending["source_provenance"])
        for _ in range(2):
            report = retry(state)
            self.assertEqual(report["added_volume_km3"], 0.)
            self.assertEqual(report["pending_magma_volume_km3"], 1250.)
            np.testing.assert_array_equal(state.native_arc_pending["xyz"], [xyz])
            np.testing.assert_array_equal(state.native_arc_pending["area"], [50.])
            self.assertEqual(state.native_arc_pending["source_provenance"], provenance)
        for name, value in arrays(state).items():
            np.testing.assert_array_equal(value, before[name], err_msg=name)

    def test_nearby_points_with_incompatible_lineage_cannot_pool_to_promote(self):
        for distinction in ("component", "episode"):
            with self.subTest(distinction=distinction):
                state = deepcopy(self.initial)
                xyz = positions_in_cell(state)[0]
                first, _ = qualified_source(state, xyz, 50.)
                second, _ = qualified_source(state, xyz, 50., links=[first["origin_id"]],
                    component=2 if distinction == "component" else 1,
                    episode=2 if distinction == "episode" else 1)
                report = supply(state, [xyz, xyz], [50., 50.], [first, second])
                self.assertEqual(report["added_volume_km3"], 0.)
                self.assertEqual(report["pending_magma_volume_km3"], 2500.)
                self.assertEqual(len(points.snapshot_fields(state)["volcanic_point_features"]), 2)

    def test_splitting_same_total_supply_does_not_create_extra_volume(self):
        outcomes = []
        for count in (1, 4):
            state = deepcopy(self.initial)
            xyz = positions_in_cell(state)[0]
            rows = []
            for _ in range(count):
                row, _ = qualified_source(state, xyz, 100. / count,
                    links=[] if not rows else [rows[-1]["origin_id"]])
                rows.append(row)
            report = supply(state, np.tile(xyz, (count, 1)), np.full(count, 100. / count), rows)
            self.assertGreater(report["added_volume_km3"], 0.)
            self.assertAlmostEqual(report["added_volume_km3"] + report["pending_magma_volume_km3"], 2500., places=7)
            outcomes.append(float(state.material_surface["area_km2"] @ state.structure["thickness_km"]))
        np.testing.assert_allclose(outcomes, [2500., 2500.], rtol=0., atol=1e-7)

    def test_reused_owner_slot_cannot_spend_old_pending_magma(self):
        state = self.state
        xyz, first = self.pending_source()
        before = arrays(state)
        owner = int(state.plate[state._indices(xyz[None])[0]])
        old_host = int(state.plate_uid[owner])
        state.plate_uid[owner] = old_host + 500
        state.t = 1.
        second, _ = qualified_source(state, xyz, 50., links=[first["origin_id"]])
        report = supply(state, [xyz], [50.], [second])
        self.assertEqual(report["added_volume_km3"], 0.)
        self.assertEqual(report["pending_magma_volume_km3"], 2500.)
        features = {row["id"]: row for row in points.snapshot_fields(state)["volcanic_point_features"]}
        self.assertEqual(features[first["origin_id"]]["owner_uid"], old_host)
        self.assertEqual(features[first["origin_id"]]["volume_km3"], 1250.)
        for name, value in arrays(state).items():
            np.testing.assert_array_equal(value, before[name], err_msg=name)

    def test_promotion_receipts_reject_unfunded_or_unconnected_claims(self):
        state = self.state
        xyz, first = self.pending_source()
        state.t = 1.
        second, _ = qualified_source(state, xyz, 50., links=[first["origin_id"]])
        report = supply(state, [xyz], [50.], [second])
        frame = dict(arcs.snapshot_fields(state), arc_material_diagnostics=report, time_myr=state.t)
        points.validate_frame(frame)

        def decision(value):
            return next(row for row in value["arc_material_diagnostics"]["emplacement_geometry"]["sources"]
                        if "point_promotion" in row)

        def receipt(value):
            return decision(value)["point_promotion"]

        def disconnect(value):
            for row in value["arc_source_placements"]:
                row["source_provenance"]["links"] = []

        changes = [
            ("unfunded volume", lambda f: receipt(f).update(source_volume_km3=9999.)),
            ("false density", lambda f: receipt(f).update(volume_density_km=99.)),
            ("false footprint", lambda f: receipt(f).update(footprint_area_km2=1.)),
            ("foreign source", lambda f: receipt(f)["source_origin_ids"].__setitem__(0, 99999)),
            ("noncanonical seed", lambda f: receipt(f).update(seed_key="01")),
            ("legacy mode", lambda f: decision(f).update(mode="growth")),
            ("disconnected source graph", disconnect),
            ("unqualified origin", lambda f: f["arc_source_placements"][0]["source_provenance"].update(valid=False)),
        ]
        for name, mutate in changes:
            with self.subTest(name=name):
                altered = deepcopy(frame)
                mutate(altered)
                with self.assertRaises(ValueError):
                    points.validate_frame(altered)

    def test_compact_foundation_accepts_later_growth_or_local_deposition_conservatively(self):
        import crust_inventory
        import native_material_evolution

        for birth_area, expected_mode in ((100., "deposition"), (1000., "growth")):
            with self.subTest(birth_area=birth_area, expected_mode=expected_mode):
                state = deepcopy(self.initial)
                xyz = positions_in_cell(state)[0]
                first, _ = qualified_source(state, xyz, birth_area)
                born = supply(state, [xyz], [birth_area], [first])
                self.assertEqual(born["new_faces"], 24)
                self.assertTrue(any("point_promotion" in row for row in
                    born["emplacement_geometry"]["sources"]))
                native_material_evolution.ensure_lineage(state)
                lineage = deepcopy(state.material_lineage)
                identity = {name: getattr(state, name).copy() for name in
                    ("parcel_patch", "parcel_arc_id", "parcel_plate", "trace_patch", "trace_id")}
                old_placements = deepcopy(state.native_arc_source_placements)
                old_mass = state.mass.copy()
                old_surface = {name: state.material_surface[name].copy()
                    for name in ("vertices", "faces", "area_km2", "reference_area_km2")}
                old_volume = float(state.material_surface["area_km2"] @
                                   state.structure["thickness_km"])
                before_grade = profile.measure(state.material_surface["vertices"],
                    state.material_surface["faces"], state.material_surface["area_km2"],
                    columns.elevation(state.structure), state.parcel_arc_basal_m)["face_grade"]

                state.t = 1.
                second, _ = qualified_source(state, xyz, 50., links=[first["origin_id"]])
                later = supply(state, [xyz], [50.], [second])
                self.assertEqual(later["new_faces"], 0)
                self.assertGreater(later["added_volume_km3"], 0.)
                self.assertEqual({row["mode"] for row in
                    later["emplacement_geometry"]["sources"]}, {expected_mode})
                self.assertAlmostEqual(later["added_volume_km3"] +
                    later["pending_magma_volume_km3"], 1250., places=7)
                actual_volume = float(state.material_surface["area_km2"] @
                                      state.structure["thickness_km"])
                self.assertAlmostEqual(actual_volume - old_volume,
                                       later["added_volume_km3"], places=7)
                self.assertAlmostEqual(float(state.mass @
                    state.structure["added_volume_km_per_reference_km2"]),
                    actual_volume, places=7)
                for name, value in identity.items():
                    np.testing.assert_array_equal(getattr(state, name), value, err_msg=name)
                for name, value in lineage.items():
                    np.testing.assert_array_equal(state.material_lineage[name], value, err_msg=name)
                self.assertEqual(state.native_arc_source_placements[:len(old_placements)],
                                 old_placements)
                placed = sum(row["volume_km3"] for row in state.native_arc_source_placements
                    if row["source_provenance"]["origin_id"] == second["origin_id"])
                held = sum(float(area) * 25. for area, origin in
                    zip(state.native_arc_pending["area"],
                        state.native_arc_pending["source_provenance"])
                    if origin["origin_id"] == second["origin_id"])
                self.assertAlmostEqual(placed + held, 1250., places=7)
                for fields in (state.structure, state.trace_structure):
                    columns._state(fields)
                    crust_inventory.validate(fields)
                    self.assertTrue(np.all((fields["thickness_km"] >= 8.) &
                                           (fields["thickness_km"] <= 75.)))
                after_grade = profile.measure(state.material_surface["vertices"],
                    state.material_surface["faces"], state.material_surface["area_km2"],
                    columns.elevation(state.structure), state.parcel_arc_basal_m)["face_grade"]
                self.assertTrue(np.all(after_grade <=
                    np.maximum(profile.MAX_GRADE, before_grade) + 1e-10))
                if expected_mode == "deposition":
                    self.assertGreater(held, 0.)
                    np.testing.assert_array_equal(state.mass, old_mass)
                    for name, value in old_surface.items():
                        np.testing.assert_array_equal(state.material_surface[name], value, err_msg=name)
                else:
                    self.assertEqual(held, 0.)
                    self.assertAlmostEqual(float(state.mass.sum() - old_mass.sum()), 50., places=8)
                    self.assertGreater(np.max(np.abs(state.material_surface["vertices"] -
                                                     old_surface["vertices"])), 0.)
                frame = dict(arcs.snapshot_fields(state), arc_material_diagnostics=later,
                             stats=deepcopy(state.process_totals), time_myr=state.t)
                points.validate_frame(frame)
                profile.validate_frame(frame)
                geometry.validate_frame(frame)

    def test_typed_checkpoint_then_next_observation_repeats_geometry_and_origin_receipts(self):
        state = self.state
        xyz, first = self.pending_source()
        compatibility = dict(engine_sha256="point-fixture", auxiliary_sources_sha256={},
                             numpy_version=np.__version__)
        manifest = dict(run_id="point-fixture", config=state.config,
                        frames=[dict(time_myr=0.)], frame_count=1, state="paused")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.npz"
            checkpoint.write_checkpoint(path, state, manifest, compatibility)
            restored, _ = checkpoint.read_checkpoint(path, compatibility, type(state))
        self.assertEqual(points.snapshot_fields(restored), points.snapshot_fields(state))
        for item in (state, restored):
            item.t = 1.
            second, _ = qualified_source(item, xyz, 50., links=[first["origin_id"]])
            supply(item, [xyz], [50.], [second])
        for name, value in arrays(state).items():
            np.testing.assert_array_equal(value, arrays(restored)[name], err_msg=name)
        self.assertEqual(restored.native_arc_source_placements, state.native_arc_source_placements)
        self.assertEqual(points.snapshot_fields(restored), points.snapshot_fields(state))
        self.assertEqual(pickle.dumps(restored.rng.bit_generator.state),
                         pickle.dumps(state.rng.bit_generator.state))


if __name__ == "__main__":
    unittest.main()
