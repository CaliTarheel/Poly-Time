"""Geographic arc admission and physical source accounting, independent of cells."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pickle
import json
import unittest
from unittest.mock import patch
import numpy as np

from test_native_processes import ocean_fixture
import arc_source_geometry as sources
import native_arc_material as arcs
import native_material_evolution as evolution
import native_material_adaptivity as adaptivity
import native_boundary_geometry as boundary
import crustal_structure as columns


def positions_in_cell(s,cell=17):
    vertices=s.native_mesh['vertices'][s.native_mesh['faces'][cell]]
    return arcs._unit(.86*s.xyz[cell]+.14*vertices[:2])


def add(s,positions,areas,owners=None):
    positions=np.asarray(positions,float).reshape(-1,3)
    cells=s._indices(positions)
    if owners is None:owners=s.plate[cells]
    return arcs.add_arc_crust(s,cells,np.asarray(areas,float),positions=positions,owners=np.asarray(owners))


# Records taken from a September 2026 recovery run. They were not published with the
# source, so the two tests that replay them skip where they are absent.
CAPTURES=Path(__file__).resolve().parents[1]/'validation/collision-repair-20260906'
RECOVERY_PATCHES=CAPTURES/'arc-recovery-birth-patches.json'
FIRST_FALSE_SOURCE=CAPTURES/'arc-first-false-source.npz'
NO_CAPTURE='This recovery-run capture is not part of the published source.'


class ArcSourceGeometryTests(unittest.TestCase):
    @unittest.skipUnless(RECOVERY_PATCHES.is_file(),NO_CAPTURE)
    def test_actual_short_recovery_step_uses_final_stored_birth_area_for_25km_source(self):
        records=json.loads(RECOVERY_PATCHES.read_text())
        s=ocean_fixture()
        points=np.array([row['centre'] for row in records])
        amounts=np.array([row['amount_km2'] for row in records])
        patches=[{name:np.asarray(value) for name,value in row['patch'].items()} for row in records]
        with patch.object(arcs,'_patch',side_effect=patches),patch.object(sources,'basement',return_value=np.array([r['basal_m'] for r in records])):
            result=add(s,points,amounts)
        expected=25.*float(amounts.sum())
        actual=float(s.material_surface['area_km2']@s.structure['thickness_km'])
        reference=float(s.mass@s.structure['added_volume_km_per_reference_km2'])
        self.assertAlmostEqual(result['added_volume_km3'],expected,places=10)
        self.assertAlmostEqual(actual,expected,places=10)
        self.assertAlmostEqual(reference,expected,places=10)
        self.assertLess(abs(result['source_volume_residual_km3']),1e-10)
        self.assertEqual(result['new_faces'],120)
        self.assertEqual(result['grown_patches'],0)
        self.assertTrue(np.all(s.structure['thickness_km']>=columns.MIN_THICKNESS_KM))
        self.assertTrue(np.all(s.structure['thickness_km']<=columns.MAX_THICKNESS_KM))

    @unittest.skipUnless(FIRST_FALSE_SOURCE.is_file(),NO_CAPTURE)
    def test_real_first_false_source_triangle_blocks_exact_180km_point(self):
        with np.load(FIRST_FALSE_SOURCE,allow_pickle=False) as data:
            s=SimpleNamespace(support=np.zeros((4,1)),
                material_surface=dict(vertices=data['vertices'],faces=data['faces']),
                kind=data['kind'],parcel_plate=data['owner'],parcel_collision_sheet=data['sheet'],collision_contacts=[])
            result=sources.classify(s,data['position'],data['owner'])
        self.assertFalse(result['eligible'][0])
        self.assertEqual(result['reason'][0],'continental_material')

    def test_birth_and_growth_time_partitions_use_identical_fixed_source_volume(self):
        totals=[]
        for chunks in ([1000.],[500.,500.],[250.]*4):
            s=ocean_fixture();point=positions_in_cell(s)[:1];reported=0.
            for amount in chunks:
                result=add(s,point,[amount]);reported+=result['added_volume_km3']
                self.assertAlmostEqual(result['supplied_magma_volume_km3'],25.*amount,places=8)
                self.assertLess(abs(result['source_volume_residual_km3']),1e-7)
            volume=float(s.material_surface['area_km2']@s.structure['thickness_km'])
            ledger=float(s.mass@s.structure['added_volume_km_per_reference_km2'])
            self.assertAlmostEqual(s.mass.sum(),1000.,places=8)
            self.assertAlmostEqual(volume,25000.,places=6)
            self.assertAlmostEqual(reported,volume,places=6)
            self.assertAlmostEqual(s.process_totals['arc_added_volume_km3'],volume,places=6)
            self.assertAlmostEqual(ledger,volume,places=6)
            self.assertTrue(np.all(s.structure['thickness_km']>=columns.MIN_THICKNESS_KM))
            self.assertTrue(np.all(s.structure['thickness_km']<=columns.MAX_THICKNESS_KM))
            totals.append(volume)
        np.testing.assert_allclose(totals,25000.,rtol=1e-12,atol=1e-7)

    def test_upper_arc_order_is_transitive_and_unordered_arcs_cannot_borrow_growth(self):
        s=ocean_fixture();point=positions_in_cell(s)[:1];add(s,point,[100.])
        triangle=s.material_surface['vertices'][s.material_surface['faces'][0]]
        # Actual upper arc 3 and lower continental sheet 1 contain the query;
        # recorded intermediate sheet 2 is spatially absent here.
        state=SimpleNamespace(support=np.zeros((1,1)),
            material_surface=dict(vertices=np.vstack((triangle,triangle)),faces=np.array([[0,1,2],[3,4,5]])),
            kind=np.array([3,1]),parcel_plate=np.array([0,0]),parcel_arc_id=np.array([9,0]),
            parcel_collision_sheet=np.array([3,1]),
            collision_contacts=[dict(top_sheet=3,under_sheet=2),dict(top_sheet=2,under_sheet=1)])
        inside=arcs._unit(triangle.sum(axis=0))[None]
        result=sources.classify(state,inside,np.array([0]))
        self.assertTrue(result['eligible'][0]);self.assertEqual(result['arc_id'][0],9)
        state.collision_contacts=[];state.kind[:]=3;state.parcel_arc_id[:]=[9,10]
        result=sources.classify(state,inside,np.array([0]))
        self.assertFalse(result['eligible'][0]);self.assertEqual(result['reason'][0],'unresolved_juvenile_material')

    def test_actual_source_position_survives_birth_without_cell_snapping(self):
        s=ocean_fixture();point=positions_in_cell(s)[:1]
        self.assertGreater(np.linalg.norm(point[0]-s.xyz[17])*6371.,10.)
        result=add(s,point,[100.])
        np.testing.assert_allclose(s.material_surface['vertices'][0],point[0],atol=2e-16)
        self.assertEqual(result['relocation_km'],0.)
        self.assertAlmostEqual(s.mass.sum(),100.,places=9)
        self.assertLess(abs(result['physical_volume_residual_km3']),1e-7)

    def test_same_cell_separated_sources_remain_independent_geographic_arcs(self):
        s=ocean_fixture();points=positions_in_cell(s)
        np.testing.assert_array_equal(s._indices(points),[17,17])
        self.assertGreater(np.linalg.norm(points[0]-points[1])*6371.,30.)
        result=add(s,points,[20.,30.])
        self.assertEqual(result['new_faces'],48)
        np.testing.assert_allclose(s.material_surface['vertices'][[0,17]],points,atol=2e-16)
        np.testing.assert_array_equal(np.unique(s.parcel_arc_id),[1,2])
        self.assertAlmostEqual(s.mass.sum(),50.,places=10)

    def test_source_grows_only_containing_component_after_same_arc_id_disconnects(self):
        s=ocean_fixture();points=positions_in_cell(s);add(s,points,[100.,100.])
        # Persistent birth identity may survive a physical fracture. Neither
        # original history nor same-owner labels join its disconnected pieces.
        s.parcel_arc_id[:]=1
        other_vertices=s.material_surface['vertices'][17:].copy()
        other_mass=s.mass[24:].copy();other_columns=s.structure['thickness_km'][24:].copy()
        result=add(s,points[:1],[50.])
        self.assertEqual(result['grown_patches'],1);self.assertEqual(result['new_faces'],0)
        np.testing.assert_array_equal(s.material_surface['vertices'][17:],other_vertices)
        np.testing.assert_array_equal(s.mass[24:],other_mass)
        np.testing.assert_array_equal(s.structure['thickness_km'][24:],other_columns)
        self.assertAlmostEqual(s.mass[:24].sum(),150.,places=8)
        self.assertAlmostEqual(result['added_volume_km3'],1250.,places=7)

    def test_continental_or_cratonic_actual_material_blocks_false_water_cell_birth(self):
        for kind in (1,2):
            with self.subTest(kind=kind):
                s=ocean_fixture();points=positions_in_cell(s)[:1];add(s,points,[100.])
                s.kind[:]=kind;s._sync_material();s.crust[:]=0
                before_mass=s.mass.copy();before_vertices=s.material_surface['vertices'].copy()
                result=add(s,points,[80.])
                self.assertEqual(result['added_area_km2'],0.)
                self.assertEqual(result['rejected_area_km2'],80.)
                self.assertEqual(result['rejected_by_reason_km2'],{'continental_material':80.})
                np.testing.assert_array_equal(s.mass,before_mass)
                np.testing.assert_array_equal(s.material_surface['vertices'],before_vertices)
                self.assertAlmostEqual(result['source_area_residual_km2'],0.)

    def test_false_continental_cell_flag_cannot_suppress_actual_ocean_source(self):
        s=ocean_fixture();s.crust[:]=2;point=positions_in_cell(s)[:1]
        self.assertTrue(sources.classify(s,point,np.array([0]))['eligible'][0])
        self.assertEqual(add(s,point,[25.])['added_area_km2'],25.)

    def test_ocean_guard_uses_continuous_owner_at_point_and_never_relocates(self):
        s=ocean_fixture(two=True);point=positions_in_cell(s)[:1]
        s.plate[:]=0;s.support[:]=0.;s.support[1]=1.
        prepared=boundary.prepare_owner_sampling(s.native_mesh,s.plate,s.support,locator=s.native_locator)
        self.assertEqual(boundary.sample_owners(point,prepared)[0],1)
        rejected=add(s,point,[40.],owners=[0])
        self.assertEqual(rejected['rejected_by_reason_km2'],{'foreign_ocean_owner':40.})
        self.assertEqual(len(s.mass),0)
        accepted=add(s,point,[40.],owners=[1])
        self.assertEqual(accepted['added_area_km2'],40.)
        self.assertTrue(np.all(s.parcel_plate==1))
        np.testing.assert_allclose(s.material_surface['vertices'][0],point[0],atol=2e-16)

    def test_growth_uses_containing_arc_and_adds_only_explicit_magma_volume(self):
        s=ocean_fixture();point=positions_in_cell(s)[:1];add(s,point,[1000.])
        inside=arcs._unit(.995*point[0]+.005*s.material_surface['vertices'][9])[None]
        query=sources.classify(s,inside,np.array([0]))
        self.assertEqual(query['arc_id'][0],1)
        s.structure['thickness_km'][:]=60.;old=float(s.material_surface['area_km2']@s.structure['thickness_km'])
        ids=s.parcel_patch.copy();result=add(s,np.vstack((point,inside)),[40.,60.])
        self.assertEqual(result['grown_patches'],1);self.assertEqual(result['new_faces'],0)
        np.testing.assert_array_equal(ids,s.parcel_patch)
        self.assertAlmostEqual(float(s.material_surface['area_km2']@s.structure['thickness_km'])-old,2500.,places=6)
        self.assertAlmostEqual(result['source_area_residual_km2'],0.,places=10)
        self.assertLess(abs(result['physical_mass_residual_km2']),1e-8)

    def test_subthreshold_pending_keeps_exact_position_and_combines_only_same_point(self):
        s=ocean_fixture();points=positions_in_cell(s)
        first=add(s,points,[.0004,.0004])
        self.assertEqual(first['added_area_km2'],0.)
        np.testing.assert_array_equal(s.native_arc_pending['xyz'],points)
        second=add(s,points[:1],[.0007])
        self.assertAlmostEqual(second['added_area_km2'],.0011,places=12)
        self.assertAlmostEqual(second['pending_area_km2'],.0004,places=12)
        np.testing.assert_array_equal(s.native_arc_pending['xyz'],points[1:])
        self.assertLess(abs(second['source_area_residual_km2']),1e-15)

    def test_previously_supplied_pending_is_retained_when_material_later_blocks_it(self):
        s=ocean_fixture();point=positions_in_cell(s)[:1];add(s,point,[100.])
        s.native_arc_pending=dict(xyz=point.copy(),owner=np.array([0],np.int16),area=np.array([80.]),
                                 emplacement_version=1,source_column_km=25.)
        s.kind[:]=1;s._sync_material()
        result=add(s,np.empty((0,3)),[],owners=np.empty(0,np.int16))
        self.assertEqual(result['added_area_km2'],0.);self.assertEqual(result['pending_area_km2'],80.)
        self.assertEqual(result['rejected_area_km2'],0.)
        self.assertEqual(result['source_area_residual_km2'],0.)
        np.testing.assert_array_equal(s.native_arc_pending['xyz'],point)

    def test_refinement_rotation_and_checkpoint_keep_exact_source_growth_identity(self):
        s=ocean_fixture();s.config.update(adaptive_refinement=1,deformation_width_km=80.)
        point=positions_in_cell(s)[:1];add(s,point,[100000.])
        s.material_deformation=dict(face_weight=np.ones(len(s.mass)),face_rigid=np.zeros(len(s.mass),bool),
                                    face_strain=np.zeros(len(s.mass)))
        s.t=2.;adaptivity.adapt(s,2.)
        s.omega[:]=[.045,.017,.002];evolution.advect(s,20.);s._sync_material()
        point=s.material_surface['vertices'][0][None].copy();ids=s.parcel_patch.copy()
        restored=pickle.loads(pickle.dumps(s,protocol=5))
        for world in (s,restored):
            result=add(world,point,[400.],owners=[0])
            self.assertEqual(result['grown_patches'],1);self.assertEqual(result['new_faces'],0)
            self.assertLess(abs(result['physical_volume_residual_km3']),1e-5)
        np.testing.assert_array_equal(ids,s.parcel_patch)
        np.testing.assert_array_equal(s.mass,restored.mass)
        np.testing.assert_array_equal(s.material_surface['vertices'],restored.material_surface['vertices'])

    def test_bad_positions_fail_before_mutation(self):
        s=ocean_fixture();before=pickle.dumps(s,protocol=5)
        with self.assertRaises(ValueError):
            arcs.add_arc_crust(s,np.array([17]),np.array([10.]),positions=[[0,0,2]],owners=np.array([0]))
        self.assertEqual(pickle.dumps(s,protocol=5),before)


if __name__=='__main__':unittest.main()
