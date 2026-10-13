"""Independent geographic/source-budget checks for the isolated candidate."""
from pathlib import Path
from types import SimpleNamespace
import json
import pickle
from copy import deepcopy
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'validation/collision-repair-after-200-20260907/candidate-arc'
sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
import native_arc_material as arcs
import arc_emplacement_geometry as geometry
import mesh_coverage
import material_surface
import mesh_geometry
import orientation
from test_native_processes import ocean_fixture
from test_arc_source_geometry import add,positions_in_cell


def state():
    s=ocean_fixture(level=2)
    s.native_arc_emplacement_version=1
    return s


def overlap_area(s):
    surface=s.material_surface
    result=mesh_coverage.material_overlaps(surface['vertices'],surface['faces'],s.parcel_arc_id)
    return float(result['area_km2'].sum())


def offset(point, direction, kilometres):
    normal=arcs._unit(direction-point*np.dot(point,direction))
    angle=kilometres/6371.
    return arcs._unit(point*np.cos(angle)+normal*np.sin(angle))


# Two arcs that overlapped in a 200 Myr run, saved before this guard existed. The capture
# was never published with the source, so the tests that read it skip where it is absent.
CAPTURE=HERE.parent/'candidate-performance/actual-arc-overlap-200.npz'
NO_CAPTURE='The 200 Myr arc-overlap capture is not part of the published source.'


def captured_intrusion():
    with np.load(CAPTURE) as data:
        old_v=data['vertices_0'].copy();old_f=data['faces_0'].copy()
        point=data['source_centre_1'].copy();new_v=data['vertices_1'].copy()
        requested=float(data['material_reference_area_km2_1'].sum())
    strike=arcs._unit(new_v[1]-point*np.dot(new_v[1],point))
    return old_v,old_f,point,strike,requested


def constructed_intrusion():
    """The same situation built from first principles, so it always runs.

    A 600 km2 footprint already exists. A 140 km2 birth is requested 10.8 km across
    strike from its centre: the source lies outside the old footprint, but the full
    requested oval intrudes into it by about 37 km2, so only a few km2 can be admitted.
    """
    centre=positions_in_cell(state())[:1][0]
    strike=arcs._unit(np.cross(centre,[0.,0.,1.]))
    old=arcs._patch(centre,strike,600.)
    old_v=old['vertices']/np.linalg.norm(old['vertices'],axis=1)[:,None]
    point=offset(centre,np.cross(centre,strike),10.8)
    for _ in range(4):
        # _patch normalizes its centre; start from a unit vector that it leaves untouched.
        again=arcs._unit(point)
        if np.array_equal(again,point): break
        point=again
    return old_v,old['faces'],point,strike,140.


class ArcEmplacementTests(unittest.TestCase):
    @unittest.skipUnless(CAPTURE.is_file(),NO_CAPTURE)
    def test_actual_200_footprint_intrusion_retains_source_and_pending_volume(self):
        self.check_intrusion_retains_source_and_pending_volume(*captured_intrusion())

    def test_constructed_footprint_intrusion_retains_source_and_pending_volume(self):
        self.check_intrusion_retains_source_and_pending_volume(*constructed_intrusion())

    def check_intrusion_retains_source_and_pending_volume(self,old_v,old_f,point,strike,requested):
        s=state()
        s.material_surface=dict(vertices=old_v,faces=old_f,radius_km=6371.)
        context=geometry.prepare(s)
        factory=lambda amount:arcs._patch(point,strike,amount)
        result=geometry.admit(context,factory,requested,0)
        self.assertGreater(result['diagnostics']['requested_footprint']['material_obstruction_km2'],27.)
        self.assertGreater(result['accepted_area_km2'],.001)
        self.assertLess(result['accepted_area_km2'],requested)
        self.assertGreater(result['pending_area_km2'],0.)
        self.assertAlmostEqual(result['accepted_area_km2']+result['pending_area_km2'],requested,places=11)
        accepted=result['diagnostics']['accepted_footprint']
        self.assertLessEqual(accepted['material_obstruction_km2'],accepted['area_tolerance_km2'])
        np.testing.assert_array_equal(result['plan']['vertices'][0],point)

    def test_same_batch_births_and_later_growth_cannot_create_overlap(self):
        s=state();point=positions_in_cell(s)[:1][0]
        second=offset(point,s.native_mesh['vertices'][0],7.)
        points=np.vstack((point,second));report=add(s,points,[100.,100.])
        self.assertEqual(report['new_faces'],48)
        self.assertGreater(report['pending_area_km2'],0.)
        self.assertLess(overlap_area(s),2e-7)
        before=float(s.material_surface['area_km2']@s.structure['thickness_km'])
        self.assertAlmostEqual(before,25.*report['added_area_km2'],places=6)
        self.assertAlmostEqual(report['added_area_km2']+report['pending_area_km2'],200.,places=9)
        snapshot=s.material_surface['vertices'][:17].copy()
        again=add(s,points[1:],[100.])
        self.assertLess(overlap_area(s),2e-7)
        self.assertGreater(again['pending_area_km2'],0.)
        self.assertLess(abs(again['source_area_residual_km2']),1e-8)
        self.assertLess(abs(again['source_volume_residual_km3']),1e-7)
        self.assertLess(abs(again['physical_volume_residual_km3']),1e-6)
        np.testing.assert_array_equal(s.material_surface['vertices'][:17],snapshot)

    def test_single_source_time_partitions_keep_full_volume_without_obstacles(self):
        for amounts in ([1000.],[500.,500.],[250.]*4):
            s=state();point=positions_in_cell(s)[:1]
            for amount in amounts:
                report=add(s,point,[amount])
                self.assertAlmostEqual(report['added_area_km2'],amount,places=9)
                self.assertEqual(report['pending_area_km2'],0.)
            self.assertEqual(len(s.mass),24)
            self.assertAlmostEqual(float(s.mass.sum()),1000.,places=8)
            self.assertAlmostEqual(float(s.material_surface['area_km2']@s.structure['thickness_km']),25000.,places=6)

    def test_existing_tectonic_overlap_does_not_block_unoccupied_growth_shell(self):
        s=state();point=positions_in_cell(s)[:1]
        # The retained old lower patch lies wholly inside the old arc. Its
        # preexisting overlap must not be mistaken for NEW juvenile intrusion.
        outer=arcs._patch(point[0],[1.,0.,0.],200.)
        inner=arcs._patch(point[0],[1.,0.,0.],20.)
        vertices=np.vstack((outer['vertices'],inner['vertices']))
        faces=np.vstack((outer['faces'],inner['faces']+17))
        s.material_surface=dict(vertices=vertices,faces=faces,radius_km=6371.)
        context=geometry.prepare(s)
        candidate=arcs._patch(point[0],[1.,0.,0.],240.)
        report=geometry.inspect(context,candidate['vertices'],candidate['faces'],0,
            old_vertices=outer['vertices'],old_faces=outer['faces'])
        self.assertTrue(report['admissible'])
        self.assertAlmostEqual(report['new_footprint_area_km2'],40.,places=6)
        self.assertLess(report['material_obstruction_km2'],1e-7)

    def test_finite_foreign_owner_coast_is_checked_even_with_valid_source_center(self):
        s=state();context=geometry.prepare(s)
        # Independent analytic P1 ownership: owner zero occupies y>=0.
        # The exact source is two kilometres inside that owner, while its
        # proposed 100-km2 oval extends into owner one's water.
        vertices=s.native_mesh['vertices']
        context['owner']['owner_slots']=np.array([0,1])
        context['owner']['scores']=np.vstack((1.+vertices[:,1],1.-vertices[:,1]))
        point=arcs._unit([1.,2./6371.,0.])
        result=geometry.admit(context,lambda area:arcs._patch(point,[0.,0.,1.],area),100.,0)
        self.assertGreater(result['diagnostics']['requested_footprint']['foreign_owner_obstruction_km2'],1.)
        self.assertGreater(result['accepted_area_km2'],0.)
        self.assertLess(result['accepted_area_km2'],100.)
        self.assertGreaterEqual(result['plan']['vertices'][:,1].min(),-2e-10)
        self.assertLessEqual(result['diagnostics']['accepted_footprint']['foreign_owner_obstruction_km2'],1e-9)

    @unittest.skipUnless(CAPTURE.is_file(),NO_CAPTURE)
    def test_refinement_and_rotation_preserve_actual_obstruction_and_admission(self):
        old_v,old_f,point,strike,_=captured_intrusion()
        self.check_refinement_and_rotation_preserve_obstruction_and_admission(
            old_v,old_f,point,strike,84.89948981534839)

    def test_refinement_and_rotation_preserve_constructed_obstruction_and_admission(self):
        self.check_refinement_and_rotation_preserve_obstruction_and_admission(*constructed_intrusion())

    def check_refinement_and_rotation_preserve_obstruction_and_admission(self,old_v,old_f,point,strike,requested):
        totals=[];full=[]
        for mode in ('original','subdivided','dateline','pole'):
            s=state();v=old_v.copy();f=old_f.copy();p=point.copy();direction=strike.copy()
            if mode=='subdivided':
                vertices=list(v);midpoints={};refined=[]
                for a,b,c in f:
                    middle=[]
                    for i,j in ((a,b),(b,c),(c,a)):
                        key=tuple(sorted((int(i),int(j))))
                        if key not in midpoints:
                            midpoints[key]=len(vertices);vertices.append(arcs._unit(v[i]+v[j]))
                        middle.append(midpoints[key])
                    ab,bc,ca=middle
                    refined.extend(((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)))
                v=np.asarray(vertices);f=np.asarray(refined)
            if mode in ('dateline','pole'):
                # Construct a proper rotation mapping the source exactly to
                # the dateline or pole; rotate the control mesh as well.
                target=np.array([-1.,0.,0.]) if mode=='dateline' else np.array([0.,0.,1.])
                before=np.column_stack((p,direction,np.cross(p,direction)))
                after_tangent=np.array([0.,1.,0.])
                after=np.column_stack((target,after_tangent,np.cross(target,after_tangent)))
                rotation=before@after.T
                v=v@rotation;p=p@rotation;direction=direction@rotation
                s.native_mesh=dict(s.native_mesh,vertices=s.native_mesh['vertices']@rotation)
                s.native_locator=mesh_geometry.build_locator(s.native_mesh['vertices'],s.native_mesh['faces'])
            s.material_surface=dict(vertices=v,faces=f,radius_km=6371.)
            context=geometry.prepare(s)
            result=geometry.admit(context,lambda area:arcs._patch(p,direction,area),requested,0)
            totals.append(result['accepted_area_km2'])
            full.append(result['diagnostics']['requested_footprint']['material_obstruction_km2'])
            self.assertTrue(result['diagnostics']['accepted_footprint']['admissible'])
        np.testing.assert_allclose(full,full[0],rtol=2e-9,atol=1e-7)
        np.testing.assert_allclose(totals,totals[0],rtol=5e-4,atol=1e-4)

    def test_obstructed_source_time_partition_keeps_accepted_and_pending_budget(self):
        totals=[]
        for amounts in ([100.],[50.,50.]):
            s=state();first=positions_in_cell(s)[:1][0]
            add(s,first[None],[100.])
            source=offset(first,s.native_mesh['vertices'][0],7.)[None]
            for amount in amounts: report=add(s,source,[amount])
            total=float(s.mass.sum())-100.
            pending=float(s.native_arc_pending['area'].sum())
            self.assertAlmostEqual(total+pending,100.,places=8)
            self.assertLess(overlap_area(s),2e-7)
            np.testing.assert_allclose(s.native_arc_pending['xyz'],source,rtol=0.,atol=2e-16)
            totals.append(total)
        np.testing.assert_allclose(totals,totals[0],rtol=1e-3,atol=1e-4)

    def test_policy_schema_rejects_tampered_geometry_and_source_ledgers(self):
        s=state();point=positions_in_cell(s)[:1];report=add(s,point,[100.])
        frame=dict(arcs.snapshot_fields(s),time_myr=2.,arc_material_diagnostics=report)
        geometry.validate_frame(frame)
        changed=[]
        item=deepcopy(frame);item.pop('arc_emplacement_version');changed.append(item)
        item=deepcopy(frame);item['arc_emplacement_version']=True;changed.append(item)
        item=deepcopy(frame);item['arc_emplacement_version']=2;changed.append(item)
        item=deepcopy(frame);item['arc_material_diagnostics'].pop('emplacement_geometry');changed.append(item)
        for key,value in [('accepted_area_km2',101.),('pending_area_km2',1.),('sequential_occupancy',False)]:
            item=deepcopy(frame);item['arc_material_diagnostics']['emplacement_geometry'][key]=value;changed.append(item)
        for key,value in [('geometry_xyz',[[0.,0.,2.]]),('owner',-1),('relocation_km',1.)]:
            item=deepcopy(frame);item['arc_material_diagnostics']['emplacement_geometry']['sources'][0][key]=value;changed.append(item)
        for key,value in [('material_obstruction_km2',1.),('admission_tolerance_km2',10.),('foreign_owner_vertices',1),
                          ('new_footprint_area_km2',float('nan'))]:
            item=deepcopy(frame)
            item['arc_material_diagnostics']['emplacement_geometry']['sources'][0]['accepted_footprint'][key]=value
            changed.append(item)
        for number,item in enumerate(changed):
            with self.subTest(mutant=number),self.assertRaises(ValueError): geometry.validate_frame(item)
        geometry.validate_frame(dict(arc_material_diagnostics={},time_myr=2.))
        geometry.validate_frame(dict(arc_emplacement_version=1,arc_material_version=1,time_myr=0.,arc_material_diagnostics={},
            arc_pending_source=dict(version=1,source_column_km=25.,geometry_xyz=[],owner=[],area_km2=[],
                                    total_area_km2=0.,total_volume_km3=0.)))

    def test_nonempty_unversioned_pending_cannot_be_reinterpreted_as_new_source(self):
        s=state();point=positions_in_cell(s)[:1]
        s.native_arc_pending=dict(xyz=point,owner=np.array([0]),area=np.array([10.]))
        before=pickle.dumps(s,protocol=5)
        with self.assertRaisesRegex(ValueError,'explicit compatible'):
            add(s,np.empty((0,3)),[],owners=np.empty(0,int))
        self.assertEqual(pickle.dumps(s,protocol=5),before)

    def test_pending_checkpoint_replay_and_motion_preserve_actual_source_position(self):
        s=state();first=positions_in_cell(s)[:1][0]
        source=offset(first,s.native_mesh['vertices'][0],7.)[None]
        add(s,np.vstack((first,source)),[100.,100.])
        restored=pickle.loads(pickle.dumps(s,protocol=5))
        for world in (s,restored): add(world,np.empty((0,3)),[],owners=np.empty(0,int))
        np.testing.assert_array_equal(s.mass,restored.mass)
        np.testing.assert_array_equal(s.material_surface['vertices'],restored.material_surface['vertices'])
        np.testing.assert_array_equal(s.native_arc_pending['xyz'],restored.native_arc_pending['xyz'])
        np.testing.assert_array_equal(s.native_arc_pending['area'],restored.native_arc_pending['area'])
        self.assertEqual(s.native_arc_emplacement_version,restored.native_arc_emplacement_version)
        from ridge_geometry import rotate
        import native_processes
        old=s.native_arc_pending['xyz'].copy();s.omega[:]=[.003,-.001,.002]
        expected=arcs._unit(rotate(old,s.omega[s.native_arc_pending['owner']]*2.))
        native_processes.advect_ocean(s,2.)
        np.testing.assert_array_equal(s.native_arc_pending['xyz'],expected)


if __name__=='__main__':
    unittest.main(verbosity=2)
