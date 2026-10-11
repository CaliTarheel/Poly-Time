"""Screening unreported admission trials changes no amount, verdict or measurement.

`original_inspect` and `original_admit` are the functions exactly as they were before
trials were screened with the exact material test alone.
"""
from pathlib import Path
import hashlib
import math
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
import arc_emplacement_geometry as geometry
import arc_material_exclusion
import native_arc_material as arcs
import native_boundary_geometry
import native_spreading as clipping
from test_native_processes import ocean_fixture
from test_arc_source_geometry import add,positions_in_cell


def original_inspect(context, vertices, faces, owner, *, old_vertices=None, old_faces=None, old_indices=None,
                     _old_cache=None):
    vertices=np.asarray(vertices,float); faces=np.asarray(faces)
    if (vertices.ndim!=2 or vertices.shape[1:]!=(3,) or not np.isfinite(vertices).all()
            or faces.ndim!=2 or faces.shape[1:]!=(3,) or faces.dtype.kind not in 'iu'
            or not np.allclose(np.linalg.norm(vertices,axis=1),1.,rtol=0.,atol=2e-12)):
        raise ValueError('Arc admission requires finite unit, shared spherical geometry.')
    triangles=vertices[faces]
    if np.any(np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1],triangles[:,2]))<=0):
        raise ValueError('Arc admission requires outward material triangles.')
    candidate_area=math.fsum(clipping._area(p,context) for p in triangles)
    old=None
    if old_vertices is not None:
        if old_faces is None: raise ValueError('An old growth footprint requires faces.')
        if _old_cache is None:
            old=geometry._context(old_vertices,old_faces,context['radius_km'])
        else:
            v=np.asarray(old_vertices,float); f=np.asarray(old_faces,np.int64)
            key=(v.shape,v.tobytes(),f.shape,f.tobytes(),context['radius_km'])
            if _old_cache.get('key')!=key:
                _old_cache.clear()
                _old_cache.update(key=key,context=geometry._context(v,f,context['radius_km']))
            old=_old_cache['context']
    shell=[]
    for polygon in triangles:
        shell.extend(geometry._remaining(polygon,[old]) if old is not None else [polygon])
    new_area=math.fsum(clipping._area(p,context) for p in shell)
    if old is None: old_area=0.
    elif _old_cache is None: old_area=math.fsum(clipping._area(p,context) for p in old['triangles'])
    else:
        if 'area' not in _old_cache:
            _old_cache['area']=math.fsum(clipping._area(p,context) for p in old['triangles'])
        old_area=_old_cache['area']
    lost=max(0.,old_area+new_area-candidate_area)
    water=[]
    base=context['material']
    if old_indices is not None:
        geometry._face_caps(base)
        indices=np.asarray(old_indices,int)
        base=dict(base,exclude=indices[indices<len(base['triangles'])])
    material_contexts=[base]
    for row in context['updates']:
        if old_indices is not None and np.all(np.isin(row['indices'],old_indices)): continue
        material_contexts.append(row['context'])
    for polygon in shell: water.extend(geometry._remaining(polygon,material_contexts))
    water_area=math.fsum(clipping._area(p,context) for p in water)
    own_area=geometry._owner_area(water,context,owner)
    water_vertices=np.concatenate(water) if water else np.empty((0,3))
    foreign_vertices=int(np.count_nonzero(native_boundary_geometry.sample_owners(
        water_vertices,context['owner'])!=owner)) if len(water_vertices) else 0
    budget_tolerance=geometry.tolerance(max(new_area,candidate_area,old_area))
    if water_area>new_area+budget_tolerance or own_area>water_area+budget_tolerance:
        raise ValueError('Arc admission clipping increased available source footprint area.')
    material_loss=max(0.,new_area-water_area)
    owner_loss=max(0.,water_area-own_area)
    admission_tolerance=max(1e-9,new_area*1e-12)
    admissible=bool(lost<=budget_tolerance and material_loss+owner_loss<=admission_tolerance
                    and foreign_vertices==0)
    exact_obstruction=None
    if admissible:
        exact_obstruction=arc_material_exclusion.obstruction(triangles,material_contexts,old)
        admissible=exact_obstruction is None
    return dict(candidate_area_km2=candidate_area,old_area_km2=old_area,new_footprint_area_km2=new_area,
        lost_old_footprint_area_km2=lost,material_obstruction_km2=material_loss,
        foreign_owner_obstruction_km2=owner_loss,foreign_owner_vertices=foreign_vertices,
        eligible_new_footprint_area_km2=own_area,
        area_tolerance_km2=budget_tolerance,admission_tolerance_km2=admission_tolerance,
        material_exclusion_version=1,exact_material_obstruction=exact_obstruction,admissible=admissible)


def original_admit(context, factory, requested_area_km2, owner, *, minimum_area_km2=.001,
                   old_vertices=None, old_faces=None, old_indices=None):
    requested=float(requested_area_km2)
    if not np.isfinite(requested) or requested<0.: raise ValueError('Arc source area must be finite and nonnegative.')
    count=0; first=None; best=None; last=None; old_cache={}
    def evaluate(amount):
        nonlocal count, first, last
        count+=1; plan=factory(amount)
        if plan is None:
            value=dict(admissible=False, reason='unrepresentable_shared_mesh')
        else:
            vertices=np.asarray(plan['vertices'],float)
            if old_vertices is None:vertices=vertices/np.linalg.norm(vertices,axis=1)[:,None]
            value=original_inspect(context,vertices,plan['faces'],owner,
                          old_vertices=old_vertices,old_faces=old_faces,old_indices=old_indices,
                          _old_cache=old_cache)
            if value['admissible'] and old_vertices is None:
                plan=dict(plan)
                plan['_emplacement_stored_triangle_sha256']=hashlib.sha256(
                    vertices[np.asarray(plan['faces'])].tobytes()).hexdigest()
        if first is None: first=value.copy()
        last=value
        return plan,value
    amount=requested; rejected=requested
    while amount>=minimum_area_km2 and count<32:
        plan,measurement=evaluate(amount)
        if measurement['admissible']:
            best=(amount,plan,measurement); break
        rejected=amount; amount*=.5
    if best is not None and best[0]<requested:
        lower=best[0]; upper=rejected
        for _ in range(20):
            if upper-lower<=max(minimum_area_km2*1e-3,requested*1e-6): break
            middle=(lower+upper)*.5; plan,measurement=evaluate(middle)
            if measurement['admissible']:
                lower=middle; best=(middle,plan,measurement)
            else: upper=middle
    accepted=0. if best is None else best[0]
    return dict(accepted_area_km2=accepted,pending_area_km2=requested-accepted,
        plan=None if best is None else best[1],
        diagnostics=dict(version=geometry.VERSION,requested_area_km2=requested,accepted_area_km2=accepted,
            pending_area_km2=requested-accepted,geometry_evaluations=count,
            requested_footprint=first,accepted_footprint=None if best is None else best[2],
            last_examined_footprint=last,relocation_km=0.))


def identical(a, b, path='result'):
    """Recursive bitwise comparison; returns the first differing path, or None."""
    if isinstance(a,dict) or isinstance(b,dict):
        if not (isinstance(a,dict) and isinstance(b,dict)) or list(a)!=list(b): return path+' keys'
        for key in a:
            found=identical(a[key],b[key],f'{path}[{key!r}]')
            if found: return found
        return None
    if isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)):
        if type(a) is not type(b) or len(a)!=len(b): return path+' length'
        for index,(x,y) in enumerate(zip(a,b)):
            found=identical(x,y,f'{path}[{index}]')
            if found: return found
        return None
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        a,b=np.asarray(a),np.asarray(b)
        return None if a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes() else path
    if isinstance(a,float) and isinstance(b,float):
        return None if np.float64(a).tobytes()==np.float64(b).tobytes() else path
    return None if type(a) is type(b) and a==b else path


def state():
    s=ocean_fixture(level=2)
    s.native_arc_emplacement_version=1
    return s


def offset(point, direction, kilometres):
    normal=arcs._unit(direction-point*np.dot(point,direction))
    angle=kilometres/6371.
    return arcs._unit(point*np.cos(angle)+normal*np.sin(angle))


def beside_existing_footprint(shift_km, requested):
    """A birth requested `shift_km` across strike from the centre of a 600 km2 footprint."""
    centre=positions_in_cell(state())[:1][0]
    strike=arcs._unit(np.cross(centre,[0.,0.,1.]))
    old=arcs._patch(centre,strike,600.)
    s=state()
    s.material_surface=dict(vertices=old['vertices']/np.linalg.norm(old['vertices'],axis=1)[:,None],
                            faces=old['faces'],radius_km=6371.)
    point=offset(centre,np.cross(centre,strike),shift_km) if shift_km else centre
    return s,lambda amount:arcs._patch(point,strike,amount),requested


class ScreenedAdmissionTests(unittest.TestCase):
    def both(self,build,**keywords):
        """Run the same admission from two fresh, identical contexts."""
        results=[]
        for admit in (original_admit,geometry.admit):
            s,factory,requested=build()
            context=geometry.prepare(s)
            with patch.object(geometry,'inspect',wraps=geometry.inspect) as measured:
                results.append((admit(context,factory,requested,0,**keywords),measured.call_count))
        (expected,_),(actual,full_measurements)=results
        self.assertIsNone(identical(actual,expected))
        return actual,full_measurements

    def test_partial_admission_through_halving_and_bisection_is_identical(self):
        result,full=self.both(lambda:beside_existing_footprint(10.8,140.))
        self.assertGreater(result['accepted_area_km2'],1.)
        self.assertLess(result['accepted_area_km2'],140.)
        evaluations=result['diagnostics']['geometry_evaluations']
        self.assertGreater(evaluations,10)
        # Rejections between the first and last trial were settled by the exact test alone.
        self.assertLess(full,evaluations)

    def test_source_inside_existing_material_is_rejected_at_every_amount_identically(self):
        result,full=self.both(lambda:beside_existing_footprint(0.,140.))
        self.assertEqual(result['accepted_area_km2'],0.)
        self.assertIsNone(result['plan'])
        evaluations=result['diagnostics']['geometry_evaluations']
        self.assertGreater(evaluations,10)
        self.assertLessEqual(full,3)   # the first trial, the last one, and nothing in between
        for name in ('requested_footprint','last_examined_footprint'):
            self.assertNotEqual(result['diagnostics'][name].get('reason'),'exact_material_obstruction')

    def test_unobstructed_birth_is_one_full_measurement(self):
        result,full=self.both(lambda:beside_existing_footprint(400.,140.))
        self.assertEqual(result['accepted_area_km2'],140.)
        self.assertEqual((result['diagnostics']['geometry_evaluations'],full),(1,1))

    def test_small_minimum_and_tight_trial_limit_still_report_the_true_last_trial(self):
        for minimum in (.001,1.,50.,139.,141.):
            with self.subTest(minimum=minimum):
                self.both(lambda:beside_existing_footprint(0.,140.),minimum_area_km2=minimum)
                self.both(lambda:beside_existing_footprint(10.8,140.),minimum_area_km2=minimum)
        # Thirty-two trials end the coarse search before the minimum does.
        result,_=self.both(lambda:beside_existing_footprint(0.,140.),minimum_area_km2=1e-12)
        self.assertEqual(result['diagnostics']['geometry_evaluations'],32)

    def test_owner_boundary_rejection_is_identical_and_stops_screening(self):
        def build():
            s=state()
            vertices=s.native_mesh['vertices']
            point=arcs._unit([1.,2./6371.,0.])
            return s,lambda area:arcs._patch(point,[0.,0.,1.],area),100.
        results=[]
        for admit in (original_admit,geometry.admit):
            s,factory,requested=build()
            context=geometry.prepare(s)
            vertices=s.native_mesh['vertices']
            context['owner']['owner_slots']=np.array([0,1])
            context['owner']['scores']=np.vstack((1.+vertices[:,1],1.-vertices[:,1]))
            with patch.object(geometry,'_exact_obstruction',wraps=geometry._exact_obstruction) as screened:
                results.append((admit(context,factory,requested,0),screened.call_count))
        (expected,_),(actual,screens)=results
        self.assertIsNone(identical(actual,expected))
        self.assertGreater(actual['diagnostics']['requested_footprint']['foreign_owner_obstruction_km2'],1.)
        self.assertGreater(actual['diagnostics']['geometry_evaluations'],5)
        # The exact material test finds nothing at an owner boundary, so it is tried once.
        self.assertEqual(screens,1)

    def test_exact_test_outside_its_domain_defers_to_the_full_measurement(self):
        outcomes=[]
        for admit in (original_admit,geometry.admit):
            s,factory,requested=beside_existing_footprint(10.8,140.)
            context=geometry.prepare(s)
            with patch.object(arc_material_exclusion,'obstruction',side_effect=ValueError('fixture domain')):
                try:
                    outcomes.append(('returned',admit(context,factory,requested,0)))
                except ValueError as error:
                    outcomes.append(('raised',str(error)))
        self.assertEqual(outcomes[0][0],outcomes[1][0])
        self.assertIsNone(identical(outcomes[1][1],outcomes[0][1]))

    def test_malformed_candidate_geometry_raises_on_a_screened_trial_too(self):
        s,factory,requested=beside_existing_footprint(0.,140.)
        context=geometry.prepare(s)
        calls=[]
        def broken(amount):
            calls.append(amount)
            plan=factory(amount)
            if len(calls)==3:
                plan=dict(plan,vertices=plan['vertices']*np.nan)
            return plan
        with self.assertRaisesRegex(ValueError,'finite unit, shared spherical geometry'):
            geometry.admit(context,broken,requested,0)
        self.assertEqual(len(calls),3)

    def test_exact_test_alone_agrees_with_the_one_inside_inspect(self):
        for shift in (0.,10.8,400.):
            s,factory,_=beside_existing_footprint(shift,140.)
            context=geometry.prepare(s)
            plan=factory(.5)
            vertices=plan['vertices']/np.linalg.norm(plan['vertices'],axis=1)[:,None]
            alone=geometry._exact_obstruction(context,vertices,plan['faces'])
            measured=original_inspect(context,vertices,plan['faces'],0)
            given=geometry.inspect(context,vertices,plan['faces'],0,_exact=alone)
            self.assertIsNone(identical(given,measured))
            if measured['exact_material_obstruction'] is not None or measured['admissible']:
                self.assertIsNone(identical(alone,measured['exact_material_obstruction']))


class ScreenedEmplacementSequenceTests(unittest.TestCase):
    """Births, later growth and deposition through add_arc_crust, original against screened."""
    def run_sequence(self,sequence,admit):
        s=state()
        with patch.object(geometry,'admit',admit):
            reports=sequence(s)
        surface=s.material_surface
        return dict(reports=reports,mass=np.asarray(s.mass),vertices=np.asarray(surface['vertices']),
            faces=np.asarray(surface['faces']),area=np.asarray(surface['area_km2']),
            arc=np.asarray(s.parcel_arc_id),thickness=np.asarray(s.structure['thickness_km']),
            pending={key:np.asarray(value) for key,value in s.native_arc_pending.items()
                     if isinstance(value,(np.ndarray,list))},
            diagnostics=getattr(s,'arc_material_diagnostics',None))

    def compare(self,sequence):
        expected=self.run_sequence(sequence,original_admit)
        actual=self.run_sequence(sequence,geometry.admit)
        self.assertIsNone(identical(actual,expected))
        return actual

    def test_same_batch_births_then_blocked_growth(self):
        def sequence(s):
            point=positions_in_cell(s)[:1][0]
            second=offset(point,s.native_mesh['vertices'][0],7.)
            points=np.vstack((point,second))
            return [add(s,points,[100.,100.]),add(s,points[1:],[100.]),add(s,points,[60.,60.])]
        result=self.compare(sequence)
        self.assertGreater(result['reports'][1]['pending_area_km2'],0.)

    def test_obstructed_source_in_one_and_two_instalments(self):
        for amounts in ([100.],[50.,50.]):
            def sequence(s):
                first=positions_in_cell(s)[:1][0]
                reports=[add(s,first[None],[100.])]
                source=offset(first,s.native_mesh['vertices'][0],7.)[None]
                return reports+[add(s,source,[amount]) for amount in amounts]
            with self.subTest(amounts=amounts):
                self.compare(sequence)

    def test_unobstructed_growth_in_instalments(self):
        def sequence(s):
            point=positions_in_cell(s)[:1]
            return [add(s,point,[amount]) for amount in (250.,250.,500.)]
        result=self.compare(sequence)
        self.assertEqual(result['reports'][-1]['pending_area_km2'],0.)


if __name__=='__main__':
    unittest.main()
