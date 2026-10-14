"""Admit finite juvenile footprints into actual unoccupied, intended-owner water.

This is a geometric emplacement constraint, not a magma reservoir or a new
collision law. A bounded search retains the proposed shared mesh and source
position. Rejected volume stays pending; it never increases the density or
thickness of the smaller accepted footprint. Existing tectonic overlap is not
removed: growth tests only the new set outside the retained component.
"""
from __future__ import annotations

import math
import hashlib
from numbers import Real, Integral
import numpy as np
import mesh_geometry
import native_boundary_geometry
import convex_partition
import native_spreading as clipping

VERSION = 1
ABSOLUTE_AREA_TOLERANCE_KM2 = 1e-7
RELATIVE_AREA_TOLERANCE = 1e-10


def validate_pending(pending):
    """A nonempty old reservoir cannot be silently reinterpreted as policy 1."""
    area=np.asarray(pending.get('area')); owners=np.asarray(pending.get('owner'))
    points=np.asarray(pending.get('xyz'))
    if (area.ndim!=1 or area.dtype.kind not in 'fiu' or not np.isfinite(area).all() or np.any(area<0)
            or owners.shape!=area.shape or owners.dtype.kind not in 'iu' or np.any(owners<0)
            or points.shape!=(len(area),3) or points.dtype.kind not in 'fiu' or not np.isfinite(points).all()
            or not np.allclose(np.linalg.norm(points,axis=1),1.,rtol=0.,atol=2e-10)):
        raise ValueError('Invalid geographic pending juvenile inventory.')
    if len(area) and (type(pending.get('emplacement_version')) is not int
                     or pending['emplacement_version']!=VERSION or pending.get('source_column_km')!=25.):
        raise ValueError('Pending juvenile material requires an explicit compatible 25-km source policy.')


def pending_snapshot(pending):
    validate_pending(pending)
    area=np.asarray(pending['area'],float)
    result=dict(version=VERSION,source_column_km=25.,geometry_xyz=np.asarray(pending['xyz']).tolist(),
                owner=np.asarray(pending['owner']).tolist(),area_km2=area.tolist(),
                total_area_km2=float(area.sum()),total_volume_km3=float(area.sum()*25.))
    if 'source_provenance' in pending:
        from copy import deepcopy
        import arc_source_cohorts
        arc_source_cohorts.validate(pending['source_provenance'],len(area))
        result['source_provenance']=deepcopy(pending['source_provenance'])
    return result


def tolerance(area):
    return max(ABSOLUTE_AREA_TOLERANCE_KM2, abs(float(area))*RELATIVE_AREA_TOLERANCE)


def _context(vertices, faces, radius):
    result=clipping.prepare_material(dict(vertices=np.asarray(vertices, float).copy(),
        faces=np.asarray(faces, np.int64).copy(), radius_km=radius))
    result['cap_center'],result['cap_chord_radius']=_cap(result['vertices'])
    result['_exact_material_cache']={}
    result['locator']=_lookup_locator(result['locator'])
    result['_candidate_boxes']={}
    result['_partition_edges']={}
    return result


def _lookup_locator(locator):
    """Private immutable bin lookup for one emplacement call, never saved state."""
    result=dict(locator)
    for key in ('keys','offsets','candidates','global_faces'):
        result[key]=np.asarray(locator[key]).copy()
        result[key].setflags(write=False)
    return result


def _candidates(polygon, context):
    """Memoize the complete sorted lookup by its unchanged conservative box."""
    if not len(context['faces']): return np.empty(0,np.int32)
    center=clipping._unit(np.sum(polygon,axis=0))
    angle=np.arccos(np.clip(polygon@center,-1.,1.)).max(initial=0.)
    locator=context['locator']; resolution=int(locator['resolution'])
    if angle>=np.pi/2-1e-8:
        key=('whole',)
    else:
        radius=2.*np.sin(angle/2.)
        lo=np.clip(np.floor((center-radius-1e-12+1)*resolution*.5).astype(int),0,resolution-1)
        hi=np.clip(np.floor((center+radius+1e-12+1)*resolution*.5).astype(int),0,resolution-1)
        key=tuple(map(int,np.r_[lo,hi]))
    if context.get('_candidate_locator') is not locator:
        context['_candidate_locator']=locator
        context['_candidate_boxes']={}
    cache=context['_candidate_boxes']
    if key not in cache:
        result=clipping._candidates(polygon,context)
        result.setflags(write=False)
        # This bounds memo storage only; evicted boxes are queried completely.
        if len(cache)>=4096: cache.pop(next(iter(cache)))
        cache[key]=result
    return cache[key]


def _cap(vertices):
    center=clipping._unit(np.sum(vertices,axis=0))
    if not len(vertices) or np.linalg.norm(center)<.5: return center,2.
    angle=np.arccos(np.clip(vertices@center,-1.,1.)).max(initial=0.)
    # A sub-hemisphere spherical cap contains the minor arcs and all convex
    # face interiors, not just their vertices. Otherwise use the whole sphere.
    return center,2.*np.sin(angle/2.) if angle<np.pi/2.-1e-8 else 2.


def prepare(s):
    """One immutable base lookup plus small, sequential accepted replacements."""
    surface=s.material_surface
    radius=float(surface.get('radius_km', 6371.))
    material=_context(surface['vertices'], surface['faces'], radius)
    material['active']=np.ones(len(surface['faces']), bool)
    owner=native_boundary_geometry.prepare_owner_sampling(s.native_mesh, s.plate, s.support,
                                                           locator=s.native_locator)
    return dict(radius_km=radius, material=material, updates=[], owner=owner,
        control=dict(vertices=np.asarray(s.native_mesh['vertices']), faces=np.asarray(s.native_mesh['faces']),
                     locator=_lookup_locator(s.native_locator),_candidate_boxes={}), owner_polygons={})


def _face_caps(context):
    """The per-face form of _cap, built once per context and kept with it."""
    if 'face_cap_center' not in context:
        triangles=context['triangles']
        total=triangles.sum(axis=1)
        center=clipping._unit(total)
        angle=np.arccos(np.clip(np.einsum('fvi,fi->fv',triangles,center),-1.,1.)).max(axis=1)
        chord=np.where((angle<np.pi/2.-1e-8)&(np.linalg.norm(total,axis=1)>=.5),
                       2.*np.sin(angle/2.), 2.)
        context['face_cap_center'],context['face_cap_chord']=center,chord
    return context['face_cap_center'],context['face_cap_chord']


def _remaining(polygon, contexts):
    """Subtract each actual material union; repeated buried sheets count once."""
    parts=[polygon]
    center,chord_radius=_cap(polygon)
    for context in contexts:
        if np.linalg.norm(center-context['cap_center'])>chord_radius+context['cap_chord_radius']+1e-11:
            continue
        candidates=_candidates(polygon, context)
        if 'active' in context:
            candidates=candidates[context['active'][candidates]]
        if 'exclude' in context:
            candidates=candidates[~np.isin(candidates,context['exclude'])]
        if len(candidates):
            # The candidate bins are a box around the whole polygon. A face
            # whose own cap is clear of the polygon's cap reaches neither the
            # polygon nor any piece of it, so drop those together rather than
            # one part at a time.
            face_center,face_chord=_face_caps(context)
            offset=face_center[candidates]-center
            reach=chord_radius+face_chord[candidates]+1e-11
            candidates=candidates[np.einsum('ij,ij->i',offset,offset)<=reach*reach]
        for face in candidates:
            # The candidate box is selected once for the whole polygon, so most
            # faces cannot reach most of its current pieces. A convex minor-arc
            # piece lying wholly outside one of a face's own edge planes cannot
            # meet that face, and needs no partition. The pieces that do survive
            # that test share one preparation of the blocker.
            planes=context['intersection_planes'][face].T
            edges=None; following=[]
            for part in parts:
                if (part@planes<0.).all(axis=0).any():
                    following.append(part); continue
                if edges is None:
                    triangle=context['triangles'][face]
                    key=triangle.tobytes(); cache=context.setdefault('_partition_edges',{})
                    previous=cache.get(int(face))
                    if previous is None or previous[0]!=key:
                        # Edge endpoints and their filtered exact predicates
                        # belong to these bytes, not to a moving source mesh.
                        previous=(key,convex_partition.prepare(triangle.copy()))
                        cache[int(face)]=previous
                    edges=previous[1]
                intersection,outside=convex_partition.partition(part,edges)
                if clipping._area(intersection,context)<=clipping.AREA_TOLERANCE_KM2:
                    following.append(part); continue
                following.extend(piece for piece in outside
                                 if clipping._area(piece,context)>clipping.AREA_TOLERANCE_KM2)
            parts=following
            if not parts: return []
    return parts


def _owner_polygon(context, cell, owner):
    key=(int(cell), int(owner))
    cache=context['owner_polygons']
    if key in cache: return cache[key]
    control=context['control']; prepared=context['owner']
    vertices=control['faces'][cell]; triangle=control['vertices'][vertices]
    scores=prepared['scores'][:, vertices]; slots=prepared['owner_slots']
    at=np.flatnonzero(slots == owner)
    polygon=triangle.copy()
    if not len(at): polygon=np.empty((0,3))
    elif np.max(scores)<=0.:
        if prepared['fallback_owner'][cell]!=owner: polygon=np.empty((0,3))
    else:
        row=int(at[0])
        for other, slot in enumerate(slots):
            if other == row: continue
            difference=scores[row]-scores[other]
            # Whole-face algebraic ties use the same stable owner order as
            # source-point sampling. Other boundaries are finite P1 contours.
            if np.max(np.abs(difference))<=1e-13:
                if slot<owner: polygon=np.empty((0,3)); break
                continue
            if np.min(difference)>=0.: continue
            if np.max(difference)<0.: polygon=np.empty((0,3)); break
            normal=np.linalg.solve(triangle, difference)
            normal/=max(float(np.linalg.norm(normal)),1e-30)
            polygon=clipping._halfspace(polygon, normal)
            if len(polygon)<3: break
    cache[key]=polygon
    return polygon


def _owner_area(polygons, context, owner):
    result=[]
    # The clipping half-spaces of a cached owner domain are as fixed as the
    # domain itself, and every shell piece crossing that cell asks for them.
    planes=context.setdefault('owner_planes', {})
    for polygon in polygons:
        for cell in _candidates(polygon, context['control']):
            key=(int(cell), int(owner))
            domain=_owner_polygon(context, int(cell), owner)
            if len(domain)<3: continue
            if key not in planes: planes[key]=clipping._polygon_planes(domain)
            intersection=clipping._intersect_planes(polygon, planes[key])
            result.append(clipping._area(intersection, context))
    return math.fsum(result)


def inspect(context, vertices, faces, owner, *, old_vertices=None, old_faces=None, old_indices=None,
            _old_cache=None):
    """Measure candidate-minus-old area and its independent material/owner losses."""
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
            old=_context(old_vertices,old_faces,context['radius_km'])
        else:
            # The factory may mutate its inputs. Reuse only byte-identical old
            # geometry, with the same radius; never freeze a changed footprint.
            v=np.asarray(old_vertices,float); f=np.asarray(old_faces,np.int64)
            key=(v.shape,v.tobytes(),f.shape,f.tobytes(),context['radius_km'])
            if _old_cache.get('key')!=key:
                _old_cache.clear()
                _old_cache.update(key=key,context=_context(v,f,context['radius_km']))
            old=_old_cache['context']
    shell=[]
    for polygon in triangles:
        shell.extend(_remaining(polygon,[old]) if old is not None else [polygon])
    new_area=math.fsum(clipping._area(p,context) for p in shell)
    if old is None: old_area=0.
    elif _old_cache is None: old_area=math.fsum(clipping._area(p,context) for p in old['triangles'])
    else:
        if 'area' not in _old_cache:
            _old_cache['area']=math.fsum(clipping._area(p,context) for p in old['triangles'])
        old_area=_old_cache['area']
    # A growth map may distort old interior triangles but must retain its old
    # outer footprint. Signed area gain alone could conceal lost old terrain.
    lost=max(0.,old_area+new_area-candidate_area)
    water=[]
    base=context['material']
    if old_indices is not None:
        # Prime the caps on the shared lookup before copying it, so a growth
        # shell inherits them instead of rebuilding them for every candidate.
        _face_caps(base)
        # Indices are in the current surface. Same-batch births are represented
        # by updates below, so only original-context indices belong in base.
        indices=np.asarray(old_indices,int)
        base=dict(base,exclude=indices[indices<len(base['triangles'])])
    material_contexts=[base]
    for row in context['updates']:
        if old_indices is not None and np.all(np.isin(row['indices'],old_indices)): continue
        material_contexts.append(row['context'])
    for polygon in shell: water.extend(_remaining(polygon,material_contexts))
    water_area=math.fsum(clipping._area(p,context) for p in water)
    own_area=_owner_area(water,context,owner)
    # An area tolerance alone could admit a very narrow foreign sliver. Use
    # the existing exact source-point owner convention at every shell-piece
    # vertex as an additional closed-footprint guard, never as a replacement
    # for integration through all crossed control domains.
    water_vertices=np.concatenate(water) if water else np.empty((0,3))
    foreign_vertices=int(np.count_nonzero(native_boundary_geometry.sample_owners(
        water_vertices,context['owner'])!=owner)) if len(water_vertices) else 0
    budget_tolerance=tolerance(max(new_area,candidate_area,old_area))
    if water_area>new_area+budget_tolerance or own_area>water_area+budget_tolerance:
        raise ValueError('Arc admission clipping increased available source footprint area.')
    material_loss=max(0.,new_area-water_area)
    owner_loss=max(0.,water_area-own_area)
    # Ledger subtraction has a larger absolute cancellation allowance than
    # admission. Do not spend that allowance as new overlap on every step.
    admission_tolerance=max(1e-9,new_area*1e-12)
    admissible=bool(lost<=budget_tolerance and material_loss+owner_loss<=admission_tolerance
                    and foreign_vertices==0)
    exact_obstruction=None
    if admissible:
        import arc_material_exclusion
        exact_obstruction=arc_material_exclusion.obstruction(triangles,material_contexts,old)
        admissible=exact_obstruction is None
    return dict(candidate_area_km2=candidate_area,old_area_km2=old_area,new_footprint_area_km2=new_area,
        lost_old_footprint_area_km2=lost,material_obstruction_km2=material_loss,
        foreign_owner_obstruction_km2=owner_loss,foreign_owner_vertices=foreign_vertices,
        eligible_new_footprint_area_km2=own_area,
        area_tolerance_km2=budget_tolerance,admission_tolerance_km2=admission_tolerance,
        material_exclusion_version=1,exact_material_obstruction=exact_obstruction,admissible=admissible)


def admit(context, factory, requested_area_km2, owner, *, minimum_area_km2=.001,
          old_vertices=None, old_faces=None, old_indices=None):
    """Return a verified amount and its original mesh plan, never a clipped mesh.

    Coarse backtracking finds a feasible lower amount. Subsequent bisection
    improves utilization but claims no global optimum for deformed footprints.
    Every returned positive amount has itself passed the exact geometry test.
    """
    requested=float(requested_area_km2)
    if not np.isfinite(requested) or requested<0.: raise ValueError('Arc source area must be finite and nonnegative.')
    count=0; first=None; best=None; last=None; old_cache={}
    def evaluate(amount):
        nonlocal count, first, last
        count+=1; plan=factory(amount)
        if plan is None:
            value=dict(admissible=False, reason='unrepresentable_shared_mesh')
        else:
            # initialize_surface performs precisely this one division on birth.
            # Inspect the geometry that append_surface will store, without
            # modifying the original plan or normalizing growth a second time.
            vertices=np.asarray(plan['vertices'],float)
            if old_vertices is None:vertices=vertices/np.linalg.norm(vertices,axis=1)[:,None]
            value=inspect(context,vertices,plan['faces'],owner,
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
        diagnostics=dict(version=VERSION,requested_area_km2=requested,accepted_area_km2=accepted,
            pending_area_km2=requested-accepted,geometry_evaluations=count,
            requested_footprint=first,accepted_footprint=None if best is None else best[2],
            last_examined_footprint=last,relocation_km=0.))


def commit(context, surface, indices):
    """Update occupancy after each accepted mutation without rebuilding the world."""
    indices=np.asarray(indices,int)
    base=context['material']; old=indices[indices<len(base['active'])]
    base['active'][old]=False
    retained=[]
    for row in context['updates']:
        if np.intersect1d(indices,row['indices']).size:
            if not np.all(np.isin(row['indices'],indices)):
                raise ValueError('Arc occupancy replacement only partly covers an earlier accepted component.')
        else: retained.append(row)
    faces=np.asarray(surface['faces'])[indices]
    vertex_ids,inverse=np.unique(faces,return_inverse=True)
    added=_context(np.asarray(surface['vertices'])[vertex_ids],inverse.reshape(-1,3),context['radius_km'])
    retained.append(dict(indices=indices.copy(),context=added))
    context['updates']=retained


def validate_deposition_policy(frame):
    """Require explicit fixed-footprint source-volume semantics in saved frames."""
    tag=frame.get('arc_deposition_version',0)
    if isinstance(tag,(bool,np.bool_)) or not isinstance(tag,Integral) or tag not in (0,1):
        raise ValueError('Unsupported saved juvenile column deposition version.')
    policy=frame.get('arc_deposition_policy')
    if not tag:
        if policy is not None:
            raise ValueError('Juvenile column deposition policy requires its version.')
        arc=frame.get('arc_material_diagnostics',{})
        if isinstance(arc,dict):
            for key in ('deposited_source_area_km2','deposited_volume_km3','deposited_patches'):
                if key in arc:
                    value=arc[key]
                    if (isinstance(value,(bool,np.bool_)) or not isinstance(value,Real)
                            or not np.isfinite(value) or value!=0.):
                        raise ValueError('Juvenile deposition totals require their saved version.')
        stats=frame.get('stats',{})
        if isinstance(stats,dict):
            for key in ('arc_deposited_source_area_km2','arc_deposited_volume_km3'):
                if key in stats:
                    value=stats[key]
                    if (isinstance(value,(bool,np.bool_)) or not isinstance(value,Real)
                            or not np.isfinite(value) or value!=0.):
                        raise ValueError('Cumulative juvenile deposition requires its saved version.')
        return 0
    expected=dict(version=1,source_column_km=25.,maximum_column_km=75.,
        minimum_source_area_km2=.001,routing='source_containing_native_face',
        fallback='failed_lateral_growth')
    if (not isinstance(policy,dict) or set(policy)!=set(expected)
            or any(isinstance(policy.get(key),(bool,np.bool_)) or policy.get(key)!=value
                   for key,value in expected.items())
            or not isinstance(policy.get('version'),Integral)):
        raise ValueError('Invalid fixed-footprint juvenile deposition policy.')
    if any(isinstance(frame.get(key),(bool,np.bool_)) or not isinstance(frame.get(key),Integral)
           or frame[key]!=1 for key in ('arc_material_version','arc_emplacement_version',
                                       'arc_birth_profile_version','arc_source_cohort_version')):
        raise ValueError('Juvenile deposition requires physical profiles and source provenance.')
    return int(tag)


def _validate_column_deposition(row, placed):
    """Check a local volume receipt without reinterpreting source area as land."""
    receipt=row.get('column_deposition')
    if (not isinstance(receipt,dict) or type(receipt.get('version')) is not int
            or receipt['version']!=1 or receipt.get('geometry_unchanged') is not True):
        raise ValueError('Juvenile deposition requires an unchanged-geometry receipt.')
    def scalar(key):
        value=receipt.get(key)
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value) or value<0:
            raise ValueError('Invalid juvenile deposition scalar: '+key)
        return float(value)
    def integers(key, count=None, *, positive=False):
        value=receipt.get(key)
        if (not isinstance(value,list) or (count is not None and len(value)!=count)
                or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Integral)
                       or x<(1 if positive else 0) for x in value)):
            raise ValueError('Invalid juvenile deposition identity: '+key)
        return value
    count=len(row['source_indices'])
    source_faces=integers('source_faces',count)
    source_ids=integers('source_face_ids',count,positive=True)
    receiving=integers('receiving_faces')
    receiving_ids=integers('receiving_face_ids',len(receiving),positive=True)
    if len(set(receiving))!=len(receiving) or len(set(receiving_ids))!=len(receiving_ids):
        raise ValueError('A juvenile deposition receipt cannot repeat a receiving face.')
    forward={};reverse={}
    for face,identity in zip(source_faces,source_ids):
        if forward.setdefault(face,identity)!=identity or reverse.setdefault(identity,face)!=face:
            raise ValueError('Juvenile source face indices and persistent identities disagree.')
    if any(forward.get(face)!=identity for face,identity in zip(receiving,receiving_ids)):
        raise ValueError('Juvenile deposition left its actual source-containing faces.')
    arrays={}
    for key in ('face_area_km2','face_reference_area_km2','face_volume_added_km3',
                'face_thickness_before_km','face_thickness_after_km'):
        raw=receipt.get(key)
        if not isinstance(raw,list) or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) for x in raw):
            raise ValueError('Invalid juvenile deposition face array: '+key)
        value=np.asarray(raw,float)
        if value.shape!=(len(receiving),) or not np.isfinite(value).all() or np.any(value<=0):
            raise ValueError('Juvenile deposition face amounts must be finite and positive.')
        arrays[key]=value
    area=arrays['face_area_km2'];volume=arrays['face_volume_added_km3']
    before=arrays['face_thickness_before_km'];after=arrays['face_thickness_after_km']
    if (np.any(after<before) or np.any(after>75.) or np.any(volume<.025-1e-7)
            or not np.allclose(area*(after-before),volume,rtol=2e-11,atol=1e-7)):
        raise ValueError('Juvenile deposited columns do not conserve their admitted source volume.')
    spent=np.asarray(row.get('source_placed_area_km2'),float)
    if spent.shape!=(count,) or not np.isfinite(spent).all() or np.any(spent<0):
        raise ValueError('Juvenile deposition needs aligned per-origin spending.')
    by_face={}
    for face,amount in zip(source_faces,spent):
        by_face[face]=by_face.get(face,0.)+25.*float(amount)
    actual_by_face=dict(zip(receiving,volume))
    if any(not np.isclose(amount,actual_by_face.get(face,0.),rtol=2e-11,atol=1e-7)
           for face,amount in by_face.items()):
        raise ValueError('Juvenile deposition borrowed volume from a different source face.')
    actual=scalar('actual_added_volume_km3');source=scalar('source_volume_km3')
    if (scalar('reference_area_change_km2')!=0. or scalar('footprint_area_change_km2')!=0.
            or not np.isclose(actual,float(volume.sum()),rtol=2e-11,atol=1e-7)
            or not np.isclose(source,25.*placed,rtol=2e-11,atol=1e-7)
            or not np.isclose(actual,source,rtol=2e-11,atol=1e-7)
            or not np.isclose(float(spent.sum()),placed,rtol=1e-11,atol=1e-7)
            or (placed>0.)!=bool(receiving)):
        raise ValueError('Juvenile deposition changed its footprint or failed its source-volume ledger.')
    if placed==0. and (actual!=0. or source!=0.):
        raise ValueError('Rejected juvenile deposition cannot consume source volume.')


def validate_frame(frame):
    """Validate marked saved policy ledgers without rerunning footprint geometry."""
    import arc_birth_footprint
    footprint_version=arc_birth_footprint.validate_frame(frame)
    deposition_version=validate_deposition_policy(frame)
    version=frame.get('arc_emplacement_version',0)
    if isinstance(version,(bool,np.bool_)) or not isinstance(version,Integral) or version not in (0,VERSION):
        raise ValueError('Unsupported saved juvenile emplacement version.')
    arc=frame.get('arc_material_diagnostics',{})
    if not isinstance(arc,dict): raise ValueError('Juvenile source diagnostics must be a mapping.')
    report=arc.get('emplacement_geometry')
    if not version:
        if report is not None or 'arc_pending_source' in frame:
            raise ValueError('Juvenile emplacement policy needs its saved version.')
        return
    if frame.get('arc_material_version')!=1:
        raise ValueError('Finite juvenile emplacement requires material arc provenance.')
    inventory=frame.get('arc_pending_source')
    if (not isinstance(inventory,dict) or type(inventory.get('version')) is not int
            or inventory['version']!=VERSION or inventory.get('source_column_km')!=25.):
        raise ValueError('A marked juvenile epoch requires its pending source inventory.')
    inventory_owners=np.asarray(inventory.get('owner'))
    if inventory_owners.shape==(0,): inventory_owners=inventory_owners.astype(np.int64)
    validate_pending(dict(xyz=np.asarray(inventory.get('geometry_xyz')).reshape(-1,3),
        owner=inventory_owners,area=np.asarray(inventory.get('area_km2')),
        emplacement_version=inventory['version'],source_column_km=inventory.get('source_column_km')))
    if report is None and frame.get('time_myr')==0. and not arc:
        if inventory.get('total_area_km2')!=0. or inventory.get('total_volume_km3')!=0. or inventory.get('area_km2'):
            raise ValueError('An initialized juvenile world cannot contain an unexplained pending reservoir.')
        return
    if not isinstance(report,dict) or report.get('version')!=VERSION or isinstance(report.get('version'),bool):
        raise ValueError('A marked juvenile epoch requires its finite-footprint ledger.')
    def scalar(row,key,*,integer=False):
        value=row.get(key)
        if (isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral if integer else Real)
                or not np.isfinite(value) or value<0):
            raise ValueError('Invalid juvenile emplacement scalar: '+key)
        return float(value)
    def near(a,b): return abs(a-b)<=tolerance(max(abs(a),abs(b)))
    pending_inventory_area=scalar(inventory,'total_area_km2')
    if (not near(pending_inventory_area,float(np.sum(inventory['area_km2'])))
            or not near(scalar(inventory,'total_volume_km3'),25.*pending_inventory_area)
            or not near(pending_inventory_area,scalar(arc,'pending_area_km2'))):
        raise ValueError('Saved pending juvenile source inventory does not reconcile.')
    requested=scalar(report,'requested_area_km2'); accepted=scalar(report,'accepted_area_km2')
    pending=scalar(report,'pending_area_km2'); evaluations=scalar(report,'geometry_evaluations',integer=True)
    if not near(pending_inventory_area,pending+scalar(report,'unexamined_pending_area_km2')):
        raise ValueError('Examined and unexamined juvenile pending stages do not reconcile.')
    if report.get('sequential_occupancy') is not True or not near(requested,accepted+pending):
        raise ValueError('Juvenile admission/pending ledger or sequential occupancy is invalid.')
    rows=report.get('sources')
    if not isinstance(rows,list): raise ValueError('Finite juvenile source rows must be a list.')
    totals=np.zeros(4)
    for row in rows:
        if not isinstance(row,dict) or row.get('version')!=VERSION or isinstance(row.get('version'),bool):
            raise ValueError('Invalid finite juvenile source record.')
        amount=scalar(row,'requested_area_km2'); placed=scalar(row,'accepted_area_km2')
        held=scalar(row,'pending_area_km2'); count=scalar(row,'geometry_evaluations',integer=True)
        scalar(row,'owner',integer=True); scalar(row,'arc_id',integer=True)
        modes=('birth','growth','deposition') if deposition_version else ('birth','growth')
        if row.get('mode') not in modes or not near(amount,placed+held):
            raise ValueError('Invalid juvenile source mode or admission arithmetic.')
        if scalar(row,'relocation_km')!=0.:
            raise ValueError('Finite juvenile emplacement cannot relocate its source.')
        indices=row.get('source_indices'); points=np.asarray(row.get('geometry_xyz'))
        if (not isinstance(indices,list) or not indices or any(isinstance(i,bool) or not isinstance(i,Integral) or i<0 for i in indices)
                or len(set(indices))!=len(indices) or points.shape!=(len(indices),3) or points.dtype.kind not in 'fiu'
                or not np.isfinite(points).all() or not np.allclose(np.linalg.norm(points,axis=1),1.,atol=2e-10,rtol=0.)):
            raise ValueError('Juvenile source positions and local batch indices must align.')
        if row['mode']=='deposition':
            _validate_column_deposition(row,placed)
        elif 'column_deposition' in row:
            raise ValueError('A column deposition receipt requires its explicit source mode.')
        measured=row.get('accepted_footprint')
        if placed>0.:
            if not isinstance(measured,dict) or measured.get('admissible') is not True:
                raise ValueError('Placed juvenile source lacks a verified finite footprint.')
            if 'material_exclusion_version' in measured:
                if (type(measured['material_exclusion_version']) is not int or measured['material_exclusion_version']!=1
                        or measured.get('exact_material_obstruction') is not None):
                    raise ValueError('Placed juvenile material failed exact new-footprint exclusion.')
            new=scalar(measured,'new_footprint_area_km2'); lost=scalar(measured,'lost_old_footprint_area_km2')
            candidate=scalar(measured,'candidate_area_km2'); old=scalar(measured,'old_area_km2')
            obstruction=scalar(measured,'material_obstruction_km2')+scalar(measured,'foreign_owner_obstruction_km2')
            eligible=scalar(measured,'eligible_new_footprint_area_km2')
            allowance=scalar(measured,'admission_tolerance_km2'); area_allowance=scalar(measured,'area_tolerance_km2')
            if (not near(allowance,max(1e-9,new*1e-12)) or allowance>max(1e-9,new*1e-12)*(1.+1e-12)
                    or area_allowance>tolerance(max(new,candidate,old))*(1.+1e-12)
                    or obstruction>allowance+1e-12 or lost>area_allowance+1e-12
                    or scalar(measured,'foreign_owner_vertices',integer=True)!=0
                    or not near(new,candidate-old+lost) or not near(new,eligible+obstruction)):
                raise ValueError('Saved juvenile footprint violates its area or obstruction contract.')
            if footprint_version and row['mode']=='birth':
                profile=row.get('profile_capacity',{})
                if (old!=0. or not near(candidate,scalar(profile,'actual_area_km2'))
                        or not near(25.*placed,scalar(profile,'actual_volume_km3'))
                        or not near(placed,scalar(profile,'source_area_km2'))
                        or not near(25.*placed,scalar(profile,'source_volume_km3'))
                        or scalar(profile,'minimum_column_km')<8.
                        or scalar(profile,'maximum_column_km')>75.
                        or candidate>25.*placed/8.+area_allowance
                        or candidate<placed-area_allowance):
                    raise ValueError('Juvenile birth footprint exceeds its fixed source volume or column bounds.')
            elif row['mode']=='deposition':
                if (candidate!=old or old<=0. or new!=0. or lost!=0. or obstruction!=0. or eligible!=0.
                        or sum(row['column_deposition']['face_area_km2'])>old+area_allowance):
                    raise ValueError('Juvenile column deposition must retain exactly the existing footprint.')
            elif not near(candidate-old,placed):
                raise ValueError('Legacy juvenile footprint must match its source equivalent area.')
        totals+=amount,placed,held,count
    if not np.allclose(totals,[requested,accepted,pending,evaluations],rtol=1e-11,atol=1e-7):
        raise ValueError('Finite juvenile aggregate does not equal its source records.')
    if not near(accepted,scalar(arc,'added_area_km2')) or not near(25.*accepted,scalar(arc,'added_volume_km3')):
        raise ValueError('Finite juvenile source ledger disagrees with actual emplacement.')
    deposition_rows=[row for row in rows if row['mode']=='deposition']
    summary=('deposited_source_area_km2','deposited_volume_km3','deposited_patches')
    # A future-only activation can retain the preceding step's legacy report.
    # Once deposition is attempted, its summary must reconcile independently.
    if deposition_rows or any(key in arc for key in summary):
        deposited=scalar(arc,summary[0]);deposited_volume=scalar(arc,summary[1])
        deposited_patches=scalar(arc,summary[2],integer=True)
        expected=sum(row['accepted_area_km2'] for row in deposition_rows)
        expected_patches=sum(row['accepted_area_km2']>0. for row in deposition_rows)
        if (not near(deposited,expected) or not near(deposited_volume,25.*expected)
                or deposited_patches!=expected_patches
                or (not deposition_version and (deposited!=0. or deposited_volume!=0. or deposited_patches!=0.))):
            raise ValueError('Juvenile deposition aggregate disagrees with its source receipts or version.')
        reference_gain=scalar(arc,'physical_mass_added_km2')
        residual=arc.get('physical_mass_residual_km2')
        expected_reference=accepted-deposited
        if (isinstance(residual,(bool,np.bool_)) or not isinstance(residual,Real)
                or not np.isfinite(residual) or not near(reference_gain,expected_reference)
                or not near(residual,reference_gain-expected_reference) or not near(residual,0.)):
            raise ValueError('Juvenile reference-area gain must exclude fixed-footprint deposition.')
