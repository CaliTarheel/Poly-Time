"""Exact ocean parts of finite native boundaries and paired swept strips.

The control contact graph remains unique and unchanged. Material triangles,
not their neighboring cell labels, determine ocean availability inside it.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np
import mesh_geometry
import mesh_coverage
import convex_partition
from ridge_geometry import rotate, finite_half_stage, normal_motion_threshold

RADIUS_KM=6371.
VERSION=1
AREA_TOLERANCE_KM2=1e-8
LOCAL_PAIRING_VERSION=1
LOCAL_PAIRING_AREA_FIELDS=('local_unpaired_area_km2','duplicate_paired_area_km2',
    'candidate_paired_area_km2','ambiguous_overlap_excluded_area_km2',
    'maximum_water_capacity_residual_km2')
LOCAL_PAIRING_MARKERS=set(LOCAL_PAIRING_AREA_FIELDS)|{
    'local_pairing_version','local_paired_patches','pairing','conflict_policy'}


def has_local_pairing_metadata(frame):
    value=frame.get('spreading_diagnostics')
    return isinstance(value,dict) and bool(LOCAL_PAIRING_MARKERS.intersection(value))


PRODUCTION_VERSION=1


def production_version(s):
    """0 keeps the replacement deposit; 1 lets a ridge fill the gap it opened."""
    value=getattr(s,'ocean_production_version',0)
    if isinstance(value,(bool,np.bool_)) or int(value)!=value or value not in (0,PRODUCTION_VERSION):
        raise ValueError('Unsupported ocean production policy version.')
    return int(value)


def _unit(value):
    value=np.asarray(value,float)
    return value/np.maximum(np.linalg.norm(value,axis=-1,keepdims=True),1e-30)


def _area(polygon,context):
    return mesh_coverage._polygon_area(polygon,context['radius_km'])


def prepare_material(surface):
    """Read-only acceleration for the union of outward minor material faces."""
    vertices=np.asarray(surface['vertices'],float)
    faces=np.asarray(surface['faces'])
    locator=mesh_geometry.build_locator(vertices,faces)
    triangles=vertices[faces]
    return dict(vertices=vertices,faces=faces,triangles=triangles,
        # Algebraically identical to cross(a,b), but preserves significant
        # bits when a short coast edge makes the two vertices nearly equal.
        planes=_unit(np.cross(triangles,np.roll(triangles,-1,axis=1)-triangles)),
        # Retain compatible conservative candidate-rejection planes separately
        # from stable coast containment and the disjoint partition predicates.
        intersection_planes=mesh_coverage._unit(np.cross(triangles,np.roll(triangles,-1,axis=1))),
        locator=locator,radius_km=float(surface.get('radius_km',RADIUS_KM)))


def _candidates(polygon,context):
    if not len(context['faces']): return np.empty(0,np.int32)
    center=_unit(np.sum(polygon,axis=0))
    angle=np.arccos(np.clip(polygon@center,-1.,1.)).max(initial=0.)
    if angle>=np.pi/2-1e-8:
        return np.arange(len(context['faces']))
    return mesh_coverage._candidates(polygon,center,2.*np.sin(angle/2.),context['locator'])


def _halfspace(polygon,normal):
    if not len(polygon): return polygon
    d=polygon@normal; inside=d>=-2e-14
    if np.all(inside): return polygon
    if not np.any(inside): return np.empty((0,3))
    result=[]
    for i in range(len(polygon)):
        j=(i-1)%len(polygon)
        if inside[i]!=inside[j]:
            fraction=d[j]/(d[j]-d[i])
            # A tolerance-classified crossing must stay on its finite source edge.
            fraction=float(np.clip(fraction,0.,1.))
            result.append(_unit(polygon[j]*(1.-fraction)+polygon[i]*fraction))
        if inside[i]: result.append(polygon[i])
    result=np.asarray(result).reshape(-1,3)
    if len(result)>1:
        result=result[np.linalg.norm(result-np.roll(result,1,axis=0),axis=1)>2e-13]
    return result


def _separated_parts(parts,planes):
    """Sufficient convex rejection only; ambiguous boundaries are never culled."""
    separated=np.zeros(len(parts),bool)
    # Bound transient work by small groups of existing convex polygons. This
    # does not change face order, part order, or the scalar clipping policy.
    for first in range(0,len(parts),256):
        batch=parts[first:first+256]
        sizes=np.fromiter((len(part) for part in batch),int,count=len(batch))
        points=np.concatenate(batch)
        starts=np.r_[0,np.cumsum(sizes[:-1])]
        maxima=np.maximum.reduceat(points@planes.T,starts,axis=0)
        separated[first:first+len(batch)]=np.any(maxima < -1e-12,axis=1)
    return separated


def uncovered_polygons(polygon_xyz,context):
    """Disjoint convex complement of the true material union inside a polygon.

    Sequential triangle subtraction handles duplicate and overlapping sheets
    without counting their covered area twice. Plane boundaries have zero area.
    """
    polygon=np.asarray(polygon_xyz,float)
    parts=[polygon] if len(polygon)>=3 else []
    candidates=_candidates(polygon,context)
    intersection_planes=context.get('intersection_planes')
    if intersection_planes is None:
        triangles=context['triangles']
        intersection_planes=mesh_coverage._unit(np.cross(triangles,np.roll(triangles,-1,axis=1)))
        context['intersection_planes']=intersection_planes
    for face in candidates:
        remaining=[]
        planes=intersection_planes[face]
        # A convex piece wholly behind any clipping plane cannot meet the
        # triangle. Batch only this sufficient rejection test. The margin is
        # deliberately wider than the unchanged scalar clipping tolerance;
        # every near-boundary case still takes the original scalar path.
        separated=_separated_parts(parts,planes)
        partitions=context.setdefault('partition_edges',{})
        if int(face) not in partitions:partitions[int(face)]=convex_partition.prepare(context['triangles'][face])
        for part,disjoint in zip(parts,separated):
            if disjoint:
                remaining.append(part); continue
            # Most bounding-box candidates do not meet this particular piece.
            intersection,outside=convex_partition.partition(part,partitions[int(face)])
            if _area(intersection,context)<=AREA_TOLERANCE_KM2:
                remaining.append(part); continue
            remaining.extend(piece for piece in outside if _area(piece,context)>AREA_TOLERANCE_KM2)
        parts=remaining
        if not parts: break
    return parts


def integrate_polygon(mesh,polygon_xyz,context,allowed_cells=None):
    """Sparse exact ocean areas and capacities in touched native cells."""
    polygon=np.asarray(polygon_xyz,float)
    if len(polygon)<3:
        return dict(cells=np.empty(0,np.int32),area_km2=np.empty(0),ocean_capacity_km2=np.empty(0))
    # A transient per-step material context also reuses static control lookup
    # and water capacities. Never rebuild the whole control mesh per strip.
    if context.get('_control_mesh_ref') is not mesh:
        context['_control_mesh_ref']=mesh
        context['_control_lookup']=dict(faces=mesh['faces'],
            locator=mesh_geometry.build_locator(mesh['vertices'],mesh['faces']))
        context['_ocean_capacity']={}
    cells=_candidates(polygon,context['_control_lookup'])
    if allowed_cells is not None: cells=cells[np.asarray(allowed_cells,bool)[cells]]
    chosen=[]; areas=[]; capacities=[]
    for cell in cells:
        cell_polygon=mesh['vertices'][mesh['faces'][cell]]
        clipped=mesh_coverage.clip_triangle(polygon,cell_polygon)
        if _area(clipped,context)<=AREA_TOLERANCE_KM2: continue
        area=math.fsum(_area(part,context) for part in uncovered_polygons(clipped,context))
        if area<=AREA_TOLERANCE_KM2: continue
        cache=context['_ocean_capacity']
        if int(cell) not in cache:
            cache[int(cell)]=math.fsum(_area(part,context) for part in uncovered_polygons(cell_polygon,context))
        capacity=cache[int(cell)]
        if area>capacity+max(1e-5,capacity*1e-10):
            raise ValueError('A swept ocean polygon exceeded its actual water capacity.')
        chosen.append(int(cell)); areas.append(min(area,capacity)); capacities.append(capacity)
    return dict(cells=np.asarray(chosen,np.int32),area_km2=np.asarray(areas),ocean_capacity_km2=np.asarray(capacities))


def ocean_intervals(start,end,context):
    """Complement of exact material-containment intervals on a minor arc."""
    start,end=np.asarray(start,float),np.asarray(end,float)
    covered=[]
    for face in _candidates(np.array([start,end]),context):
        before=context['planes'][face]@start; after=context['planes'][face]@end
        # Use the same boundary tolerance in both containment and crossing.
        # Otherwise two roundoff signs on a genuine coast fabricate water.
        before[np.abs(before)<=2e-14]=0.
        after[np.abs(after)<=2e-14]=0.
        low,high=0.,1.
        for a,b in zip(before,after):
            if a < -2e-14 and b < -2e-14: high=-1.;break
            if a<0.<=b: low=max(low,float(-a/(b-a)))
            elif b<0.<=a: high=min(high,float(-a/(b-a)))
        if high-low>1e-12: covered.append((low,high))
    result=[]; cursor=0.
    for low,high in sorted(covered):
        if low>cursor+1e-12: result.append((cursor,low))
        cursor=max(cursor,high)
    if cursor<1.-1e-12: result.append((cursor,1.))
    return result


def _piece_arrays(rows,s):
    count=len(rows)
    parent=np.array([row[0] for row in rows],np.int32)
    start=np.array([row[1] for row in rows],float).reshape(count,3)
    end=np.array([row[2] for row in rows],float).reshape(count,3)
    mid=_unit(start+end)
    normal=np.array([row[3] for row in rows],float).reshape(count,3)
    normal=_unit(normal-mid*np.sum(normal*mid,axis=1)[:,None])
    length=RADIUS_KM*np.arctan2(np.linalg.norm(np.cross(start,end),axis=1),np.sum(start*end,axis=1))
    return dict(parent=parent,start=start,end=end,mid=mid,normal=normal,length=length,
        owner_a=s.bp[parent].copy(),owner_b=s.bq[parent].copy())


def prepare(s):
    """Current exact material/ocean intervals on the unmodified contact graph."""
    contour=s.native_boundary_geometry; surface=s.material_surface
    digest=hashlib.sha256()
    for array in (surface['vertices'],surface['faces'],contour['segments_start'],
                  contour['segments_end'],contour['segment_normals'],contour['contact_index'],s.bp,s.bq):
        digest.update(np.asarray(array).tobytes())
    signature=digest.hexdigest()
    if signature!=getattr(s,'_native_spreading_geometry_signature',None):
        context=prepare_material(surface)
        ocean=[]; continental=[]
        for start,end,normal,parent in zip(contour['segments_start'],contour['segments_end'],
                                           contour['segment_normals'],contour['contact_index']):
            wet=ocean_intervals(start,end,context)
            dry=[]; cursor=0.
            for low,high in wet:
                if low>cursor+1e-12: dry.append((cursor,low))
                cursor=high
            if cursor<1.-1e-12: dry.append((cursor,1.))
            for destination,intervals in ((ocean,wet),(continental,dry)):
                for low,high in intervals:
                    a=_unit((1-low)*start+low*end);b=_unit((1-high)*start+high*end)
                    if np.linalg.norm(a-b)>1e-12: destination.append((int(parent),a,b,normal))
        s.native_spreading_geometry=dict(version=VERSION,ocean=_piece_arrays(ocean,s),
            continental=_piece_arrays(continental,s))
        s._native_spreading_geometry_signature=signature
    result=s.native_spreading_geometry
    for name in ('ocean','continental'):
        row=result[name]
        velocity=np.cross(s.omega[row['owner_b']]-s.omega[row['owner_a']],row['mid'])*RADIUS_KM
        normal=np.sum(velocity*row['normal'],axis=1)
        shear=np.linalg.norm(velocity-normal[:,None]*row['normal'],axis=1)
        row['normal_speed']=normal
        row['divergent']=normal>normal_motion_threshold(shear,
            spreading_version=getattr(s,'normal_spreading_version',0))
        row['active']=s.active[row['owner_a']] & s.active[row['owner_b']]
        row['active']&=(s.plate[s.ba[row['parent']]]==row['owner_a']) & (s.plate[s.bb[row['parent']]]==row['owner_b'])
    result['parent_ocean_length_km']=np.bincount(result['ocean']['parent'],weights=result['ocean']['length'],minlength=len(s.ba))
    result['parent_continental_length_km']=np.bincount(result['continental']['parent'],weights=result['continental']['length'],minlength=len(s.ba))
    return result


def boundary_segments(s):
    """Review pieces retain exact geometry and original parent contact indices."""
    import boundary_labels
    geometry=prepare(s); rows=[]
    display=boundary_labels.codes(s)
    for kind in ('ocean','continental'):
        values=geometry[kind]
        for i,parent in enumerate(values['parent']):
            code=int(display[parent])
            if values['divergent'][i] and values['active'][i]: code=1 if kind=='ocean' else 5
            elif code in (1,5): code=3
            rows.append(dict(geometry_xyz=[values['start'][i].tolist(),values['end'][i].tolist()],
                normal=values['normal'][i].tolist(),code=code,kinematic_code=int(s.bcode[parent]),
                owner_a=int(values['owner_a'][i]),owner_b=int(values['owner_b'][i]),
                down=int(s.down[parent]),contact_index=int(parent),
                material_side_classification=kind,geometry_version=VERSION))
    return rows


def validate_frame(frame):
    """Reject malformed optional exact native spreading geometry metadata."""
    normal_motion_threshold(0., spreading_version=frame.get('normal_spreading_version',0))
    if frame.get('normal_spreading_version',0) and frame.get('native_spreading_version',0)!=VERSION:
        raise ValueError('Normal spreading policy requires native spreading geometry.')
    version=frame.get('native_spreading_version',0)
    if version==0:
        if has_local_pairing_metadata(frame) or any(isinstance(row,dict) and ('geometry_version' in row or 'material_side_classification' in row)
               for row in frame.get('boundary_segments',[])):
            raise ValueError('Exact native boundary metadata requires its spreading version.')
        return
    if not isinstance(version,(int,np.integer)) or isinstance(version,(bool,np.bool_)) or version!=VERSION:
        raise ValueError('Unsupported native spreading geometry version.')
    if has_local_pairing_metadata(frame):
        diagnostic=frame['spreading_diagnostics']
        policy=diagnostic.get('local_pairing_version')
        if not isinstance(policy,(int,np.integer)) or isinstance(policy,(bool,np.bool_)) or policy!=LOCAL_PAIRING_VERSION:
            raise ValueError('Local paired spreading diagnostics require their policy version.')
        patches=diagnostic.get('local_paired_patches')
        if not isinstance(patches,(int,np.integer)) or isinstance(patches,(bool,np.bool_)) or patches<0:
            raise ValueError('Local paired spreading patch count must be a nonnegative integer.')
        fields=LOCAL_PAIRING_AREA_FIELDS+('generated_area_km2','requested_area_km2',
            'excluded_material_area_km2','side_p_area_km2','side_q_area_km2')
        for name in fields:
            value=diagnostic.get(name)
            if (not isinstance(value,(int,float,np.integer,np.floating)) or isinstance(value,(bool,np.bool_))
                    or not np.isfinite(value) or value<0.):
                raise ValueError('Local paired spreading needs finite nonnegative '+name+'.')
        tolerance=max(1e-5,diagnostic['requested_area_km2']*1e-10)
        correction=diagnostic.get('capacity_roundoff_removed_area_km2',0.)
        if (isinstance(correction,(bool,np.bool_)) or not isinstance(correction,(int,float,np.integer,np.floating))
                or not np.isfinite(correction) or correction<0.):
            raise ValueError('Spreading capacity roundoff correction must be finite and nonnegative.')
        if (abs(diagnostic['candidate_paired_area_km2']-diagnostic['generated_area_km2']-
                diagnostic['ambiguous_overlap_excluded_area_km2']-correction)>tolerance
                or abs(diagnostic['side_p_area_km2']-diagnostic['side_q_area_km2'])>tolerance
                or abs(diagnostic['generated_area_km2']-diagnostic['side_p_area_km2']-
                    diagnostic['side_q_area_km2'])>tolerance
                or sum(diagnostic[name] for name in ('candidate_paired_area_km2','duplicate_paired_area_km2',
                    'local_unpaired_area_km2','excluded_material_area_km2'))>diagnostic['requested_area_km2']+tolerance
                or diagnostic['maximum_water_capacity_residual_km2']>1e-5):
            raise ValueError('Local paired spreading budget does not close.')
    rows=frame.get('boundary_segments')
    if not isinstance(rows,list): raise ValueError('Exact native spreading history needs boundary segments.')
    owners_a=np.asarray(frame.get('native_boundary_owner_a',[]))
    owners_b=np.asarray(frame.get('native_boundary_owner_b',[]))
    if (owners_a.ndim!=1 or owners_b.shape!=owners_a.shape
            or owners_a.dtype.kind not in 'iu' or owners_b.dtype.kind not in 'iu'
            or np.any(owners_a<0) or np.any(owners_b<0)):
        raise ValueError('Exact native segments need aligned parent owners.')
    totals={1:0.,5:0.}
    def integer(value): return isinstance(value,(int,np.integer)) and not isinstance(value,(bool,np.bool_))
    # Boundary label version 2 adds one display code, for convergence that no trench takes up.
    labels=(1,2,3,4,5,6) if frame.get('boundary_label_version',0)==2 else (1,2,3,4,5)
    for row in rows:
        if not isinstance(row,dict): raise ValueError('Native boundary segment must be an object.')
        kind=row.get('material_side_classification'); code=row.get('code');parent=row.get('contact_index')
        if (not integer(row.get('geometry_version')) or row.get('geometry_version')!=VERSION
                or kind not in ('ocean','continental') or not integer(code) or code not in labels
                or not integer(parent) or not 0<=parent<len(owners_a)
                or not integer(row.get('owner_a')) or not integer(row.get('owner_b'))
                or row['owner_a']!=int(owners_a[parent]) or row['owner_b']!=int(owners_b[parent])
                or (code==1 and kind!='ocean') or (code==5 and kind!='continental')):
            raise ValueError('Invalid exact native boundary classification or parent identity.')
        try:
            points=np.asarray(row['geometry_xyz'],float); normal=np.asarray(row['normal'],float)
        except (KeyError,TypeError,ValueError) as exc:
            raise ValueError('Invalid exact native boundary coordinates.') from exc
        if (points.shape!=(2,3) or normal.shape!=(3,) or not np.isfinite(points).all()
                or not np.isfinite(normal).all() or np.any(np.abs(np.linalg.norm(points,axis=1)-1.)>1e-6)
                or abs(np.linalg.norm(normal)-1.)>1e-6
                or abs(float(_unit(points.sum(axis=0))@normal))>1e-6):
            raise ValueError('Exact native boundaries need finite unit spherical geometry.')
        length=RADIUS_KM*math.atan2(float(np.linalg.norm(np.cross(*points))),float(points[0]@points[1]))
        if not 0.<length<math.pi*RADIUS_KM: raise ValueError('Exact native boundary must have finite positive minor length.')
        if code in totals: totals[code]+=length
    stats=frame.get('stats',{})
    for code,name in ((1,'ridge_length_km'),(5,'rift_length_km')):
        value=stats.get(name)
        if (isinstance(value,bool) or not isinstance(value,(int,float,np.number))
                or not np.isfinite(value) or not math.isclose(float(value),totals[code],rel_tol=1e-8,abs_tol=1e-5)):
            raise ValueError('Exact native boundary length statistic disagrees with saved pieces.')


def dry_control_cells(s,land):
    """Fully covered control cells; retain water history in every partial cell.

    The exact intersection sum is the union on nonoverlapping material. Every
    cell involving a face with a measured sheet overlap uses explicit union
    subtraction instead, so stacked sheets cannot hide a remaining water gap.
    """
    tolerance=np.maximum(AREA_TOLERANCE_KM2,s.cell_area*1e-11)
    dry=np.asarray(land,bool)&(s.land_mass>=s.cell_area-tolerance)
    coverage=s._material_coverage
    overlap=getattr(s,'_collision_overlap',None)
    involved=np.empty(0,int) if overlap is None else np.union1d(overlap['first'],overlap['second'])
    uncertain=np.zeros(s.n,bool)
    uncertain[coverage['control_index'][np.isin(coverage['material_index'],involved)]]=True
    uncertain|=s.land_mass>s.cell_area+tolerance
    candidates=np.flatnonzero(dry&uncertain)
    if len(candidates):
        context=prepare_material(s.material_surface)
        for cell in candidates:
            polygon=s.native_mesh['vertices'][s.native_mesh['faces'][cell]]
            water=math.fsum(_area(part,context) for part in uncovered_polygons(polygon,context))
            dry[cell]=water<=AREA_TOLERANCE_KM2
    return dry


def _strip_polygon(axis,shore):
    polygon=np.array([axis[0],shore[0],shore[1],axis[1]])
    if np.linalg.det(polygon[:3])<0: polygon=polygon[::-1]
    return polygon


def _polygon_planes(polygon):
    planes=_unit(np.cross(polygon,np.roll(polygon,-1,axis=0)-polygon))
    center=np.sum(polygon,axis=0)
    return planes*np.where(planes@center<0.,-1.,1.)[:,None]


def _intersect_planes(polygon,planes):
    for plane in planes:
        polygon=_halfspace(polygon,plane)
        if len(polygon)<3: break
    return polygon


def _intersect_polygon(polygon,clip):
    return _intersect_planes(polygon,_polygon_planes(clip))


def _subtract_polygons(polygon,blockers,context):
    """Exact convex decomposition after a union of convex footprints."""
    parts=[polygon]
    for blocker in blockers:
        remaining=[]
        edges=convex_partition.prepare(blocker)
        for part in parts:
            # One immutable operand, many current complement pieces. Reuse
            # these exact predicates for intersection and subtraction within this
            # blocker visit only; no geometry cache survives motion or a call.
            intersection,outside=convex_partition.partition(part,edges)
            if _area(intersection,context)<=AREA_TOLERANCE_KM2:
                remaining.append(part);continue
            remaining.extend(piece for piece in outside if _area(piece,context)>AREA_TOLERANCE_KM2)
        parts=remaining
        if not parts: break
    return parts


def _water_cell_polygons(mesh,polygon,context):
    """Actual water pieces with their control cells; no area-only pooling."""
    if context.get('_control_mesh_ref') is not mesh:
        context['_control_mesh_ref']=mesh
        context['_control_lookup']=dict(faces=mesh['faces'],
            locator=mesh_geometry.build_locator(mesh['vertices'],mesh['faces']))
        context['_ocean_capacity']={}
    result=[]
    for cell in _candidates(polygon,context['_control_lookup']):
        cell_polygon=mesh['vertices'][mesh['faces'][cell]]
        clipped=mesh_coverage.clip_triangle(polygon,cell_polygon)
        if _area(clipped,context)<=AREA_TOLERANCE_KM2: continue
        water=uncovered_polygons(clipped,context)
        if not water: continue
        if int(cell) not in context['_ocean_capacity']:
            context['_ocean_capacity'][int(cell)]=math.fsum(
                _area(part,context) for part in uncovered_polygons(cell_polygon,context))
        result.extend((int(cell),part) for part in water)
    return result


def _bounded_pair_areas(cells, paired, capacity, cell_area):
    """Reconcile geometric roundoff while retaining equal paired shore areas.

    Reject real overfill first. Reduce each pair by the tighter of its two
    receiving bounds; birth fractions and ledgers use the same accepted areas.
    """
    paired=np.asarray(paired,float)
    ends=np.asarray(cells).reshape(-1,2)
    limit=np.minimum(capacity,cell_area)
    deposited=np.bincount(ends.ravel(),weights=np.repeat(paired,2),minlength=len(limit))
    if (not np.isfinite(paired).all() or np.any(paired<0.)
            or not np.isfinite(limit).all() or np.any(limit<0.)
            or not np.isfinite(deposited).all()
            or np.any(deposited>limit+np.maximum(1e-5,limit*1e-10))):
        raise ValueError('Locally disjoint paired ocean footprints exceeded true water capacity.')
    over=deposited>limit
    scale=np.ones(len(limit))
    if np.any(over):
        counts=np.bincount(ends.ravel(),minlength=len(limit))
        # Bound re-summation of rounded products using a margin proportional
        # only to machine precision and the number of incident deposits.
        margin=np.finfo(float).eps*(counts[over]+4)
        scale[over]=np.maximum(0.,(limit[over]/deposited[over])*(1.-margin))
    factors=np.minimum(scale[ends[:,0]],scale[ends[:,1]])
    accepted=paired*factors
    deposited=np.bincount(ends.ravel(),weights=np.repeat(accepted,2),minlength=len(limit))
    if np.any(deposited>limit):
        raise ValueError('Paired spreading roundoff correction did not satisfy water capacity.')
    return accepted,factors,deposited


def advance(s,transported,dt,arrivals=None):
    """Admit exact, local paired water footprints without area-only pooling.

    H = Rmid Rp^-1 maps the p quadrilateral congruently onto its q partner:
    p shore -> axis and axis -> q shore. This is finite half-stage geometric
    correspondence, not a solved pairing of magma parcels with the same age.
    """
    pieces=prepare(s)['ocean']
    selected=np.flatnonzero(pieces['active'] & pieces['divergent'])
    birth=np.zeros(s.n)
    base=dict(geometry_version=VERSION,local_pairing_version=LOCAL_PAIRING_VERSION,
        active_segments=int(len(selected)),
        reconstructed_cells=0,generated_area_km2=0.,requested_area_km2=0.,
        excluded_material_area_km2=0.,local_unpaired_area_km2=0.,
        duplicate_paired_area_km2=0.,candidate_paired_area_km2=0.,
        ambiguous_overlap_excluded_area_km2=0.,local_paired_patches=0,
        maximum_water_capacity_residual_km2=0.,capacity_roundoff_removed_area_km2=0.,
        candidate_claims=0,admitted_claims=0,rejected_foreign_claims=0,
        rejected_buoyant_claims=0,rejected_foreign_owner_claims=0,
        rejected_incoming_foreign_claims=0,rejected_foreign_neighbor_claims=0,
        side_p_area_km2=0.,side_q_area_km2=0.,
        geometry='exact finite ocean intervals and spherical swept-strip/material-union subtraction',
        pairing='local congruent finite half-stage polygons; p shore maps to axis and axis to q shore',
        conflict_policy='union same-pair duplicate footprints; exclude contested physical footprints and their local partners',
        claim_count_basis='receiving cell and finite-strip pairs; foreign reasons may overlap',
        parent_contact_graph_unchanged=True,continental_material_unchanged=True)
    s.spreading_diagnostics=base
    s._spreading_pending=None
    if not len(selected): return birth
    surface=s.material_surface
    moved_vertices=rotate(surface['vertices'],s.omega[surface['vertex_owner']]*dt)
    context=prepare_material(dict(vertices=moved_vertices,faces=surface['faces'],radius_km=RADIUS_KM))
    groups={};records=[];raw_paired=[];individual_water=[];requested=[]
    for index in selected:
        oldp,oldq=int(pieces['owner_a'][index]),int(pieces['owner_b'][index])
        p,q=sorted((oldp,oldq));key=(p,q)
        if key not in groups:
            rotation=finite_half_stage(s.omega[p],s.omega[q],dt)
            rp=rotate(np.eye(3),s.omega[p]*dt).T
            rm=rotate(np.eye(3),rotation).T
            h=rm@rp.T
            owner=s.plate
            masks=dict(foreign_owner=(owner!=p)&(owner!=q),
                foreign_neighbor=np.any((owner[s.native_mesh['face_neighbors']]!=p)&(owner[s.native_mesh['face_neighbors']]!=q),axis=1),
                incoming_foreign=np.zeros(s.n,bool) if arrivals is None else arrivals.sum(axis=0)-arrivals[p]-arrivals[q]>=1e-6)
            groups[key]=dict(h=h,rotation=rotation,masks=masks,
                allowed=~np.logical_or.reduce(list(masks.values())),union_by_cell={})
        group=groups[key];h=group['h'];rotation=group['rotation']
        endpoints=np.array([pieces['start'][index],pieces['end'][index]])
        normal=pieces['normal'][index]*(1. if oldp==p else -1.)
        normal=_unit(rotate(normal,rotation))
        axis=rotate(endpoints,rotation)
        shore_p=rotate(endpoints,s.omega[p]*dt);shore_q=rotate(endpoints,s.omega[q]*dt)
        opening=(shore_q-shore_p)@normal
        if np.all(opening<=0.): continue
        if np.any(opening<0.):
            fraction=float(-opening[0]/(opening[1]-opening[0]))
            crossing=_unit(endpoints[0]*(1.-fraction)+endpoints[1]*fraction)
            endpoints[0 if opening[0]<0. else 1]=crossing
            axis=rotate(endpoints,rotation)
            shore_p=rotate(endpoints,s.omega[p]*dt);shore_q=rotate(endpoints,s.omega[q]*dt)
        waters=[]
        for shore in (shore_p,shore_q):
            polygon=_strip_polygon(axis,shore);area=_area(polygon,context)
            requested.append(area)
            water=_water_cell_polygons(s.native_mesh,polygon,context)
            base['excluded_material_area_km2']+=max(0.,area-math.fsum(_area(poly,context) for _,poly in water))
            cells=np.array(sorted({cell for cell,_ in water}),int)
            base['candidate_claims']+=len(cells)
            base['admitted_claims']+=int(np.count_nonzero(group['allowed'][cells]))
            base['rejected_foreign_claims']+=int(np.count_nonzero(~group['allowed'][cells]))
            for label,mask in group['masks'].items():
                base['rejected_'+label+'_claims']+=int(np.count_nonzero(mask[cells]))
            water=[(cell,poly) for cell,poly in water if group['allowed'][cell]]
            individual_water.extend(_area(poly,context) for _,poly in water)
            waters.append(water)
        # Every surviving polygon carries both actual receiving cells. Its
        # opposite footprint is an exact rotation, so area cannot be borrowed
        # from a remote portion of this boundary or another control cell.
        q_in_p=[(cell,poly@h) for cell,poly in waters[1]]
        for pc,ppoly in waters[0]:
            for qc,qpoly in q_in_p:
                common=_intersect_polygon(ppoly,qpoly)
                area=_area(common,context)
                if area<=AREA_TOLERANCE_KM2: continue
                raw_paired.append(2.*area)
                existing=group['union_by_cell'].setdefault(pc,[])
                for part in _subtract_polygons(common,existing,context):
                    existing.append(part)
                    records.append(dict(key=key,pc=pc,qc=qc,polygon=part))
    base['requested_area_km2']=math.fsum(requested)
    raw_total=math.fsum(raw_paired)
    base['local_unpaired_area_km2']=max(0.,math.fsum(individual_water)-raw_total)
    candidate=2.*math.fsum(_area(row['polygon'],context) for row in records)
    base['candidate_paired_area_km2']=candidate
    base['duplicate_paired_area_km2']=max(0.,raw_total-candidate)
    if not records: return birth
    # Potential physical footprints are partitioned by actual control cells.
    # Same-pair same-side duplicates were unioned above. A point claimed by
    # distinct front sides or owner pairs is physically ambiguous: exclude
    # that point AND its corresponding partner, locally and symmetrically.
    occupied={}
    for row in records:
        key=row['key'];poly=row['polygon'];h=groups[key]['h']
        occupied.setdefault(row['pc'],[]).append((key,0,poly))
        occupied.setdefault(row['qc'],[]).append((key,1,poly@h.T))
    cells=[];makers=[];areas=[];paired=[];measured_q=[]
    for row in records:
        key=row['key'];h=groups[key]['h'];pc,qc=row['pc'],row['qc']
        blockers=[poly for other,side,poly in occupied[pc] if other!=key or side!=0]
        blockers.extend(poly@h for other,side,poly in occupied[qc] if other!=key or side!=1)
        for part in _subtract_polygons(row['polygon'],blockers,context):
            area=_area(part,context);qarea=_area(part@h.T,context)
            if not math.isclose(area,qarea,rel_tol=1e-9,abs_tol=1e-6):
                raise ValueError('Local finite half-stage polygons lost congruent area.')
            # key=(p,q) and the strips are ordered p shore then q shore, so each
            # half of a congruent patch records the plate whose shore made it.
            paired.append(area);measured_q.append(qarea)
            cells.extend((pc,qc));makers.extend(key);areas.extend((area,area))
    final=2.*math.fsum(paired)
    base['ambiguous_overlap_excluded_area_km2']=max(0.,candidate-final)
    base['local_paired_patches']=len(paired)
    if not cells: return birth
    cells=np.asarray(cells,int);areas=np.asarray(areas,float)
    deposited=np.bincount(cells,weights=areas,minlength=s.n)
    capacity=np.zeros(s.n)
    for cell,value in context['_ocean_capacity'].items(): capacity[cell]=value
    original_paired=np.asarray(paired,float)
    paired,factors,deposited=_bounded_pair_areas(cells,original_paired,capacity,s.cell_area)
    areas=np.repeat(paired,2)
    measured_q=np.asarray(measured_q)*factors
    base['capacity_roundoff_removed_area_km2']=2.*math.fsum(original_paired-paired)
    birth=deposited/s.cell_area
    # New crust belongs to the plate whose shore produced it. The local pairing
    # already resolved that side geometrically, so ownership follows material.
    # Re-splitting a receiving cell's whole ocean support from a support
    # contrast field instead reassigned pre-existing crust, handing it across
    # the axis. This birth transaction must not reassign pre-existing ocean
    # support; explicit ridge jumps, capture, or other ownership handoffs belong
    # to separate processes. Only the newly created fraction is placed; older
    # crust is diluted by it but keeps its own plate.
    at=np.unique(cells)
    created=np.zeros_like(transported[:,at])
    np.add.at(created,(np.asarray(makers,int),np.searchsorted(at,cells)),np.asarray(areas,float))
    total=transported[:,at].sum(axis=0)
    # One budget, spent in the right order. A pair's opening first fills the
    # hole the opening itself made, and only what is left over repaints the
    # axis. Previously the whole budget repainted: the cell was scaled down by
    # exactly what was added, so a ridge could not change any plate's area and
    # the hole was left for support normalization, which fills it in
    # proportion to what each plate already holds and therefore awards new
    # seafloor to whichever flank is already the larger.
    if production_version(s)==1:
        produced=created.sum(axis=0)/s.cell_area[at]
        share=created/np.maximum(created.sum(axis=0),1e-300)
        fill=np.minimum(np.maximum(1.-total,0.),produced)
        repaint=produced-fill
        transported[:,at]*=(1.-repaint)
        transported[:,at]+=share*(total*repaint+fill)
        # The newborn fraction is of the cell's FINAL cover, which the fill raised.
        covered=total+fill
        birth[at]=np.divide(total*repaint+fill,covered,out=np.zeros_like(covered),where=covered>1e-300)
    else:
        transported[:,at]*=(1.-birth[at])
        transported[:,at]+=total*created/s.cell_area[at]
    generated=float(s.cell_area@birth)
    s.process_totals['ocean_created_km2']+=generated
    base.update(generated_area_km2=generated,reconstructed_cells=len(at),
        # Deposits use the shared congruent area above; these diagnostics retain
        # the separate measurements of both polygons, including their roundoff.
        side_p_area_km2=math.fsum(paired),side_q_area_km2=math.fsum(measured_q),
        maximum_capacity_fraction=float(np.max(capacity/s.cell_area,initial=0.)),
        maximum_water_capacity_residual_km2=float(np.max(deposited-capacity,initial=0.)),
        partly_ocean_receiving_cells=int(np.count_nonzero((deposited>0.)&(capacity<s.cell_area-1e-5))),
        mean_full_rate_cm_yr=float(base['requested_area_km2']/max(dt*math.fsum(pieces['length'][selected]),1e-30)/10.))
    return birth
