"""Physical coverage of native control faces by moving spherical triangles.

Candidate boxes only accelerate the search. Each reported area comes from
convex polygon clipping by great-circle planes on the sphere. Separate material
overlaps remain separate intersections: choosing the exposed sheet is the
engine's policy, not an implicit area normalization in this geometry helper.
"""
from __future__ import annotations
from decimal import Decimal, localcontext
from fractions import Fraction
from math import atan2, fsum
import numpy as np

from mesh_geometry import RADIUS_KM, _triangles, build_locator
from parallel_runtime import active_runtime, share_inputs, read_inputs
from spherical_predicates import edge_distances, plane_distances

MIN_PARALLEL_MATERIAL_FACES = 2048
MIN_PARALLEL_CONTROL_FACES = 2048

def _unit(points):
    points=np.asarray(points,float)
    return points/np.maximum(np.linalg.norm(points,axis=-1,keepdims=True),1e-30)


def _triangle_planes(triangles):
    """Great-circle normals without cancellation on short spherical edges.

    a x (b-a) equals a x b, but the latter subtracts products of global
    coordinates to recover an edge-sized quantity. Difference vectors retain
    the represented edge before normalization magnifies that roundoff.
    """
    triangles=np.asarray(triangles,float)
    return _unit(np.cross(triangles,np.roll(triangles,-1,axis=-2)-triangles))


def _polygon_area(polygon,radius_km):
    if len(polygon)<3:
        return 0.
    a=polygon[0]
    b,c=polygon[1:-1],polygon[2:]
    # The same solid-angle determinant in local difference coordinates. A
    # narrow overlap must not be the residual of O(1) global cross products.
    numerator=np.abs(np.einsum('i,ij->j',a,np.cross(b-a,c-a).T))
    denominator=1+b@a+c@a+np.einsum('ij,ij->i',b,c)
    return float(np.sum(2*np.arctan2(numerator,denominator))*radius_km**2)


def clip_triangle(subject,clipper):
    """Return the minor spherical intersection polygon of two unit triangles.

    Both triangles must be outward oriented and lie inside an open hemisphere.
    A clipping crossing is the normalized positive linear combination lying
    on the clipping plane: it is the exact great-circle arc/plane intersection,
    not an intersection on a projected longitude/latitude map.
    """
    clipper=np.asarray(clipper,float)
    planes=_triangle_planes(clipper)
    return _clip_planes(subject,planes,clipper)


def _clip_planes(subject,planes,clipper=None,*,edge_pairs=None):
    """Clip physical halfspaces, using original edges when available.

    With no edge provenance, planes denote their stored normal coordinates;
    they do not reconstruct the exact edge that produced a rounded normal.
    """
    polygon=np.asarray(subject,float).copy()
    for index,plane in enumerate(planes):
        if not len(polygon):
            break
        if edge_pairs is not None:
            distance=edge_distances(polygon,*edge_pairs[index])
        elif clipper is not None:
            distance=edge_distances(polygon,clipper[index],clipper[(index+1)%len(clipper)])
        else:
            distance=plane_distances(polygon,plane)
        inside=distance>=0.
        if np.all(inside):
            continue
        if not np.any(inside):
            return np.empty((0,3))
        output=[]
        for i in range(len(polygon)):
            before=(i-1)%len(polygon)
            if inside[i]!=inside[before]:
                fraction=distance[before]/(distance[before]-distance[i])
                fraction=float(np.clip(fraction,0.,1.))
                crossing=(polygon[before] if fraction==0. else polygon[i] if fraction==1.
                          else _unit((1-fraction)*polygon[before]+fraction*polygon[i]))
                output.append(crossing)
            if inside[i]:
                output.append(polygon[i])
        polygon=np.asarray(output).reshape(-1,3)
        if len(polygon)>1:
            # A shared vertex may be emitted twice. A fixed angular deletion
            # tolerance can erase a real thin overlap; remove exact copies only.
            distinct=np.any(polygon!=np.roll(polygon,1,axis=0),axis=1)
            polygon=polygon[distinct]
    return polygon


def _precise_intersection_area(subject, clipper, radius_km):
    """Measure the original represented rays without rounding new vertices.

    Halfspace decisions and crossings are exact rationals. Normalize only for
    the solid-angle integral, at 80 digits; no cached area or closure target
    enters the calculation. This is the same physical great-circle geometry
    as the vectorized path, with its coordinate-return roundoff removed.
    """
    def cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

    def dot(a, b):
        return sum(x*y for x, y in zip(a, b))

    polygon = [tuple(Fraction(float(value)) for value in point) for point in subject]
    triangle = [tuple(Fraction(float(value)) for value in point) for point in clipper]
    for a, b in zip(triangle, triangle[1:]+triangle[:1]):
        if len(polygon) < 3:
            return 0.
        plane = cross(a, b)
        distances = [dot(plane, point) for point in polygon]
        inside = [distance >= 0 for distance in distances]
        output = []
        for index, point in enumerate(polygon):
            previous = (index-1) % len(polygon)
            if inside[index] != inside[previous]:
                before, after = distances[previous], distances[index]
                output.append(tuple((x*before-y*after)/(before-after)
                                    for x, y in zip(point, polygon[previous])))
            if inside[index]:
                output.append(point)
        polygon = [point for index, point in enumerate(output) if point != output[index-1]]
    if len(polygon) < 3:
        return 0.
    with localcontext() as context:
        context.prec = 80
        points = []
        for point in polygon:
            ray = [Decimal(value.numerator)/Decimal(value.denominator) for value in point]
            length = dot(ray, ray).sqrt()
            points.append([value/length for value in ray])
        a = points[0]
        angles = []
        for index, (b, c) in enumerate(zip(points[1:-1], points[2:]), 1):
            winding = dot(polygon[0], cross(polygon[index], polygon[index+1]))
            if winding == 0:
                continue
            numerator = dot(a, cross(b, c))
            denominator = 1+dot(a, b)+dot(b, c)+dot(c, a)
            if denominator <= 0 or winding < 0 or numerator <= 0:
                raise ValueError('Exact intersection needs a positively wound minor convex polygon.')
            angles.append(2*atan2(float(numerator), float(denominator)))
        return fsum(angles)*float(radius_km)**2


def _triangle_area_condition(triangles):
    """Filter coordinate rounding relative to each source solid angle."""
    a, b, c = np.moveaxis(triangles, 1, 0)
    numerator = np.abs(np.einsum('ni,ni->n', a, np.cross(b-a, c-a)))
    denominator = 1+np.einsum('ni,ni->n', b+c, a)+np.einsum('ni,ni->n', b, c)
    angle = 2*np.arctan2(numerator, denominator)
    perimeter = np.linalg.norm(triangles-np.roll(triangles, 1, axis=1), axis=2).sum(axis=1)
    return 32*np.finfo(float).eps*perimeter > 2e-10*angle


def _intersection_areas(subjects,clippers,radius_km):
    """Vectorized paired triangle clipping with a bounded polygon buffer.

    A convex triangle intersection needs at most six distinct vertices. Twelve
    slots also accommodate temporary duplicate crossings on coincident edges;
    duplicates are removed after each clipping plane. No pairing is invented
    here: callers supply only the sparse geometric candidates.
    """
    count=len(subjects)
    capacity=12
    polygon=np.zeros((count,capacity,3))
    polygon[:,:3]=subjects
    sizes=np.full(count,3,int)
    cut=np.zeros(count,bool)
    rows=np.arange(count)[:,None]
    slots=np.arange(capacity)[None,:]
    for k in range(3):
        valid=slots<sizes[:,None]
        distance=edge_distances(polygon,clippers[:,k,None,:],
                                clippers[:,(k+1)%3,None,:],valid=valid)
        inside=distance>=0.
        previous=(slots-1)%np.maximum(sizes[:,None],1)
        before=polygon[rows,previous]
        before_distance=distance[rows,previous]
        crossing=(inside!=inside[rows,previous])&valid
        cut |= np.any(crossing,axis=1)
        fraction=np.divide(before_distance,before_distance-distance,
                           out=np.zeros_like(distance),where=before_distance!=distance)
        fraction=np.clip(fraction,0.,1.)
        crossing_point=_unit(before*(1-fraction[:,:,None])+polygon*fraction[:,:,None])
        crossing_point=np.where((fraction==0.)[:,:,None],before,
                                np.where((fraction==1.)[:,:,None],polygon,crossing_point))
        candidates=np.stack((crossing_point,polygon),axis=2).reshape(count,capacity*2,3)
        chosen=np.stack((crossing,inside&valid),axis=2).reshape(count,capacity*2)
        rank=np.cumsum(chosen,axis=1)-1
        r,c=np.nonzero(chosen)
        sizes=chosen.sum(axis=1)
        if np.any(sizes>capacity):
            raise ValueError('A convex spherical triangle intersection exceeded its vertex bound.')
        output=np.zeros_like(polygon)
        output[r,rank[r,c]]=candidates[r,c]
        previous=(slots-1)%np.maximum(sizes[:,None],1)
        different=np.any(output!=output[rows,previous],axis=2)
        chosen=different&(slots<sizes[:,None])
        rank=np.cumsum(chosen,axis=1)-1
        r,c=np.nonzero(chosen)
        polygon[:]=0.
        polygon[r,rank[r,c]]=output[r,c]
        sizes=chosen.sum(axis=1)
    a=polygon[:,0]
    b,c=polygon[:,1:-1],polygon[:,2:]
    numerator=np.abs(np.einsum('ni,nvi->nv',a,np.cross(b-a[:,None],c-a[:,None])))
    denominator=1+np.einsum('nvi,ni->nv',b+c,a)+np.einsum('nvi,nvi->nv',b,c)
    valid=np.arange(capacity-2)[None,:]<sizes[:,None]-2
    angle=np.sum(np.where(valid,2*np.arctan2(numerator,denominator),0.),axis=1)
    areas=angle*radius_km**2
    # Moving a returned unit coordinate by roundoff sweeps an area proportional
    # to polygon perimeter. Thin source faces and thin intersections amplify
    # that error even when every orientation predicate is confidently positive.
    # Refine the measurement, not the geometry or any acceptance tolerance.
    previous=(slots-1)%np.maximum(sizes[:,None],1)
    length=np.linalg.norm(polygon-polygon[rows,previous],axis=2)
    perimeter=np.sum(np.where(slots<sizes[:,None],length,0.),axis=1)
    sensitive=(angle>0.) & (32*np.finfo(float).eps*perimeter > 2e-10*angle)
    sensitive |= cut & (_triangle_area_condition(subjects)|_triangle_area_condition(clippers))
    for index in np.flatnonzero(sensitive):
        areas[index]=_precise_intersection_area(subjects[index],clippers[index],radius_km)
    return areas


def _candidates(triangle,centre,chord_radius,locator):
    resolution=int(locator['resolution'])
    lo=np.clip(np.floor((centre-chord_radius-1e-12+1)*resolution*.5).astype(int),0,resolution-1)
    hi=np.clip(np.floor((centre+chord_radius+1e-12+1)*resolution*.5).astype(int),0,resolution-1)
    if int(np.prod(hi-lo+1))>4096:
        # A genuinely large source face can cover a substantial portion of the
        # sphere. Iterate its potential controls once, still never allocating a
        # global material-by-control matrix or expanding a huge box of bins.
        return np.arange(int(locator['face_count']),dtype=np.int32)
    x,y,z=[np.arange(a,b+1) for a,b in zip(lo,hi)]
    keys=(x[:,None,None]+resolution*(y[None,:,None]+resolution*z[None,None,:])).ravel()
    at=np.searchsorted(locator['keys'],keys)
    valid=at<len(locator['keys'])
    chosen=at[valid]
    chosen=chosen[locator['keys'][chosen] == keys[valid]]
    parts=[locator['candidates'][locator['offsets'][i]:locator['offsets'][i+1]] for i in chosen]
    if len(locator['global_faces']):
        parts.append(locator['global_faces'])
    return np.unique(np.concatenate(parts)) if parts else np.empty(0,np.int32)


def _candidate_pairs(centres,chord_radius,locator,*,pair_limit=32768):
    """Yield bounded source/target buffers with the exact scalar bin policy.

    Tiny adjacent material triangles often query the identical Cartesian box.
    Enumerate each distinct box once; retain every original triangle and its
    complete sorted candidate set. Buffer size changes work partition only.
    """
    if not len(centres) or not int(locator['face_count']):return
    resolution=int(locator['resolution'])
    lower=np.clip(np.floor((centres-chord_radius[:,None]-1e-12+1)*resolution*.5).astype(int),0,resolution-1)
    upper=np.clip(np.floor((centres+chord_radius[:,None]+1e-12+1)*resolution*.5).astype(int),0,resolution-1)
    _,first,inverse=np.unique(np.column_stack((lower,upper)),axis=0,return_index=True,return_inverse=True)
    order=np.argsort(inverse,kind='stable')
    starts=np.r_[0,np.cumsum(np.bincount(inverse,minlength=len(first)))]
    for group,representative in enumerate(first):
        candidates=_candidates(None,centres[representative],chord_radius[representative],locator)
        if not len(candidates):continue
        sources=order[starts[group]:starts[group+1]]
        # Slice the flattened Cartesian product, including the exceptional
        # all-controls fallback, without allocating that whole product.
        total=len(sources)*len(candidates)
        for start in range(0,total,pair_limit):
            flat=np.arange(start,min(start+pair_limit,total),dtype=np.int64)
            yield sources[flat//len(candidates)],candidates[flat%len(candidates)]


def _ordered_pairs(first,second):
    """Restore the established source-major, target-minor clipping order."""
    if not first:return np.empty(0,np.int32),np.empty(0,np.int32)
    first=np.concatenate(first).astype(np.int32,copy=False)
    second=np.concatenate(second).astype(np.int32,copy=False)
    order=np.lexsort((second,first))
    return first[order],second[order]


def _intersections_serial(control_mesh,material_vertices,material_faces,*,locator=None):
    """Return sparse exact intersection areas, in deterministic material order.

    Result arrays are ``control_index``, ``material_index`` and ``area_km2``.
    Edge/vertex-only contact has zero area and is omitted. No material is filled
    across holes, promoted to a full cell, or merged with another overlapping
    sheet. Native control geometry and optional locator may be rigidly rotated.
    """
    vertices,faces,triangles=_triangles(material_vertices,material_faces)
    control=np.asarray(control_mesh['vertices'])[control_mesh['faces']]
    radius=float(control_mesh.get('radius_km',RADIUS_KM))
    if not np.isfinite(radius) or radius<=0:
        raise ValueError('Coverage needs a finite positive sphere radius.')
    if len(triangles):
        det=np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1],triangles[:,2]))
        if np.any(det<=0):
            raise ValueError('Material coverage triangles must be outward oriented.')
    if locator is None:
        locator=build_locator(control_mesh['vertices'],control_mesh['faces'])
    if (int(locator['face_count'])!=len(control)
            or not np.allclose(locator['face_vertices'],control,rtol=0.,atol=2e-14)):
        raise ValueError('Coverage locator must match the current control geometry.')
    centres=_unit(triangles.sum(axis=1))
    cc=_unit(control.sum(axis=1))
    control_angle=np.max(np.arccos(np.clip(np.einsum('fvi,fi->fv',control,cc),-1.,1.)),axis=1)
    chord=np.max(np.linalg.norm(triangles-centres[:,None,:],axis=2),axis=1) if len(triangles) else np.empty(0)
    angle=2*np.arcsin(np.clip(chord*.5,0.,1.))
    planes=_triangle_planes(control)
    own_planes=_triangle_planes(triangles)
    control_index,material_index=[],[]
    for material,candidates in _candidate_pairs(centres,chord,locator):
        inside_cap=np.einsum('ij,ij->i',cc[candidates],centres[material]) >= np.cos(np.minimum(np.pi,angle[material]+control_angle[candidates]))-2e-14
        material,candidates=material[inside_cap],candidates[inside_cap]
        if not len(candidates):continue
        # Reject disjoint triangles by either face's separating edge planes
        # before the small variable-length polygon operation.
        outside_control=np.any(np.max(np.einsum('fei,fvi->fev',planes[candidates],triangles[material]),axis=2)<-5e-14,axis=1)
        outside_material=np.any(np.max(np.einsum('fvi,fei->fev',control[candidates],own_planes[material]),axis=2)<-5e-14,axis=1)
        keep=~(outside_control|outside_material)
        if len(candidates):
            control_index.append(candidates[keep])
            material_index.append(material[keep])
    material_index,control_index=_ordered_pairs(material_index,control_index)
    candidate_count=len(control_index)
    areas=np.empty(len(control_index))
    for start in range(0,len(areas),1024):
        chunk=slice(start,start+1024)
        areas[chunk]=_intersection_areas(triangles[material_index[chunk]],control[control_index[chunk]],radius)
    # Floating-point fan triangles at shared-edge-only contacts can have an
    # immeasurably small residual determinant: omit <=0.0001 square metre.
    positive=areas>1e-10
    return dict(control_index=control_index[positive],
                material_index=material_index[positive],area_km2=areas[positive],
                candidate_pairs=int(candidate_count),method='spherical convex great-circle polygon intersection')


def _coverage_range(job):
    shared, start, stop = job
    with read_inputs(shared) as data:
        result = _intersections_serial(data['control_mesh'], data['vertices'],
            data['faces'][start:stop], locator=data['locator'])
        result['material_index'] += start
        return result


def intersections(control_mesh,material_vertices,material_faces,*,locator=None):
    """Exact ordered coverage; active runtimes may solve independent ranges.

    Range boundaries change dispatch only. Each range uses the original exact
    kernel; concatenation restores source-major order without a new reduction.
    """
    runtime = active_runtime()
    count = len(material_faces)
    if (runtime is None or runtime.policy.status()['requested_workers'] == 1
            or count < MIN_PARALLEL_MATERIAL_FACES
            or len(control_mesh['faces']) < MIN_PARALLEL_CONTROL_FACES):
        return _intersections_serial(control_mesh,material_vertices,material_faces,locator=locator)
    # Validate the complete material set before any dispatch, then share one
    # immutable stage copy rather than repeatedly pickling the control mesh.
    _triangles(material_vertices,material_faces)
    if locator is None:
        locator = build_locator(control_mesh['vertices'], control_mesh['faces'])
    pieces = min(runtime.policy.status()['max_workers']*2, max(2, count//2048))
    cuts = np.linspace(0,count,pieces+1,dtype=int)
    with share_inputs(dict(control_mesh=control_mesh,vertices=np.asarray(material_vertices),
                           faces=np.asarray(material_faces),locator=locator)) as shared:
        rows = runtime.map(_coverage_range, ((shared,int(a),int(b)) for a,b in zip(cuts[:-1],cuts[1:])))
    result = {key: np.concatenate([row[key] for row in rows])
              for key in ('control_index','material_index','area_km2')}
    result.update(candidate_pairs=sum(row['candidate_pairs'] for row in rows),
                  method='spherical convex great-circle polygon intersection')
    return result


def material_overlaps(vertices, faces, sheet_ids, *, radius_km=RADIUS_KM, include_touching=False):
    """Sparse exact positive-area intersections between distinct material sheets.

    Sheets remain separate; these areas are a contact ledger, never subtracted
    from reference material or summed as a geometric union. Edge-only touching
    has no buried area. Candidate buffers remain bounded during clipping.

    ``include_touching=True`` is a force-only candidate stencil: retain every
    nonrejected broadphase pair, including tiny or zero physical area. Boundary
    birth can have nonzero one-sided area work even at zero current area. Such
    candidates are not contact inventory and do not change the default ledger.
    """
    vertices, faces, triangles = _triangles(vertices, faces)
    sheets = np.asarray(sheet_ids)
    if sheets.shape != (len(faces),) or not np.issubdtype(sheets.dtype, np.integer):
        raise ValueError('Material contact sheets must align with faces.')
    locator = build_locator(vertices, faces)
    centres = _unit(triangles.sum(axis=1))
    chord = np.max(np.linalg.norm(triangles-centres[:, None], axis=2), axis=1) if len(faces) else np.empty(0)
    planes = _triangle_planes(triangles)
    first, second = [], []
    for source,target in _candidate_pairs(centres,chord,locator):
        keep=(target>source)&(sheets[target]!=sheets[source])
        source,target=source[keep],target[keep]
        if not len(source):continue
        outside_a=np.any(np.max(np.einsum('fei,fvi->fev',planes[target],triangles[source]),axis=2)<-5e-14,axis=1)
        outside_b=np.any(np.max(np.einsum('fvi,fei->fev',triangles[target],planes[source]),axis=2)<-5e-14,axis=1)
        keep=~(outside_a|outside_b)
        first.append(source[keep]);second.append(target[keep])
    a,b=_ordered_pairs(first,second)
    area = np.empty(len(a))
    for start in range(0, len(a), 1024):
        part = slice(start, start+1024)
        area[part] = _intersection_areas(triangles[a[part]], triangles[b[part]], radius_km)
    retained = np.ones(len(area),bool) if include_touching else area > 1e-8
    result=dict(first=a[retained], second=b[retained], area_km2=area[retained],
                candidate_pairs=int(len(a)), locator=locator)
    if include_touching:result['complete_force_stencil']=True
    return result


def same_sheet_overlaps(vertices, faces, sheet_ids, *, radius_km=RADIUS_KM):
    """Exact positive-area intersections inside one indexed material sheet.

    Ordinary outward manifold neighbours share a complete indexed edge but
    have zero intersection area.  They therefore pass without a topological
    exemption, while malformed coincident or folded edge pairs remain visible.
    Faces which share only a vertex are candidates: two free boundaries can
    meet there and then penetrate without inverting either triangle.  The same
    sparse locator, separating planes and spherical clipping kernel used by
    the inter-sheet contact ledger decide the geometry.
    """
    vertices, faces, triangles = _triangles(vertices, faces)
    sheets = np.asarray(sheet_ids)
    if sheets.shape != (len(faces),) or not np.issubdtype(sheets.dtype, np.integer):
        raise ValueError('Material contact sheets must align with faces.')
    locator = build_locator(vertices, faces)
    centres = _unit(triangles.sum(axis=1))
    chord = (np.max(np.linalg.norm(triangles-centres[:, None], axis=2), axis=1)
             if len(faces) else np.empty(0))
    planes = _triangle_planes(triangles)
    first, second = [], []
    for source, target in _candidate_pairs(centres, chord, locator):
        keep = (target > source) & (sheets[target] == sheets[source])
        source, target = source[keep], target[keep]
        if not len(source):
            continue
        # Do not exempt indexed neighbours here.  A proper oppositely directed
        # manifold edge clips to exact zero, whereas a same-directed duplicate
        # or folded edge is a real positive-area invariant violation.
        outside_a = np.any(np.max(
            np.einsum('fei,fvi->fev', planes[target], triangles[source]), axis=2) < -5e-14, axis=1)
        outside_b = np.any(np.max(
            np.einsum('fvi,fei->fev', triangles[target], planes[source]), axis=2) < -5e-14, axis=1)
        keep = ~(outside_a | outside_b)
        first.append(source[keep]); second.append(target[keep])
    a, b = _ordered_pairs(first, second)
    area = np.empty(len(a))
    for start in range(0, len(a), 1024):
        part = slice(start, start+1024)
        area[part] = _intersection_areas(triangles[a[part]], triangles[b[part]], radius_km)
    # The source mesh and indexed shared boundaries resolve to exact zero in
    # this same clipper.  Keep the invariant literal: every represented
    # positive area is penetration, independent of an unrelated lower face's
    # size in the later burial/phase stack.
    positive = area > 0.
    return dict(first=a[positive], second=b[positive], area_km2=area[positive],
                candidate_pairs=int(len(a)), locator=locator)
