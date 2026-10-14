"""Finite incoming-ocean capture on native contours, without a grid-sized belt.

This removes available fractional support where a finite convergent interface
sweeps actual downgoing ocean. It is a reduced kinematic sink, not a pressure or
slab-volume solution. Upper-plate continent is permitted; only the incoming
owner's continental union excludes ocean. Raster and unversioned native worlds
keep their historical operator.
"""
from __future__ import annotations

from collections import defaultdict
from copy import copy, deepcopy
import hashlib
import math
import numpy as np
import mesh_coverage
import native_spreading as exact
import finite_capture_geometry
import slab_memory
from ridge_geometry import rotate

VERSION=1
RADIUS_KM=6371.
AREA_FIELDS=(
    'raw_pair_capture_area_km2','local_water_area_before_union_km2',
    'unique_downgoing_water_area_km2','duplicate_capture_area_removed_km2',
    'rejected_unrelated_owner_water_area_km2','maturity_weighted_water_area_km2',
    'available_total_overlap_km2','requested_removal_area_km2','removed_area_km2',
    'unsupported_or_unavailable_capture_area_km2')
SUPPLY_VERSION=1
SUPPLY_AREAS=('raw_geometric_source_area_km2','admitted_unique_water_area_km2',
    'maturity_weighted_water_area_km2','donor_limited_request_area_km2',
    'requested_after_local_overlap_cap_km2','realized_removal_area_km2',
    'donor_reduction_area_km2','local_overlap_reduction_area_km2','competition_reduction_area_km2',
    'local_water_before_union_km2','rejected_unrelated_owner_water_area_km2','duplicate_water_removed_km2')
SUPPLY_EPOCH='source-step-before-endpoint-rematch'
SUPPLY_CAP_ORDER=('min(transported donor support, maturity-weighted unique water / cell area)',
    'min(donor-limited request, shared cell excess * admitted mean maturity)',
    'simultaneous proportional allocation when summed local requests exceed shared cell excess')


def _supply_version(s):
    version=getattr(s,'native_subduction_supply_version',0)
    if isinstance(version,(bool,np.bool_)) or not isinstance(version,(int,np.integer)) or version not in (0,SUPPLY_VERSION):
        raise ValueError('Unsupported native subduction supply diagnostic version.')
    return int(version)


def _source_trench(s,parent):
    identities=getattr(s,'trench_id',None)
    return int(identities[parent]) if identities is not None and identities[parent]>0 else None


def _supply_front(s,parent):
    """Freeze the source front, never substitute its endpoint descendants."""
    p=int(s.down[parent]);q=int(s.bq[parent] if p==s.bp[parent] else s.bp[parent])
    contour=s.native_boundary_geometry
    indices=np.flatnonzero(np.asarray(contour['contact_index'])==parent)
    starts=np.asarray(contour['segments_start'])[indices];ends=np.asarray(contour['segments_end'])[indices]
    normals=np.asarray(contour['segment_normals'])[indices]*(1. if p==s.bp[parent] else -1.)
    length=RADIUS_KM*math.fsum(math.atan2(float(np.linalg.norm(np.cross(a,b))),float(a@b)) for a,b in zip(starts,ends))
    return dict(source_parent_index=int(parent),source_trench_id=_source_trench(s,parent),
        source_downgoing_plate_uid=int(s.plate_uid[p]),source_overriding_plate_uid=int(s.plate_uid[q]),
        source_geometry=dict(segment_start_xyz=starts.tolist(),segment_end_xyz=ends.tolist(),
            segment_normal_xyz=normals.tolist(),length_km=length,
            normal_convention='Stored contour normal oriented incoming-to-overriding before endpoint tangent projection.'),
        source_nominal_maturity=float(s.trench_maturity[parent]),
        source_parent_normal_speed_km_myr=float(s.normal_speed[parent]),
        source_parent_boundary_code=int(s.bcode[parent]),
        geometry_witnesses=[dict(source_parent_index=int(parent),source_trench_id=_source_trench(s,parent))],
        **dict.fromkeys(SUPPLY_AREAS,0.))


def _empty_supply():
    return dict(version=SUPPLY_VERSION,attribution_epoch=SUPPLY_EPOCH,cap_order=list(SUPPLY_CAP_ORDER),
        source_identity_scope='Chosen polygon parent at removal time; winding witnesses are not additional feed recipients. Endpoint splits or joins do not relabel this record.',
        pre_cap_request='maturity_weighted_water_area_km2; admitted unique water before donor and overlap limits',
        availability_scope='Caps are measured in each owner/control-cell, then attributed in accepted maturity-weighted water proportions. Shared excess is not separately available to every front.',
        zero_source_scope='No raw capture can reflect opening, zero admitted maturity, an ineligible front or winding cancellation. Source context is retained; no unique cause is inferred.',
        water_exclusion_scope='Raw geometry minus local and unrelated-owner water includes incoming material exclusion and clipping residual, not a diagnosed collision cause.',
        maturity_scope='Source nominal maturity precedes any intact-neck gating; admitted water is weighted by the actual capture maturity.',
        speed_scope='Signed parent normal speed is source context only; capture uses finite Euler motion, not this scalar alone.',
        limitation='Diagnostics of the existing finite-volume sink; zero admitted or realized supply is not guaranteed feed and adds no force.',
        fronts=[],trenches=[],totals=dict.fromkeys(SUPPLY_AREAS,0.))


def _finish_supply(diagnostics,values,weighted,area,requests,removed):
    """Attribute existing caps; this helper never changes a sink or slab feed."""
    fronts=diagnostics.pop('_supply_fronts')
    contributions=diagnostics.pop('_supply_contributions')
    stages=defaultdict(lambda:defaultdict(list))
    donor=np.minimum(values,weighted/area)
    for parent,p,cell,amount in contributions:
        for name,stage in (('donor_limited_request_area_km2',donor),
                           ('requested_after_local_overlap_cap_km2',requests),
                           ('realized_removal_area_km2',removed)):
            # Same accepted-source proportional attribution as the existing feed.
            value=amount*stage[p,cell]*area[cell]/weighted[p,cell] if weighted[p,cell]>0 else 0.
            stages[parent][name].append(value)
    groups=defaultdict(list)
    for parent,row in sorted(fronts.items()):
        for name in ('donor_limited_request_area_km2','requested_after_local_overlap_cap_km2','realized_removal_area_km2'):
            row[name]=math.fsum(stages[parent][name])
        row['duplicate_water_removed_km2']=max(0.,row['local_water_before_union_km2']-row['admitted_unique_water_area_km2'])
        # Roundoff in area/fraction conversions cannot represent negative losses.
        for name,before,after in (
            ('donor_reduction_area_km2','maturity_weighted_water_area_km2','donor_limited_request_area_km2'),
            ('local_overlap_reduction_area_km2','donor_limited_request_area_km2','requested_after_local_overlap_cap_km2'),
            ('competition_reduction_area_km2','requested_after_local_overlap_cap_km2','realized_removal_area_km2')):
            row[name]=max(0.,row[before]-row[after])
        key=tuple(row[name] for name in ('source_trench_id','source_downgoing_plate_uid','source_overriding_plate_uid'))
        groups[key].append(row)
    ledger=_empty_supply();ledger['fronts']=[fronts[key] for key in sorted(fronts)]
    for key,rows in groups.items():
        ledger['trenches'].append(dict(zip(('source_trench_id','source_downgoing_plate_uid','source_overriding_plate_uid'),key),
            source_parent_indices=[row['source_parent_index'] for row in rows],
            **{name:math.fsum(row[name] for row in rows) for name in SUPPLY_AREAS}))
    ledger['totals']={name:math.fsum(row[name] for row in ledger['fronts']) for name in SUPPLY_AREAS}
    diagnostics['supply_ledger']=ledger


def enabled(s):
    return isinstance(getattr(s,'native_mesh',None),dict) and getattr(s,'native_subduction_version',0)==VERSION


def _unit(x):
    x=np.asarray(x,float)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-30)


def _owners(s):
    surface=s.material_surface
    result=np.asarray(getattr(s,'parcel_plate',surface.get('face_owner',np.empty(0,int))))
    if result.shape!=(len(surface['faces']),):
        raise ValueError('Incoming-water material owners must align with actual faces.')
    return result


def _material(s,p,dt=0.):
    surface=s.material_surface
    vertices=np.asarray(surface['vertices'],float)
    if dt:vertices=rotate(vertices,np.asarray(s.omega[p])*dt)
    return exact.prepare_material(dict(vertices=vertices,faces=np.asarray(surface['faces'])[_owners(s)==p],
                                       radius_km=surface.get('radius_km',RADIUS_KM)))


def edge_ocean_fraction(s,down=None):
    """Exact incoming-owner water length fraction on each unchanged parent edge.

    A scalar owner queries every parent for that candidate polarity. An owner
    array selects its local polarity per parent; omitted uses current s.down.
    No center-cell crust flag participates in this new version.

    Where the reconstruction carries the division of its pieces among contacts, each
    piece's length and wet length go to contacts in those shares, so the fraction is
    that of the ground the contact's length came from. Otherwise a piece counts whole
    for the one contact it is listed under, and a contact with no piece reads dry.
    """
    contour=s.native_boundary_geometry
    requested=np.asarray(s.down if down is None else down)
    if requested.ndim==0:requested=np.full(len(s.ba),int(requested))
    if requested.shape!=(len(s.ba),):raise ValueError('Incoming owner must align with parent contacts.')
    digest=hashlib.sha256()
    import native_boundary_geometry
    shared=native_boundary_geometry.has_shares(contour)
    for array in (np.array([len(s.ba)]),s.bp,s.bq,s.material_surface['vertices'],s.material_surface['faces'],_owners(s),
                  contour['segments_start'],contour['segments_end'],contour['contact_index'],
                  *((contour['share_segment'],contour['share_contact'],contour['share_weight']) if shared else ())):
        digest.update(np.asarray(array).tobytes())
    signature=digest.hexdigest()
    if signature!=getattr(s,'_native_subduction_water_signature',None):
        s._native_subduction_water_signature=signature;s._native_subduction_water_cache={}
    cache=s._native_subduction_water_cache
    result=np.zeros(len(s.ba))
    for p in np.unique(requested[requested>=0]):
        p=int(p)
        if p not in cache:
            context=_material(s,p);total=np.zeros(len(s.ba));wet=np.zeros(len(s.ba))
            pieces=len(contour['contact_index']);piece_total=np.zeros(pieces);piece_wet=np.zeros(pieces)
            for index,(a,b,parent) in enumerate(zip(contour['segments_start'],contour['segments_end'],contour['contact_index'])):
                if p not in (s.bp[parent],s.bq[parent]):continue
                length=RADIUS_KM*math.atan2(float(np.linalg.norm(np.cross(a,b))),float(a@b))
                total[parent]+=length;piece_total[index]=length
                for low,high in exact.ocean_intervals(a,b,context):
                    first=_unit((1-low)*a+low*b);last=_unit((1-high)*a+high*b)
                    extent=RADIUS_KM*math.atan2(float(np.linalg.norm(np.cross(first,last))),float(first@last))
                    wet[parent]+=extent;piece_wet[index]+=extent
            if shared:
                # A piece only shares with contacts of its own plate pair, so these sums stay
                # within the contacts that involve this owner.
                total=native_boundary_geometry.shares(contour,piece_total,len(s.ba))
                wet=native_boundary_geometry.shares(contour,piece_wet,len(s.ba))
            cache[p]=np.clip(np.divide(wet,total,out=np.zeros_like(wet),where=total>0),0.,1.)
        take=requested==p;result[take]=cache[p][take]
    return result


def _geometry_version(s):
    value=getattr(s,'native_subduction_geometry_version',0)
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value not in (0,1):
        raise ValueError('Unsupported finite subduction geometry policy.')
    return int(value)


def _attached_capture_state(s):
    """Gate local attachment before geometric union/maturity selection.

    Preserve failed patches in the spatial partition. Coincident intact/failed
    histories admit only the intact width share; a private maturity array makes
    repeated capture queries idempotent without changing trench history.
    """
    import slab_tether_local as local_necks
    import slab_tether_history as necks
    rows=[r for r in getattr(s,'trench_systems',()) if local_necks.enabled(r)]
    if not rows:return s
    staged=copy(s);staged.trench_maturity=np.asarray(s.trench_maturity,float).copy()
    for row in rows:
        edges=np.flatnonzero(np.asarray(s.trench_id)==row['id'])
        if not len(edges):continue
        portions=local_necks.allocation(row,np.asarray(s.bmid)[edges],np.ones(len(edges)))
        intact=np.array([c['neck']['damage']<1. for c in row[necks.FIELD]])
        staged.trench_maturity[edges]*=portions[intact].sum(axis=0)
    return staged


def capture_polygons(s,dt):
    staged=_attached_capture_state(s)
    if _geometry_version(s)==1:return finite_capture_geometry.capture_polygons(staged,dt)
    return _capture_polygons_legacy(staged,dt)


def _capture_polygons_legacy(s,dt):
    """Finite converging p/q shore polygons with parent/maturity provenance."""
    if not np.isfinite(dt) or dt<=0:raise ValueError('Native capture needs positive finite time.')
    contour=s.native_boundary_geometry
    maturity=np.asarray(getattr(s,'trench_maturity',np.zeros(len(s.ba))),float)
    if maturity.shape!=(len(s.ba),) or not np.isfinite(maturity).all() or np.any((maturity<0)|(maturity>1)):
        raise ValueError('Native trench maturity must be finite and lie in [0,1].')
    import normal_partition
    candidates=normal_partition.capture_candidates(s)
    rows=[]
    for a,b,normal,parent in zip(contour['segments_start'],contour['segments_end'],
                                 contour['segment_normals'],contour['contact_index']):
        parent=int(parent);p=int(s.down[parent])
        if not candidates[parent] or maturity[parent]<=0 or p not in (s.bp[parent],s.bq[parent]):continue
        q=int(s.bq[parent] if p==s.bp[parent] else s.bp[parent])
        if not (s.active[p] and s.active[q]):continue
        endpoints=np.array([a,b],float)
        normal=np.asarray(normal,float)*(1. if p==s.bp[parent] else -1.)
        normals=_unit(normal-endpoints*(endpoints@normal)[:,None])
        incoming=rotate(endpoints,np.asarray(s.omega[p])*dt)
        overriding=rotate(endpoints,np.asarray(s.omega[q])*dt)
        convergence=np.sum((incoming-overriding)*normals,axis=1)
        if np.all(convergence<=0.):continue
        if np.any(convergence<0.):
            fraction=float(-convergence[0]/(convergence[1]-convergence[0]))
            crossing=_unit(endpoints[0]*(1-fraction)+endpoints[1]*fraction)
            endpoints[0 if convergence[0]<0. else 1]=crossing
            incoming=rotate(endpoints,np.asarray(s.omega[p])*dt)
            overriding=rotate(endpoints,np.asarray(s.omega[q])*dt)
        polygon=np.array([incoming[0],incoming[1],overriding[1],overriding[0]])
        if np.linalg.det(polygon[:3])<0:polygon=polygon[::-1]
        # Geodesic Euler shores share each internal subdivision boundary;
        # a finite front has no artificial round cap beyond either endpoint.
        area=mesh_coverage._polygon_area(polygon,RADIUS_KM)
        if area<=exact.AREA_TOLERANCE_KM2:continue
        rows.append(dict(parent=parent,down=p,over=q,maturity=float(maturity[parent]),polygon=polygon,
                         raw_area_km2=float(area)))
    # The maximum local maturity owns any geometrically coincident footprint.
    # Ties do not change its union integral, irrespective of contact ordering.
    rows.sort(key=lambda row:(-row['maturity'],row['down'],row['over'],
                              tuple(np.round(row['polygon'].mean(axis=0),14))))
    return rows


def _polygon_context(polygons):
    """Retain accepted convex clipping pieces without inventing a mesh fan.

    Finite clipping can retain consecutive collinear vertices in a positive-
    area polygon. A fan about its first corner then contains zero-area faces;
    additionally a genuine small clipping piece can lie below the material
    mesh locator's determinant floor. Neither means the water polygon is
    empty. Direct convex subtraction uses the original polygon halfspaces
    and the existing area tolerance, without deleting positive polygon area.
    """
    return dict(polygons=tuple(np.asarray(polygon,float).copy() for polygon in polygons),
                radius_km=RADIUS_KM)


def integrate_capture(s,rows,dt):
    """Exact unique incoming-water area and maturity-weighted area per owner/cell.

    Previous pieces are subtracted per incoming owner and actual receiving
    control cell. This avoids duplicate sinks at shared fronts or junctions,
    while a blocked unrelated-owner piece cannot suppress an eligible piece.
    """
    shape=np.asarray(s.support).shape
    admitted=np.zeros(shape);weighted=np.zeros(shape)
    contexts={};accepted=defaultdict(list);prior_context={}
    raw=0.;water_before_union=0.;foreign=0.;pieces=0
    track_slab=slab_memory.enabled(s)
    contributions=[]
    supply_fronts=None;supply_contributions=[]
    if _supply_version(s):
        supply_fronts={int(parent):_supply_front(s,int(parent))
            for parent in np.unique(s.native_boundary_geometry['contact_index'])
            if int(s.down[parent])>=0 and s.down[parent] in (s.bp[parent],s.bq[parent])}
    for row in rows:
        p,q=row['down'],row['over'];polygon=row['polygon'];raw+=row['raw_area_km2']
        if supply_fronts is not None:
            source_parent=int(row['parent'])
            if source_parent not in supply_fronts:supply_fronts[source_parent]=_supply_front(s,source_parent)
            front=supply_fronts[source_parent]
            front['raw_geometric_source_area_km2']+=row['raw_area_km2']
            witnesses={r['source_parent_index']:r for r in front['geometry_witnesses']}
            for parent in row.get('source_parents',[source_parent]):
                witnesses[int(parent)]=dict(source_parent_index=int(parent),source_trench_id=_source_trench(s,int(parent)))
            front['geometry_witnesses']=[witnesses[parent] for parent in sorted(witnesses)]
        if p not in contexts:contexts[p]=_material(s,p,dt)
        context=contexts[p]
        local=(np.asarray(s.plate)==p)|(np.asarray(s.plate)==q)
        # Incoming support is fractional. A categorical incoming-arrival field
        # must not erase valid minority ocean in a partly continental cell.
        hit=exact.integrate_polygon(s.native_mesh,polygon,context)
        for cell,amount in zip(hit['cells'],hit['area_km2']):
            cell=int(cell)
            if not local[cell]:
                foreign+=float(amount)
                if supply_fronts is not None:front['rejected_unrelated_owner_water_area_km2']+=float(amount)
                continue
            clipped=mesh_coverage.clip_triangle(polygon,s.native_mesh['vertices'][s.native_mesh['faces'][cell]])
            water=exact.uncovered_polygons(clipped,context);water_before_union+=float(amount)
            if supply_fronts is not None:front['local_water_before_union_km2']+=float(amount)
            key=(p,cell)
            if accepted[key]:
                if key not in prior_context:prior_context[key]=_polygon_context(accepted[key])
                previous=prior_context[key]
                water=[part for patch in water for part in
                       exact._subtract_polygons(patch,previous['polygons'],previous)]
            area=math.fsum(mesh_coverage._polygon_area(part,RADIUS_KM) for part in water)
            if area<=exact.AREA_TOLERANCE_KM2:continue
            admitted[p,cell]+=area;weighted[p,cell]+=area*row['maturity'];pieces+=len(water)
            if supply_fronts is not None:
                front['admitted_unique_water_area_km2']+=area
                front['maturity_weighted_water_area_km2']+=area*row['maturity']
                supply_contributions.append((source_parent,p,cell,area*row['maturity']))
            if track_slab:
                parent=int(row['parent']);ident=int(s.trench_id[parent])
                if ident<=0:raise ValueError('Accepted slab feed lacks a local trench identity.')
                below=int(s.ba[parent] if s.bp[parent]==p else s.bb[parent])
                age=float(s.age[below])
                if not math.isfinite(age) or age<0:raise ValueError('Slab feed requires finite nonnegative incoming cooling age.')
                contributions.append((ident,p,cell,area*row['maturity'],min(math.sqrt(age/80.),1.8),parent,age))
            accepted[key].extend(water);prior_context.pop(key,None)
    diagnostics=dict(capture_polygons=len(rows),unique_water_parts=pieces,
        raw_pair_capture_area_km2=raw,local_water_area_before_union_km2=water_before_union,
        unique_downgoing_water_area_km2=float(admitted.sum()),
        duplicate_capture_area_removed_km2=water_before_union-float(admitted.sum()),
        rejected_unrelated_owner_water_area_km2=foreign)
    if track_slab:diagnostics['_slab_contributions']=contributions
    if supply_fronts is not None:
        diagnostics['_supply_fronts']=supply_fronts
        diagnostics['_supply_contributions']=supply_contributions
    return admitted,weighted,diagnostics


def removal(s,transported,dt,arrivals=None):
    """Return bounded per-owner removed support; do not mutate its input."""
    values=np.asarray(transported,float)
    if values.shape!=np.asarray(s.support).shape or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('Captured incoming support must be finite, nonnegative and aligned.')
    rows=capture_polygons(s,dt)
    admitted,weighted,diagnostics=integrate_capture(s,rows,dt)
    area=np.asarray(s.cell_area,float)
    if area.shape!=(values.shape[1],) or not np.isfinite(area).all() or np.any(area<=0.):
        raise ValueError('Finite capture requires aligned positive control-cell areas.')
    excess=np.maximum(values.sum(axis=0)-1.,0.)
    effective=np.divide(weighted,admitted,out=np.zeros_like(weighted),where=admitted>0)
    # A trench may only consume double cover, and the double cover it made does
    # not stay where it made it: the transport is first-order upwind, so the
    # arriving slab is smeared a cell or two behind the front and barely a
    # third of the sweep is left in the front's own cells. Capping each request
    # at the LOCAL excess therefore throttles the sink to what happens to have
    # stayed put. The allowance is extended to the immediate neighbours the
    # material actually went to - cells this plate arrived in this step and
    # where the other side already holds at least as much - while the removal
    # itself still happens where this plate's support is, so nothing is taken
    # from a cell that does not hold it.
    allowance=np.repeat(excess[None,:],len(values),axis=0)
    if getattr(s,'ocean_production_version',0)==1 and arrivals is not None:
        incoming=np.asarray(arrivals,float)
        for neighbour in np.asarray(s.native_mesh['face_neighbors']).T:
            gate=(incoming[:,neighbour]>1e-3)&(values[:,neighbour]<=1.-values[:,neighbour])
            allowance=allowance+excess[neighbour][None,:]*gate
    requests=np.minimum(np.minimum(values,weighted/area),allowance*effective)
    total=requests.sum(axis=0)
    ceiling=allowance.max(axis=0)
    scale=np.minimum(1.,np.divide(ceiling,total,out=np.ones_like(total),where=total>0))
    removed=requests*scale[None,:]
    if np.any(removed>values+1e-12) or np.any(removed.sum(axis=0)>ceiling+1e-12):
        raise ValueError('Finite native subduction exceeded incoming support or shared overlap.')
    if '_slab_contributions' in diagnostics:
        shares=defaultdict(list)
        import slab_tether_local as local_necks
        import slab_tether_history as necks
        histories={r['id']:r for r in s.trench_systems}
        local_shares=defaultdict(list); allocation_cache={}
        import slab_anchors
        located=defaultdict(lambda:defaultdict(lambda:[0.,0.])) if slab_anchors.enabled(s) else None
        for ident,p,cell,amount,buoyancy,parent,age in diagnostics.pop('_slab_contributions'):
            # Attribute the already-computed sink in its exact accepted-water
            # proportions. No changes to clipping, maturity, requests or caps.
            realized=amount*removed[p,cell]*area[cell]/weighted[p,cell] if weighted[p,cell]>0 else 0.
            shares[(ident,p)].append((realized,realized*buoyancy))
            record=histories[ident]
            if located is not None and realized>0. and not local_necks.enabled(record):
                # Where this slab went down: its accepted boundary contact.
                entry=located[(ident,p)][int(parent)]
                entry[0]+=realized
                entry[1]+=realized*1e6*float(slab_memory.excess_mass_per_area_kg_m2(age))
            if local_necks.enabled(record) and realized>0.:
                key=(ident,parent)
                if key not in allocation_cache:
                    portions=local_necks.allocation(record,np.asarray(s.bmid)[[parent]],[1.])[:,0]
                    intact=np.array([c['neck']['damage']<1. for c in record[necks.FIELD]])
                    portions*=intact
                    # The accepted geometry was already reduced by this intact
                    # width share. Attribute that reduced feed conditionally;
                    # applying the share a second time would lose inventory.
                    intact_share=float(portions.sum())
                    allocation_cache[key]=portions/intact_share if intact_share>0. else portions
                portions=allocation_cache[key]
                if abs(float(portions.sum())-1.)>1e-12:
                    raise ValueError('Accepted local slab feed has no resolved neck patch at its parent front.')
                for index in np.flatnonzero(portions>0):
                    if record[necks.FIELD][index]['neck']['damage']==1.:
                        raise ValueError('Accepted capture cannot reattach a ruptured slab neck.')
                    a=realized*float(portions[index])
                    mass=a*1e6*float(slab_memory.excess_mass_per_area_kg_m2(age))
                    local_shares[(ident,int(index))].append((a,a*buoyancy,mass))
        diagnostics['removed_area_by_trench']=[dict(trench_id=ident,
            downgoing_plate_uid=int(s.plate_uid[p]),area_km2=math.fsum(v[0] for v in share),
            buoyancy_area_km2=math.fsum(v[1] for v in share))
            for (ident,p),share in sorted(shares.items())]
        if located is not None:
            slots={int(s.plate_uid[q]):int(q) for q in np.flatnonzero(s.active)}
            for feed in diagnostics['removed_area_by_trench']:
                points=located.get((feed['trench_id'],slots[feed['downgoing_plate_uid']]),{})
                feed['located']=[dict(contact=int(c),xyz=np.asarray(s.bmid)[c].tolist(),area_km2=float(v[0]),
                                      mass_weight_kg=float(v[1])) for c,v in sorted(points.items())]
        for feed in diagnostics['removed_area_by_trench']:
            ident=feed['trench_id']
            if local_necks.enabled(histories[ident]):
                feed['neck_feeds']=[dict(channel_index=index,area_km2=math.fsum(v[0] for v in values),
                    buoyancy_area_km2=math.fsum(v[1] for v in values),excess_mass_kg=math.fsum(v[2] for v in values))
                    for (identity,index),values in sorted(local_shares.items()) if identity==ident]
        diagnostics['trench_attribution']='accepted unique polygon parent; availability-capped owner/cell removal in accepted weighted-area proportions'
    diagnostics.update(version=VERSION,state='measured_step',
        model='finite Euler convergence footprint, exact incoming continental-union exclusion, bounded fractional support sink',
        step_start_myr=float(getattr(s,'t',0.)),step_end_myr=float(getattr(s,'t',0.))+float(dt),
        step_duration_myr=float(dt),grid_radius_used=False,center_crust_gate_used=False,
        owner_slots=list(range(values.shape[0])),
        maturity_weighted_water_area_km2=float(weighted.sum()),
        available_total_overlap_km2=float(area@excess),
        requested_removal_area_km2=float(area@total),removed_area_km2=float(area@removed.sum(axis=0)),
        removed_area_by_owner_km2=(removed@area).tolist(),
        unsupported_or_unavailable_capture_area_km2=float(weighted.sum()-area@removed.sum(axis=0)),
        maximum_support_bound_residual=float(np.max(removed-values,initial=0.)),
        maximum_shared_excess_bound_residual=float(np.max(removed.sum(axis=0)-excess,initial=0.)),
        maturity_combination='maximum per unique geometric footprint; area-weighted within each receiving cell',
        owner_competition='simultaneous proportional allocation of shared available excess',
        incoming_arrival_categories_used=False,force_parent_graph_unchanged=True,
        limitation='Reduced kinematic capture of available finite-volume support; remnant transport/normalization error is separately recorded, not declared physical subduction.')
    if _geometry_version(s)==1:
        diagnostics.update(capture_geometry_version=1,capture_geometry_method='connected-front-positive-winding')
    if '_supply_fronts' in diagnostics:
        _finish_supply(diagnostics,values,weighted,area,requests,removed)
    s.native_subduction_diagnostics=diagnostics
    return removed


def snapshot_metadata(s):
    """Saved law and last measured sink; an initial epoch has explicit zeros."""
    supply_version=_supply_version(s)
    if not enabled(s):
        if supply_version:raise ValueError('Supply diagnostics require finite native subduction.')
        return {}
    record=getattr(s,'native_subduction_diagnostics',None)
    if record is None:
        if float(s.t)!=0.:
            raise ValueError('A progressed finite-subduction world lacks its measured step record.')
        record=dict(version=VERSION,state='initialized',step_start_myr=0.,step_end_myr=0.,step_duration_myr=0.,
            owner_slots=list(range(len(s.support))),removed_area_by_owner_km2=[0.]*len(s.support),
            capture_polygons=0,unique_water_parts=0,grid_radius_used=False,center_crust_gate_used=False,
            incoming_arrival_categories_used=False,force_parent_graph_unchanged=True,
            maximum_support_bound_residual=0.,maximum_shared_excess_bound_residual=0.,
            model='finite Euler convergence footprint; no transport step yet',**dict.fromkeys(AREA_FIELDS,0.))
        if supply_version:record['supply_ledger']=_empty_supply()
    result=dict(native_subduction_version=VERSION,native_subduction_diagnostics=deepcopy(record))
    if supply_version:result['native_subduction_supply_version']=supply_version
    # A topology event can grow the slot buffer after transport. New slots did
    # not participate in the recorded sink and need explicit zero rows; the
    # existing per-owner amounts and total are never redistributed.
    row=result['native_subduction_diagnostics']
    missing=[p for p in range(len(s.support)) if p not in row['owner_slots']]
    if missing:
        row['owner_slots'].extend(missing)
        row['removed_area_by_owner_km2'].extend([0.]*len(missing))
    if _geometry_version(s)==1:
        result['native_subduction_geometry_version']=1
        if row['state']=='initialized':
            row.update(capture_geometry_version=1,capture_geometry_method='connected-front-positive-winding')
    import normal_partition
    result.update(normal_partition.snapshot(s))
    validate_frame(dict(result,time_myr=float(s.t)))
    return result


def _validate_supply_ledger(record,tol):
    ledger=record.get('supply_ledger')
    if (not isinstance(ledger,dict) or type(ledger.get('version')) is not int
            or ledger['version']!=SUPPLY_VERSION or ledger.get('attribution_epoch')!=SUPPLY_EPOCH
            or ledger.get('cap_order')!=list(SUPPLY_CAP_ORDER)):
        raise ValueError('Supply ledger requires its exact version, source epoch and cap order.')
    def identity(value,*,nullable=False,positive=False):
        return ((nullable and value is None) or
            (type(value) is int and (not positive or value>0)))
    def amounts(row):
        if not isinstance(row,dict):raise ValueError('Supply ledger rows must be objects.')
        for name in SUPPLY_AREAS:
            value=row.get(name)
            if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.number))
                    or not math.isfinite(value) or value<0.):
                raise ValueError('Invalid supply ledger area: '+name)
        for before,after in zip(SUPPLY_AREAS[:5],SUPPLY_AREAS[1:6]):
            if row[after]>row[before]+tol:raise ValueError('Supply ledger exceeds a source or cap.')
        for before,after,reduction in (
            ('maturity_weighted_water_area_km2','donor_limited_request_area_km2','donor_reduction_area_km2'),
            ('donor_limited_request_area_km2','requested_after_local_overlap_cap_km2','local_overlap_reduction_area_km2'),
            ('requested_after_local_overlap_cap_km2','realized_removal_area_km2','competition_reduction_area_km2')):
            if abs(row[before]-row[after]-row[reduction])>tol:
                raise ValueError('Supply ledger cap reductions do not close.')
        if (row['local_water_before_union_km2']+row['rejected_unrelated_owner_water_area_km2']>row['raw_geometric_source_area_km2']+tol
                or row['admitted_unique_water_area_km2']>row['local_water_before_union_km2']+tol
                or abs(row['local_water_before_union_km2']-row['admitted_unique_water_area_km2']-row['duplicate_water_removed_km2'])>tol):
            raise ValueError('Supply ledger locality or unique-water accounting disagrees.')
    def source_key(row):
        key=tuple(row.get(name) for name in ('source_trench_id','source_downgoing_plate_uid','source_overriding_plate_uid'))
        if not identity(key[0],nullable=True,positive=True) or not all(identity(uid) for uid in key[1:]):
            raise ValueError('Supply ledger requires explicit source trench and plate identities.')
        return key
    fronts=ledger.get('fronts');trenches=ledger.get('trenches');totals=ledger.get('totals')
    if not isinstance(fronts,list) or not isinstance(trenches,list):
        raise ValueError('Supply ledger requires source front and trench lists.')
    groups=defaultdict(list);seen=set()
    for front in fronts:
        amounts(front);key=source_key(front);parent=front.get('source_parent_index')
        if not identity(parent) or parent<0 or parent in seen:
            raise ValueError('Supply ledger requires unique source parent indices.')
        seen.add(parent);groups[key].append(front)
        witnesses=front.get('geometry_witnesses')
        if not isinstance(witnesses,list) or not witnesses:
            raise ValueError('Supply ledger front lacks source geometry witnesses.')
        witness_map={}
        for witness in witnesses:
            if not isinstance(witness,dict):raise ValueError('Invalid supply geometry witness.')
            index=witness.get('source_parent_index');ident=witness.get('source_trench_id')
            if (not identity(index) or index<0 or index in witness_map
                    or not identity(ident,nullable=True,positive=True)):
                raise ValueError('Invalid supply geometry witness identity.')
            witness_map[index]=ident
        if parent not in witness_map or witness_map[parent]!=key[0]:
            raise ValueError('Supply source identity disagrees with its chosen geometry witness.')
        geometry=front.get('source_geometry')
        if not isinstance(geometry,dict):raise ValueError('Supply front lacks its immutable source geometry.')
        arrays=[]
        for name in ('segment_start_xyz','segment_end_xyz','segment_normal_xyz'):
            array=np.asarray(geometry.get(name))
            if (array.ndim!=2 or array.shape[1:]!=(3,) or not len(array)
                    or not np.issubdtype(array.dtype,np.number) or not np.isfinite(array).all()
                    or not np.allclose(np.linalg.norm(array,axis=1),1.,rtol=1e-10,atol=1e-10)):
                raise ValueError('Invalid supply source contour coordinates or normals.')
            arrays.append(array)
        if not all(array.shape==arrays[0].shape for array in arrays):
            raise ValueError('Supply source contour arrays must align.')
        length=geometry.get('length_km')
        measured=RADIUS_KM*math.fsum(math.atan2(float(np.linalg.norm(np.cross(a,b))),float(a@b)) for a,b in zip(arrays[0],arrays[1]))
        if (isinstance(length,bool) or not isinstance(length,(int,float)) or not math.isfinite(length)
                or length<=0. or abs(length-measured)>max(1e-8,measured*2e-11)):
            raise ValueError('Supply source length differs from its saved contour.')
        maturity=front.get('source_nominal_maturity');speed=front.get('source_parent_normal_speed_km_myr')
        if (isinstance(maturity,bool) or not isinstance(maturity,(int,float)) or not math.isfinite(maturity)
                or not 0.<=maturity<=1. or isinstance(speed,bool) or not isinstance(speed,(int,float))
                or not math.isfinite(speed) or not identity(front.get('source_parent_boundary_code'))):
            raise ValueError('Invalid supply source maturity or parent motion context.')
    seen_trenches=set()
    for trench in trenches:
        amounts(trench);key=source_key(trench)
        if key in seen_trenches or key not in groups:
            raise ValueError('Supply ledger trench groups duplicate or omit source genealogy.')
        seen_trenches.add(key);rows=groups[key]
        parents=trench.get('source_parent_indices')
        if (not isinstance(parents,list) or not all(identity(parent) for parent in parents)
                or parents!=[row['source_parent_index'] for row in rows]):
            raise ValueError('Supply ledger trench must preserve its exact source parents.')
        for name in SUPPLY_AREAS:
            if abs(trench[name]-math.fsum(row[name] for row in rows))>tol:
                raise ValueError('Supply ledger trench amounts differ from its source fronts.')
    if seen_trenches!=set(groups):raise ValueError('Supply ledger omits a source trench group.')
    amounts(totals)
    for name in SUPPLY_AREAS:
        if abs(totals[name]-math.fsum(row[name] for row in fronts))>tol:
            raise ValueError('Supply ledger totals differ from its source fronts.')
    for name,saved in (
        ('raw_geometric_source_area_km2','raw_pair_capture_area_km2'),
        ('admitted_unique_water_area_km2','unique_downgoing_water_area_km2'),
        ('maturity_weighted_water_area_km2','maturity_weighted_water_area_km2'),
        ('local_water_before_union_km2','local_water_area_before_union_km2'),
        ('rejected_unrelated_owner_water_area_km2','rejected_unrelated_owner_water_area_km2'),
        ('duplicate_water_removed_km2','duplicate_capture_area_removed_km2'),
        ('requested_after_local_overlap_cap_km2','requested_removal_area_km2'),
        ('realized_removal_area_km2','removed_area_km2')):
        if abs(totals[name]-record[saved])>tol:
            raise ValueError('Supply ledger differs from the measured native sink: '+name)
    if 'removed_area_by_trench' in record:
        actual=defaultdict(list)
        for trench in trenches:
            actual[(trench['source_trench_id'],trench['source_downgoing_plate_uid'])].append(trench['realized_removal_area_km2'])
        feeds={(row['trench_id'],row['downgoing_plate_uid']):row['area_km2'] for row in record['removed_area_by_trench']}
        for key in actual.keys()|feeds.keys():
            if abs(math.fsum(actual[key])-feeds.get(key,0.))>tol:
                raise ValueError('Supply ledger differs from unchanged source-trench slab feed.')


def validate_frame(frame):
    """Reject malformed versioned sink evidence without changing legacy frames."""
    import normal_partition
    normal_partition.validate_frame(frame)
    geometry_version=frame.get('native_subduction_geometry_version',0)
    if isinstance(geometry_version,(bool,np.bool_)) or not isinstance(geometry_version,(int,np.integer)) or geometry_version not in (0,1):
        raise ValueError('Unsupported finite subduction geometry policy.')
    geometry_row=frame.get('native_subduction_diagnostics')
    supply_version=frame.get('native_subduction_supply_version',0)
    if (isinstance(supply_version,(bool,np.bool_)) or not isinstance(supply_version,(int,np.integer))
            or supply_version not in (0,SUPPLY_VERSION)):
        raise ValueError('Unsupported native subduction supply diagnostic version.')
    supply_marked=isinstance(geometry_row,dict) and 'supply_ledger' in geometry_row
    if supply_version:
        if frame.get('native_subduction_version')!=VERSION or not supply_marked:
            raise ValueError('Supply diagnostic version requires its measured supply ledger and native sink.')
    elif supply_marked:
        raise ValueError('Supply ledger requires its saved diagnostic version.')
    marked=isinstance(geometry_row,dict) and bool({'capture_geometry_version','capture_geometry_method'}.intersection(geometry_row))
    if geometry_version:
        if (frame.get('native_subduction_version')!=VERSION or not isinstance(geometry_row,dict)
                or isinstance(geometry_row.get('capture_geometry_version'),(bool,np.bool_))
                or not isinstance(geometry_row.get('capture_geometry_version'),(int,np.integer))
                or geometry_row.get('capture_geometry_version')!=1
                or geometry_row.get('capture_geometry_method')!='connected-front-positive-winding'):
            raise ValueError('Versioned capture geometry requires its exact measured policy tags.')
    elif marked:
        raise ValueError('Capture geometry diagnostic tags require their saved policy version.')
    version=frame.get('native_subduction_version',0)
    if isinstance(version,(bool,np.bool_)) or not isinstance(version,(int,np.integer)) or version not in (0,VERSION):
        raise ValueError('Unsupported native subduction version.')
    if not version:
        if 'native_subduction_diagnostics' in frame:
            raise ValueError('Finite native subduction diagnostics require their subduction version.')
        return
    row=frame.get('native_subduction_diagnostics')
    if not isinstance(row,dict) or row.get('version')!=VERSION or isinstance(row.get('version'),(bool,np.bool_)) or not isinstance(row.get('version'),(int,np.integer)):
        raise ValueError('Finite native subduction needs a versioned diagnostic record.')
    def number(name, *, nonnegative=True):
        value=row.get(name)
        if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.number))
                or not np.isfinite(value) or (nonnegative and value < -1e-5)):
            raise ValueError('Invalid finite subduction diagnostic: '+name)
        return float(value)
    for name in AREA_FIELDS:number(name)
    for name in ('capture_polygons','unique_water_parts'):
        value=row.get(name)
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<0:
            raise ValueError('Finite capture counts must be nonnegative integers.')
    for name in ('grid_radius_used','center_crust_gate_used','incoming_arrival_categories_used'):
        if row.get(name) is not False:raise ValueError('Finite capture cannot claim a categorical or radius operator.')
    if row.get('force_parent_graph_unchanged') is not True:
        raise ValueError('Finite capture must retain its parent force graph.')
    slots=np.asarray(row.get('owner_slots',[]));amount=np.asarray(row.get('removed_area_by_owner_km2',[]))
    if (slots.ndim!=1 or not len(slots) or not np.issubdtype(slots.dtype,np.integer)
            or np.any(slots<0) or np.any(np.diff(slots)<=0) or amount.shape!=slots.shape
            or not np.issubdtype(amount.dtype,np.number) or not np.isfinite(amount).all() or np.any(amount<0.)):
        raise ValueError('Finite capture owner rows need aligned finite nonnegative areas and unique integer slots.')
    for key in ('mesh_owner_slots','mesh_plate','native_boundary_owner_a','native_boundary_owner_b'):
        if key in frame and not np.isin(frame[key],slots).all():
            raise ValueError('Finite capture owner rows omit a recorded native owner.')
    scale=max(1.,max(number(key) for key in AREA_FIELDS));tol=max(1e-5,scale*2e-11)
    if 'removed_area_by_trench' in row:
        feeds=row['removed_area_by_trench'];seen=set();fed=[]
        if not isinstance(feeds,list):raise ValueError('Local slab feed must be a list of accepted trench contributions.')
        for feed in feeds:
            ident=feed.get('trench_id');uid=feed.get('downgoing_plate_uid')
            if (isinstance(ident,bool) or not isinstance(ident,int) or ident<=0 or ident in seen
                    or isinstance(uid,bool) or not isinstance(uid,int)):
                raise ValueError('Local slab feed needs unique positive trench IDs and integer incoming identities.')
            seen.add(ident)
            a,b=feed.get('area_km2'),feed.get('buoyancy_area_km2')
            if (not isinstance(a,(int,float)) or not isinstance(b,(int,float))
                    or not np.isfinite([a,b]).all() or a<0 or b<0 or b>1.8*a+tol):
                raise ValueError('Local slab feed exceeds its actual area or cooling buoyancy bound.')
            if 'neck_feeds' in feed:
                import slab_tether_local
                slab_tether_local.validate_feed(feed)
            fed.append(a)
        if abs(math.fsum(fed)-number('removed_area_km2'))>tol:
            raise ValueError('Local slab feeding differs from recorded actual ocean removal.')
    equalities=(
        (float(amount.sum()),number('removed_area_km2')),
        (number('local_water_area_before_union_km2')-number('unique_downgoing_water_area_km2'),number('duplicate_capture_area_removed_km2')),
        (number('maturity_weighted_water_area_km2')-number('removed_area_km2'),number('unsupported_or_unavailable_capture_area_km2')))
    if any(abs(a-b)>tol for a,b in equalities):
        raise ValueError('Finite subduction area components disagree.')
    bounds=(('local_water_area_before_union_km2','raw_pair_capture_area_km2'),
            ('unique_downgoing_water_area_km2','local_water_area_before_union_km2'),
            ('maturity_weighted_water_area_km2','unique_downgoing_water_area_km2'),
            ('requested_removal_area_km2','maturity_weighted_water_area_km2'),
            ('removed_area_km2','requested_removal_area_km2'),('removed_area_km2','available_total_overlap_km2'))
    if any(number(a)>number(b)+tol for a,b in bounds):
        raise ValueError('Finite subduction exceeded a measured water, maturity, request or overlap bound.')
    if (number('local_water_area_before_union_km2')+number('rejected_unrelated_owner_water_area_km2')
            >number('raw_pair_capture_area_km2')+tol):
        raise ValueError('Finite capture local and rejected water exceed the actual polygon area.')
    if ((number('raw_pair_capture_area_km2')>tol and row['capture_polygons']==0)
            or (number('unique_downgoing_water_area_km2')>tol and row['unique_water_parts']==0)):
        raise ValueError('Nonzero capture area requires actual polygon parts.')
    for key in ('maximum_support_bound_residual','maximum_shared_excess_bound_residual'):
        if abs(number(key,nonnegative=False))>1e-12:
            raise ValueError('Finite capture violated its per-cell support bound.')
    start,end,dt=(number(key) for key in ('step_start_myr','step_end_myr','step_duration_myr'))
    if min(start,end,dt)<0. or abs(end-start-dt)>1e-9:
        raise ValueError('Finite capture step times are inconsistent.')
    state=row.get('state')
    if state=='initialized':
        if (end!=0. or start!=0. or dt!=0. or any(number(key)!=0. for key in AREA_FIELDS) or np.any(amount)
                or row['capture_polygons']!=0 or row['unique_water_parts']!=0):
            raise ValueError('Unstepped finite capture must have zero elapsed time and zero sinks.')
    elif state!='measured_step' or dt<=0.:
        raise ValueError('Finite capture needs a positive measured step or explicit initial state.')
    epoch=frame.get('time_myr')
    if epoch is not None and (isinstance(epoch,(bool,np.bool_)) or not isinstance(epoch,(int,float,np.number))
            or not np.isfinite(epoch) or abs(float(epoch)-end)>1e-6):
        raise ValueError('Finite capture record does not end at its saved epoch.')
    if supply_version:_validate_supply_ledger(row,tol)
