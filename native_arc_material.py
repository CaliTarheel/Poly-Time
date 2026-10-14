"""Connected volcanic foundations with explicit lateral magma-volume budgets.

New arcs are finite-width two-ring material patches. Growth moves shared
outer-band vertices once and mixes supplied juvenile crust with the old column
volume. A mountain's current thickness never determines the magma source.
"""
from __future__ import annotations

import numpy as np
import normal_partition
import material_surface
import structure_engine
import crustal_structure as columns

RADIUS_KM = 6371.
JUVENILE_THICKNESS_KM = 25.
MIN_PATCH_AREA_KM2 = .001
# Report keys held OUT of per-owner event details.
#
# Both island_arc recorders emit one event per receiving owner and embed the
# whole shared `report` in each. That report carries emplacement_geometry, whose
# `sources` list ran to 3,207 entries and 7.2 MB at 340 Myr -- so a three-owner
# step wrote three copies of it into an append-only event log, and every later
# frame re-serialised all of them. Ten such events held 98% of a 51.6 MB frame,
# and one 5.4 MB payload recorded at 300 Myr was still being rewritten forty
# frames later.
#
# The canonical copy stays in s.arc_material_diagnostics, which is where every
# consumer already reads it: arc_birth_profile.py:240, arc_emplacement_geometry
# .py:278, and the frame validators in mesh_history and native_frame_sampling.
# Nothing reads it from an event, so nothing loses information here.
EVENT_DETAIL_EXCLUDED = ('emplacement_geometry',)


def _event_details(report, **extra):
    """Per-owner event details, without the shared bulk diagnostics."""
    return dict({k: v for k, v in report.items() if k not in EVENT_DETAIL_EXCLUDED}, **extra)
RAYS = 8


def _unit(value):
    value=np.asarray(value,float)
    return value/np.maximum(np.linalg.norm(value,axis=-1,keepdims=True),1e-30)


def ensure_fields(s):
    count=len(s.mass)
    for name,dtype in (('parcel_arc_id',np.int64),('parcel_arc_basal_m',float)):
        old=getattr(s,name,None)
        if old is None:setattr(s,name,np.zeros(count,dtype=dtype))
        elif len(old)!=count:raise ValueError('Arc provenance must align with material faces.')
    s.native_arc_material_version=1
    if not hasattr(s,'next_arc_material_id'):
        s.next_arc_material_id=int(s.parcel_arc_id.max(initial=0))+1
    if not hasattr(s,'native_arc_birth_faces'):s.native_arc_birth_faces=0


def snapshot_fields(s):
    ensure_fields(s)
    result=dict(arc_material_version=1,material_arc_id=s.parcel_arc_id.copy(),
                material_arc_basal_m=s.parcel_arc_basal_m.copy())
    policy=getattr(s,'native_arc_emplacement_version',0)
    if isinstance(policy,(bool,np.bool_)) or not isinstance(policy,(int,np.integer)) or policy not in (0,1):
        raise ValueError('Unsupported juvenile emplacement policy.')
    if policy==1:
        import arc_emplacement_geometry as emplacement
        result['arc_emplacement_version']=1
        pending=getattr(s,'native_arc_pending',dict(xyz=np.empty((0,3)),owner=np.empty(0,np.int16),area=np.empty(0)))
        result['arc_pending_source']=emplacement.pending_snapshot(pending)
    import arc_birth_profile
    result.update(arc_birth_profile.snapshot_fields(s))
    import arc_birth_footprint
    result.update(arc_birth_footprint.snapshot_fields(s))
    import arc_local_deposition
    result.update(arc_local_deposition.snapshot_fields(s))
    import arc_point_nucleation
    result.update(arc_point_nucleation.snapshot_fields(s))
    return result


def _solve_scale(evaluate, low, high, target, *, low_area=None, high_area=None, seed=None):
    """Return the scale whose spherical faces sum to `target`.

    Area is monotone in the scale and grows as its square in the planar limit.
    A secant step on that smooth curve converges superlinearly, so this reaches
    the answer in a handful of evaluations where the fifty-two step bisection it
    replaces needed fifty-two. Each evaluation rebuilds the whole patch's
    spherical areas, and this runs thousands of times per step, so the count is
    the cost.

    A bracket is carried throughout. A secant step that leaves it, or that
    cannot be formed because the curve is locally flat, is replaced by the
    bisection step, so a stalled or badly conditioned curve still converges.
    """
    area_low = evaluate(low)[1].sum() if low_area is None else low_area
    area_high = evaluate(high)[1].sum() if high_area is None else high_area
    if not area_low <= target <= area_high:
        raise ValueError('The requested area is not bracketed by the supplied scales.')
    # A relative area residual of 1e-11 moves a vertex by far less than a
    # millimetre on a 30 km patch, well below anything the model represents,
    # and costs several evaluations fewer than chasing the last digits the
    # bisection happened to land on.
    tolerance = max(abs(target)*1e-11, 1e-9)
    resolution = (high-low)*1e-15
    history = []
    guess = seed if seed is not None and low < seed < high else None
    for _ in range(64):
        if high-low <= resolution:
            break
        if guess is None:
            guess = _secant(history, low, high, (low, area_low-target), (high, area_high-target))
        area = evaluate(guess)[1].sum()
        if abs(area-target) <= tolerance:
            return guess
        history = [*history[-1:], (guess, area-target)]
        if area < target:
            low, area_low = guess, area
        else:
            high, area_high = guess, area
        guess = None
    return (low+high)*.5


def _secant(history, low, high, *ends):
    """A secant step through the last two evaluations, else the bracket ends.

    Falling back to the midpoint of a bracket whose upper end is far beyond the
    answer wastes the first evaluations, so before there is any history the step
    interpolates the bracket instead, which is already close for a monotone
    curve. Anything non-finite or outside the bracket bisects.
    """
    middle = (low+high)*.5
    (x0, f0), (x1, f1) = history[-2:] if len(history) >= 2 else ends
    if f1 == f0 or not np.isfinite(f1-f0):
        return middle
    step = x1-f1*(x1-x0)/(f1-f0)
    margin = (high-low)*1e-9
    return step if low+margin < step < high-margin and np.isfinite(step) else middle


def _patch(centre,direction,area):
    """Exact-area convex spherical oval with an interior and a shared apron."""
    centre=_unit(centre);along=_unit(direction-centre*np.dot(centre,direction))
    if np.linalg.norm(along)<.5:
        along=_unit(np.cross(centre,[0.,0.,1.] if abs(centre[2])<.9 else [0.,1.,0.]))
    across=np.cross(centre,along)
    phi=np.arange(RAYS)*2*np.pi/RAYS
    # Broad asymmetry removes a repeated regular-star seed without stochastic
    # per-step jitter. The oval's long axis is the local trench strike.
    modulation=1.+.075*np.cos(3*phi+.4)
    xy=np.column_stack((1.5*np.cos(phi),np.sin(phi)/1.5))*modulation[:,None]
    coordinates=np.vstack((np.zeros((1,2)),.48*xy,xy))
    radius=np.r_[0.,np.full(RAYS,.48),np.ones(RAYS)]
    faces=[]
    for i in range(RAYS):
        j=(i+1)%RAYS
        faces.extend(((0,1+i,1+j),(1+i,1+RAYS+i,1+RAYS+j),(1+i,1+RAYS+j,1+j)))
    faces=np.asarray(faces,np.int64)
    tangents=coordinates[:,0,None]*along+coordinates[:,1,None]*across
    distance=np.linalg.norm(tangents,axis=1)
    bearing=tangents/np.maximum(distance[:,None],1e-30)
    def evaluate(scale):
        angles=distance*scale
        vertices=centre*np.cos(angles)[:,None]+bearing*np.sin(angles)[:,None]
        vertices[0]=centre
        return vertices,material_surface.spherical_face_areas(vertices,faces)
    low,high=0.,1.1/distance.max()
    outer=evaluate(high)[1].sum()
    if not 0<area<outer:raise ValueError('Arc seed must fit within a minor spherical cap.')
    # Area is quadratic in the scale for a small cap, so this seed is already
    # close and the bracketed solve only has to polish it.
    vertices,areas=evaluate(_solve_scale(evaluate,low,high,area,low_area=0.,high_area=outer,
                                         seed=high*np.sqrt(area/outer)))
    return dict(vertices=vertices,faces=faces,area_km2=areas,
                reference_area_km2=areas*(area/areas.sum()),profile_radius=radius[faces].mean(axis=1))


def _strike(s,cell,owner,position=None):
    centre=s.xyz[cell] if position is None else position
    eligible=np.flatnonzero(normal_partition.subduction(s)&((s.bp==owner)|(s.bq==owner))) if hasattr(s,'bcode') else np.empty(0,int)
    if len(eligible):
        edge=eligible[np.argmax(s.bmid[eligible]@centre)]
        return np.cross(s.bmid[edge],s.bn[edge])
    return s.native_mesh['vertices'][s.native_mesh['faces'][cell,0]]


def _basement(s,cell):
    # The same thermal ocean floor law used by the engine, sampled at birth.
    age=max(float(s.age[cell]),0.)
    depth=-2600.-365.*np.sqrt(age) if age<20. else -5651.+2473.*np.exp(-.0278*age)
    return min(-200.,depth+float(s.ocean_relief[cell]))


def _grow_geometry(surface,selected,addition):
    """Expand a connected shared outer band without moving its central core."""
    faces=surface['faces'][selected]
    vertex_ids,local=np.unique(faces,return_inverse=True)
    faces=local.reshape(-1,3)
    points=surface['vertices'][vertex_ids]
    old_area=surface['area_km2'][selected]
    centre=_unit(np.sum(_unit(points[faces].sum(axis=1))*old_area[:,None],axis=0))
    dot=points@centre
    tangent=points-centre*dot[:,None]
    angle=np.arctan2(np.linalg.norm(tangent,axis=1),dot)
    if angle.max(initial=0.)>=1.1:return None
    bearing=_unit(tangent)
    # The anchored inner core does not expand laterally merely because magma
    # arrived at its surrounding apron. Refined vertices follow the same map.
    edges=np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    boundary=unique[counts==1]
    if not len(boundary):return None
    boundary_vertices=np.unique(boundary)
    width=.3*angle[boundary_vertices].min()
    distance=np.full(len(points),np.inf)
    for a,b in boundary:
        middle=_unit(points[a]+points[b]);normal=_unit(np.cross(points[a],points[b]))
        tangent=np.cross(normal,middle)
        half=.5*np.arccos(np.clip(np.dot(points[a],points[b]),-1.,1.))
        along=np.arctan2(points@tangent,points@middle)
        across=np.arcsin(np.clip(np.abs(points@normal),0.,1.))
        to_edge=np.where(np.abs(along)<=half,across,
            np.arccos(np.clip(np.maximum(points@points[a],points@points[b]),-1.,1.)))
        distance=np.minimum(distance,to_edge)
    weight=np.clip(1.-distance/max(width,1e-30),0.,1.)
    weight[boundary_vertices]=1.
    band=angle*weight
    target=float(old_area.sum()+addition)
    def evaluate(scale):
        radius=angle+(scale-1.)*band
        result=centre*np.cos(radius)[:,None]+bearing*np.sin(radius)[:,None]
        fixed=band==0.;result[fixed]=points[fixed]
        return result,material_surface.spherical_face_areas(result,faces)
    high=1.+max(0.,1.1-angle.max())/max(band.max(initial=0.),1e-30)
    outer=evaluate(high)[1].sum()
    if outer<target:return None
    low=1.
    vertices,areas=evaluate(_solve_scale(evaluate,low,high,target,
                                         low_area=float(old_area.sum()),high_area=outer))
    signed=np.einsum('fi,fi->f',vertices[faces[:,0]],np.cross(vertices[faces[:,1]],vertices[faces[:,2]]))
    delta=areas-old_area
    if np.any(signed<=1e-15) or np.any(delta < -old_area*.01):
        return None
    # Tiny spherical-area cancellation is corrected only in the accounting
    # fractions; physical geometry remains the exact solved spherical patch.
    delta=np.maximum(delta,0.);delta*=addition/delta.sum()
    return dict(vertex_ids=vertex_ids,vertices=vertices,area_km2=areas,added_area_km2=delta)


def _grow(s,selected,addition,*,prepared_plan=None):
    surface=s.material_surface
    plan=_grow_geometry(surface,selected,addition) if prepared_plan is None else prepared_plan
    if plan is None:return None
    old_triangles=surface['vertices'][surface['faces'][selected]].copy()
    old_area=surface['area_km2'][selected].copy()
    old_mass=s.mass[selected].copy()
    old_t=s.structure['thickness_km'][selected].copy()
    extra=plan['added_area_km2'];new_mass=old_mass+extra
    old_volume=float(old_area@old_t)
    supplied=float(extra.sum()*JUVENILE_THICKNESS_KM)
    thickness=(old_area*old_t+extra*JUVENILE_THICKNESS_KM)/plan['area_km2']
    import crust_inventory
    candidate={name:np.asarray(value)[selected].copy() for name,value in s.structure.items()}
    candidate['thickness_km']=thickness.copy()
    candidate['area_factor']=plan['area_km2']/new_mass
    import numerical_accuracy
    if numerical_accuracy.version(s):candidate[numerical_accuracy.AREA]=plan['area_km2'].copy()
    crust_inventory.grow_reference(candidate,np.arange(len(selected)),old_mass,new_mass,
                                    extra*JUVENILE_THICKNESS_KM)
    try:columns._state(candidate)
    except ValueError:return None
    before_elevation=columns.elevation({name:s.structure[name][selected] for name in s.structure})
    patches=s.parcel_patch[selected]
    trace=np.flatnonzero(np.isin(s.trace_patch,patches))
    order=np.argsort(patches)
    local=order[np.searchsorted(patches[order],s.trace_patch[trace])]
    trace_thickness=(old_area[local]*s.trace_structure['thickness_km'][trace]+extra[local]*JUVENILE_THICKNESS_KM)/plan['area_km2'][local]
    trace_candidate={name:np.asarray(value)[trace].copy() for name,value in s.trace_structure.items()}
    trace_candidate['thickness_km']=trace_thickness.copy()
    trace_candidate['area_factor']=plan['area_km2'][local]/new_mass[local]
    if numerical_accuracy.version(s):trace_candidate[numerical_accuracy.AREA]=plan['area_km2'][local].copy()
    crust_inventory.grow_reference(trace_candidate,np.arange(len(trace)),old_mass[local],new_mass[local],
                                    extra[local]*JUVENILE_THICKNESS_KM)
    try:columns._state(trace_candidate)
    except ValueError:return None
    weights=np.linalg.solve(old_triangles[local].transpose(0,2,1),s.trace_xyz[trace,:,None])[...,0]
    weights/=weights.sum(axis=1)[:,None]
    if np.any(weights < -1e-7):raise ValueError('Arc growth marker escaped its material triangle.')
    surface['vertices'][plan['vertex_ids']]=plan['vertices']
    material_surface.refresh_geometry(surface)
    surface['reference_area_km2'][selected]=new_mass
    s.mass[selected]=new_mass
    s.structure['thickness_km'][selected]=thickness
    s.structure['area_factor'][selected]=surface['area_km2'][selected]/new_mass
    if numerical_accuracy.version(s):s.structure[numerical_accuracy.AREA][selected]=surface['area_km2'][selected]
    crust_inventory.grow_reference(s.structure,selected,old_mass,new_mass,extra*JUVENILE_THICKNESS_KM)
    # Like the composition inventory, this ledger is volume per reference area. Cumulative denudation,
    # rebound and thermal surface metres belong to enduring chart points;
    # diluting them would invent negative erosion in history transport.
    s.structure['added_volume_km_per_reference_km2'][selected]*=old_mass/new_mass
    s.structure['added_volume_km_per_reference_km2'][selected]+=extra*JUVENILE_THICKNESS_KM/new_mass
    after_elevation=columns.elevation({name:s.structure[name][selected] for name in s.structure})
    s.relief[selected]+=after_elevation-before_elevation
    new_triangles=surface['vertices'][surface['faces'][selected]]
    s.trace_xyz[trace]=_unit(np.einsum('ni,nij->nj',weights,new_triangles[local]))
    from native_material_evolution import _transport_tangent
    s.trace_rift_tangent[trace]=_transport_tangent(old_triangles[local],new_triangles[local],s.trace_rift_tangent[trace],s.trace_xyz[trace])
    # Each retained point keeps its own history; its newly mixed thickness
    # follows the local realized area and the same juvenile-volume source.
    if len(trace):
        before=columns.elevation({name:s.trace_structure[name][trace] for name in s.trace_structure})
        s.trace_structure['thickness_km'][trace]=trace_thickness
        s.trace_structure['area_factor'][trace]=s.structure['area_factor'][selected[local]]
        if numerical_accuracy.version(s):s.trace_structure[numerical_accuracy.AREA][trace]=surface['area_km2'][selected[local]]
        crust_inventory.grow_reference(s.trace_structure,trace,old_mass[local],new_mass[local],extra[local]*JUVENILE_THICKNESS_KM)
        s.trace_structure['added_volume_km_per_reference_km2'][trace]*=old_mass[local]/new_mass[local]
        s.trace_structure['added_volume_km_per_reference_km2'][trace]+=extra[local]*JUVENILE_THICKNESS_KM/new_mass[local]
        after=columns.elevation({name:s.trace_structure[name][trace] for name in s.trace_structure})
        s.trace_relief_m[trace]+=after-before
        s.trace_adjustment_m[trace]+=after-before
    actual=float(surface['area_km2'][selected]@s.structure['thickness_km'][selected])
    if not np.isclose(actual,old_volume+supplied,rtol=1e-11,atol=1e-7):
        raise ValueError('Arc growth failed its explicit magma-volume budget.')
    return dict(added_volume_km3=supplied,volume_error_km3=actual-old_volume-supplied,
                moved_vertices=len(plan['vertex_ids']),retained_faces=len(selected))


def _birth(s,cell,owner,amount,*,position=None,basal_m=None,fixed_source_volume=False,prepared_patch=None):
    import arc_surface
    import arc_birth_profile
    centre=s.xyz[cell] if position is None else position
    patch=_patch(centre,_strike(s,cell,owner,position),amount) if prepared_patch is None else prepared_patch
    # Geographic admission certifies the once-normalized footprint that
    # append_surface stores. Reject a stale/tampered plan before any state or
    # source identifiers are advanced; legacy non-geographic births are unmarked.
    if '_emplacement_stored_triangle_sha256' in patch:
        import hashlib
        stored=np.asarray(patch['vertices'],float)
        stored=stored/np.linalg.norm(stored,axis=1)[:,None]
        digest=hashlib.sha256(stored[np.asarray(patch['faces'])].tobytes()).hexdigest()
        if digest!=patch['_emplacement_stored_triangle_sha256']:
            raise ValueError('Admitted juvenile stored footprint changed before birth.')
    count=len(patch['faces']);start=len(s.mass)
    basal=_basement(s,cell) if basal_m is None else float(basal_m)
    physical_profile=arc_birth_profile.version(s)==1
    if physical_profile:
        candidate=arc_birth_profile.birth(patch,amount,basal)
        if not candidate['diagnostics']['admissible']:
            raise ValueError('Unadmitted juvenile birth exceeds its physical profile capacity.')
    ids=np.arange(s.next_patch_uid,s.next_patch_uid+count,dtype=np.int64)
    arc_id=int(s.next_arc_material_id);s.next_arc_material_id+=1;s.next_patch_uid+=count
    profile=arc_surface.birth_profile(patch['profile_radius'],basal)
    if physical_profile:profile=candidate
    height=np.asarray(profile['height_m']);thickness=np.asarray(profile['thickness_km'])
    if fixed_source_volume:
        # Birth and growth consume the same magma volume per supplied area.
        # Keep the relative column profile and its reference topography, but
        # normalize against the ACTUAL spherical footprint, not face count.
        thickness*=amount*JUVENILE_THICKNESS_KM/float(patch['area_km2']@thickness)
        if np.any(thickness<columns.MIN_THICKNESS_KM) or np.any(thickness>columns.MAX_THICKNESS_KM):
            raise ValueError('The fixed-volume juvenile birth profile exceeds physical column bounds.')
    surface=s.material_surface
    column_fields={name:np.zeros((count,)+value.shape[1:],dtype=value.dtype) for name,value in surface['columns'].items()}
    provenance={name:np.zeros((count,)+value.shape[1:],dtype=value.dtype) for name,value in surface['provenance'].items()}
    if 'source_face_id' in provenance:provenance['source_face_id']=ids.copy()
    material_surface.append_surface(surface,patch['vertices'],patch['faces'],np.full(count,owner,int),
        np.full(count,3,np.uint8),face_id=ids,columns=column_fields,provenance=provenance,
        reference_area_km2=patch['reference_area_km2'])
    if fixed_source_volume:
        # Appending normalizes spherical vertices. For a small newborn patch,
        # that last rounding changes its physical areas enough to matter to
        # the explicit source ledger. Use the final stored footprint before
        # assigning any parcel or marker columns; retain the source guard.
        thickness*=amount*JUVENILE_THICKNESS_KM/float(surface['area_km2'][start:]@thickness)
        if np.any(thickness<columns.MIN_THICKNESS_KM) or np.any(thickness>columns.MAX_THICKNESS_KM):
            raise ValueError('The stored fixed-volume juvenile profile exceeds physical column bounds.')
    if physical_profile:
        profile=arc_birth_profile.birth(patch,amount,basal,area_km2=surface['area_km2'][start:])
        thickness=profile['thickness_km'];height=profile['height_m']
        if not profile['diagnostics']['admissible']:
            raise ValueError('Stored juvenile footprint exceeded its physical profile capacity.')
    reference=surface['area_km2'][start:]*(amount/surface['area_km2'][start:].sum())
    surface['reference_area_km2'][start:]=reference
    values=dict(pos=material_surface.face_centres(surface)[start:],mass=reference,
        parcel_plate=np.full(count,owner,dtype=s.parcel_plate.dtype),kind=np.full(count,3,np.uint8),
        relief=height-120.,suture=np.full(count,.55),parcel_birth=np.full(count,s.t),
        parcel_patch=ids,parcel_craton=np.full(count,-1,np.int32),parcel_arc_id=np.full(count,arc_id,np.int64),
        parcel_arc_basal_m=np.full(count,basal),**s._empty_rift_material(count))
    for name,value in values.items():setattr(s,name,np.concatenate((getattr(s,name),value)))
    s.native_arc_birth_faces+=count
    structure_engine.append_parcels(s,count)
    for name in ('thickness_km','reference_thickness_km'):s.structure[name][start:]=thickness
    s.structure['reference_elevation_m'][start:]=height
    s.structure['area_factor'][start:]=surface['area_km2'][start:]/s.mass[start:]
    s.structure['added_volume_km_per_reference_km2'][start:]=thickness*s.structure['area_factor'][start:]
    import crust_inventory
    if crust_inventory.present(s.structure):
        import dense_crust
        # Replace only the neutral newborn placeholder, after its final profile
        # and footprint are known. No pre-existing inventory is reset.
        newborn={name:value[start:].copy() for name,value in s.structure.items()
                 if name not in crust_inventory.FIELDS+dense_crust.FIELDS}
        crust_inventory.initialize(newborn)
        for name in crust_inventory.FIELDS:s.structure[name][start:]=newborn[name]
        if dense_crust.present(s.structure):
            dense_crust.initialize(newborn, 1200.)
            for name in dense_crust.FIELDS:s.structure[name][start:]=newborn[name]
    before=len(s.trace_patch)
    s._new_arc_traces(np.full(count,cell,int),np.full(count,owner,int),ids)
    if len(s.trace_patch)>before:
        selected=start+np.searchsorted(ids,s.trace_patch[before:])
        s.trace_xyz[before:]=s.pos[selected]
        s.trace_relief_m[before:]=s.relief[selected]
        for name in s.structure:s.trace_structure[name][before:]=s.structure[name][selected]
    result=dict(new_faces=count,added_volume_km3=float(surface['area_km2'][start:]@thickness),arc_id=arc_id)
    if physical_profile:result['birth_profile']=profile['diagnostics']
    return result


def _cell_add_arc_crust(s,cells,additions):
    ensure_fields(s)
    cells,additions=np.asarray(cells),np.asarray(additions,float)
    if (cells.ndim!=1 or cells.dtype.kind not in 'iu' or additions.shape!=cells.shape or
        np.any(cells<0) or np.any(cells>=s.n) or not np.isfinite(additions).all() or np.any(additions<0)):
        raise ValueError('Arc additions need valid native cells and finite nonnegative areas.')
    pending=getattr(s,'native_arc_pending',dict(xyz=np.empty((0,3)),owner=np.empty(0,np.int16),area=np.empty(0)))
    pc=s._indices(pending['xyz']) if len(pending['owner']) else np.empty(0,int)
    incoming_cells=np.r_[cells,pc];incoming_owners=np.r_[s.plate[cells],pending['owner']]
    incoming_area=np.r_[additions,pending['area']]
    positive=incoming_area>0
    keys,inverse=np.unique(incoming_owners[positive].astype(np.int64)*s.n+incoming_cells[positive],return_inverse=True)
    amount=np.bincount(inverse,weights=incoming_area[positive],minlength=len(keys))
    cells,owners=(keys%s.n).astype(int),(keys//s.n).astype(int)
    refs=np.bincount(s.material_surface['faces'].ravel(),minlength=len(s.material_surface['vertices']))
    by_key={}
    for arc_id in np.unique(s.parcel_arc_id[s.parcel_arc_id>0]):
        selected=np.flatnonzero(s.parcel_arc_id==arc_id)
        if not len(selected) or np.any(s.kind[selected]!=3) or np.any(s.parcel_plate[selected]!=s.parcel_plate[selected[0]]):continue
        vertex_ids,counts=np.unique(s.material_surface['faces'][selected],return_counts=True)
        if np.any(refs[vertex_ids]!=counts):continue
        centre=_unit(np.sum(s.pos[selected]*s.material_surface['area_km2'][selected,None],axis=0))
        key=int(s.parcel_plate[selected[0]])*s.n+int(s._indices(centre[None])[0])
        by_key.setdefault(key,selected)
    unresolved=[];new_faces=0;grown=0;added=0.;volume=0.;errors=[];owner_totals={}
    for cell,owner,area,key in zip(cells,owners,amount,keys):
        if area<MIN_PATCH_AREA_KM2:
            unresolved.append((cell,owner,area));continue
        selected=by_key.get(int(key))
        if selected is not None:
            result=_grow(s,selected,float(area))
            if result is None:
                unresolved.append((cell,owner,area));continue
            grown+=1;errors.append(result['volume_error_km3'])
        else:
            result=_birth(s,int(cell),int(owner),float(area));new_faces+=result['new_faces']
        added+=float(area);volume+=result['added_volume_km3']
        owner_total=owner_totals.setdefault(int(owner),dict(added_area_km2=0.,added_volume_km3=0.))
        owner_total['added_area_km2']+=float(area);owner_total['added_volume_km3']+=result['added_volume_km3']
    s.native_arc_pending=dict(xyz=np.array([s.xyz[cell] for cell,_,_ in unresolved]).reshape(-1,3),
        owner=np.array([owner for _,owner,_ in unresolved],np.int16),area=np.array([area for _,_,area in unresolved]))
    s.process_totals['arc_added_km2']+=added
    s.process_totals['arc_added_volume_km3']=s.process_totals.get('arc_added_volume_km3',0.)+volume
    s._sync_material();s._coverage_signature=None;s._owner_occupancy_signature=None
    for name in ('_material_occupancy_hits','_material_coverage','exposed_material'):
        if hasattr(s,name):delattr(s,name)
    report=dict(added_area_km2=added,added_volume_km3=volume,pending_area_km2=float(s.native_arc_pending['area'].sum()),
        new_faces=new_faces,grown_patches=grown,patch_count=len(np.unique(s.parcel_arc_id[s.parcel_arc_id>0])),
        maximum_growth_volume_error_km3=max(map(abs,errors),default=0.),juvenile_thickness_km=JUVENILE_THICKNESS_KM,
        representation='connected finite-width volcanic cores, flanks and submerged aprons')
    s.arc_material_diagnostics=report
    if added:
        for owner in sorted(owner_totals):
            take=owners==owner
            s._record('island_arc',f'Subduction magma built connected volcanic core and apron material on {s.names[owner]}.',
                ('arc',int(s.plate_uid[owner]),int(s.t//100)),plates=(int(owner),),
                xyz=np.sum(s.xyz[cells[take]]*amount[take,None],axis=0),details=_event_details(report,**owner_totals[owner]))
    return report


def add_arc_crust(s,cells,additions,*,positions=None,owners=None,source_provenance=None):
    """Place supplied juvenile area at real sources, retaining the old API.

    Explicit positions activate geographic routing. Cell-only calls retain
    their historical semantics for old experiments and direct legacy callers.
    A source over an existing native arc grows only that containing arc ID;
    separate ocean points never coalesce because they share a cell or owner.
    """
    policy=getattr(s,'native_arc_emplacement_version',0)
    import arc_birth_profile
    profile_version=arc_birth_profile.version(s)
    import arc_local_deposition
    deposition_version=arc_local_deposition.version(s)
    import arc_point_nucleation
    point_version=arc_point_nucleation.version(s)
    if isinstance(policy,(bool,np.bool_)) or not isinstance(policy,(int,np.integer)) or policy not in (0,1):
        raise ValueError('Unsupported juvenile emplacement policy.')
    if positions is None:
        if owners is not None:raise ValueError('Explicit arc owners require actual source positions.')
        if getattr(s,'native_arc_emplacement_version',0)!=1:
            return _cell_add_arc_crust(s,cells,additions)
        supplied_cells=np.asarray(cells)
        if (supplied_cells.ndim!=1 or supplied_cells.dtype.kind not in 'iu'
                or np.any(supplied_cells<0) or np.any(supplied_cells>=s.n)):
            raise ValueError('Arc additions need valid native cells.')
        # A new marked world cannot bypass finite-footprint admission through
        # its compatibility API. That API declares a cell-centre source;
        # historical unmarked states retain their original implementation.
        positions=s.xyz[supplied_cells].copy();owners=s.plate[supplied_cells].copy()
    import arc_source_geometry as sources
    cells=np.asarray(cells);additions=np.asarray(additions,float)
    if (cells.ndim!=1 or cells.dtype.kind not in 'iu' or additions.shape!=cells.shape
            or np.any(cells<0) or np.any(cells>=s.n)
            or not np.isfinite(additions).all() or np.any(additions<0)):
        raise ValueError('Arc additions need valid native cells and finite nonnegative areas.')
    if owners is None:raise ValueError('Actual arc source positions require intended owners.')
    points,owners=sources.validate_positions(positions,owners,len(cells),len(s.support))
    ensure_fields(s)
    pending=getattr(s,'native_arc_pending',dict(xyz=np.empty((0,3)),owner=np.empty(0,np.int16),area=np.empty(0)))
    pp,po=sources.validate_positions(pending['xyz'],pending['owner'],len(pending['area']),len(s.support))
    if getattr(s,'native_arc_emplacement_version',0)==1:
        import arc_emplacement_geometry as emplacement
        emplacement.validate_pending(pending)
    pa=np.asarray(pending['area'],float)
    if not np.isfinite(pa).all() or np.any(pa<0):raise ValueError('Pending arc source area must be finite and nonnegative.')
    incoming=float(additions.sum());prior=float(pa.sum())
    all_points=np.vstack((points,pp));all_owners=np.r_[owners,po]
    all_area=np.r_[additions,pa];old_pending=np.arange(len(all_area))>=len(additions)
    route=sources.classify(s,all_points,all_owners)
    all_provenance=None;host_invalid=np.zeros(len(all_area),bool)
    if profile_version:
        import arc_source_cohorts as source_cohorts
        import arc_cohort_footprint as cohort_geometry
        if len(pa) and 'source_provenance' not in pending:
            raise ValueError('Pending magma without proven source provenance cannot be upgraded to the new profile.')
        current=source_cohorts.normalize(s,source_provenance,len(points),owners=owners,areas=additions)
        all_provenance=source_cohorts.concat(current,pending.get('source_provenance',[]))
        host_invalid=~source_cohorts.host_mask(s,all_owners,all_provenance)
        route['eligible'][host_invalid]=False;route['reason'][host_invalid]='inactive_or_changed_source_host'
    placement_version=getattr(s,'native_arc_emplacement_version',0)
    if isinstance(placement_version,bool) or placement_version not in (0,1):
        raise ValueError('Unsupported juvenile emplacement policy.')
    placement=None;placement_rows=[]
    if placement_version==1:
        import arc_emplacement_geometry as emplacement
        placement=emplacement.prepare(s)
    # A newly rejected candidate never becomes a juvenile source. Previously
    # supplied pending magma remains accounted for at its own moving position.
    rejected=~route['eligible']&~old_pending&~host_invalid
    rejected_area=float(all_area[rejected].sum())
    unresolved=[(all_points[i].copy(),int(all_owners[i]),float(all_area[i]),int(i))
                for i in np.flatnonzero(~route['eligible']&(old_pending|host_invalid)&(all_area>0))]
    refs=np.bincount(s.material_surface['faces'].ravel(),minlength=len(s.material_surface['vertices']))
    growable={};component_at_face=np.full(len(s.mass),-1,np.int64)
    for arc_id in np.unique(route['arc_id'][route['eligible']]):
        if arc_id<=0:continue
        selected=np.flatnonzero(s.parcel_arc_id==arc_id)
        # A persistent birth ID can outlive a physical fracture. Actual shared
        # edges and owner identity define each independently growable piece.
        from collision_contacts import _new_components
        components=_new_components(s.material_surface['faces'][selected],s.parcel_plate[selected])
        for component in np.unique(components):
            piece=selected[components==component];identity=int(piece[0])
            component_at_face[piece]=identity
            vertices,counts=np.unique(s.material_surface['faces'][piece],return_counts=True)
            if np.all(s.kind[piece]==3) and np.all(refs[vertices]==counts):
                growable[(int(arc_id),identity)]=piece
    # Freeze routing before mutation. Existing patches collect their actual
    # contained source contributions; exact duplicate ocean points can combine.
    # A genuine source history can connect distinct geographic observations.
    # Local membership still requires the final physical birth footprint and
    # the induced adjacency graph, never owner, distance or trench ID alone.
    basal=sources.basement(s,all_points) if len(all_points) else np.empty(0)
    import arc_birth_footprint
    footprint_version=arc_birth_footprint.version(s)
    birth_candidates={}
    def birth_candidate(anchor,amount,strategy=None):
        # Shared cohort and admission queries reuse the exact funded proposal.
        # The cache lasts only this immutable routing batch, not later epochs.
        if strategy is None:
            if point_version and all_provenance[anchor]['valid']:
                compact=birth_candidate(anchor,amount,'point')
                if compact['profile']['diagnostics']['admissible']:return compact
            strategy='legacy'
        identity=(int(anchor),float(amount),strategy)
        if identity not in birth_candidates:
            position=all_points[anchor];cell=int(s._indices(position[None])[0])
            strike=_strike(s,cell,int(all_owners[anchor]),position)
            factory=lambda physical_area:_patch(position,strike,physical_area)
            if strategy=='point':
                seed=arc_point_nucleation.seed_key(s,[all_provenance[anchor]])
                proposal=arc_point_nucleation.propose(position,strike,amount,basal[anchor],seed)
            elif footprint_version:
                proposal=arc_birth_footprint.propose(factory,amount,basal[anchor])
            else:
                patch=factory(amount)
                proposal=dict(plan=patch,nominal_plan=patch,
                    profile=arc_birth_profile.birth(patch,amount,basal[anchor]),search=None)
            birth_candidates[identity]=proposal
        return birth_candidates[identity]
    groups={};group_links={};cofunded=set()
    if profile_version:
        proposed=source_cohorts.groups(s,all_points,all_owners,all_area,all_provenance,
                                      route['eligible']&(route['arc_id']==0))
        for cohort in proposed:
            if not cohort['qualified']:continue
            def factory(anchor,amount):
                proposal=birth_candidate(anchor,amount)
                # Infeasible sources still undergo the same membership pruning
                # before the capacity predicate leaves their volume pending.
                return proposal['plan'] if proposal['plan'] is not None else proposal['nominal_plan']
            def capacity(patch,anchor,amount):
                return arc_birth_profile.birth(patch,amount,basal[anchor])['diagnostics']['admissible']
            for local in cohort_geometry.local_groups(all_points,all_area,cohort,factory,capacity):
                key=('cohort',len(groups));groups[key]=local['indices'];group_links[key]=local['links']
                cofunded.update(local['indices'])
    for i in np.flatnonzero(route['eligible']&(all_area>0)):
        if int(i) in cofunded:continue
        arc=int(route['arc_id'][i]);owner=int(all_owners[i])
        component=int(component_at_face[route['material_index'][i]]) if arc else -1
        key=('arc',arc,component) if arc else ('point',owner,*map(float,all_points[i]))
        groups.setdefault(key,[]).append(int(i))
    def point_birth_attempt(take,key):
        first=int(take[0]);owner=int(all_owners[first]);area=float(all_area[take].sum())
        evaluations=0;attempts=[]
        for strategy in ('point','legacy'):
            proposal=birth_candidate(first,area,strategy)
            row=dict(version=1,mode='birth',arc_id=0,owner=owner,
                source_indices=take.tolist(),geometry_xyz=all_points[take].tolist(),
                requested_area_km2=area,accepted_area_km2=0.,pending_area_km2=area,
                geometry_evaluations=evaluations,relocation_km=0.,requested_footprint=None,
                accepted_footprint=None,last_examined_footprint=None,
                profile_capacity=proposal['profile']['diagnostics'],footprint_capacity=proposal['search'])
            plan=None;spending=take.copy()
            if proposal['profile']['diagnostics']['admissible']:
                target=area
                while True:
                    admission=emplacement.admit(placement,
                        lambda amount:birth_candidate(first,amount,strategy)['plan'],target,owner,
                        minimum_area_km2=MIN_PATCH_AREA_KM2)
                    evaluations+=admission['diagnostics']['geometry_evaluations']
                    if admission['accepted_area_km2']<=0:break
                    inside=spending[cohort_geometry.contained(admission['plan'],all_points[spending])]
                    inside=cohort_geometry.connected(inside,group_links.get(key),first)
                    if len(inside)==len(spending):break
                    spending=inside
                    if not len(inside):
                        admission['accepted_area_km2']=0.;admission['plan']=None;break
                    target=float(all_area[inside].sum())
                accepted=float(admission['accepted_area_km2']);plan=admission['plan']
                row.update(admission['diagnostics'],requested_area_km2=area,accepted_area_km2=accepted,
                           pending_area_km2=area-accepted,geometry_evaluations=evaluations)
                if accepted>0:
                    chosen=birth_candidate(first,accepted,strategy)
                    row['profile_capacity']=arc_birth_profile.birth(plan,accepted,basal[first])['diagnostics']
                    row['footprint_capacity']=chosen['search']
                    if not row['profile_capacity']['admissible']:
                        row.update(accepted_area_km2=0.,pending_area_km2=area,
                                   physical_emplacement_rejected='constructive_profile_capacity')
                        plan=None
                else:
                    row['profile_capacity']=dict(version=1,admissible=False,
                        reason='source_containment' if not len(spending) else 'no_geographic_footprint')
            if row['accepted_area_km2']<=0.:row['accepted_footprint']=None
            attempts.append(dict(strategy=strategy,accepted_area_km2=float(row['accepted_area_km2']),
                reason=row['profile_capacity']['reason']))
            if row['accepted_area_km2']>0:
                if strategy=='point':
                    volume=25.*float(row['accepted_area_km2']);footprint=float(plan['area_km2'].sum())
                    row['point_promotion']=dict(version=1,
                        seed_key=str(arc_point_nucleation.seed_key(s,[all_provenance[first]])),
                        source_origin_ids=[int(all_provenance[i]['origin_id']) for i in take if i in set(spending)],
                        source_volume_km3=volume,footprint_area_km2=footprint,
                        volume_density_km=volume/footprint)
                break
        row['point_attempts']=attempts
        return row,plan,spending

    before_volume=float(s.material_surface['area_km2']@s.structure['thickness_km'])
    before_mass=float(s.mass.sum())
    before_footprint=float(s.material_surface['area_km2'].sum())
    new_faces=0;grown=0;added=0.;volume=0.;errors=[];owner_totals={}
    deposited_area=0.;deposited_patches=0
    born_profiles=[]
    # Basements are sampled at the supplied points before any material changes.
    for key,indices in groups.items():
        take=np.asarray(indices,int);area=float(all_area[take].sum());first=take[0]
        owner=int(all_owners[first]);point=all_points[first];arc=int(route['arc_id'][first])
        spending_take=take.copy();spent_override=None;mode='growth' if arc else 'birth'
        if area<MIN_PATCH_AREA_KM2:
            unresolved.extend((all_points[i].copy(),owner,float(all_area[i]),int(i)) for i in take);continue
        if arc:
            selected=growable.get((arc,int(component_at_face[route['material_index'][first]])))
            plan=None;result=None;accepted=area
            if placement is not None and selected is not None:
                surface=s.material_surface
                old_ids,old_faces=np.unique(surface['faces'][selected],return_inverse=True)
                old_vertices=surface['vertices'][old_ids].copy();old_faces=old_faces.reshape(-1,3)
                def growth_factory(amount):
                    candidate=_grow_geometry(surface,selected,amount)
                    if candidate is not None:
                        candidate['faces']=np.searchsorted(candidate['vertex_ids'],surface['faces'][selected])
                    return candidate
                admission=emplacement.admit(placement,growth_factory,area,owner,
                    minimum_area_km2=MIN_PATCH_AREA_KM2,old_vertices=old_vertices,old_faces=old_faces,
                    old_indices=selected)
                placement_rows.append(dict(mode='growth',arc_id=arc,owner=owner,source_indices=take.tolist(),
                                           geometry_xyz=all_points[take].tolist(),
                                           **admission['diagnostics']))
                accepted=float(admission['accepted_area_km2'])
                if accepted<=0.:
                    if profile_version:
                        placement_rows[-1]['profile_capacity']=dict(version=1,admissible=False,reason='no_geographic_footprint')
                else:plan=admission['plan']
                if profile_version and accepted>0.:
                    capacity=arc_birth_profile.growth(s,selected,plan)
                    placement_rows[-1]['profile_capacity']=capacity
                    if not capacity['admissible']:
                        placement_rows[-1].update(accepted_area_km2=0.,pending_area_km2=area,
                            physical_emplacement_rejected='constructive_profile_capacity')
                        accepted=0.
            else: accepted=area
            if selected is not None and accepted>0.:
                result=_grow(s,selected,accepted,prepared_plan=plan)
            if result is None:
                if placement is not None and selected is not None and accepted>0.:
                    placement_rows[-1].update(accepted_area_km2=0.,pending_area_km2=area,
                                               physical_emplacement_rejected='column_or_trace_bounds')
                if deposition_version and placement is not None and selected is not None:
                    lateral=placement_rows[-1]
                    deposited=arc_local_deposition.propose(s,selected,route['material_index'][take],all_area[take])
                    accepted=float(deposited['accepted_area_km2']);mode='deposition'
                    placement_rows[-1]=dict(version=1,mode=mode,arc_id=arc,owner=owner,
                        source_indices=take.tolist(),geometry_xyz=all_points[take].tolist(),
                        requested_area_km2=area,accepted_area_km2=accepted,pending_area_km2=area-accepted,
                        geometry_evaluations=lateral['geometry_evaluations'],
                        column_evaluations=deposited['column_evaluations'],relocation_km=0.,
                        requested_footprint=None,accepted_footprint=deposited['accepted_footprint'],
                        last_examined_footprint=None,profile_capacity=deposited['profile_capacity'],
                        column_deposition=deposited['column_deposition'],lateral_attempt=lateral)
                    if accepted>0.:
                        result=arc_local_deposition.commit(s,deposited)
                        spent_override=deposited['spent_area_km2']
                        deposited_area+=accepted;deposited_patches+=1
            if result is None:
                unresolved.extend((all_points[i].copy(),owner,float(all_area[i]),int(i)) for i in take);continue
            if placement is not None and mode!='deposition':
                emplacement.commit(placement,s.material_surface,selected)
            if mode!='deposition':
                grown+=1;errors.append(result['volume_error_km3'])
        else:
            # The containing cell only seeds marker bookkeeping. Both actual
            # geometry and basal height use the original geographic position.
            cell=int(s._indices(point[None])[0])
            plan=None;accepted=area;start=len(s.mass)
            if point_version and all_provenance[first]['valid']:
                row,plan,spending_take=point_birth_attempt(take,key)
                placement_rows.append(row);accepted=float(row['accepted_area_km2'])
                if accepted<=0.:
                    unresolved.extend((all_points[i].copy(),owner,float(all_area[i]),int(i)) for i in take);continue
            else:
                if placement is not None:
                    strike=_strike(s,cell,owner,point)
                    if profile_version:
                        proposal=birth_candidate(first,area)
                        requested_profile=proposal['profile']
                        if not requested_profile['diagnostics']['admissible']:
                            placement_rows.append(dict(version=1,mode='birth',arc_id=0,owner=owner,
                                source_indices=take.tolist(),geometry_xyz=all_points[take].tolist(),
                                requested_area_km2=area,accepted_area_km2=0.,pending_area_km2=area,
                                geometry_evaluations=0,relocation_km=0.,requested_footprint=None,
                                accepted_footprint=None,last_examined_footprint=None,
                                profile_capacity=requested_profile['diagnostics']))
                            if footprint_version:placement_rows[-1]['footprint_capacity']=proposal['search']
                            unresolved.extend((all_points[i].copy(),owner,float(all_area[i]),int(i)) for i in take);continue
                    target=area;evaluations=0
                    while True:
                        factory=(lambda amount:birth_candidate(first,amount)['plan']) if profile_version else (lambda amount:_patch(point,strike,amount))
                        admission=emplacement.admit(placement,factory,target,owner,
                                                     minimum_area_km2=MIN_PATCH_AREA_KM2)
                        evaluations+=admission['diagnostics']['geometry_evaluations']
                        if not profile_version or admission['accepted_area_km2']<=0:break
                        inside=spending_take[cohort_geometry.contained(admission['plan'],all_points[spending_take])]
                        inside=cohort_geometry.connected(inside,group_links.get(key),first)
                        if len(inside)==len(spending_take):break
                        spending_take=inside
                        if not len(inside):
                            admission['accepted_area_km2']=0.;admission['plan']=None;break
                        target=float(all_area[inside].sum())
                    admission['diagnostics'].update(requested_area_km2=area,
                        pending_area_km2=area-admission['accepted_area_km2'],geometry_evaluations=evaluations)
                    placement_rows.append(dict(mode='birth',arc_id=0,owner=owner,source_indices=take.tolist(),
                                               geometry_xyz=all_points[take].tolist(),
                                               **admission['diagnostics']))
                    accepted=float(admission['accepted_area_km2']);plan=admission['plan']
                    if accepted<=0.:
                        if profile_version:
                            placement_rows[-1]['profile_capacity']=dict(version=1,admissible=False,reason='no_geographic_footprint')
                        unresolved.extend((all_points[i].copy(),owner,float(all_area[i]),int(i)) for i in take);continue
                    if profile_version:
                        capacity=arc_birth_profile.birth(plan,accepted,basal[first])['diagnostics']
                        placement_rows[-1]['profile_capacity']=capacity
                        if footprint_version:placement_rows[-1]['footprint_capacity']=birth_candidate(first,accepted)['search']
                        if not capacity['admissible']:
                            placement_rows[-1].update(accepted_area_km2=0.,pending_area_km2=area,
                                physical_emplacement_rejected='constructive_profile_capacity')
                            unresolved.extend((all_points[i].copy(),owner,float(all_area[i]),int(i)) for i in take);continue
            result=_birth(s,cell,owner,accepted,position=point,basal_m=basal[first],
                          fixed_source_volume=True,prepared_patch=plan)
            if placement is not None:
                emplacement.commit(placement,s.material_surface,np.arange(start,len(s.mass)))
            new_faces+=result['new_faces']
            if profile_version:
                born_profiles.append(result['birth_profile'])
                placement_rows[-1]['profile_capacity']=result['birth_profile']
        spent=np.zeros(len(take)) if spent_override is None else spent_override
        if spent_override is None:
            eligible_spending=np.isin(take,spending_take)
            spent[eligible_spending]=all_area[take[eligible_spending]]*(accepted/float(all_area[spending_take].sum()))
        held=all_area[take]-spent
        unresolved.extend((all_points[i].copy(),owner,float(amount),int(i)) for i,amount in zip(take,held) if amount>0.)
        if profile_version:
            from copy import deepcopy
            row=placement_rows[-1]
            row.update(arc_id=int(arc or result['arc_id']),source_origin_ids=[int(all_provenance[i]['origin_id']) for i in take],
                source_placed_area_km2=spent.tolist(),source_pending_area_km2=held.tolist(),
                source_available_area_km2=all_area[take].tolist(),
                cohort_qualified=key in group_links,
                consumed_origins_contained=True,source_containment_tolerance=cohort_geometry.CONTAINMENT_TOLERANCE)
            ledger=getattr(s,'native_arc_source_placements',[])
            for i,amount in zip(take,spent):
                if amount<=0:continue
                entry=dict(placement_id=len(ledger)+1,arc_id=int(arc or result['arc_id']),
                    placement_time_myr=float(s.t),source_provenance=deepcopy(all_provenance[i]),
                    geometry_xyz=[all_points[i].tolist()],area_km2=float(amount),volume_km3=float(25.*amount),
                    mode=mode)
                if mode=='deposition':entry['receiving_face_id']=int(s.parcel_patch[route['material_index'][i]])
                ledger.append(entry)
            s.native_arc_source_placements=ledger
        area=accepted
        added+=area;volume+=result['added_volume_km3']
        entry=owner_totals.setdefault(owner,dict(area=0.,volume=0.,xyz=np.zeros(3)))
        entry['area']+=area;entry['volume']+=result['added_volume_km3']
        entry['xyz']+=np.sum(all_points[take]*spent[:,None],axis=0)
    s.native_arc_pending=dict(xyz=np.array([p for p,_,_,_ in unresolved],float).reshape(-1,3),
        owner=np.array([o for _,o,_,_ in unresolved],np.int16),area=np.array([a for _,_,a,_ in unresolved],float))
    if placement is not None:
        s.native_arc_pending.update(emplacement_version=1,source_column_km=JUVENILE_THICKNESS_KM)
    if profile_version:
        s.native_arc_pending['source_provenance']=source_cohorts.subset(all_provenance,[i for _,_,_,i in unresolved])
    pending_total=float(s.native_arc_pending['area'].sum())
    residual=incoming+prior-rejected_area-added-pending_total
    actual_volume=float(s.material_surface['area_km2']@s.structure['thickness_km'])-before_volume
    mass_added=float(s.mass.sum())-before_mass
    if not np.isclose(residual,0.,rtol=0.,atol=max(1e-9,(incoming+prior)*1e-12)):
        raise ValueError('Geographic arc source area failed its admission/pending ledger.')
    if not np.isclose(actual_volume,volume,rtol=2e-11,atol=max(1e-7,abs(before_volume)*2e-15)):
        raise ValueError('Geographic arc emplacement failed its physical column-volume ledger.')
    supplied_volume=added*JUVENILE_THICKNESS_KM
    if not np.isclose(volume,supplied_volume,rtol=2e-11,atol=1e-7):
        raise ValueError('Geographic arc birth/growth must use the same explicit magma source volume.')
    s.process_totals['arc_added_km2']+=added
    s.process_totals['arc_added_volume_km3']=s.process_totals.get('arc_added_volume_km3',0.)+volume
    if deposition_version:
        s.process_totals['arc_deposited_source_area_km2']=s.process_totals.get('arc_deposited_source_area_km2',0.)+deposited_area
        s.process_totals['arc_deposited_volume_km3']=s.process_totals.get('arc_deposited_volume_km3',0.)+deposited_area*JUVENILE_THICKNESS_KM
    s._sync_material();s._coverage_signature=None;s._owner_occupancy_signature=None
    for name in ('_material_occupancy_hits','_material_coverage','exposed_material'):
        if hasattr(s,name):delattr(s,name)
    reasons={str(reason):float(all_area[rejected&(route['reason']==reason)].sum())
             for reason in np.unique(route['reason'][rejected])}
    report=dict(source_geometry_version=1,added_area_km2=added,added_volume_km3=volume,
        requested_area_km2=incoming,previous_pending_area_km2=prior,rejected_area_km2=rejected_area,
        rejected_by_reason_km2=reasons,pending_area_km2=pending_total,source_area_residual_km2=residual,
        physical_mass_added_km2=mass_added,physical_mass_residual_km2=mass_added-(added-deposited_area),
        physical_footprint_area_added_km2=float(s.material_surface['area_km2'].sum())-before_footprint,
        physical_volume_added_km3=actual_volume,physical_volume_residual_km3=actual_volume-volume,
        supplied_magma_volume_km3=supplied_volume,source_volume_residual_km3=volume-supplied_volume,
        pending_magma_volume_km3=pending_total*JUVENILE_THICKNESS_KM,
        new_faces=new_faces,grown_patches=grown,patch_count=len(np.unique(s.parcel_arc_id[s.parcel_arc_id>0])),
        maximum_growth_volume_error_km3=max(map(abs,errors),default=0.),juvenile_thickness_km=JUVENILE_THICKNESS_KM,
        representation='exact geographic source points and contained connected juvenile footprints',
        relocation_km=0.,grouping='same contained shared-edge arc component or identical ocean point; no owner/cell merging')
    if deposition_version:
        report.update(deposited_source_area_km2=deposited_area,
            deposited_volume_km3=deposited_area*JUVENILE_THICKNESS_KM,deposited_patches=deposited_patches)
    if placement is not None:
        report['emplacement_geometry']=dict(version=1,
            policy='exact new footprint in uncovered intended-owner water; shared mesh backtracking; pending at original source',
            sequential_occupancy=True,geometry_evaluations=sum(row['geometry_evaluations'] for row in placement_rows),
            requested_area_km2=sum(row['requested_area_km2'] for row in placement_rows),
            accepted_area_km2=sum(row['accepted_area_km2'] for row in placement_rows),
            pending_area_km2=sum(row['pending_area_km2'] for row in placement_rows),
            unexamined_pending_area_km2=max(0.,pending_total-sum(row['pending_area_km2'] for row in placement_rows)),
            sources=placement_rows)
    if profile_version:
        # Rejected and below-capacity proposals also retain an explicit origin
        # partition. No source volume is lost merely because no face was born.
        for row in placement_rows:
            if 'source_origin_ids' in row:continue
            indices=np.asarray(row['source_indices'],int)
            row.update(source_origin_ids=[int(all_provenance[i]['origin_id']) for i in indices],
                source_placed_area_km2=np.zeros(len(indices)).tolist(),
                source_pending_area_km2=all_area[indices].tolist(),source_available_area_km2=all_area[indices].tolist(),
                cohort_qualified=all(all_provenance[i]['valid'] for i in indices),
                consumed_origins_contained=True,source_containment_tolerance=cohort_geometry.CONTAINMENT_TOLERANCE)
        report['birth_profile_capacity']=dict(version=1,
            requested_birth_area_km2=sum(row['requested_area_km2'] for row in placement_rows if row['mode']=='birth'),
            placed_birth_area_km2=sum(row['accepted_area_km2'] for row in placement_rows if row['mode']=='birth'),
            profile_pending_area_km2=sum(row['pending_area_km2'] for row in placement_rows
                if row.get('profile_capacity',{}).get('reason') in ('constructive_slope_capacity','physical_column_capacity')),
            maximum_accepted_birth_grade=max((row['maximum_grade'] for row in born_profiles),default=0.),
            reference_self_GPE_created_km4=sum(row['reference_self_GPE_created_km4'] for row in born_profiles),
            interpretation='Initial column-derived constructive relief; rejected source remains pending; self-only created reference GPE does not include growth or tectonic work')
    s.arc_material_diagnostics=report
    for owner,entry in sorted(owner_totals.items()):
        s._record('island_arc',f'Subduction magma built connected volcanic core and apron material on {s.names[owner]}.',
            ('arc',int(s.plate_uid[owner]),int(s.t//100)),plates=(owner,),xyz=entry['xyz'],
            details=_event_details(report,added_area_km2=entry['area'],added_volume_km3=entry['volume']))
    return report
