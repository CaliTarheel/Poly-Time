"""Conservative native connected split commits; no raster fracture handlers."""
from __future__ import annotations
from copy import deepcopy
import numpy as np

import ocean_rifting
import progressive_rifting
import backarc
import trench_history
from material_geometry import splits_protected_groups
from ridge_interaction import reassign_ridge_episodes


def _unit(value):
    value=np.asarray(value,float)
    return value/np.maximum(np.linalg.norm(value,axis=-1,keepdims=True),1e-30)


def _viable(s,p,region):
    return progressive_rifting.partition_viability(s,s.plate==p,region)['viable']


def _trace_side(s,parcel_side,region):
    side=region[s._indices(s.trace_xyz)].copy()
    if len(s.parcel_patch) and len(s.trace_patch):
        order=np.argsort(s.parcel_patch);ids=s.parcel_patch[order]
        at=np.searchsorted(ids,s.trace_patch)
        valid=(at<len(ids))&(ids[np.minimum(at,len(ids)-1)]==s.trace_patch)
        side[valid]=parcel_side[order[at[valid]]]
    return side


def _intact_material(s,p,parcel_side):
    selected=s.parcel_plate==p
    return (parcel_side.shape==selected.shape
            and not splits_protected_groups(s.parcel_craton[selected],parcel_side[selected]))


def _arc_hits_triangles(a,b,triangles):
    """Exact minor-great-circle intersection, including touching/coincident arcs."""
    if not len(triangles):return False
    normals=np.cross(triangles,np.roll(triangles,-1,axis=1))
    if np.any(np.all(normals@a>=-1e-11,axis=1)) or np.any(np.all(normals@b>=-1e-11,axis=1)):
        return True
    first=triangles.reshape(-1,3);second=np.roll(triangles,-1,axis=1).reshape(-1,3)
    cut_normal=_unit(np.cross(a,b));face_normal=_unit(np.cross(first,second))
    cross=np.cross(face_normal,cut_normal);size=np.linalg.norm(cross,axis=1)
    middle=_unit(a+b);threshold=float(np.dot(a,middle))
    face_middle=_unit(first+second);face_threshold=np.sum(first*face_middle,axis=1)
    parallel=size<1e-11
    if np.any(parallel):
        same_plane=np.abs(first[parallel]@cut_normal)<1e-10
        on_cut=((first[parallel]@middle>=threshold-1e-11)|(second[parallel]@middle>=threshold-1e-11))
        cut_on_face=((face_middle[parallel]@a>=face_threshold[parallel]-1e-11)
                     |(face_middle[parallel]@b>=face_threshold[parallel]-1e-11))
        if np.any(same_plane&(on_cut|cut_on_face)):return True
    point=cross/np.maximum(size[:,None],1e-30)
    for sign in (1.,-1.):
        x=point*sign
        on_cut=x@middle>=threshold-1e-11
        on_face=np.sum(x*face_middle,axis=1)>=face_threshold-1e-11
        if np.any(~parallel&on_cut&on_face):return True
    return False


def _ocean_cut_clear(s,edge_indices):
    """Every proposed cut segment must avoid the actual buoyant triangles."""
    geometry=s.native_mesh
    control=geometry['edge_faces'][edge_indices]
    if np.any(s.crust[control]>0):return False
    surface=s.material_surface
    triangles=surface['vertices'][surface['faces']]
    if not len(triangles):return True
    centres=_unit(triangles.sum(axis=1))
    radii=np.max(np.arccos(np.clip(np.einsum('nvi,ni->nv',triangles,centres),-1,1)),axis=1)
    vertices=geometry['vertices'];pairs=geometry['edge_vertices'][edge_indices]
    for a,b in vertices[pairs]:
        midpoint=_unit(a+b)
        half=np.arctan2(np.linalg.norm(np.cross(a,b)),np.dot(a,b))*.5
        candidate=centres@midpoint>=np.cos(np.minimum(radii+half+1e-10,np.pi))
        if _arc_hits_triangles(a,b,triangles[candidate]):return False
    return True


def _material_partition_clear(s,p,region,parcel_side):
    # Face vertices straddling the native partition signal an underresolved
    # buoyant triangle. Reject it; do not infer an ocean cut from its centroid.
    material=s.material_surface
    selected=(s.parcel_plate==p)&np.isin(s.kind,(1,2))
    if not np.any(selected):return True
    vertices=material['vertices'][material['faces'][selected]]
    sides=region[s._indices(vertices.reshape(-1,3))].reshape(-1,3)
    desired=parcel_side[selected]
    return bool(np.all(sides==desired[:,None]))


def _allocate(s,p):
    available=np.flatnonzero(~s.active)
    if not len(available):
        s._ensure_plate_capacity(s.capacity+1)
        available=np.flatnonzero(~s.active)
    if not len(available):return None
    q=int(available[0]);s.count=max(s.count,q+1)
    s.active[q]=True;s._new_plate_identity(q,p);s.split_serial+=1
    s.polarity[q,:]=s.polarity[:,q]=-1
    s.collision_clock[q,:]=s.collision_clock[:,q]=0
    s.support[q]=0.
    return q


def _commit(s,p,region,parcel_side,trace_side,rotations,setting,loading,center,require_viability=True):
    # partition_viability judges a CUT through a connected plate, and by design rejects
    # "partition only reassigns existing disconnected islands". An isolated region is
    # exactly that, so the test can never pass for it -- which left _commit_isolated
    # dead from 338 Myr on. Only that caller opts out; material integrity still applies.
    isolated=(loading or {}).get('cause')=='isolated_region'
    if require_viability:
        viability=progressive_rifting.partition_viability(s,s.plate==p,region)
        if not viability['viable']:return False
    else:
        viability=dict(viable=True,model='not applicable',reason='region already disconnected; no cut is made')
    if not _intact_material(s,p,parcel_side):return False
    if np.asarray(rotations).shape!=(2,3) or not np.isfinite(rotations).all():return False
    q=_allocate(s,p)
    if q is None:return False
    old_omega=s.omega[p].copy();old_mantle=s.mantle[p].copy()
    if isolated:
        s.names[q]=f"{'Ocean basin' if setting=='oceanic' else 'Detached plate'} {s.split_serial:02d}"
    else:
        s.names[q]=f"{'Ocean rift' if setting=='oceanic' else 'Rift plate'} {s.split_serial:02d}"
    s.plate[region]=q
    s.support[q,region]=s.support[p,region];s.support[p,region]=0.
    s.parcel_plate[(s.parcel_plate==p)&parcel_side]=q
    s.trace_plate[(s.trace_plate==p)&trace_side]=q
    # Trenches are matched by plate-ID pair, so a split must carry the trench
    # records, their maturity and their slabs to whichever daughter now holds
    # each subducting or overriding edge. Uses the pre-split boundary cache.
    trench_history.transfer_split(s,p,q)
    s.omega[p]=old_omega+rotations[0];s.omega[q]=old_omega+rotations[1]
    s.mantle[q]=old_mantle
    for slot in (p,q):
        selected=s.plate==slot
        centre=np.sum(s.xyz[selected]*s.cell_area[selected,None],axis=0)
        s.centres[slot]=_unit(centre) if np.linalg.norm(centre)>1e-10 else s.xyz[np.flatnonzero(selected)[0]]
        if slot==q:s.born[slot]=s.t
        s.rift_clock[slot]=s.ocean_rift_clock[slot]=0.
    s._transfer_ridge_hosts(p,q)
    s._sync_material()
    s.event_keys={key for key in s.event_keys if not(isinstance(key,tuple) and key[0]=='collision' and q in key[1:])}
    s.process_totals['rift_events']+=1
    # Counterpart of the backarc increment: this is the one that means a plate actually
    # divided. See backarc.py for why the undifferentiated total is not usable evidence.
    # An isolated split divides a plate without any rifting, so it is counted apart.
    counter='plate_splits_isolated' if isolated else 'rift_events_breakup'
    s.process_totals[counter]=s.process_totals.get(counter,0.)+1
    if isolated:
        description=(f"{s.names[p]} had become geographically disconnected; its separated "
                     f"{'basin' if setting=='oceanic' else 'region'} became {s.names[q]}.")
        geometry,model='no fracture: region already disconnected from its plate','topological isolation'
    else:
        description=(f"A developed {'oceanic' if setting=='oceanic' else 'continental'} deformation belt "
                     f"divided {s.names[p]}, creating {s.names[q]}.")
        geometry,model='connected native material deformation belt','local spherical tensile deformation and inherited weakness'
    s._record('rift',description,
        plates=(p,q),xyz=center,details=dict(setting=setting,parent_plate_uid=int(s.plate_uid[p]),
        new_plate_uid=int(s.plate_uid[q]),new_plate_area_fraction=float(s.cell_area[region].sum()/s.cell_area[np.isin(s.plate,[p,q])].sum()),
        fracture_geometry=geometry,rift_belt_relief_reduction_m=0.,
        daughter_viability=viability,
        loading_model=model,loading=deepcopy(loading or {})))
    return True


def split_continent(s,p,chosen,loading=None):
    """Commit a developed material cut, preserving every face and fitted motion."""
    if chosen is None or not s.active[p]:return False
    side=np.asarray(chosen['parcel_side'],bool);trace=np.asarray(chosen['trace_side'],bool)
    if side.shape!=(len(s.mass),) or trace.shape!=(len(s.trace_patch),):return False
    selected=s.parcel_plate==p
    if not np.any(selected&side) or not np.any(selected&~side):return False
    region=(s.plate==p)&(np.asarray(chosen['grid_distance'])>0)
    if not _viable(s,p,region) or not _intact_material(s,p,side):return False
    original=s.parcel_plate==p;original_trace=s.trace_plate==p
    crack=chosen['crack'];center=np.asarray(crack.center,float)
    accepted=_commit(s,p,region,side,trace,np.asarray(chosen['rotations']), 'continental',loading,center)
    if not accepted:return False
    q=int(s.parcel_plate[np.flatnonzero(original&side)[0]])
    system=next((row for row in s.rift_systems if row['id']==(loading or {}).get('rift_system_id')),None)
    rid=system.get('source_rift_id') if system else None
    if rid is None:rid=s._new_rift_record('continental breakup',(p,q),center)
    for prefix,points,selected,kinds in (('',s.pos,original,s.kind),('trace_',s.trace_xyz,original_trace,s.trace_kind)):
        belt=selected&(np.abs(crack.signed_distance(points))<.09)
        ids=getattr(s,prefix+'rift_id');new=belt&(ids<0)&(kinds!=3)
        ids[new]=rid;getattr(s,prefix+'rift_birth_myr')[new]=s.t
        getattr(s,prefix+'rift_tangent')[new]=s._rift_tangents(crack,points[new])
        sutures=getattr(s,prefix+'suture');sutures[belt]=np.maximum(sutures[belt],.8)
    return True


def _commit_ocean(s,proposal):
    p=int(proposal['parent_slot'])
    if not s.active[p] or int(s.plate_uid[p])!=int(proposal['parent_uid']):return False
    region=np.zeros(s.n,bool);region[np.asarray(proposal['child_faces'],int)]=True
    if not _viable(s,p,region) or not _ocean_cut_clear(s,np.asarray(proposal['failed_edge_indices'],int)):return False
    parcel_side=region[s.parcel_cell]
    if not _intact_material(s,p,parcel_side) or not _material_partition_clear(s,p,region,parcel_side):return False
    traces=_trace_side(s,parcel_side,region)
    center=_unit(np.sum(s.xyz[region]*s.cell_area[region,None],axis=0))
    return _commit(s,p,region,parcel_side,traces,np.asarray(proposal['delta_omega']), 'oceanic',
        dict(mean_opening_km_myr=float(proposal['mean_opening_km_myr']),
             failed_native_edges=np.asarray(proposal['failed_edge_indices']).tolist(),
             cause=proposal['cause']),center)


# A detached body must reach this area before it earns its own plate.
#
# Absolute rather than a fraction of planet area, so the threshold means the same
# thing as a real plate size regardless of world radius. 2.5e6 km2 sits just
# BELOW the Indian plate (3.29e6), which is the useful reference: India is
# unambiguously a real plate, so a floor above it could refuse to recognise one.
#
# A floor is required, not optional. Mesh noise routinely leaves one- and two-face
# orphans -- at 338 Myr plate 1 carried three and plate 2 one, each a single face
# of a few thousand km2 -- and spawning a plate for each would churn through
# _allocate. It also has to stay clear of _retire_remnants, which absorbs whole
# tiny plates below REMNANT_AREA_FRACTION of planet area (~1.3e3 km2 here), so
# the two rules are three orders of magnitude apart and cannot contest a body.
#
# Expected to come down substantially later in the run. Fine microplate and
# archipelago structure -- a Philippines or a Banda Sea -- lives nearer 1e5 km2,
# and resolving it late (roughly 900-1000 Myr) needs a floor around there. Left
# high for now because a coarse early world has no such structure to represent
# and every extra plate costs a capacity slot and a capacity^2 polarity matrix.
#
# LOWERED to 1e5 km2 at the 394+ Myr migration, at the user's direction: isolation
# is the criterion, and this floor exists only so mesh speckle is not plated. At
# 392 Myr the detached bodies were one 3.87e6 km2 ocean basin and nine one- and
# two-face specks of 5.9e3-1.2e4 km2; 1e5 km2 (about sixteen level-6 faces) keeps
# every speck unplated and is still an order of magnitude above the ~9.3e3 km2
# effective _retire_remnants floor below, so the two rules cannot contest a body.
ISOLATED_SPLIT_AREA_KM2 = 1e5

# Below this share of planet area a whole plate is absorbed by its neighbour
# rather than kept. Set a little under the North Galapagos microplate (~1.3e3 km2
# here), so a body that small is treated as a real feature, not debris.
#
# Was .001 -- about 5.1e5 km2, larger than Spain, so anything short of a major
# plate was liable to be eaten. Survivable only while nothing ever split; once
# fragmentation works it would absorb genuine microplates as fast as they form,
# and it sat five times ABOVE the ~1e5 km2 floor intended for late-run
# archipelago resolution, so the two rules would have fought every step.
REMNANT_AREA_FRACTION = 2.5e-6
# ...but a threshold finer than the mesh is inert. One level-6 face is ~6.2e3 km2,
# so a 1.3e3 km2 cut-off can never match anything and debris would accumulate for
# ever. The effective floor is therefore whichever is LARGER: the physical size
# above, or a small multiple of a face. At 1.5 faces this still absorbs a
# single-face remnant -- which is the resolution limit, not a resolved microplate
# -- while keeping anything two faces or bigger. Once the mesh is fine enough
# that 1.5 faces falls below 1.3e3 km2, the physical number takes over on its own.
REMNANT_MIN_FACES = 1.5


def _commit_isolated(s):
    """Split a plate that has become geographically disconnected.

    The stress-driven paths ask "is this being torn apart?". This asks "is this
    still one piece?", which is a different question with a different answer.

    Old ocean lithosphere is strong -- cooling_strength suppresses damage growth,
    correctly, and intra-oceanic fission is genuinely rare on Earth. So a basin
    can be severed by continental drift while its own crust is nowhere near
    failing: at 338 Myr the ocean solve reported max_damage 0.059 against a 0.95
    threshold and max_tensile_strain 0.042 against 0.35, hence zero failed links
    and zero proposals -- while plate 0 was ALREADY in two connected components,
    a 5.2e6 km2 body at -58..-33 degrees rotating about the same Euler pole as a
    main body reaching +89.

    A region with no face-adjacency path to the rest of its plate cannot transmit
    stress to it, so sharing one rigid rotation is unphysical however undamaged
    the crust is. This is a topological backstop, deliberately last: it runs only
    when no stress-driven path has claimed the step, so a real rift always takes
    precedence and keeps its own provenance.

    The detached piece inherits its parent's motion exactly (zero delta_omega).
    Separation grants independence to respond to forces; it does not itself
    impart velocity, and inventing one here would be fabricating kinematics the
    solve never asked for.
    """
    a, b = s.native_mesh['edge_faces'].T
    internal = s.plate[a] == s.plate[b]
    s.isolated_split_diagnostics = []
    for p in np.flatnonzero(s.active):
        faces = np.flatnonzero(s.plate == p)
        if len(faces) < 2:
            continue
        label = np.full(s.n, -1, np.int64)
        label[faces] = np.arange(len(faces))
        # Union-find over same-plate adjacency only.
        parent = np.arange(len(faces))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        pair = internal & (s.plate[a] == p)
        for u, v in zip(label[a[pair]], label[b[pair]]):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[rv] = ru
        roots = np.array([find(i) for i in range(len(faces))])
        if len(np.unique(roots)) < 2:
            continue
        # Detach the largest NON-primary body: the parent keeps its main mass and
        # its identity, so history and naming stay with the bigger piece.
        order = sorted(np.unique(roots), key=lambda r: -s.cell_area[faces[roots == r]].sum())
        for r in order[1:]:
            member = faces[roots == r]
            area = float(s.cell_area[member].sum())
            # Three gates below can each veto a detached body, and until now all three
            # did it silently -- so a plate visibly in two pieces produced no split and
            # no explanation. Record the reason: without it the only observable is
            # active_plates staying flat, which is indistinguishable from "nothing was
            # disconnected". Purely diagnostic; no gate changes here.
            def _decline(reason):
                s.isolated_split_diagnostics.append(dict(
                    epoch_myr=float(s.t), plate=int(p), plate_uid=int(s.plate_uid[p]),
                    detached_area_km2=area, parent_area_km2=float(s.cell_area[faces].sum()),
                    threshold_km2=float(ISOLATED_SPLIT_AREA_KM2), declined_by=reason))
            if area < ISOLATED_SPLIT_AREA_KM2:
                _decline('below_area_threshold')
                continue
            region = np.zeros(s.n, bool)
            region[member] = True
            # No partition_viability here: it rejects every already-disconnected island by
            # construction, so as a gate it made this whole path unreachable (declined
            # 'not_viable' at 394 Myr on a 3.81e6 km2 basin). Isolation plus the area floor
            # is the criterion; the two material-integrity guards below still apply.
            parcel_side = region[s.parcel_cell]
            if not _intact_material(s, p, parcel_side):
                _decline('material_not_intact')
                continue
            if not _material_partition_clear(s, p, region, parcel_side):
                _decline('material_partition_unclear')
                continue
            traces = _trace_side(s, parcel_side, region)
            center = _unit(np.sum(s.xyz[region]*s.cell_area[region, None], axis=0))
            ocean_share = float(s.cell_area[region & (s.crust == 0)].sum()) / area
            setting = 'oceanic' if ocean_share >= .5 else 'continental'
            if _commit(s, p, region, parcel_side, traces, np.zeros((2, 3)), setting,
                       dict(cause='isolated_region',
                            detached_area_km2=area,
                            parent_area_km2=float(s.cell_area[faces].sum()),
                            ocean_area_fraction=ocean_share,
                            mean_opening_km_myr=0.), center, require_viability=False):
                return True
            _decline('commit_refused')
    return False


def _retire_empty_plates(s):
    """Remove empty Euler unknowns before any post-transport force balance.

    Advection can consume the last fractional ocean footprint while its slot
    is still active. There is then no plate to move and no basal resistance.
    Keep every positive fractional footprint, however small, as well as all
    material and trace owners. Invalid support must still fail validation.
    """
    for p in np.flatnonzero(s.active):
        if (np.any(s.plate == p) or not np.all(s.support[p] == 0.)
                or np.any(s.parcel_plate == p) or np.any(s.trace_plate == p)):
            continue
        s.active[p] = False


def _retire_remnants(s):
    """Absorb only whole, tiny, material-free plates; keep named other domains."""
    _retire_empty_plates(s)
    # Eligibility is geometric/material, not elapsed integration-step count.
    # A 20-step gate kept identical remnants alive for different physical ages
    # when dt or an authored transition split changed the step schedule.
    area=np.bincount(s.plate,weights=s.cell_area,minlength=s.capacity)
    a,b=s.native_mesh['edge_faces'].T
    floor=max(s.earth_area*REMNANT_AREA_FRACTION,REMNANT_MIN_FACES*float(s.cell_area.mean()))
    for p in np.flatnonzero(s.active):
        if (area[p]>floor or np.any(s.parcel_plate==p) or np.any(s.trace_plate==p)
                or np.any((s.plate==p)&(s.crust>0))):continue
        boundary=((s.plate[a]==p)&(s.plate[b]!=p))|((s.plate[b]==p)&(s.plate[a]!=p))
        neighbours=np.where(s.plate[a[boundary]]==p,s.plate[b[boundary]],s.plate[a[boundary]])
        if not len(neighbours):
            continue
        q=int(np.bincount(neighbours,weights=s.native_mesh['edge_length'][boundary],minlength=s.capacity).argmax())
        s.plate[s.plate==p]=q;s.support[q]+=s.support[p];s.support[p]=0.
        # The absorbed remnant's trench edges now belong to q. Carry their
        # records and slabs across before p is retired; otherwise each one is
        # shut down as 'plate_owner_lost' and restarts from zero maturity.
        trench_history.transfer_split(s,p,q)
        s.active[p]=False
        reassign_ridge_episodes(s,int(s.plate_uid[p]),int(s.plate_uid[q]))
        s._record('plate_absorption',f'The tiny oceanic remnant of {s.names[p]} joined {s.names[q]}.',plates=(p,q),xyz=s.centres[p],
                  details=dict(source_plate_uid=int(s.plate_uid[p]),target_plate_uid=int(s.plate_uid[q])))


def update_topology(s,dt,*,backarc_pending=None):
    """Advance every ocean history, then prefer one backarc/continental cut.

    The ocean solve uses the pre-commit owners/boundaries. A successful backarc
    changes topology, but cannot erase the elapsed ocean damage/healing step.
    """
    _retire_empty_plates(s)
    boundary={name:getattr(s,name) for name in ('ba','bb','bp','bq','bmid','bn','bl')}
    boundary['valid']=s._valid_loading_edges()
    result=ocean_rifting.update(s.native_mesh,dict(owner=s.plate,ocean=s.crust==0,age_myr=s.age),
        dict(omega=s.omega,uids=s.plate_uid,active=s.active),boundary,s.ocean_history,dt,
        strength_scale=float(s.config['rift_strength']),min_fraction=0.)
    s.ocean_history=result['history']
    # Advection records consumed/generated area and unresolved overlaps before
    # this stage. Retain that accounting alongside the current fracture solve.
    s.ocean_diagnostics=dict(getattr(s,'ocean_diagnostics',{}),**result['diagnostics'])
    import force_rifting
    force_rifting.update(s,dt)
    if backarc.commit_pending(s,backarc_pending):return True
    # A force-limit tear is the explicit load-driven breakup; when enabled it
    # precedes the velocity-loaded continental path in the one-split step.
    if force_rifting.commit(s):return True
    if progressive_rifting.commit(s):return True
    for proposal in result['proposals']:
        if _commit_ocean(s,proposal):return True
    # Last, so a stress-driven cut always wins the step and keeps its provenance.
    if _commit_isolated(s):return True
    _retire_remnants(s)
    return False
