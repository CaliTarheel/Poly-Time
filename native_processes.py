"""Native ocean transport, column processes and physical juvenile arc surfaces.

The computational mesh is authoritative. No displayed grid is sampled during
transport, ridge creation, deformation or material birth. Existing worldbuilding
process coefficients are retained; geometry is expressed in kilometres.
"""
from __future__ import annotations
import numpy as np
import mesh_transport
import material_surface
import mesh_geometry
import structure_engine
import progressive_rifting
import local_accretion
import trench_history
import native_subduction
from native_spreading import production_version as native_spreading_production
import normal_partition
import backarc
from ridge_geometry import finite_half_stage, rotate
from ridge_spreading import _active

RADIUS_KM=6371.


def _unit(v):
    v=np.asarray(v,float)
    return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-30)


def _cap_candidates(centres,radii,points):
    """Sparse native face candidates; exact spherical caps and bounded buffers."""
    radii=np.broadcast_to(np.asarray(radii,float),(len(centres),))
    segment,cells=[],[]
    for start in range(0,len(centres),128):
        local=centres[start:start+128]
        cosine=np.cos(np.minimum(np.pi,radii[start:start+128]/RADIUS_KM))
        for begin in range(0,len(points),4096):
            row,col=np.nonzero(local@points[begin:begin+4096].T >= cosine[:,None]-2e-14)
            if len(row):
                segment.append(row+start);cells.append(col+begin)
    if not segment:
        return np.empty(0,int),np.empty(0,int)
    return np.concatenate(segment),np.concatenate(cells)


def _spreading_advance(s, transported, dt, arrivals=None):
    """Reconstruct moving fronts and return the new-crust fraction per cell.

    Membership changes are restricted to existing local two-plate territory.
    A real third owner, including at a junction, blocks reconstruction. Unlike
    a fractional-support threshold, this guard cannot admit a remote tail.
    """
    if getattr(s,'native_spreading_version',0)==1:
        from native_spreading import advance
        return advance(s,transported,dt,arrivals)
    birth = np.zeros(s.n)
    s.spreading_diagnostics = dict(active_segments=0, reconstructed_cells=0,
        generated_area_km2=0., rejected_foreign_claims=0, candidate_claims=0,
        admitted_claims=0, rejected_buoyant_claims=0,
        rejected_foreign_owner_claims=0,rejected_incoming_foreign_claims=0,
        rejected_foreign_neighbor_claims=0,
        claim_count_basis='finite-strip candidate cell/segment pairs; foreign reasons may overlap')
    edge = _active(s)
    if not len(edge):
        s._spreading_pending = None
        return birth
    p, q = s.bp[edge], s.bq[edge]
    mid, normal = s.bmid[edge].copy(), s.bn[edge].copy()
    # Cached intersections belong to specific contact faces and persistent UIDs.
    old = getattr(s, 'spreading_fronts', None)
    if old is not None and len(old['key']):
        key = s.ba[edge].astype(np.int64)*s.n+s.bb[edge]
        at = np.searchsorted(old['key'], key)
        at = np.minimum(at, len(old['key'])-1)
        use = ((old['key'][at] == key) & (old['p_uid'][at] == s.plate_uid[p])
               & (old['q_uid'][at] == s.plate_uid[q]))
        mid[use], normal[use] = old['mid'][at[use]], old['normal'][at[use]]
    rotation = finite_half_stage(s.omega[p], s.omega[q], dt)
    moved = rotate(mid, rotation)
    normals = _unit(rotate(normal, rotation))
    tangent = _unit(np.cross(moved, normals))
    shore_p, shore_q = rotate(mid, s.omega[p]*dt), rotate(mid, s.omega[q]*dt)
    opening = np.maximum(0., np.sum((shore_q-shore_p)*normals, axis=1)*RADIUS_KM)
    length = s.bl[edge]
    spacing = float(np.sqrt(np.mean(s.cell_area)))
    # A compact linear transition preserves continuous motion below one pixel.
    band = np.maximum(spacing*1.5, opening/2+spacing)
    radius = np.hypot(band, length/2+spacing)
    segment, cells = _cap_candidates(moved, radius, s.xyz)
    owner = s.plate
    pp, qq = p[segment], q[segment]
    foreign_owner = (owner[cells] != pp) & (owner[cells] != qq)
    foreign_incoming = np.zeros(len(cells),bool)
    if arrivals is not None:
        incoming_other = arrivals.sum(axis=0)[cells]-arrivals[pp, cells]-arrivals[qq, cells]
        foreign_incoming = ~(incoming_other < 1e-6)
    # Check true neighboring owners, not hidden fractional claims. Pole samples
    # use the same spherical continuation as transport.
    foreign_neighbor = np.zeros(len(cells),bool)
    for neighbor in s.native_mesh['face_neighbors'][cells].T:
        foreign_neighbor |= (owner[neighbor] != pp) & (owner[neighbor] != qq)
    signed = np.arcsin(np.clip(np.sum(s.xyz[cells]*normals[segment], axis=1), -1, 1))*RADIUS_KM
    along = np.arcsin(np.clip(np.sum(s.xyz[cells]*tangent[segment], axis=1), -1, 1))*RADIUS_KM
    spatial = (np.abs(signed) <= band[segment]) & (np.abs(along) <= length[segment]/2+spacing*.6)
    foreign = foreign_owner | foreign_incoming | foreign_neighbor
    buoyant = s.crust[cells] != 0
    eligible = spatial & ~foreign & ~buoyant
    claim_counts=dict(candidate_claims=int(spatial.sum()),admitted_claims=int(eligible.sum()),
        rejected_foreign_claims=int(np.sum(spatial & foreign)),
        rejected_buoyant_claims=int(np.sum(spatial & ~foreign & buoyant)),
        rejected_foreign_owner_claims=int(np.sum(spatial & foreign_owner)),
        rejected_incoming_foreign_claims=int(np.sum(spatial & foreign_incoming)),
        rejected_foreign_neighbor_claims=int(np.sum(spatial & foreign_neighbor)),
        claim_count_basis='finite-strip candidate cell/segment pairs; foreign reasons may overlap')
    distance = signed*signed + np.maximum(np.abs(along)-length[segment]/2, 0.)**2
    segment, cells, signed, distance = (v[eligible] for v in (segment, cells, signed, distance))
    nearest = np.full(s.n, -1, np.int32)
    if len(cells):
        order = np.lexsort((segment, distance, cells))
        first = np.r_[True, np.diff(cells[order]) != 0]
        chosen = order[first]
        cells, segment, signed = cells[chosen], segment[chosen], signed[chosen]
        nearest[cells] = segment
        # Advect one continuous contrast field per neighboring plate pair.
        # Extrapolating an independent fitted plane from each raster face made
        # neighboring faces disagree, seeding alternating ownership stripes.
        # The half-stage inverse lookup preserves the curved shared interface
        # and its subcell motion without inventing new zero crossings.
        back = rotate(s.xyz[cells], -rotation[segment])
        fraction_q = np.zeros(len(cells))
        for sample, weight in s._sample_coordinates(back):
            a = s.support[p[segment], sample]
            b = s.support[q[segment], sample]
            fraction_q += weight*np.divide(b, a+b, out=np.full(len(cells), .5), where=a+b > 1e-12)
        fraction_q = np.clip(fraction_q, 0., 1.)
        transported[:, cells] = 0.
        transported[p[segment], cells] = 1-fraction_q
        transported[q[segment], cells] = fraction_q
    # Integrate each finite segment's two new strips. Quadrature is refined if
    # the opening exceeds a pixel; bilinear deposition conserves their area even
    # when each newborn strip is much narrower than a map cell.
    deposit_cells, deposit_segments, deposit_sides, deposit_areas = [], [], [], []
    counts = np.maximum(1, np.ceil(opening/(spacing*.5)).astype(int))
    for count in np.unique(counts):
        chosen = np.flatnonzero(counts == count)
        for side in (0, 1):
            for j in range(int(count)):
                offset = (j+.5)/count*opening[chosen]/2
                angle = offset/RADIUS_KM*(1 if side else -1)
                points = np.cos(angle)[:, None]*moved[chosen]+np.sin(angle)[:, None]*normals[chosen]
                area = opening[chosen]*length[chosen]/(2*count)
                for cell, weight in s._sample_coordinates(points):
                    target = nearest[cell]
                    safe = target >= 0
                    at = np.maximum(target, 0)
                    safe &= (((p[at] == p[chosen]) & (q[at] == q[chosen]))
                             | ((p[at] == q[chosen]) & (q[at] == p[chosen])))
                    gain = area*weight*safe
                    deposit_cells.append(cell)
                    deposit_segments.append(chosen)
                    deposit_sides.append(np.full(len(chosen), side))
                    deposit_areas.append(gain)
    dc, ds, side, gain = map(np.concatenate, (deposit_cells, deposit_segments, deposit_sides, deposit_areas))
    paired_area = np.zeros((len(edge), 2))
    np.add.at(paired_area, (ds, side), gain)
    # If a real junction clips one strip, admit the same amount on its partner.
    common = paired_area.min(axis=1)
    gain *= np.divide(common[ds], paired_area[ds, side], out=np.zeros(len(gain)),
                      where=paired_area[ds, side] > 0)
    deposited = np.bincount(dc, weights=gain, minlength=s.n)
    # One scaling per segment keeps both halves paired even in a saturated cell.
    cell_scale = np.minimum(1., s.cell_area/np.maximum(deposited, 1e-30))
    segment_scale = np.ones(len(edge))
    np.minimum.at(segment_scale, ds[gain > 0], cell_scale[dc[gain > 0]])
    gain *= segment_scale[ds]
    deposited = np.bincount(dc, weights=gain, minlength=s.n)
    paired_area[:] = 0
    np.add.at(paired_area, (ds, side), gain)
    # Multiple finite segment strips may meet inside a coarse cell. Sum their
    # distinct areas once, bounded by that cell's actual area.
    birth = np.minimum(deposited/s.cell_area, 1.)
    generated = float(np.sum(birth*s.cell_area))
    s.process_totals['ocean_created_km2'] += generated
    s._spreading_pending = dict(mid=moved, normal=normals, p=p, q=q,
                               p_uid=s.plate_uid[p].copy(), q_uid=s.plate_uid[q].copy(),
                               nearest=nearest)
    full_speed = opening/dt
    axis_speed = np.arccos(np.clip(np.sum(mid*moved, axis=1), -1, 1))*RADIUS_KM/dt
    s.spreading_diagnostics = dict(active_segments=int(len(edge)), reconstructed_cells=int(len(cells)),
        generated_area_km2=generated, requested_area_km2=float(np.sum(opening*length)),
        side_p_area_km2=float(paired_area[:, 0].sum()), side_q_area_km2=float(paired_area[:, 1].sum()),
        mean_full_rate_cm_yr=float(np.average(full_speed, weights=length)/10),
        mean_axis_speed_cm_yr=float(np.average(axis_speed, weights=length)/10),
        **claim_counts)
    return birth




def _close_partition(s, moved, iterations=4000, tolerance=1e-3):
    """Force full cover without letting the closure decide who owns the world.

    Every cell must read 1.0, or plates would slowly evaporate. But scaling a
    cell's plates by a common factor is precisely what awards a gap by
    incumbency. This instead fits both margins at once - alternating cell
    scaling against a coverage target of 1 and plate scaling against the areas
    the ridge and trench operators just set - so coverage is restored while
    each plate keeps the area the physics gave it.

    Falls back to the plain column scaling if it does not converge, so it can
    never stop a run.
    """
    area = np.asarray(s.cell_area, float)
    target = moved@area
    live = target > 0.
    scale = float(area.sum()/target.sum()) if target.sum() > 0. else 1.
    target = target*scale
    fitted = moved.copy()
    for _ in range(iterations):
        column = fitted.sum(axis=0)
        if np.any(column <= 1e-14):
            return moved/np.maximum(moved.sum(axis=0), 1e-300)[None, :], dict(converged=False, reason='empty_cell')
        fitted /= column[None, :]
        row = fitted@area
        residual = float(np.abs(row[live]-target[live]).max()) if np.any(live) else 0.
        if residual <= tolerance:
            fitted /= np.maximum(fitted.sum(axis=0), 1e-300)[None, :]
            return fitted, dict(converged=True, row_residual_km2=residual, global_scale=scale)
        adjust = np.ones(len(fitted))
        adjust[live] = target[live]/np.maximum(row[live], 1e-300)
        fitted *= adjust[:, None]
    fitted /= np.maximum(fitted.sum(axis=0), 1e-300)[None, :]
    return fitted, dict(converged=False, reason='iteration_limit',
                        row_residual_km2=float(np.abs((fitted@area)[live]-target[live]).max()) if np.any(live) else 0.)


def _ocean_ledger_stage(s,weights,conditional,active):
    """Observe support/moments without changing transport or owner selection."""
    total=weights.sum(axis=0)
    moments=np.einsum('pf,pfk,f->pk',weights[active],conditional,s.cell_area)
    return dict(owner_support_area_km2=weights[active]@s.cell_area,
        owner_moments=moments,moment_totals=moments.sum(axis=0),
        support_area_km2=float(s.cell_area@total),
        excess_area_km2=float(s.cell_area@np.maximum(total-1.,0.)),
        deficit_area_km2=float(s.cell_area@np.maximum(1.-total,0.)))


def advect_ocean(s, dt):
    """Carry support and ocean material moments through spherical face fluxes.

    Age, primordial provenance and structural memory are transported together.
    Relative plate motion can create overlapping fractional claims; only local
    mature subduction removes incoming material as recorded consumption. The
    remaining contact normalization is reported separately. Measured paired
    ridge strips alone reset a fraction of thermal age and material memory.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Native ocean transport needs a positive finite timestep.')
    fields = ['age', 'ocean_relief', 'primordial_fraction', *s.ocean_history]
    initial = np.column_stack((s.age+dt, s.ocean_relief,
                               s.primordial_fraction,
                               *[s.ocean_history[name] for name in s.ocean_history]))
    moved = np.zeros_like(s.support, dtype=float)
    # Only active owners need conditional property buffers. The reconstruction
    # keeps their transported material through trench removal and fractional
    # front repositioning, without allocating buffers for every spare slot.
    active = np.flatnonzero(s.active)
    conditional = np.zeros((len(active), s.n, len(fields)), dtype=float)
    input_stage=_ocean_ledger_stage(s,s.support,np.broadcast_to(initial,conditional.shape),active)
    operator_moments=np.zeros((len(active),len(fields)))
    old_owner = s.plate.copy()
    arrivals = np.zeros_like(s.support, dtype=float)
    for row,p in enumerate(active):
        input_fields = np.column_stack((s.support[p], s.support[p,None].T*initial,
                                        (old_owner == p).astype(float)))
        output = mesh_transport.advect(s.native_mesh, input_fields, s.omega[p], dt)
        operator_moments[row]=s.cell_area@output[:,1:-1]
        moved[p] = output[:,0]
        conditional[row] = np.divide(output[:,1:-1], moved[p,:,None],
                                    out=initial.copy(),where=moved[p,:,None] > 1e-14)
        arrivals[p] = output[:,-1]
    transport_stage=_ocean_ledger_stage(s,moved,conditional,active)
    before = moved.sum(axis=0)
    excess = np.maximum(before-1.,0.)
    ocean = s.crust == 0
    removed = np.zeros(s.n)
    removed_moments=np.zeros(len(fields))
    # Remove an incoming claim preferentially and by an explicit amount,
    # rather than counting every numerical overlap as subducted material.
    if native_subduction.enabled(s):
        capture = native_subduction.removal(s,moved,dt,arrivals=arrivals)
        for row,p in enumerate(active):
            amount=capture[p]
            moved[p]-=amount;removed+=amount
            removed_moments+=(s.cell_area*amount)@conditional[row]
    else:
        for row,p in enumerate(active):
            maturity = trench_history.downgoing_weight(s,int(p))
            available = np.maximum(moved.sum(axis=0)-1.,0.)
            amount = np.minimum(moved[p], available*maturity)*ocean
            moved[p] -= amount
            removed += amount
            removed_moments+=(s.cell_area*amount)@conditional[row]
    consumed = float(s.cell_area@removed)
    s.process_totals['ocean_consumed_km2'] += consumed
    overlap_mask = np.ones(s.n,bool) if native_subduction.enabled(s) else ocean
    residual_overlap = float(s.cell_area@(np.maximum(moved.sum(axis=0)-1.,0.)*overlap_mask))
    removal_stage=_ocean_ledger_stage(s,moved,conditional,active)
    newborn = _spreading_advance(s,moved,dt,arrivals=arrivals)
    spreading_stage=_ocean_ledger_stage(s,moved,conditional,active)
    total = moved.sum(axis=0)
    if np.any(total <= 1e-14):
        raise ValueError('Native surface transport lost all plate support in a cell.')
    if native_spreading_production(s)==1:
        s.support, s.partition_closure = _close_partition(s, moved)
    else:
        s.support = moved/total[None,:]
    normalization_stage=_ocean_ledger_stage(s,s.support,conditional,active)
    values = np.einsum('pf,pfk->fk',s.support[active],conditional)
    before_newborn=values.copy()
    # Production at a finite ridge is paired and conservatively area-limited.
    # New seafloor has zero inherited damage/weakness and mean age dt/2.
    values[:,0] = values[:,0]*(1-newborn)+dt*.5*newborn
    values[:,1:] *= (1-newborn[:,None])
    s.age = np.maximum(values[:,0],0.)
    s.ocean_relief = values[:,1]
    s.primordial_fraction = np.clip(values[:,2],0.,1.)
    for i,name in enumerate(s.ocean_history, start=3):
        s.ocean_history[name] = np.maximum(values[:,i],0.)
        if name in ('damage','weakness'):
            s.ocean_history[name] = np.minimum(s.ocean_history[name],1.)
    s.plate = np.argmax(s.support,axis=0).astype(np.int16)
    final_values=np.column_stack((s.age,s.ocean_relief,s.primordial_fraction,
                                  *[s.ocean_history[name] for name in s.ocean_history]))
    final_stage=_ocean_ledger_stage(s,s.support,np.broadcast_to(final_values,conditional.shape),active)
    stages=dict(before_transport_after_age_advance=input_stage,
                after_transport_and_conditional_reconstruction=transport_stage,
                after_measured_subduction=removal_stage,after_spreading_replacement=spreading_stage,
                after_support_normalization=normalization_stage,after_newborn_reset=final_stage)
    expected_arrivals=np.array([s.cell_area[old_owner==p].sum() for p in active])
    actual_arrivals=arrivals[active]@s.cell_area
    expected_newborn_delta=np.r_[s.cell_area@(newborn*(dt*.5-before_newborn[:,0])),
                                 -(s.cell_area*newborn)@before_newborn[:,1:]]
    def serial(stage):
        return {key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in stage.items()}
    ledger=dict(version=1,
        basis='All native control-cell fractional support and support-weighted property moments; these are not continental rock mass or a fully resolved ocean lithosphere volume.',
        owner_slots=active.tolist(),owner_uids=s.plate_uid[active].tolist(),moment_fields=fields,
        moment_units='km2 multiplied by the named stored field unit; age is Myr, ocean_relief is m, primordial_fraction/damage/weakness are dimensionless.',
        stages={name:serial(stage) for name,stage in stages.items()},
        source_owner_footprint_area_km2=expected_arrivals.tolist(),
        transported_arrival_area_km2=actual_arrivals.tolist(),
        arrival_area_residual_by_owner_km2=(actual_arrivals-expected_arrivals).tolist(),
        transport_support_area_residual_by_owner_km2=(transport_stage['owner_support_area_km2']-input_stage['owner_support_area_km2']).tolist(),
        transport_operator_output_moments_by_owner=operator_moments.tolist(),
        transport_operator_moment_residuals_by_owner=(operator_moments-input_stage['owner_moments']).tolist(),
        conditional_reconstruction_moment_delta=(transport_stage['moment_totals']-operator_moments.sum(axis=0)).tolist(),
        measured_subduction_area_km2=consumed,measured_subduction_removed_moments=removed_moments.tolist(),
        subduction_area_residual_km2=removal_stage['support_area_km2']-transport_stage['support_area_km2']+consumed,
        subduction_moment_residuals=(removal_stage['moment_totals']-transport_stage['moment_totals']+removed_moments).tolist(),
        spreading_replacement_area_delta_km2=spreading_stage['support_area_km2']-removal_stage['support_area_km2'],
        spreading_replacement_moment_deltas=(spreading_stage['moment_totals']-removal_stage['moment_totals']).tolist(),
        normalization_area_delta_km2=normalization_stage['support_area_km2']-spreading_stage['support_area_km2'],
        normalization_moment_deltas=(normalization_stage['moment_totals']-spreading_stage['moment_totals']).tolist(),
        normalization_area_policy_residual_km2=normalization_stage['support_area_km2']-spreading_stage['support_area_km2']
            -(spreading_stage['deficit_area_km2']-spreading_stage['excess_area_km2']),
        newborn_reset_moment_deltas=(final_stage['moment_totals']-normalization_stage['moment_totals']).tolist(),
        newborn_reset_policy_residuals=(final_stage['moment_totals']-normalization_stage['moment_totals']-expected_newborn_delta).tolist(),
        normalization_is_not_subduction=True,step_duration_myr=float(dt))
    s.ocean_diagnostics = dict(transport='native conservative finite-volume material moments',
        consumed_area_km2=consumed, generated_area_km2=float(s.cell_area@newborn),
        unresolved_contact_overlap_km2=residual_overlap,
        raw_overlap_km2=float(s.cell_area@(excess*overlap_mask)),
        newborn_primordial_fraction=0., display_raster_used=False,support_moment_ledger=ledger)
    if native_subduction.enabled(s):
        s.ocean_diagnostics['overlap_basis']='all actual transported fractional supports; finite measured incoming-water sink is separately bounded'
    for p in np.flatnonzero(s.active):
        s.centres[p] = rotate(s.centres[p],s.omega[p]*dt)
    pending = getattr(s,'native_arc_pending',None)
    if getattr(s,'native_arc_birth_profile_version',0)==1:
        import arc_source_cohorts
        arc_source_cohorts.advect(s,dt)
    elif pending is not None and len(pending['owner']):
        pending['xyz'] = _unit(rotate(pending['xyz'],s.omega[pending['owner']]*dt))
    backarc.advect_records(s,dt,rotate)


def deform_and_accrete(s, dt):
    """Retain column, rift, collision and arc physics on native geometry."""
    budget_before = {name: s.structure[name].copy() for name in
                    ('added_volume_km_per_reference_km2', 'denudation_m', 'foundered_m')
                     if name in s.structure}
    if getattr(s,'retained_dense_crust_version',0)==1:
        import dense_crust
        budget_before.update({name:s.structure[name].copy() for name in
                              (dense_crust.CONVERTED,dense_crust.ERODED,dense_crust.RETURNED)})
    collision, subduction, rifting = normal_partition.collision(s), normal_partition.subduction(s), s.bcode == 5
    convergence = np.maximum(-s.normal_speed,0.)
    retreat = getattr(s,'trench_retreat_speed',np.empty(0))
    if len(retreat) == len(convergence):
        convergence += np.where(subduction,retreat,0.)
    extension = np.maximum(s.normal_speed,0.)
    shortening,stretch,volcanic = np.zeros((3,s.n))
    np.maximum.at(shortening,s.ba[collision],convergence[collision])
    np.maximum.at(shortening,s.bb[collision],convergence[collision])
    np.maximum.at(stretch,s.ba[rifting],extension[rifting])
    np.maximum.at(stretch,s.bb[rifting],extension[rifting])
    s.ocean_relief *= np.exp(-dt/45.)
    arc_cells, additions = np.empty(0,int),np.empty(0)
    exact_arc=getattr(s,'native_arc_source_geometry_version',0)==1
    source_positions=np.empty((0,3));source_owners=np.empty(0,int);source_provenance=None
    if exact_arc:
        s.arc_source_candidate_diagnostics=dict(version=1,candidates=0,eligible=0,
            potential_area_km2=0.,eligible_area_km2=0.,excluded_area_km2=0.)
    if np.any(subduction):
        edges = np.flatnonzero(subduction)
        maturity = trench_history.weights(s)[edges]
        down_a = s.down[edges] == s.bp[edges]
        below = np.where(down_a,s.ba[edges],s.bb[edges])
        above = np.where(down_a,s.bb[edges],s.ba[edges])
        np.minimum.at(s.ocean_relief,below,-np.clip(convergence[edges]*25.,400.,1900.))
        direction = s.bn[edges]*np.where(down_a,1.,-1.)[:,None]
        angle = 180./RADIUS_KM
        arc_xyz = np.cos(angle)*s.bmid[edges]+np.sin(angle)*direction
        at = s._indices(arc_xyz)
        if not exact_arc: at = np.where(s.plate[at] == s.plate[above],at,above)
        np.maximum.at(volcanic,at,convergence[edges]*1.25*maturity)
        if exact_arc:
            from arc_source_geometry import classify
            intended=s.plate[above]
            routes=classify(s,arc_xyz,intended)
            juvenile=routes['eligible']
            source_positions=arc_xyz[juvenile];source_owners=intended[juvenile]
            potential=(convergence[edges]+8.)*s.bl[edges]*dt*.015*maturity
            s.arc_source_candidate_diagnostics=dict(version=1,candidates=len(edges),eligible=int(juvenile.sum()),
                potential_area_km2=float(potential.sum()),eligible_area_km2=float(potential[juvenile].sum()),
                excluded_area_km2=float(potential[~juvenile].sum()),
                reason_counts={str(reason):int(np.count_nonzero(routes['reason']==reason)) for reason in np.unique(routes['reason'])},
                position_policy='Actual180km overriding-side points; no control-center placement or fallback.')
        else:
            juvenile = (s.crust[at] == 0)|(s.crust[at] == 3)
        arc_cells = at[juvenile]
        additions = (convergence[edges][juvenile]+8.)*s.bl[edges][juvenile]*dt*.015*maturity[juvenile]
        if exact_arc and getattr(s,'native_arc_birth_profile_version',0)==1:
            import arc_source_cohorts
            source_provenance=arc_source_cohorts.build(s,edges[juvenile],positions=source_positions,
                owners=source_owners,areas=additions)
    if exact_arc and getattr(s,'native_arc_birth_profile_version',0)==1 and source_provenance is None:
        import arc_source_cohorts
        source_provenance=arc_source_cohorts.build(s,np.empty(0,int),positions=source_positions,
            owners=source_owners,areas=additions)
    # Physical belt sampling acts directly at moving material points. There is
    # no pixel cross-blur; these native control fields only supply the retained
    # bookkeeping/legacy explicit inputs to the column/rift adapters.
    deformation = structure_engine.deform(s,shortening,stretch,volcanic,dt,
                                           spherical_boundaries=True)
    # Physical volume closes after the one geometric strain, explicit magma,
    # actual exposed-rock removal and the crust returned to the mantle. Arc
    # birth/growth occurs below and has its separate juvenile source ledger; no
    # collision cap invents a sink.
    old_volume = getattr(s, 'material_pre_motion_volume_km3', None)
    if old_volume is not None:
        magma_volume = float(s.mass@(s.structure['added_volume_km_per_reference_km2']
                                    -budget_before['added_volume_km_per_reference_km2']))
        removed_volume = float(s.material_surface['area_km2']@
                               (s.structure['denudation_m']-budget_before['denudation_m']))/1000.
        # Eclogitic foundering is a SECOND physical outflow from the column
        # reservoir and is not denudation: no rock reaches the surface, so it
        # cannot share the erosion term. Without it the residual would report the
        # whole sink -- some 1e7 km3 a step -- as a failure of an identity that
        # otherwise closes to about 1e-6 km3.
        foundered_volume = (float(s.material_surface['area_km2']@
                            (s.structure['foundered_m']-budget_before['foundered_m']))/1000.
                            if 'foundered_m' in budget_before else 0.)
        actual_volume = float(s.material_surface['area_km2']@s.structure['thickness_km'])
        contraction=0.
        if getattr(s,'retained_dense_crust_version',0)==1:
            contraction=float(s.mass@(s.structure[dense_crust.CONVERTED]-budget_before[dense_crust.CONVERTED]))*(1./dense_crust.RATIO-1.)
        s.material_column_budget = dict(before_motion_volume_km3=float(old_volume),
            magmatic_volume_km3=magma_volume, eroded_volume_km3=removed_volume,
            foundered_volume_km3=foundered_volume,
            phase_contraction_volume_km3=contraction,
            after_columns_volume_km3=actual_volume,
            residual_km3=actual_volume-(old_volume+magma_volume-removed_volume-foundered_volume-contraction),
            juvenile_arc_source_accounted_separately=True)
        if getattr(s,'retained_dense_crust_version',0)==1:
            density_extra=1./dense_crust.RATIO-1.
            exported=removed_volume+density_extra*float(s.mass@(s.structure[dense_crust.ERODED]-budget_before[dense_crust.ERODED]))
            returned=float(s.mass@(s.structure[dense_crust.RETURNED]-budget_before[dense_crust.RETURNED]))/dense_crust.RATIO
            actual_mass=float(s.mass@dense_crust.mass_volume(s.structure))*2800.*1e9
            expected_mass=s.material_pre_motion_mass_kg+(magma_volume-exported-returned)*2800.*1e9
            s.material_column_budget.update(after_columns_mass_kg=actual_mass,
                before_motion_mass_kg=s.material_pre_motion_mass_kg,phase_mass_residual_kg=actual_mass-expected_mass)
    progressive_rifting.update(s,dt,stretch,realized_extension=(
        deformation['parcel']['extension_strain'],deformation['trace']['extension_strain']))
    for plan in local_accretion.plan_accretions(s,dt):
        local_accretion.apply_accretion(s,plan)
    account_source_work = (getattr(s,'material_mechanics_version',0)==1
        and getattr(s,'native_arc_birth_profile_version',0)==1
        and bool(s.config.get('deforming_regions',1)))
    if account_source_work:
        from native_material_evolution import reference_energy_state
        before_arc_energy = reference_energy_state(s)
    if exact_arc:
        source_arguments=dict(positions=source_positions,owners=source_owners)
        if source_provenance is not None:source_arguments['source_provenance']=source_provenance
        import arc_ribbon
        if arc_ribbon.version(s):source_arguments['dt']=dt
        add_arc_crust(s,arc_cells,additions,**source_arguments)
    else:
        add_arc_crust(s,arc_cells,additions)
    if account_source_work:
        after_arc_energy = reference_energy_state(s)
        gravity = s.deformation_diagnostics['gravitational_relaxation']
        gravity['post_transport_column_and_tectonic_energy_change_km4'] = before_arc_energy-gravity['energy_after_km4']
        gravity['arc_emplacement_reference_energy_before_km4'] = before_arc_energy
        gravity['arc_emplacement_reference_energy_after_km4'] = after_arc_energy
        gravity['arc_emplacement_reference_energy_change_km4'] = after_arc_energy-before_arc_energy
        gravity['arc_emplacement_reference_energy_change_j'] = (after_arc_energy-before_arc_energy)*2800.*9.81*(1-2800./3300.)*1e12
        gravity['source_work_scope'] = 'Measured full reference-energy change across the arc batch, including growth geometry; pending magma is excluded from physical columns. Column/rift/accretion changes before that batch are recorded separately.'
    if np.any(collision):
        pairs = np.unique(np.sort(np.column_stack((s.bp[collision],s.bq[collision])),axis=1),axis=0)
        for p,q in pairs:
            key=('collision',int(p),int(q))
            # collision_events is deduplicated by event_key, so it counts DISTINCT
            # plate pairs that have ever collided and saturates -- it read 1.0 through
            # 5,000 km of active shortening and was mistaken for "collision is not
            # happening". Count the occurrences separately so the two questions
            # ("which pairs" vs "how much, now") have separate answers.
            s.process_totals['collision_pair_occurrences'] = s.process_totals.get(
                'collision_pair_occurrences', 0.)+1
            if key in s.event_keys:
                continue
            pair=collision&(((s.bp == p)&(s.bq == q))|((s.bp == q)&(s.bq == p)))
            s.process_totals['collision_events'] += 1
            s._record('collision',f'{s.names[p]} and {s.names[q]} began continental shortening and mountain building.',
                key,plates=(p,q),xyz=np.sum(s.bmid[pair]*s.bl[pair,None],axis=0),
                details=dict(boundary_length_km=float(s.bl[pair].sum()),
                             mean_convergence_km_myr=float(np.average(convergence[pair],weights=s.bl[pair]))))


def _arc_triangles(centres, direction, areas):
    """Equilateral minor spherical triangles of the requested physical area."""
    centres = _unit(centres)
    east = _unit(direction-centres*np.sum(direction*centres,axis=1)[:,None])
    north = np.cross(centres,east)
    phi = np.arange(3)*2*np.pi/3
    bearings = east[:,None,:]*np.cos(phi)[None,:,None]+north[:,None,:]*np.sin(phi)[None,:,None]
    def evaluate(radius):
        triangle = centres[:,None,:]*np.cos(radius)[:,None,None]+bearings*np.sin(radius)[:,None,None]
        a,b,c = triangle.transpose(1,0,2)
        determinant = np.einsum('ij,ij->i',a,np.cross(b,c))
        denominator = 1+np.einsum('ij,ij->i',a,b)+np.einsum('ij,ij->i',b,c)+np.einsum('ij,ij->i',c,a)
        area = 2*np.arctan2(determinant,denominator)*RADIUS_KM**2
        return triangle,area
    low, high = np.zeros(len(areas)), np.full(len(areas),1.2)
    if np.any(areas <= 0) or np.any(areas > evaluate(high)[1]):
        raise ValueError('Juvenile triangle area must be positive and fit inside its spherical cap.')
    for _ in range(48):
        middle = (low+high)*.5
        less = evaluate(middle)[1] < areas
        low,high = np.where(less,middle,low),np.where(less,high,middle)
    return evaluate((low+high)*.5)[0]


def _grow_arc_triangles(triangles, areas):
    """Expand an existing arc without regularizing its deformed shape.

    All corner bearings and ratios of their angular distances from the current
    spherical centroid survive. One bounded geodesic scale matches the target
    area. The cap remains inside an open hemisphere, so outward scaling nests
    these convex triangles and the area search is monotonic.
    """
    triangles, areas = np.asarray(triangles, float), np.asarray(areas, float)
    if (areas.ndim != 1 or triangles.shape != (len(areas), 3, 3)
            or not np.isfinite(triangles).all() or not np.isfinite(areas).all()
            or not np.allclose(np.linalg.norm(triangles, axis=2), 1., atol=1e-10, rtol=0.)):
        raise ValueError('Arc growth requires finite unit triangular footprints and target areas.')
    centres = _unit(triangles.sum(axis=1))
    dot = np.einsum('nvi,ni->nv', triangles, centres)
    tangent = triangles-centres[:, None, :]*dot[:, :, None]
    length = np.linalg.norm(tangent, axis=2)
    angle = np.arctan2(length, dot)
    if np.any(length <= 1e-15) or np.any(angle >= 1.2):
        raise ValueError('Growing arc footprints must lie inside their minor spherical cap.')
    bearings = tangent/length[:, :, None]

    def evaluate(scale):
        radius = angle*scale[:, None]
        points = centres[:, None, :]*np.cos(radius)[:, :, None]+bearings*np.sin(radius)[:, :, None]
        a, b, c = points.transpose(1, 0, 2)
        determinant = np.einsum('ij,ij->i', a, np.cross(b, c))
        denominator = 1+np.einsum('ij,ij->i', a, b)+np.einsum('ij,ij->i', b, c)+np.einsum('ij,ij->i', c, a)
        return points, 2*np.arctan2(determinant, denominator)*RADIUS_KM**2

    low = np.ones(len(areas)); high = 1.2/np.max(angle, axis=1)
    initial = evaluate(low)[1]
    if np.any(initial <= 0.) or np.any(areas < initial*(1-1e-11)) or np.any(areas > evaluate(high)[1]):
        raise ValueError('Arc growth target must increase area and fit inside its spherical cap.')
    for _ in range(52):
        middle = (low+high)*.5
        less = evaluate(middle)[1] < areas
        low, high = np.where(less, middle, low), np.where(less, high, middle)
    result = evaluate((low+high)*.5)[0]
    unchanged = np.isclose(areas, initial, rtol=1e-12, atol=0.)
    result[unchanged] = triangles[unchanged]
    return result


def _legacy_add_arc_crust(s, cells, additions):
    """Accumulate magma as exact-area movable triangles, including tiny islands.

    One current owner/cell has at most one new growing private triangle. Existing
    continental faces are never resized. Areas smaller than 0.001 km² remain in
    an owner-attached material budget until representable; they are not promoted
    to a whole computational cell or discarded. Every materialized addition
    appears in both the reference-area ledger and an actual spherical face.
    """
    cells, additions = np.asarray(cells),np.asarray(additions,float)
    if (cells.ndim != 1 or additions.shape != cells.shape or cells.dtype.kind not in 'iu'
            or np.any(cells < 0) or np.any(cells >= s.n)
            or not np.isfinite(additions).all() or np.any(additions < 0)):
        raise ValueError('Arc additions need valid native cells and finite nonnegative areas.')
    pending = getattr(s,'native_arc_pending',dict(xyz=np.empty((0,3)),owner=np.empty(0,np.int16),area=np.empty(0)))
    owners = s.plate[cells]
    incoming_cells = np.r_[cells,s._indices(pending['xyz']) if len(pending['owner']) else np.empty(0,int)]
    incoming_owners = np.r_[owners,pending['owner']]
    incoming_areas = np.r_[additions,pending['area']]
    if not len(incoming_cells) or not np.any(incoming_areas > 0):
        return dict(added_area_km2=0.,pending_area_km2=float(pending['area'].sum()),new_faces=0)
    valid = incoming_areas > 0
    keys,inverse = np.unique(incoming_owners[valid].astype(np.int64)*s.n+incoming_cells[valid],return_inverse=True)
    amount = np.bincount(inverse,weights=incoming_areas[valid])
    cells,owners = (keys%s.n).astype(int),(keys//s.n).astype(np.int16)
    resolved = amount >= .001
    s.native_arc_pending = dict(xyz=s.xyz[cells[~resolved]].copy(),owner=owners[~resolved].copy(),area=amount[~resolved].copy())
    cells,owners,amount,keys = cells[resolved],owners[resolved],amount[resolved],keys[resolved]
    if not len(cells):
        return dict(added_area_km2=0.,pending_area_km2=float(s.native_arc_pending['area'].sum()),new_faces=0)
    surface=s.material_surface
    known=np.asarray(getattr(s,'native_arc_patch_ids',np.empty(0,np.int64)),np.int64)
    references=np.bincount(surface['faces'].ravel(),minlength=len(surface['vertices']))
    private=np.all(references[surface['faces']] == 1,axis=1)
    existing=np.flatnonzero((s.kind == 3)&np.isin(s.parcel_patch,known)&private)
    destinations=np.full(len(cells),-1,int)
    if len(existing):
        existing_keys=s.parcel_plate[existing].astype(np.int64)*s.n+s._indices(s.pos[existing])
        old_keys,first=np.unique(existing_keys,return_index=True)
        at=np.searchsorted(old_keys,keys)
        match=(at < len(old_keys))&(old_keys[np.minimum(at,len(old_keys)-1)] == keys)
        destinations[match]=existing[first[at[match]]]
    growing=destinations >= 0
    if np.any(growing):
        chosen=destinations[growing]
        # Magma adds area to an already deformed footprint. Reference area
        # is a material ledger and no longer equals current geometric area.
        old_area=surface['area_km2'][chosen].copy()
        old_reference=s.mass[chosen].copy()
        target=old_area+amount[growing]
        vertices=surface['faces'][chosen]
        old_triangles=surface['vertices'][vertices].copy()
        geometry=_grow_arc_triangles(old_triangles,target)
        surface['vertices'][vertices]=geometry
        surface['reference_area_km2'][chosen]=old_reference+amount[growing]
        s.mass[chosen]=old_reference+amount[growing]
        material_surface.refresh_geometry(surface)
        s.structure['area_factor'][chosen]=surface['area_km2'][chosen]/s.mass[chosen]
        patched=s.parcel_patch[chosen]
        traces=np.flatnonzero(np.isin(s.trace_patch,patched)&(s.trace_kind == 3))
        if len(traces):
            by_patch={int(uid):i for i,uid in enumerate(patched)}
            local=np.array([by_patch[int(uid)] for uid in s.trace_patch[traces]])
            weights=np.linalg.solve(np.swapaxes(old_triangles[local],1,2),s.trace_xyz[traces,:,None])[...,0]
            weights/=weights.sum(axis=1)[:,None]
            s.trace_xyz[traces]=_unit(np.einsum('ni,nij->nj',weights,geometry[local]))
            from native_material_evolution import _transport_tangent
            s.trace_rift_tangent[traces]=_transport_tangent(old_triangles[local],geometry[local],
                s.trace_rift_tangent[traces],s.trace_xyz[traces])
            s.trace_structure['area_factor'][traces]=s.structure['area_factor'][chosen[local]]
        # Existing column thickness is preserved during lateral magma growth;
        # the added volume is the new area times that column's thickness.
        # Rebase per-reference volume counters to the enlarged reference area.
        ledger=s.structure['added_volume_km_per_reference_km2']
        ledger[chosen]=(ledger[chosen]*old_reference+amount[growing]*s.structure['thickness_km'][chosen])/s.mass[chosen]
        structure_engine.replenish(s,chosen,traces,700.)
    newborn=~growing
    count=int(np.count_nonzero(newborn))
    if count:
        nc,no,na=cells[newborn],owners[newborn],amount[newborn]
        centres=s.xyz[nc]
        direction=s.native_mesh['vertices'][s.native_mesh['faces'][nc,0]]
        triangles=_arc_triangles(centres,direction,na)
        patches=np.arange(s.next_patch_uid,s.next_patch_uid+count,dtype=np.int64)
        s.next_patch_uid += count
        column_fields={name:np.zeros((count,)+value.shape[1:],dtype=value.dtype) for name,value in surface['columns'].items()}
        provenance={name:np.zeros((count,)+value.shape[1:],dtype=value.dtype) for name,value in surface['provenance'].items()}
        material_surface.append_surface(surface,triangles.reshape(-1,3),np.arange(3*count).reshape(-1,3),
            no,np.full(count,3,np.uint8),face_id=patches,columns=column_fields,provenance=provenance,
            reference_area_km2=na)
        s.native_arc_patch_ids=np.r_[known,patches]
        values=dict(pos=centres,mass=na,parcel_plate=no,kind=np.full(count,3,np.uint8),
            relief=np.full(count,850.),suture=np.full(count,.55),parcel_birth=np.full(count,s.t),
            parcel_patch=patches,parcel_craton=np.full(count,-1,np.int32),
            **s._empty_rift_material(count))
        for name,value in values.items():
            setattr(s,name,np.concatenate((getattr(s,name),value)))
        structure_engine.append_parcels(s,count)
        s._new_arc_traces(nc,no,patches)
    added=float(amount.sum())
    s.process_totals['arc_added_km2'] += added
    s._sync_material()
    # Exact occupancy is rebuilt at the next native exposure refresh. Do not
    # allow a mid-step trench-sweep birth to reuse old local-accretion contacts.
    s._owner_occupancy_signature=None
    for p in np.unique(owners):
        select=owners == p
        s._record('island_arc',f'Subduction-generated magma added persistent buoyant island-arc crust on {s.names[p]}.',
            ('arc',int(s.plate_uid[p]),int(s.t//100)),plates=(p,),
            xyz=np.sum(s.xyz[cells[select]]*amount[select,None],axis=0),
            details=dict(added_area_this_step_km2=float(amount[select].sum()),reporting_interval_myr=100,
                         representation='exact-area connected spherical material triangles'))
    return dict(added_area_km2=added,pending_area_km2=float(s.native_arc_pending['area'].sum()),new_faces=count)


def add_arc_crust(s,cells,additions,*,positions=None,owners=None,source_provenance=None,dt=None):
    """Build connected volcanic foundations; preserve explicit legacy opt-in."""
    if getattr(s,'native_arc_material_version',1)==0:
        return _legacy_add_arc_crust(s,cells,additions)
    from native_arc_material import add_arc_crust as add_connected_arc
    # Only the arc ribbon limits what one interval may place, so only it is told the interval.
    interval={} if dt is None else dict(dt=dt)
    if source_provenance is not None:
        return add_connected_arc(s,cells,additions,positions=positions,owners=owners,
            source_provenance=source_provenance,**interval)
    return add_connected_arc(s,cells,additions,positions=positions,owners=owners,**interval)

