"""Finite half-stage spreading fronts and locally paired ocean production.

The front is a compact spherical reconstruction, independent of the advected
ocean properties. Its subcell position survives between steps. Continental
parcels, mantle forcing and trench retreat are not changed here.
"""
import numpy as np

from ridge_geometry import candidate_cells, finite_half_stage, rotate, normal_motion_threshold

RADIUS_KM = 6371.


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


def _active(s):
    """Current velocity, not last step's boundary label, controls production."""
    valid = ((s.plate[s.ba] == s.bp) & (s.plate[s.bb] == s.bq)
             & s.active[s.bp] & s.active[s.bq])
    dv = np.cross(s.omega[s.bq]-s.omega[s.bp], s.bmid)*RADIUS_KM
    normal = np.sum(dv*s.bn, axis=1)
    shear = np.linalg.norm(dv-normal[:, None]*s.bn, axis=1)
    threshold = (normal_motion_threshold(shear,
                    spreading_version=getattr(s,'normal_spreading_version',0)) if hasattr(s, 'native_mesh')
                 else normal_motion_threshold(shear, obliquity=.35))
    valid &= normal > threshold
    valid &= (s.crust[s.ba] == 0) & (s.crust[s.bb] == 0)
    return np.flatnonzero(valid)


def advance(s, transported, dt, arrivals=None):
    """Reconstruct moving fronts and return the new-crust fraction per cell.

    Membership changes are restricted to existing local two-plate territory.
    A real third owner, including at a junction, blocks reconstruction. Unlike
    a fractional-support threshold, this guard cannot admit a remote tail.
    """
    birth = np.zeros(s.n)
    s.spreading_diagnostics = dict(active_segments=0, reconstructed_cells=0,
                                  generated_area_km2=0., rejected_foreign_claims=0)
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
    spacing = np.pi*RADIUS_KM/s.h
    # A compact linear transition preserves continuous motion below one pixel.
    band = np.maximum(spacing*1.5, opening/2+spacing)
    radius = np.hypot(band, length/2+spacing)
    segment, cells = candidate_cells(moved, radius, s.w, s.h)
    owner = s.plate
    pp, qq = p[segment], q[segment]
    eligible = ((owner[cells] == pp) | (owner[cells] == qq)) & (s.crust[cells] == 0)
    if arrivals is not None:
        incoming_other = arrivals.sum(axis=0)[cells]-arrivals[pp, cells]-arrivals[qq, cells]
        eligible &= incoming_other < 1e-6
    # Check true neighboring owners, not hidden fractional claims. Pole samples
    # use the same spherical continuation as transport.
    for neighbor in (s.east[cells], s.west[cells], s.north[cells], s.south[cells]):
        eligible &= (owner[neighbor] == pp) | (owner[neighbor] == qq)
    signed = np.arcsin(np.clip(np.sum(s.xyz[cells]*normals[segment], axis=1), -1, 1))*RADIUS_KM
    along = np.arcsin(np.clip(np.sum(s.xyz[cells]*tangent[segment], axis=1), -1, 1))*RADIUS_KM
    eligible &= (np.abs(signed) <= band[segment]) & (np.abs(along) <= length[segment]/2+spacing*.6)
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
        rejected_foreign_claims=0)
    return birth


def refresh(s):
    """Resample surviving moved fronts onto their new contact faces.

    Intersections retain the moved spherical axis, rather than snapping back
    to a grid midpoint. Changed owners, closing fronts and new junctions expire
    the old front locally; a newly opened contact starts from its own geometry.
    """
    if getattr(s,'native_spreading_version',0)==1:
        from native_spreading import prepare
        prepare(s)
        # Exact native pieces follow the advected continuous owner-support
        # contours. They never share a cached midpoint by duplicate cell key.
        s._spreading_pending=None
        return
    edge = _active(s)
    mid, normal = s.bmid[edge].copy(), s.bn[edge].copy()
    pending = getattr(s, '_spreading_pending', None)
    if pending is not None and len(edge):
        for cell in (s.ba[edge], s.bb[edge]):
            at = pending['nearest'][cell]
            safe = at >= 0
            at = np.maximum(at, 0)
            pu, qu = s.plate_uid[s.bp[edge]], s.plate_uid[s.bq[edge]]
            direct = (pu == pending['p_uid'][at]) & (qu == pending['q_uid'][at])
            reverse = (pu == pending['q_uid'][at]) & (qu == pending['p_uid'][at])
            n = pending['normal'][at]*np.where(direct, 1., -1.)[:, None]
            a, b = s.xyz[s.ba[edge]], s.xyz[s.bb[edge]]
            fa, fb = np.sum(a*n, axis=1), np.sum(b*n, axis=1)
            fraction = np.divide(-fa, fb-fa, out=np.full(len(edge), -1.), where=np.abs(fb-fa) > 1e-12)
            use = safe & (direct | reverse) & (fraction >= 0) & (fraction <= 1)
            mid[use] = _unit(a[use]*(1-fraction[use, None])+b[use]*fraction[use, None])
            normal[use] = n[use]
    key = s.ba[edge].astype(np.int64)*s.n+s.bb[edge]
    order = np.argsort(key)
    s.spreading_fronts = dict(key=key[order], mid=mid[order], normal=normal[order],
                             p_uid=s.plate_uid[s.bp[edge]][order].copy(),
                             q_uid=s.plate_uid[s.bq[edge]][order].copy())
    # This is an intra-step lookup, not persistent history/checkpoint state.
    s._spreading_pending = None
