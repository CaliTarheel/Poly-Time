"""Continuous two-owner interfaces inside a native spherical control mesh.

Control edges carry flux and contact identity; their three preferred mesh
directions are not physical fault normals. Reconstruct the equal-support
contour of shared, area-weighted vertex scores instead. Each triangle holds
linear barycentric scores, so its contour is a finite great-circle segment.
Clip that segment wherever a third owner's score dominates the tied pair.

The reconstructed length is apportioned once among nearby existing contact
keys. This preserves the ridge front cache without mutating labels, support,
velocities or material. Existing force laws use the corrected geometry; the
reconstruction prescribes no extra forcing. Contacts too small to be represented by the
vertex field retain explicit raw geometry rather than disappearing silently. A raw
categorical seam incompatible with an already represented local P1 interface is
instead retained as a zero-length contact; it must not add a second physical trace.

Each contour piece is listed once, under the contact that takes most of it
(``contact_index``), while its length is divided among every nearby contact of
the pair. A contact can therefore hold length and own no piece. Anything then
measured on the pieces listed under a contact, such as the ocean fraction of its
incoming side, describes other ground than the length it multiplies, and is zero
for a contact with no piece: a trench contact reads as blocked by a continent
that is not there, and a control cell changing owner moves hundreds of
kilometres between a pulling trench and a jammed one. Contact share version 1
returns the division itself (``share_segment``, ``share_contact``,
``share_weight``), so a measure can be taken on the ground a contact's length
came from. `shares` gives the per-contact sum of any per-piece quantity.
"""
from __future__ import annotations

import numpy as np


SHARE_VERSION = 1


def _unit(value):
    return value/np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-30)


def share_version(s):
    value = getattr(s, 'contact_share_version', 0)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value not in (0, SHARE_VERSION):
        raise ValueError('Unsupported contact share version.')
    return int(value)


def upgrade_shares(s):
    """Measure contacts on the ground their length came from, from the next reconstruction on."""
    if share_version(s) == SHARE_VERSION:
        return dict(getattr(s, 'contact_share_migration', {}))
    report = dict(version=SHARE_VERSION, time_myr=float(s.t),
                  selection='Per-contact measures use the division of contour pieces that gives each contact its length.',
                  effective_from='the next boundary reconstruction; the geometry in hand carries no shares')
    s.contact_share_version = SHARE_VERSION
    s.contact_share_migration = report
    return dict(report)


def has_shares(geometry):
    return isinstance(geometry, dict) and 'share_segment' in geometry


def shares(geometry, values, count):
    """Per-contact sum of a per-piece quantity, each piece divided as its length was."""
    values = np.asarray(values, float)
    if values.shape != (len(geometry['contact_index']),):
        raise ValueError('A shared quantity needs one value for each contour piece.')
    return np.bincount(geometry['share_contact'], weights=geometry['share_weight']*values[geometry['share_segment']],
                       minlength=count)


def _vertex_support(mesh, support):
    faces = np.asarray(mesh['faces'])
    vertex = faces.ravel()
    area = np.asarray(mesh['area_km2'])
    denominator = np.bincount(vertex, weights=np.repeat(area, 3), minlength=len(mesh['vertices']))
    result = np.zeros((len(support), len(denominator)))
    for owner in range(len(support)):
        result[owner] = np.divide(np.bincount(vertex, weights=np.repeat(support[owner]*area, 3),
                                             minlength=len(denominator)), denominator,
                                  out=np.zeros(len(denominator)), where=denominator > 0)
    return result


def prepare_owner_sampling(mesh, plate, support, *, locator=None):
    """Prepare the same shared P1 scores used by the boundary contours.

    This is a display/diagnostic reconstruction. It neither changes finite
    volume ownership nor advects material. Persisting these vertex scores and
    their owner slots allows saved epochs to reproduce the same continuous
    ocean ownership without depending on their display raster resolution.
    """
    plate = np.asarray(plate)
    support = np.asarray(support, float)
    faces = np.asarray(mesh['faces'])
    if (plate.shape != (len(faces),) or not np.issubdtype(plate.dtype, np.integer)
            or support.ndim != 2 or support.shape[1] != len(faces)
            or not np.isfinite(support).all() or np.any(support < 0)
            or np.any(plate < 0) or np.any(plate >= len(support))):
        raise ValueError('Owner sampling requires finite native owner/support fields.')
    slots = np.union1d(np.flatnonzero(np.max(support, axis=1) > 0), np.unique(plate)).astype(np.int32)
    return owner_sampling_context(mesh, slots, _vertex_support(mesh, support[slots]), plate, locator=locator)


def owner_sampling_context(mesh, owner_slots, scores, fallback_owner, *, locator=None):
    """Prepare saved compact vertex scores, including local domain lookup.

    Scores are not recomputed here. The same context supports live snapshots
    and saved-epoch exports. An owner's domain source is selected from actual
    nearby winning faces, never from a remote plate with the same name.
    """
    from mesh_geometry import build_locator, geometry
    from mesh_transport import face_vertex_stencil
    slots = np.asarray(owner_slots)
    scores = np.asarray(scores, float)
    plate = np.asarray(fallback_owner)
    faces = np.asarray(mesh['faces'])
    if (slots.ndim != 1 or not len(slots) or not np.issubdtype(slots.dtype, np.integer)
            or np.any(slots < 0) or np.any(np.diff(slots) <= 0)
            or scores.shape != (len(slots), len(mesh['vertices'])) or not np.isfinite(scores).all()
            or np.any(scores < 0) or plate.shape != (len(faces),)
            or not np.issubdtype(plate.dtype, np.integer)):
        raise ValueError('Saved owner reconstruction needs sorted slots and finite vertex scores.')
    vertex_faces, weights = face_vertex_stencil(mesh)
    neighbours = mesh.get('face_neighbors')
    if neighbours is None:
        neighbours = geometry(mesh['vertices'], faces)['face_neighbors']
    return dict(scores=scores, owner_slots=slots, faces=faces, fallback_owner=plate.copy(),
                locator=build_locator(mesh['vertices'], faces) if locator is None else locator,
                vertex_faces=vertex_faces, vertex_face_valid=weights > 0,
                face_neighbors=np.asarray(neighbours),
                face_xyz=_unit(np.asarray(mesh['vertices'])[faces].sum(axis=1)))


def _owner_cells(points, cells, owners, prepared):
    """Resolve a domain source locally; -1 means no local winner is known."""
    labels = prepared['fallback_owner']
    result = cells.copy()
    changed = np.flatnonzero(labels[cells] != owners)
    for row in changed:
        vertices = prepared['faces'][cells[row]]
        nearby = prepared['vertex_faces'][vertices][prepared['vertex_face_valid'][vertices]]
        nearby = np.unique(nearby)
        selected = nearby[labels[nearby] == owners[row]]
        for _ in range(2):
            if len(selected):
                break
            neighbours = prepared['face_neighbors'][nearby].ravel()
            nearby = np.unique(np.r_[nearby, neighbours[neighbours >= 0]])
            selected = nearby[labels[nearby] == owners[row]]
        # Sorted IDs provide deterministic tie order. This bounded graph
        # search cannot borrow a distant parent domain for an isolated sheet.
        if len(selected):
            distance = np.sum((prepared['face_xyz'][selected]-points[row])**2, axis=1)
            nearest = np.flatnonzero(distance <= distance.min()+1e-14)[0]
            result[row] = selected[nearest]
        else:
            result[row] = -1
    return result


def sample_owners(points, prepared, *, chunk_size=65536, return_cells=False):
    """Argmax continuous support with stable ties and bounded working memory.

    Each owner is evaluated separately inside a query chunk: memory scales as
    O(owners*vertices + queries + chunk_size), never queries*owners. Exact or
    roundoff-scale ties choose the lowest owner slot; this matches the contour
    convention that shared score differences below 1e-13 are algebraic ties.
    The fallback applies only where every interpolated score is zero. With
    return_cells=True, also return a nearby native face whose winning owner
    matches the selected owner. Unresolved local ownership has cell=-1; this
    never changes the selected owner or searches the remote parent plate.
    """
    from mesh_geometry import locate_points
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Owner samples require finite (N,3) spherical points.')
    if isinstance(chunk_size, bool) or int(chunk_size) != chunk_size or chunk_size < 1:
        raise ValueError('Owner sampling chunk size must be a positive integer.')
    scores = prepared['scores']
    slots = np.asarray(prepared['owner_slots'])
    result = np.empty(len(points), dtype=prepared['fallback_owner'].dtype)
    source_cells = np.empty(len(points), np.int32) if return_cells else None
    for start in range(0, len(points), int(chunk_size)):
        stop = min(start+int(chunk_size), len(points))
        cells, bary = locate_points(points[start:stop], prepared['locator'])
        if np.any(cells < 0):
            raise ValueError('Saved native control mesh does not cover an owner query.')
        vertices = prepared['faces'][cells]
        best = np.full(len(cells), -np.inf)
        owner = np.full(len(cells), slots[0], result.dtype)
        for row, slot in enumerate(slots):
            value = np.sum(scores[row, vertices]*bary, axis=1)
            stronger = value > best+1e-13
            owner[stronger] = slot
            best[stronger] = value[stronger]
        missing = best <= 0
        owner[missing] = prepared['fallback_owner'][cells[missing]]
        result[start:stop] = owner
        if return_cells:
            source_cells[start:stop] = _owner_cells(points[start:stop], cells, owner, prepared)
    return (result, source_cells) if return_cells else result


def sample_owner_cells(points, prepared, *, chunk_size=65536):
    """Owner/domain-source arrays with an explicit unresolved-query count."""
    owners, cells = sample_owners(points, prepared, chunk_size=chunk_size, return_cells=True)
    return dict(owners=owners, cells=cells,
                diagnostics=dict(unresolved_queries=int(np.count_nonzero(cells < 0)), queries=len(cells)))


def _pair_segments(mesh, scores, p, q):
    """Return dominant p=q contour pieces, including proper triple clipping."""
    faces = np.asarray(mesh['faces'])
    vertices = np.asarray(mesh['vertices'])
    phi = scores[q, faces]-scores[p, faces]
    # Shared algebraic ties must stay shared at native vertices. Tiny signed
    # summation noise otherwise emits overlapping near-zero corner slivers.
    phi[np.abs(phi) < 1e-13] = 0.
    candidates = np.flatnonzero((phi.min(axis=1) <= 0) & (phi.max(axis=1) >= 0)
                                & (np.max(np.abs(phi), axis=1) > 1e-13))
    if not len(candidates):
        return None
    value = phi[candidates]
    triangle = vertices[faces[candidates]]
    intersections = np.zeros((len(candidates), 3, 3))
    valid = np.zeros((len(candidates), 3), bool)
    for edge, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
        fa, fb = value[:, a], value[:, b]
        fraction = np.divide(-fa, fb-fa, out=np.zeros(len(fa)), where=np.abs(fb-fa) > 1e-15)
        valid[:, edge] = (fa*fb <= 0) & ((fa != 0) | (fb != 0))
        intersections[:, edge, a] = 1-fraction
        intersections[:, edge, b] = fraction
    # A zero-valued vertex can occur on two edges. Select the farthest valid
    # pair; this also represents an entire zero-valued edge exactly once here.
    pair = np.array([[0, 1], [1, 2], [2, 0]])
    point = np.einsum('nki,nij->nkj', intersections, triangle)
    distance = np.sum((point[:, pair[:, 0]]-point[:, pair[:, 1]])**2, axis=2)
    distance = np.where(valid[:, pair[:, 0]] & valid[:, pair[:, 1]], distance, -1.)
    choice = distance.argmax(axis=1)
    keep = distance[np.arange(len(choice)), choice] > 1e-24
    candidates, triangle, value = candidates[keep], triangle[keep], value[keep]
    selected = pair[choice[keep]]
    intersections = intersections[keep]
    rows = np.arange(len(candidates))
    first = intersections[rows, selected[:, 0]]
    second = intersections[rows, selected[:, 1]]
    lower = np.zeros(len(candidates)); upper = np.ones(len(candidates))
    # Scores remain linear along the unnormalised barycentric chord. Clip
    # there before normalising endpoints onto the sphere.
    for other in range(len(scores)):
        if other == p or other == q:
            continue
        difference = scores[p, faces[candidates]]-scores[other, faces[candidates]]
        d0 = np.sum(difference*first, axis=1)
        d1 = np.sum(difference*second, axis=1)
        crossing = np.divide(d0, d0-d1, out=np.zeros(len(d0)), where=np.abs(d0-d1) > 1e-15)
        lower = np.where((d0 < -1e-12) & (d1 >= -1e-12), np.maximum(lower, crossing), lower)
        upper = np.where((d1 < -1e-12) & (d0 >= -1e-12), np.minimum(upper, crossing), upper)
        upper[(d0 < -1e-12) & (d1 < -1e-12)] = -1.
    keep = upper-lower > 1e-12
    if not keep.any():
        return None
    clipped = int(np.count_nonzero(keep & ((lower > 1e-10) | (upper < 1-1e-10))))
    candidates, triangle, value = candidates[keep], triangle[keep], value[keep]
    first, second, lower, upper = first[keep], second[keep], lower[keep], upper[keep]
    difference = second-first
    a = _unit(np.einsum('ni,nij->nj', first+lower[:, None]*difference, triangle))
    b = _unit(np.einsum('ni,nij->nj', first+upper[:, None]*difference, triangle))
    # The homogeneous linear score difference is g dot x. Its zero plane has
    # the exact p-to-q normal, independent of the mesh coordinate basis.
    normal = _unit(value[:, 0, None]*np.cross(triangle[:, 1], triangle[:, 2])
                   + value[:, 1, None]*np.cross(triangle[:, 2], triangle[:, 0])
                   + value[:, 2, None]*np.cross(triangle[:, 0], triangle[:, 1]))
    middle = _unit(a+b)
    length = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a*b, axis=1))*float(mesh.get('radius_km', 6371.))
    # On an exact native edge both adjoining triangles emit the same piece.
    # Deduplicate only that geometric identity, not nearby distinct branches.
    rounded_a, rounded_b = np.round(a, 12), np.round(b, 12)
    swap = np.array([tuple(x) > tuple(y) for x, y in zip(rounded_a, rounded_b)])
    key = np.concatenate((np.where(swap[:, None], rounded_b, rounded_a),
                          np.where(swap[:, None], rounded_a, rounded_b)), axis=1)
    _, unique = np.unique(key, axis=0, return_index=True)
    unique.sort()
    return dict(face=candidates[unique], start=a[unique], end=b[unique], mid=middle[unique],
                normal=normal[unique], length=length[unique], clipped=clipped)


def _represented_crossing(mesh, scores, p, q, edge, incident_faces, pieces):
    """Prove a strict two-face P1 crossing through a categorical raw edge.

    Nearby pieces of the same pair are insufficient: they might belong to a
    different component or terminate at a triple point. Both actual incident
    triangles must contain one finite piece, joined at an interior crossing of
    their shared edge, with the pair strictly dominant over every third owner.
    The score/endpoint tolerances only test the existing P1 algebra and geometric
    identity; they do not change reconstruction, ownership or force parameters.
    """
    vertices = np.asarray(mesh['vertices'])
    ends = np.asarray(mesh['edge_vertices'])[edge]
    phi = scores[q, ends]-scores[p, ends]
    if not ((phi[0] > 1e-13 and phi[1] < -1e-13)
            or (phi[0] < -1e-13 and phi[1] > 1e-13)):
        return None
    fraction = float(-phi[0]/(phi[1]-phi[0]))
    if not 1e-10 < fraction < 1.-1e-10:
        return None
    crossing = _unit((1.-fraction)*vertices[ends[0]]+fraction*vertices[ends[1]])
    other = np.flatnonzero((np.arange(len(scores)) != p) & (np.arange(len(scores)) != q))
    matched = []
    for face in incident_faces:
        rows = np.flatnonzero(pieces['face'] == face)
        if len(rows) != 1:
            return None
        row = int(rows[0])
        points = np.array([pieces['start'][row], pieces['end'][row]])
        if (np.linalg.norm(points[0]-points[1]) <= 1e-10
                or np.min(np.linalg.norm(points-crossing, axis=1)) > 1e-10):
            return None
        triangle = vertices[np.asarray(mesh['faces'])[face]]
        bary = np.linalg.solve(triangle.T, points.T).T
        bary /= bary.sum(axis=1)[:, None]
        values = scores[:, np.asarray(mesh['faces'])[face]]@bary.T
        pair = np.minimum(values[p], values[q])
        if (np.any(pair <= 1e-13) or np.any(np.abs(values[p]-values[q]) > 1e-12)
                or (len(other) and np.any(pair <= values[other].max(axis=0)+1e-12))):
            return None
        matched.append(row)
    return np.asarray(matched, np.int64)


def reconstruct(mesh, plate, support, edge_indices=None, *, with_shares=False):
    """Return geometry aligned to existing contact edges plus display pieces.

    ``edge_indices`` selects native ``edge_faces`` in their existing a-to-b
    orientation; omission selects all different-owner edges. The caller keeps
    those a/b identities and classifies motion using returned normals/midpoints.
    Piece ``contact_index`` addresses this aligned output, not a native edge ID.
    Cost is sparse local contact work plus native score reconstruction; there
    is no geographic raster and no all-pairs boundary-distance matrix.
    """
    plate = np.asarray(plate)
    support = np.asarray(support, float)
    faces = np.asarray(mesh['faces'])
    edges = np.asarray(mesh['edge_faces'])
    if (plate.shape != (len(faces),) or support.ndim != 2 or support.shape[1] != len(faces)
            or not np.isfinite(support).all() or np.any(support < 0)
            or np.any(plate < 0) or np.any(plate >= len(support))):
        raise ValueError('Boundary reconstruction requires finite native owner/support fields.')
    if edge_indices is None:
        edge_indices = np.flatnonzero(plate[edges[:, 0]] != plate[edges[:, 1]])
    edge_indices = np.asarray(edge_indices, np.int64)
    a, b = edges[edge_indices].T
    if np.any(plate[a] == plate[b]) or len(np.unique(edge_indices)) != len(edge_indices):
        raise ValueError('Only unique existing two-owner contacts can be reconstructed.')
    count = len(a)
    raw_mid = np.asarray(mesh['edge_mid'])[edge_indices]
    raw_normal = np.asarray(mesh['edge_normal'])[edge_indices]
    raw_length = np.asarray(mesh['edge_length'])[edge_indices]
    mid, normal, length = raw_mid.copy(), raw_normal.copy(), raw_length.copy()
    refined = np.zeros(count, bool)
    if not count:
        return dict(midpoints=mid, normals=normal, lengths=length, refined=refined,
                    segments_start=np.empty((0, 3)), segments_end=np.empty((0, 3)),
                    segment_normals=np.empty((0, 3)), contact_index=np.empty(0, np.int32),
                    diagnostics=dict(representation='native support contours', edges=0, refined_edges=0,
                                     fallback_edges=0, contour_segments=0, raw_length=0., corrected_length=0.,
                                     zero_geometry_edges=0, suppressed_redundant_contacts=[],
                                     suppressed_raw_length_km=0.,
                                     suppression_reason='strict two-face P1 crossing fully represented on orientation-compatible contacts'))
    scores = _vertex_support(mesh, support)
    p, q = np.minimum(plate[a], plate[b]), np.maximum(plate[a], plate[b])
    sign = np.where(plate[a] == p, 1., -1.)
    contact_at_edge = np.full(len(edges), -1, np.int32)
    contact_at_edge[edge_indices] = np.arange(count)
    sums = np.zeros(count); sum_mid = np.zeros((count, 3)); sum_normal = np.zeros((count, 3))
    segments, unmapped, clipped = [], 0, 0
    divided, listed = [], 0
    redundancy_candidates = []
    neighbours = np.asarray(mesh['face_neighbors'])
    for pp, qq in np.unique(np.column_stack((p, q)), axis=0):
        pieces = _pair_segments(mesh, scores, int(pp), int(qq))
        if pieces is None:
            continue
        clipped += pieces['clipped']
        # The contour traverses native faces. Only nearby actual contacts of
        # the same pair may receive its length; third-owner links are excluded.
        nearby = np.column_stack((pieces['face'], neighbours[pieces['face']]))
        nearby = np.concatenate((nearby, neighbours[nearby].reshape(len(nearby), -1)), axis=1)
        candidate = contact_at_edge[np.asarray(mesh['face_edges'])[nearby].reshape(len(nearby), -1)]
        candidate.sort(axis=1)
        valid = candidate >= 0
        valid[:, 1:] &= candidate[:, 1:] != candidate[:, :-1]
        safe = np.maximum(candidate, 0)
        valid &= (p[safe] == pp) & (q[safe] == qq)
        alignment = np.sum(raw_normal[safe]*pieces['normal'][:, None, :], axis=2)*sign[safe]
        distance2 = np.sum((raw_mid[safe]-pieces['mid'][:, None, :])**2, axis=2)
        scale2 = np.maximum(np.asarray(mesh['area_km2'])[pieces['face']]/float(mesh.get('radius_km', 6371.))**2, 1e-20)
        weight = np.exp(-distance2/(2*scale2[:, None]))*np.maximum(alignment, 0)*valid
        denominator = weight.sum(axis=1)
        usable = denominator > 1e-14
        unmapped += int(np.count_nonzero(~usable))
        weight = np.divide(weight, denominator[:, None], out=np.zeros_like(weight), where=denominator[:, None] > 1e-14)
        assigned = weight*pieces['length'][:, None]
        selected = assigned > 0
        target = candidate[selected]
        np.add.at(sums, target, assigned[selected])
        for axis in range(3):
            np.add.at(sum_mid[:, axis], target, (assigned*pieces['mid'][:, None, axis])[selected])
            np.add.at(sum_normal[:, axis], target, (assigned*pieces['normal'][:, None, axis]*sign[safe])[selected])
        best = candidate[np.arange(len(candidate)), weight.argmax(axis=1)]
        segments.append((pieces['start'][usable], pieces['end'][usable],
                         pieces['normal'][usable]*sign[best[usable], None], best[usable]))
        if with_shares:
            # Position of each usable piece in the returned list, and where its length went.
            position = listed+np.cumsum(usable)-1
            rows, columns = np.nonzero(selected)
            divided.append((position[rows], candidate[rows, columns], weight[rows, columns]))
        listed += int(np.count_nonzero(usable))
        # A categorical contact can point across the already reconstructed
        # interface in the opposite orientation. Do not append its unrelated
        # raw mesh edge when its actual two-face P1 crossing is wholly present
        # on compatible same-pair contacts. Keep genuine unresolved contacts.
        pair_contacts = np.flatnonzero((p == pp) & (q == qq) & (sums <= 1e-12))
        for contact in pair_contacts:
            associated = valid & (candidate == contact)
            affected = np.any(associated, axis=1)
            if (not np.any(affected) or not np.all(usable[affected])
                    or np.any(alignment[associated] > 0.)):
                continue
            crossing_rows = _represented_crossing(mesh, scores, int(pp), int(qq),
                edge_indices[contact], (a[contact], b[contact]), pieces)
            if crossing_rows is None:
                continue
            if any(not np.any(associated[row]) or np.any(alignment[row, associated[row]] >= -1e-13)
                   for row in crossing_rows):
                continue
            # Every nearby piece that could have mapped to this raw contact
            # must already be fully allocated. The recipients' normal sums
            # are checked after all pair reconstruction has completed.
            if not np.all(np.abs(weight[affected].sum(axis=1)-1.) <= 1e-12):
                continue
            recipients = np.unique(candidate[affected][assigned[affected] > 0.])
            if len(recipients):
                redundancy_candidates.append((int(contact), recipients))
    refined = (sums > 1e-12) & (np.linalg.norm(sum_normal, axis=1) > .25*sums)
    mid[refined] = _unit(sum_mid[refined])
    fitted = sum_normal-mid*np.sum(sum_normal*mid, axis=1)[:, None]
    normal[refined] = _unit(fitted[refined])
    length[refined] = sums[refined]
    redundant = np.zeros(count, bool)
    for contact, recipients in redundancy_candidates:
        if sums[contact] <= 1e-12 and np.all(refined[recipients]):
            redundant[contact] = True
    length[redundant] = 0.
    # Fallback is reviewable and keeps unresolved microplate contacts present.
    fallback = np.flatnonzero(~refined & ~redundant)
    endpoints = np.asarray(mesh['vertices'])[np.asarray(mesh['edge_vertices'])[edge_indices[fallback]]]
    segments.append((endpoints[:, 0], endpoints[:, 1], raw_normal[fallback], fallback))
    start, end, segment_normal, contacts = (np.concatenate([entry[k] for entry in segments]) for k in range(4))
    extra = {}
    if with_shares:
        # A contact that keeps its raw edge takes that edge whole and nothing of the contour:
        # its length is the edge's, whatever the contour offered it.
        piece, contact, weight = (np.concatenate([entry[k] for entry in divided]) if divided else np.empty(0)
                                  for k in range(3))
        keep = refined[contact.astype(np.int64)] if len(contact) else np.zeros(0, bool)
        extra = dict(share_segment=np.concatenate((piece[keep], listed+np.arange(len(fallback)))).astype(np.int32),
                     share_contact=np.concatenate((contact[keep], fallback)).astype(np.int32),
                     share_weight=np.concatenate((weight[keep], np.ones(len(fallback)))).astype(float))
    return dict(midpoints=mid, normals=normal, lengths=length, refined=refined,
                segments_start=start, segments_end=end, segment_normals=segment_normal,
                contact_index=contacts.astype(np.int32), **extra,
                diagnostics=dict(representation='dominant-pair native P1 support contours', edges=count,
                                 refined_edges=int(refined.sum()), fallback_edges=len(fallback),
                                 contour_segments=len(start)-len(fallback), third_owner_clipped_segments=clipped,
                                 unmapped_segments=unmapped, raw_length=float(raw_length.sum()),
                                 zero_geometry_edges=int(redundant.sum()),
                                 suppressed_redundant_contacts=np.flatnonzero(redundant).tolist(),
                                 suppressed_raw_length_km=float(raw_length[redundant].sum()),
                                 suppression_reason='strict two-face P1 crossing fully represented on orientation-compatible contacts',
                                 corrected_length=float(length.sum())))
