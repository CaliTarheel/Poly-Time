"""Resolved juvenile volcanic profiles and their submerged open boundaries.

Birth columns are a schematic initial volcanic structure, not a magma chamber
or an isostatic emplacement solve. Their heights and thicknesses are stored in
the material state and subsequently evolve normally. Reconstruction only ties
true open arc-boundary vertices to the local ocean, retaining the interior
columns and actual mesh connectivity. It is not a mass or erosion ledger.
"""
from __future__ import annotations

import numpy as np
import material_reconstruction


def birth_profile(radius, basement_m, *, peak_m=970.):
    """Return neutral summit, sloping flank and submerged-apron birth columns.

    Radius is normalized to the outer volcanic footprint. The central third
    remains a summit; a smooth taper reaches the basal seafloor at radius one.
    The caller initializes both reference/current thickness and reference
    elevation from these values, and accounts for the actual inserted volume.
    """
    radius, basement = np.broadcast_arrays(np.asarray(radius, float), np.asarray(basement_m, float))
    if (not np.isfinite(radius).all() or not np.isfinite(basement).all()
            or np.any(radius < 0.) or not np.isfinite(peak_m)
            or np.any(basement >= peak_m)):
        raise ValueError('Arc profiles require finite nonnegative radius and a basal surface below the summit.')
    t = np.clip((radius-1/3)/(2/3), 0., 1.)
    fraction = 1-t*t*(3-2*t)
    return dict(height_m=basement+(float(peak_m)-basement)*fraction,
                thickness_km=8.+17.*fraction, construction_fraction=fraction)


def prepare(stencil, arc_ids):
    """Find actual open arc edges, without treating adjacent sheets as joined.

    A shared edge beside continental crust or another connected arc is not an
    open coastline. Persistent arc identity selects new profiles; legacy arcs
    and other material use zero. Same-owner reconstruction nodes preserve
    continuity at any genuinely shared junctions.
    """
    faces = np.asarray(stencil['face_vertices'])
    ids = np.asarray(arc_ids)
    if ids.shape != (len(faces),) or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
        raise ValueError('Arc identities must be nonnegative integers aligned with material faces.')
    original = np.asarray(stencil['vertex_index'])[faces]
    pairs = np.array([[0, 1], [1, 2], [2, 0]])
    # Ordering each edge's two ends is a minimum and a maximum; sorting a length-two
    # axis row by row is the same result at many times the cost.
    first, second = original, original[:, [1, 2, 0]]
    edges = np.stack((np.minimum(first, second), np.maximum(first, second)), axis=2).reshape(-1, 2)
    # Only equality/counts matter here, not the order of unique edges.
    if edges.dtype.kind in 'iu' and edges.size and 0 <= edges.min() and edges.max() < 1 << 31:
        # Both ends fit in 31 bits, so one int64 per edge is an exact, collision-free key.
        records = (edges[:, 0].astype(np.int64) << 32) | edges[:, 1].astype(np.int64)
    else:
        # Fixed width integer byte records avoid NumPy's structured-field row sorting;
        # unlike arithmetic packing, they cannot overflow for large vertex IDs.
        records = np.ascontiguousarray(edges).view(np.dtype((np.void, 2*edges.dtype.itemsize))).ravel()
    _, inverse, counts = np.unique(records, return_inverse=True, return_counts=True)
    open_arc = (counts[inverse] == 1) & np.repeat(ids > 0, 3)
    nodes = faces[:, pairs].reshape(-1, 2)[open_arc]
    boundary = np.zeros(stencil['vertex_count'], bool)
    boundary[nodes.ravel()] = True
    return dict(stencil=stencil, boundary_nodes=boundary, face_boundary_mask=boundary[faces],
                open_edge_count=int(open_arc.sum()), affected_faces=np.any(boundary[faces], axis=1))


def ocean_weight(prepared, face_indices, barycentric_weights):
    faces = np.asarray(face_indices)
    weights = np.asarray(barycentric_weights, float)
    if (faces.ndim != 1 or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0) or np.any(faces >= len(prepared['face_boundary_mask']))
            or weights.shape != (len(faces), 3) or not np.isfinite(weights).all()
            or np.any(weights < -1e-10)
            or not np.allclose(weights.sum(axis=1), 1., atol=1e-9, rtol=0.)):
        raise ValueError('Arc surface reconstruction needs actual barycentric triangle hits.')
    weights = np.maximum(weights, 0.)
    weights /= weights.sum(axis=1)[:, None]
    return np.sum(weights*prepared['face_boundary_mask'][faces], axis=1)


def apply(prepared, nodal_heights, face_indices, barycentric_weights, ocean_height_at_queries):
    """Match the entire open boundary to ocean while retaining interior nodes.

    Boundary-node barycentric weight takes the ocean function at the query,
    rather than a linear approximation between two sampled ocean endpoints.
    Consequently both sides converge to the same depth everywhere along an
    edge, including curved bathymetry. Only boundary-incident faces change.
    """
    faces, weights = np.asarray(face_indices), np.asarray(barycentric_weights, float)
    base = material_reconstruction.sample_hits(prepared['stencil'], nodal_heights, faces, weights)
    ocean = np.asarray(ocean_height_at_queries, float)
    if ocean.shape != base.shape or not np.isfinite(ocean).all():
        raise ValueError('Arc boundary reconstruction requires finite ocean heights at its queries.')
    fraction = ocean_weight(prepared, faces, weights)
    weights = np.maximum(weights, 0.)
    weights /= weights.sum(axis=1)[:, None]
    nodes = prepared['stencil']['face_vertices'][faces]
    boundary_part = np.sum(weights*prepared['face_boundary_mask'][faces]*np.asarray(nodal_heights)[nodes], axis=1)
    return base-boundary_part+fraction*ocean
