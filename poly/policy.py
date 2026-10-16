"""Explicit plate-retention policy and geometry-based absorption recipients."""
from dataclasses import dataclass
import math

import numpy as np
import spherely

from poly import sphere


@dataclass(frozen=True)
class PlatePolicy:
    """Area cutoffs in km²; numerical cleanup is distinct from plate retention.

    The default retains plates down to 250,000 km². Smaller
    resolved plates can be retained by lowering ``minimum_plate_km2``; this
    does not lower the geometry engine's independent level of detail.
    ``legacy()`` restores the previous one-million-km² absorption threshold.
    """
    minimum_plate_km2: float = 250_000.
    numerical_crumb_km2: float = 100.

    def __post_init__(self):
        if (not math.isfinite(self.minimum_plate_km2)
                or not math.isfinite(self.numerical_crumb_km2)
                or self.numerical_crumb_km2 < sphere.DETAIL_KM2
                or self.minimum_plate_km2 <= self.numerical_crumb_km2):
            raise ValueError('plate policy requires minimum_plate_km2 > numerical_crumb_km2 '
                             '>= geometry DETAIL_KM2, all finite')

    @classmethod
    def legacy(cls):
        return cls(minimum_plate_km2=1_000_000., numerical_crumb_km2=100.)


# One centimetre: enough for frame roundoff, far below outline/detail scales.
CONTACT_TOLERANCE_KM = 1e-5


def _boundary_edges(shape):
    """Only area-bearing polygon rings; ignore incidental S2 line/point debris."""
    rings = [ring for polygon in sphere.polygons(shape) for ring in polygon]
    if not rings:
        return np.empty((0, 3)), np.empty((0, 3))
    a = np.vstack(rings)
    b = np.vstack([np.roll(ring, -1, axis=0) for ring in rings])
    keep = sphere.arc_km(a, b) > 1e-9
    return a[keep], b[keep]


def _shared_boundary_km(edges, other_edges):
    """Length of great-circle arc overlaps, allowing centimetre frame roundoff.

    S2's exact line intersection can lose an entire shared edge after a harmless
    rotation and inverse rotation. Match endpoints to an edge's great-circle
    plane, project their short arc onto that edge, and union the overlap
    intervals. A crossing point contributes no length. Unioning intervals also
    prevents double counting where an adjacent outline subdivides an edge.
    """
    c, d = other_edges
    if not len(c):
        return 0.
    total = 0.
    tolerance = CONTACT_TOLERANCE_KM / sphere.RADIUS_KM
    for a, b in zip(*edges):
        normal = np.cross(a, b)
        sine = np.linalg.norm(normal)
        theta = np.arctan2(sine, a @ b)
        normal /= sine
        on = (np.abs(c @ normal) <= tolerance) & (np.abs(d @ normal) <= tolerance)
        if not on.any():
            continue
        tangent = np.cross(normal, a)
        start = np.arctan2(c[on] @ tangent, c[on] @ a)
        end = np.arctan2(d[on] @ tangent, d[on] @ a)
        delta = (end - start + np.pi) % (2. * np.pi) - np.pi
        lower, upper = np.minimum(start, start + delta), np.maximum(start, start + delta)
        intervals = []
        for shift in (-2. * np.pi, 0., 2. * np.pi):
            lo, hi = np.maximum(lower + shift, 0.), np.minimum(upper + shift, theta)
            intervals.extend(zip(lo[hi > lo], hi[hi > lo]))
        reach = 0.
        for lo, hi in sorted(intervals):
            total += max(hi - max(lo, reach), 0.)
            reach = max(reach, hi)
    return total * sphere.RADIUS_KM


def main_neighbours(pieces, plate, regions):
    """Recipients with the longest shared boundary; ties use plate id.

    Polygon distance cannot distinguish neighbours that all touch. Measure
    shared great-circle arc coverage within CONTACT_TOLERANCE_KM instead. A
    detached numerical crumb may have a small gap to every region after
    cleanup; only if there is no shared arc do we use distance, with the same
    stable id tie-break. No plate is returned when the donor is the only plate.
    """
    others = sorted(p for p in regions if p != plate)
    if not others:
        return [None] * len(pieces)
    shapes = np.array([regions[p] for p in others], dtype=object)
    outlines = [_boundary_edges(g) for g in shapes]
    result = []
    for piece in pieces:
        edges = _boundary_edges(piece)
        lengths = np.array([_shared_boundary_km(edges, outline) for outline in outlines])
        largest = float(lengths.max())
        if largest > CONTACT_TOLERANCE_KM:
            candidates = np.flatnonzero(np.isclose(lengths, largest, rtol=1e-10,
                                                    atol=CONTACT_TOLERANCE_KM))
        else:
            distances = np.asarray(spherely.distance(piece, shapes), float)
            candidates = np.flatnonzero(np.isclose(distances, distances.min(), rtol=1e-10, atol=1e-5))
        result.append(others[int(candidates[0])])
    return result
