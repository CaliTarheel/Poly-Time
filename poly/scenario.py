"""Geometry and contacts for chosen starting states, never for an evolved world.

The spherical twists vary an initial layout while retaining a shared boundary
network. They are not a substitute for plate motion or fabricated prehistory.
"""
from __future__ import annotations

import numpy as np

from poly import sphere
from poly.build import kinds_along
from poly.world import Crust


class SphericalWarp:
    """Smooth invertible twists about three axes on the unit sphere.

    Each twist changes longitude by a function of latitude about its own axis.
    It preserves spherical area in the continuous limit. Native polygons still
    use finite great-circle edges; their actual areas are measured separately.
    """

    def __init__(self, seed, strength=1.):
        rng = np.random.default_rng(seed)
        self.axes = np.linalg.qr(rng.normal(size=(3, 3)))[0].T
        self.phases = rng.uniform(0., 2. * np.pi, 3)
        self.amplitudes = strength * np.array([.55, .45, .35])
        self.frequencies = np.array([2.1, 3.7, 5.3])

    def __call__(self, points, inverse=False):
        points = np.asarray(points, float)
        shape = points.shape
        out = points.reshape(-1, 3).copy()
        order = range(2, -1, -1) if inverse else range(3)
        for k in order:
            axis = self.axes[k]
            latitude = out @ axis
            angle = self.amplitudes[k] * np.sin(self.frequencies[k] * latitude + self.phases[k])
            if inverse:
                angle = -angle
            cosine, sine = np.cos(angle)[:, None], np.sin(angle)[:, None]
            out = (out * cosine + np.cross(axis, out) * sine
                   + latitude[:, None] * axis * (1. - cosine))
            out = sphere.unit(out)
        return out.reshape(shape)

    def line(self, points, spacing_km, tolerance_km=2.):
        """Sample a warped geodesic line to its length and curvature limits."""
        source = np.asarray(points, float).reshape(-1, 3)
        if spacing_km <= 0. or tolerance_km <= 0.:
            raise ValueError('Positive starting-boundary spacing and tolerance are required.')
        for _ in range(12):
            mapped = self(source)
            if len(source) < 2:
                return mapped
            middle = sphere.unit(source[:-1] + source[1:])
            actual = self(middle)
            chord = sphere.unit(mapped[:-1] + mapped[1:])
            split = ((sphere.arc_km(mapped[:-1], mapped[1:]) > spacing_km)
                     | (sphere.arc_km(actual, chord) > tolerance_km))
            if not split.any():
                return mapped
            rows = []
            for k, point in enumerate(source[:-1]):
                rows.append(point)
                if split[k]:
                    rows.append(middle[k])
            source = np.vstack((*rows, source[-1]))
        raise ValueError('A curved starting boundary did not reach its sampling tolerance.')


def curve_network(template, seed, spacing_km):
    """Curve a fresh network once, sharing the same vertices on both sides.

    Motions are then classified on the new geometry. Initial trench polarities
    are provisional until the actual continents have been installed.
    """
    if template.time_myr != 0. or getattr(template, 'ledger', []):
        raise ValueError('Only a fresh starting network can be reshaped.')
    if any(piece.kind != 'ocean' for piece in template.crust):
        raise ValueError('Starting geometry requires an all-ocean template, before continents are placed.')
    ages = {}
    for piece in template.crust:
        if piece.plate in ages and ages[piece.plate] != piece.born_myr:
            raise ValueError('Starting geometry requires one ocean age per plate.')
        ages[piece.plate] = piece.born_myr
    if any(not np.allclose(frame, np.eye(3), atol=1e-14, rtol=0.)
           for frame in template.frames.values()):
        raise ValueError('Starting geometry requires unrotated initial frames.')
    network = template.network
    warp = SphericalWarp(seed + 3911, strength=.35)
    lines = {ident: warp.line(network.points(ident), spacing_km)
             for ident in network.sections}
    junctions = {ident: warp(point) for ident, point in network.junctions.items()}
    network.junctions = junctions
    for ident, section in network.sections.items():
        section.interior = lines[ident][1:-1]
    # from_network copies the template's material, so keep even this temporary
    # all-ocean state consistent with its changed native boundary network.
    template.crust = [Crust(plate, 'ocean', region, born_myr=ages[plate])
                      for plate, region in network.polygons().items()]
    return template


def initialize_contacts(world, rotations):
    """Declare contacts from fresh geometry and the actual continental crust.

    Continental collision has no downgoing ocean slab. At a continent/ocean
    trench the continental side initially overrides. Existing ocean/ocean
    polarity is retained; equal-age new trenches choose the higher plate ID.
    This chooses an initial situation, not polarity reversal during evolution.
    """
    if world.time_myr != 0. or world.ledger:
        raise ValueError('Initial contacts cannot replace evolved boundary history.')
    for boundary in world.boundaries:
        kinds = kinds_along(boundary.points, rotations[boundary.left], rotations[boundary.right])
        for k, kind in enumerate(kinds):
            if kind != 'trench':
                boundary.kinds[k], boundary.overriding[k] = kind, None
                continue
            left, right = world._sides(boundary, k)
            if left == right == 'continent':
                boundary.kinds[k], boundary.overriding[k] = 'collision', None
            else:
                boundary.kinds[k] = 'trench'
                if left == 'continent':
                    boundary.overriding[k] = boundary.left
                elif right == 'continent':
                    boundary.overriding[k] = boundary.right
                elif boundary.overriding[k] not in (boundary.left, boundary.right):
                    boundary.overriding[k] = max(boundary.left, boundary.right)
        boundary.closed_km[:] = 0.

