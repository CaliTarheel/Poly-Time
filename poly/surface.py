"""Sample crust and surface height independently of the map's colour scheme.

Heights use the existing model's zero datum: 35 km continental crust has height
zero, and ocean height is minus GDH1 depth. A specified sea level is an output
scenario relative to that datum; it does not change crust or conserve water.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spherely

from poly import sphere
from poly.mesh import ISOSTASY, REFERENCE_KM
from poly.ocean import depth_km


def elevation_km(piece, local_points, time_myr):
    """Height above the reference datum at points in the piece's plate frame.

The caller supplies points belonging to this piece. Crust classification does
not determine emergence: thin continental crust can be below sea level.
"""
    points = np.asarray(local_points, float).reshape(-1, 3)
    if piece.kind == 'continent':
        mesh = getattr(piece, 'mesh', None)
        thickness = mesh.sample(points) if mesh is not None else piece.thickness_km
        return np.broadcast_to(ISOSTASY * (np.asarray(thickness) - REFERENCE_KM),
                               (len(points),)).copy()
    if piece.kind == 'ocean':
        age = 0. if piece.born_myr is None else max(time_myr - piece.born_myr, 0.)
        return np.full(len(points), -float(depth_km(age)))
    raise ValueError('Unknown crust kind: %r' % piece.kind)


@dataclass
class SurfaceSample:
    plate: np.ndarray
    crust_kind: np.ndarray
    elevation_km: np.ndarray
    sea_level_km: float

    @property
    def height_above_water_km(self):
        return self.elevation_km - self.sea_level_km

    @property
    def land(self):
        """Emergent points, including height exactly equal to sea level."""
        return self.height_above_water_km >= 0.

    @property
    def water(self):
        return self.height_above_water_km < 0.

    @property
    def continental_crust(self):
        return self.crust_kind == 'continent'


def sample(world, world_points, sea_level_km=0.):
    """Sample world-frame unit vectors without changing the world.

Points without crust have NaN height and empty crust kind. They are neither
land nor water, allowing callers to diagnose geometry gaps. Plate zero means
no containing plate; a crust gap within a plate retains that plate's ID. The
returned arrays are flat in the input point order.
"""
    if not np.isfinite(sea_level_km):
        raise ValueError('Sea level must be finite.')
    xyz = np.asarray(world_points, float).reshape(-1, 3)
    lon, lat = sphere.to_lonlat(xyz)
    geographies = spherely.points(lon, lat)
    plates = np.zeros(len(xyz), int)
    kinds = np.full(len(xyz), '', dtype='<U9')
    height = np.full(len(xyz), np.nan)
    for plate, region in world.plate_polygons().items():
        spherely.prepare(region)
        inside = np.asarray(spherely.contains(region, geographies))
        indices = np.flatnonzero(inside)
        if not len(indices):
            continue
        plates[indices] = plate
        local = sphere.rotate(xyz[indices], world.frames.get(plate, np.eye(3)).T)
        llon, llat = sphere.to_lonlat(local)
        points = spherely.points(llon, llat)
        for piece in (p for p in world.crust if p.plate == plate):
            spherely.prepare(piece.shape)
            hit = np.asarray(spherely.contains(piece.shape, points))
            if not hit.any():
                continue
            kinds[indices[hit]] = piece.kind
            height[indices[hit]] = elevation_km(piece, local[hit], world.time_myr)
    return SurfaceSample(plates, kinds, height, float(sea_level_km))
