"""Equirectangular maps of a world: crust by kind and ocean age, boundaries by kind."""
from __future__ import annotations

import numpy as np
import spherely
from PIL import Image, ImageDraw

from poly import sphere

COLOURS = dict(ridge=(220, 40, 40), trench=(120, 40, 160), transform=(30, 150, 60),
               converging=(240, 140, 20), collision=(120, 70, 30), suture=(60, 30, 10))
CONTINENT = (196, 170, 120)
THICKENED = np.array([120, 85, 55])  # continent at twice its usual thickness or more
CONTINENT_KM = 35.
YOUNG, OLD = np.array([150, 215, 235]), np.array([20, 50, 110])
OCEAN_SPAN_MYR = 150.


def draw(world, path, width=720, boundaries=True):
    """Write a PNG map of `world` to `path` and return the per-pixel plate array."""
    height = width // 2
    lon = np.linspace(-180., 180., width, endpoint=False) + 180. / width
    lat = np.linspace(90., -90., height, endpoint=False) - 90. / height
    lon, lat = np.meshgrid(lon, lat)
    xyz = sphere.from_lonlat(lon.ravel(), lat.ravel())
    image = np.zeros((height * width, 3), np.uint8)
    plate_of = np.zeros(height * width, int)
    world_points = spherely.points(lon.ravel(), lat.ravel())
    for plate, region in world.plate_polygons().items():
        spherely.prepare(region)
        inside = np.asarray(spherely.contains(region, world_points))
        plate_of[inside] = plate
        if not inside.any():
            continue
        local = sphere.rotate(xyz[inside], world.frame(plate).T)
        llon, llat = sphere.to_lonlat(local)
        points = spherely.points(llon, llat)
        colour = np.tile(np.array(OLD, np.uint8), (inside.sum(), 1))
        for piece in (p for p in world.crust if p.plate == plate):
            spherely.prepare(piece.shape)
            hit = np.asarray(spherely.contains(piece.shape, points))
            if not hit.any():
                continue
            if piece.kind == 'continent':
                f = min(max(getattr(piece, 'thickness_km', CONTINENT_KM) / CONTINENT_KM - 1., 0.), 1.)
                colour[hit] = np.round(np.array(CONTINENT) * (1 - f) + THICKENED * f).astype(np.uint8)
            else:
                age = 0. if piece.born_myr is None else max(world.time_myr - piece.born_myr, 0.)
                f = min(age / OCEAN_SPAN_MYR, 1.)
                colour[hit] = np.round(YOUNG * (1 - f) + OLD * f).astype(np.uint8)
        image[inside] = colour
    picture = Image.fromarray(image.reshape(height, width, 3))
    if boundaries:
        pen = ImageDraw.Draw(picture)
        if hasattr(world, 'boundary_lines'):
            lines = list(world.boundary_lines())
        else:
            lines = [(world.network.points(i), s.kind) for i, s in world.network.sections.items()]
        if hasattr(world, 'suture_lines'):
            lines += [(points, 'suture') for points, _ in world.suture_lines()]
        for points, kind in lines:
            plon, plat = sphere.to_lonlat(points)
            x = (plon + 180.) / 360. * width
            y = (90. - plat) / 180. * height
            for k in range(len(x) - 1):
                if abs(x[k + 1] - x[k]) > width / 2:
                    continue  # crosses the map edge
                pen.line([(x[k], y[k]), (x[k + 1], y[k + 1])], fill=COLOURS[kind], width=2 if kind != 'suture' else 1)
    picture.save(path)
    return plate_of.reshape(height, width)
