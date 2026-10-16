"""Equirectangular maps of a world: crust by kind and ocean age, boundaries by kind."""
from __future__ import annotations

import numpy as np
import spherely
from PIL import Image, ImageDraw

from poly import sphere
from poly.mesh import ISOSTASY
from poly.ocean import depth_km

COLOURS = dict(ridge=(220, 40, 40), trench=(120, 40, 160), transform=(30, 150, 60),
               converging=(240, 140, 20), collision=(120, 70, 30), suture=(60, 30, 10))
CONTINENT = (196, 170, 120)
# Hypsometric tints for continent by isostatic elevation (km): lowland, upland, mountain, high peaks.
RELIEF = [(0., (168, 190, 120)), (.5, (196, 170, 120)), (1.5, (140, 100, 60)), (3., (245, 245, 245))]
CONTINENT_KM = 35.
# Ocean tinted by depth: shallow (ridge crests, 2.6 km) to deep (old floor, 5.6 km).
SHALLOW, DEEP = np.array([150, 215, 235]), np.array([20, 50, 110])
SHALLOW_KM, DEEP_KM = 2.5, 5.7


def _tint(elevation_km):
    """Colours for elevations, interpolated along RELIEF."""
    levels = np.array([e for e, _ in RELIEF])
    colours = np.array([c for _, c in RELIEF], float)
    out = np.zeros((len(elevation_km), 3))
    for k in range(3):
        out[:, k] = np.interp(elevation_km, levels, colours[:, k])
    return np.round(out).astype(np.uint8)


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
        colour = np.tile(np.array(DEEP, np.uint8), (inside.sum(), 1))
        for piece in (p for p in world.crust if p.plate == plate):
            spherely.prepare(piece.shape)
            hit = np.asarray(spherely.contains(piece.shape, points))
            if not hit.any():
                continue
            if piece.kind == 'continent':
                mesh = getattr(piece, 'mesh', None)
                thick = mesh.sample(local[hit]) if mesh is not None else getattr(piece, 'thickness_km', CONTINENT_KM)
                colour[hit] = _tint(ISOSTASY * (np.asarray(thick, float) - CONTINENT_KM) * np.ones(int(hit.sum())))
            else:
                age = 0. if piece.born_myr is None else max(world.time_myr - piece.born_myr, 0.)
                f = float(np.clip((depth_km(age) - SHALLOW_KM) / (DEEP_KM - SHALLOW_KM), 0., 1.))
                colour[hit] = np.round(SHALLOW * (1 - f) + DEEP * f).astype(np.uint8)
        image[inside] = colour
    picture = Image.fromarray(image.reshape(height, width, 3))
    pen = ImageDraw.Draw(picture)
    # Islands smaller than a few pixels would fall between pixel centres: a dot each.
    pixel_km2 = sphere.SPHERE_KM2 / (width * height)
    for piece in world.crust:
        if piece.kind == 'continent' and getattr(piece, 'mesh', None) is None and piece.area_km2() < 4. * pixel_km2:
            centre = spherely.centroid(piece.shape)
            local = sphere.from_lonlat(float(spherely.get_x(centre)), float(spherely.get_y(centre)))
            plon, plat = sphere.to_lonlat(sphere.rotate(local, world.frame(piece.plate)))
            x, y = (float(plon) + 180.) / 360. * width, (90. - float(plat)) / 180. * height
            pen.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=tuple(int(v) for v in _tint(np.array([0.]))[0]))
    if boundaries:
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
