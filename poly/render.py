"""Equirectangular surface maps: land and water from height, boundaries by kind."""
from __future__ import annotations

import numpy as np
import spherely
from PIL import Image, ImageDraw

from poly import sphere, surface

COLOURS = dict(ridge=(220, 40, 40), trench=(120, 40, 160), transform=(30, 150, 60),
               converging=(240, 140, 20), collision=(120, 70, 30), suture=(60, 30, 10))
CONTINENT = (196, 170, 120)
# Hypsometric tints for continent by isostatic elevation (km): lowland, upland, mountain, high peaks.
RELIEF = [(0., (168, 190, 120)), (.5, (196, 170, 120)), (1.5, (140, 100, 60)), (3., (245, 245, 245))]
# Water over either continental or oceanic crust, tinted by depth below sea level.
BATHYMETRY = [(0., (190, 230, 235)), (.2, (150, 215, 235)),
              (2.5, (85, 155, 200)), (5.7, (20, 50, 110))]


def _tint(elevation_km, palette=RELIEF):
    """Colours for heights or depths, interpolated along the selected palette."""
    levels = np.array([e for e, _ in palette])
    colours = np.array([c for _, c in palette], float)
    out = np.zeros((len(elevation_km), 3))
    for k in range(3):
        out[:, k] = np.interp(elevation_km, levels, colours[:, k])
    return np.round(out).astype(np.uint8)


MESH = (90, 70, 40)  # the continental mesh, when asked for


def draw(world, path, width=720, boundaries=True, mesh=False, sea_level_km=0.):
    """Write a PNG map of `world` to `path` and return the per-pixel plate array.

    With `mesh`, the triangles of every continental mesh are outlined, to show the level of detail.
    Sea level is an output scenario relative to the existing height datum. Tiny
    emerged islands receive visibility markers; use surface.sample for exact masks.
    """
    height = width // 2
    lon = np.linspace(-180., 180., width, endpoint=False) + 180. / width
    lat = np.linspace(90., -90., height, endpoint=False) - 90. / height
    lon, lat = np.meshgrid(lon, lat)
    xyz = sphere.from_lonlat(lon.ravel(), lat.ravel())
    image = np.zeros((height * width, 3), np.uint8)
    sampled = surface.sample(world, xyz, sea_level_km)
    plate_of = sampled.plate
    relative_height = sampled.height_above_water_km
    image[sampled.land] = _tint(relative_height[sampled.land])
    image[sampled.water] = _tint(-relative_height[sampled.water], BATHYMETRY)
    picture = Image.fromarray(image.reshape(height, width, 3))
    pen = ImageDraw.Draw(picture)
    # Islands smaller than a few pixels would fall between pixel centres: a dot each, at the
    # island's own size with a floor of a pixel and a half.
    pixel_km2 = sphere.SPHERE_KM2 / (width * height)
    pixel_km = 2. * np.pi * sphere.RADIUS_KM / width
    for piece in world.crust:
        area = piece.area_km2()
        if piece.kind == 'continent' and getattr(piece, 'mesh', None) is None and area < 4. * pixel_km2:
            centre = spherely.centroid(piece.shape)
            local = sphere.from_lonlat(float(spherely.get_x(centre)), float(spherely.get_y(centre)))
            elevation = float(surface.elevation_km(piece, local[None], world.time_myr)[0]) - sea_level_km
            if elevation < 0.:
                continue
            plon, plat = sphere.to_lonlat(sphere.rotate(local, world.frames.get(piece.plate, np.eye(3))))
            x, y = (float(plon) + 180.) / 360. * width, (90. - float(plat)) / 180. * height
            r = max(.75, np.sqrt(area / np.pi) / pixel_km)
            pen.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(v) for v in _tint(np.array([elevation]))[0]))
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
    if mesh:
        for piece in world.crust:
            field = getattr(piece, 'mesh', None)
            if field is None or len(field.triangles) == 0:
                continue
            tri = np.asarray(field.triangles)
            edges = np.unique(np.sort(np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]])), axis=1), axis=0)
            plon, plat = sphere.to_lonlat(sphere.rotate(field.nodes, world.frames.get(piece.plate, np.eye(3))))
            x = (plon + 180.) / 360. * width
            y = (90. - plat) / 180. * height
            for i, j in edges:
                if abs(x[i] - x[j]) > width / 2:
                    continue  # crosses the map edge
                pen.line([(x[i], y[i]), (x[j], y[j])], fill=MESH, width=1)
    picture.save(path)
    return plate_of.reshape(height, width)
