"""Strength-guided spherical fracture paths for the optional artist profile.

Great-circle cuts screen the far-field loading. A supercritical candidate is
routed within a finite corridor, then scored again using its actual partition,
length and local normals. The routing grid is temporary; the resulting shared
spherical polygon boundary is the mechanical geometry.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import spherely
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from poly import sphere

REACH_KM = 350.
ALONG_KM = 100.
ACROSS_KM = 50.
SMOOTH_KM = 200.
ANCHOR_WEIGHT = .08
BEND_WEIGHT = .3


def _points(xyz):
    lon, lat = sphere.to_lonlat(xyz)
    return spherely.points(lon, lat)


def _material(world, plate, points, rifting):
    """Effective resistance: resolved thickness and persistent weak sutures.

    Thickness is an artist-model strength proxy, not a lithospheric yield law.
    One kilometre is a resistance floor, not a modification of the rock field.
    """
    local = sphere.rotate(points, world.frame(plate).T)
    probes = _points(local)
    strength = np.full(len(points), rifting.ocean_strength_n_per_m)
    continental = np.zeros(len(points), bool)
    for body in world.crust:
        if body.plate != plate or body.kind != 'continent':
            continue
        spherely.prepare(body.shape)
        hit = np.asarray(spherely.contains(body.shape, probes))
        if not hit.any():
            continue
        thickness = body.thickness_km if body.mesh is None else body.mesh.sample(local[hit])
        strength[hit] = rifting.continent_strength_n_per_m * np.maximum(thickness, 1.) / 35.
        continental[hit] = True
    lines = [sphere.rotate(f['points'], world.frame(plate))
             for f in world.geology['features'] if f['plate'] == plate and f['kind'] == 'collision']
    lines += [sphere.rotate(f['points'], world.frame(plate))
              for f in world.sutures if f['plate'] == plate]
    if lines:
        from poly.build import densify
        nodes = np.vstack([densify(line, 50.) for line in lines])
        distance = cKDTree(nodes).query(points)[0] * sphere.RADIUS_KM
        near = distance <= rifting.suture_reach_km
        strength[near] = np.minimum(strength[near], rifting.suture_strength_n_per_m)
    return strength, continental


def _route(cost, offsets, spacing_km):
    """Minimum-action path with limited slope, followed by regional smoothing."""
    rows, columns = cost.shape
    middle = columns // 2
    prior = np.full(columns, np.inf)
    prior[middle] = 0.
    parent = np.zeros((rows, columns), int)
    delta = offsets[:, None] - offsets[None, :]
    slope = delta / spacing_km
    length = np.sqrt(1. + slope ** 2)
    allowed = np.abs(delta) <= ACROSS_KM + 1e-9
    anchor = ANCHOR_WEIGHT * (offsets / REACH_KM) ** 2
    for k in range(1, rows):
        edge = .5 * (cost[k - 1, :, None] + cost[k, None, :]) * length
        action = prior[:, None] + edge + BEND_WEIGHT * slope ** 2 + anchor[None]
        action[~allowed] = np.inf
        parent[k] = np.argmin(action, axis=0)
        prior = action[parent[k], np.arange(columns)]
    chosen = np.zeros(rows, int)
    chosen[-1] = middle
    for k in range(rows - 1, 0, -1):
        chosen[k - 1] = parent[k, chosen[k]]
    return gaussian_filter1d(offsets[chosen], SMOOTH_KM / spacing_km, mode='wrap')


def routed_cut(balance, world, rotations, cut, rifting):
    from poly.dynamics import R_M, SECONDS_PER_MYR, plate_torques
    centre, pole = sphere.unit(cut.centre), sphere.unit(cut.pole)
    tangent = sphere.unit(np.cross(pole, centre))
    count = int(np.ceil(2. * np.pi * sphere.RADIUS_KM / ALONG_KM))
    angles = np.linspace(-np.pi, np.pi, count, endpoint=False)
    along = np.cos(angles)[:, None] * centre + np.sin(angles)[:, None] * tangent
    offsets = np.arange(-REACH_KM, REACH_KM + ACROSS_KM / 2., ACROSS_KM)
    across = offsets / sphere.RADIUS_KM
    grid = along[:, None, :] * np.cos(across)[None, :, None] + pole * np.sin(across)[None, :, None]
    region = world.regions[cut.plate]
    spherely.prepare(region)
    inside = np.asarray(spherely.contains(region, _points(grid.reshape(-1, 3)))).reshape(count, -1)
    strength, _ = _material(world, cut.plate, grid.reshape(-1, 3), rifting)
    cost = strength.reshape(count, -1) / rifting.continent_strength_n_per_m
    cost[~inside] = 0.
    # Close the loop in a row outside the plate when possible, avoiding an
    # arbitrary anchor through the middle of the actual continental fracture.
    outside = np.flatnonzero(~inside.any(axis=1))
    shift = int(outside[0]) if len(outside) else 0
    offset = _route(np.roll(cost, -shift, axis=0), offsets,
                    2. * np.pi * sphere.RADIUS_KM / count)
    offset = np.roll(offset, shift) / sphere.RADIUS_KM
    path = along * np.cos(offset)[:, None] + pole * np.sin(offset)[:, None]
    half = sphere.polygon(path)
    part = sphere.intersection(region, half)
    area = sphere.area_km2(part)
    if min(area, sphere.area_km2(region) - area) < rifting.minimum_part_km2:
        return None
    # Re-evaluate the force balance on the actual curved partition. A curve
    # never inherits the straight candidate's old tension or strength score.
    sampled = np.asarray(spherely.contains(region, balance.geographies))
    points = balance.points[sampled]
    side = np.asarray(spherely.contains(half, balance.geographies[sampled]))
    a = points[~side]
    drag = balance.parameters.basal_drag_pa_s_per_m * R_M ** 2 * balance.point_area_m2
    block = drag * (np.eye(3) * len(a) - a.T @ a)
    mids, torques = plate_torques(balance, world, rotations, cut.plate)
    other = ~np.asarray(spherely.contains(half, _points(mids)))
    torque = torques[other].sum(axis=0) - block @ (np.asarray(rotations[cut.plate]) / SECONDS_PER_MYR)
    following = np.roll(path, -1, axis=0)
    mid = sphere.unit(path + following)
    on = np.asarray(spherely.contains(region, _points(mid)))
    if on.sum() < 2:
        return None
    lengths = sphere.arc_km(path, following)[on] * 1000.
    normals = sphere.unit(np.cross(path, following))[on]
    arm = (np.cross(mid[on], normals) * lengths[:, None]).sum(axis=0)
    norm = float(arm @ arm)
    tension = float(-torque @ arm / (R_M * norm)) if norm > 0. else 0.
    strength, continental = _material(world, cut.plate, mid[on], rifting)
    return replace(cut, path=path, length_km=float(lengths.sum() / 1000.),
                   tension_n_per_m=tension,
                   strength_n_per_m=float(lengths @ strength / lengths.sum()),
                   continent_fraction=float(lengths @ continental / lengths.sum()))
