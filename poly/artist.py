"""Artist scenarios and material-attached relief over the spherical polygon model.

The mechanical field remains crustal thickness on moving meshes. Sub-mesh relief is
explicitly an output amplification, not additional crust or a geodynamic prediction.
Water is calibrated once from equal-area samples, then its volume is held fixed.
"""
from __future__ import annotations

import numpy as np
import spherely
from scipy.spatial import cKDTree

from poly import geology, scenario, sphere, surface
from poly.build import densify, voronoi_world
from poly.mesh import ContinentMesh, ISOSTASY, REFERENCE_KM
from poly.regions import from_network
from poly.world import Crust


def equal_area_points(count=32768):
    """Deterministic Fibonacci quadrature; every point has the same spherical weight."""
    i = np.arange(count, dtype=float)
    z = 1. - 2. * (i + .5) / count
    angle = i * (np.pi * (3. - np.sqrt(5.)))
    radius = np.sqrt(1. - z * z)
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))


def _noise(points, seed):
    """Smooth hashed lattice noise in 3D, with no longitude seam or polar singularity."""
    cell = np.floor(points).astype(np.int64)
    fraction = points - cell
    blend = fraction ** 3 * (fraction * (fraction * 6. - 15.) + 10.)
    result = np.zeros(len(points))
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                corner = (cell + [x, y, z]).astype(np.uint32)
                key = (corner[:, 0] * np.uint32(374761393) ^ corner[:, 1] * np.uint32(668265263)
                       ^ corner[:, 2] * np.uint32(2246822519) ^ np.uint32(seed % 2**32))
                key = (key ^ (key >> 13)) * np.uint32(1274126177)
                key ^= key >> 16
                weight = np.prod(np.where(np.array([x, y, z]), blend, 1. - blend), axis=1)
                result += weight * (key.astype(float) / (2**31 - .5) - 1.)
    return result


def texture(points, seed, scale=1.):
    """Bounded seamless fractal noise, evaluated in the rock's reference frame."""
    points = np.asarray(points).reshape(-1, 3)
    total = np.zeros(len(points))
    norm = 0.
    for octave, (frequency, amplitude) in enumerate(((12., 1.), (27., .5), (59., .25), (127., .125), (263., .0625))):
        total += amplitude * _noise(points * frequency * scale, seed + octave * 71)
        norm += amplitude
    return total / norm


def mountain_detail(points, seed):
    """Ridged domain-warped detail in material coordinates; output relief only."""
    warp = np.column_stack([_noise(points * 11., seed + 1901 + k * 73) for k in range(3)])
    warped = points + .035 * warp
    result = np.zeros(len(points))
    for octave, (frequency, amplitude) in enumerate(((35., .6), (85., .27), (190., .13))):
        ridges = (1. - np.abs(_noise(warped * frequency, seed + 5009 + octave * 71))) ** 2
        result += amplitude * (ridges - .55)
    return result


def _land_shape(seed, fraction, layout='varied'):
    """A few lobed continental masses independent of the plate tessellation."""
    if not 0.05 <= fraction <= .85:
        raise ValueError('Scenario continental coverage must be between 5% and 85%.')
    rng = np.random.default_rng(seed + 811)
    centres = sphere.from_lonlat([-150., -82., -15., 59., 127., 170.],
                                 [22., -33., 41., -27., 39., -44.])
    rotation = sphere.rotation(sphere.unit(rng.normal(size=3)) * rng.uniform(0., np.pi))
    centres = sphere.rotate(centres, rotation)
    if layout not in ('simple', 'varied'):
        raise ValueError('Unknown starting layout: %s' % layout)
    warp = scenario.SphericalWarp(seed + 1069) if layout == 'varied' else None
    angles = np.linspace(0., 2. * np.pi, 384 if warp else 192, endpoint=False)
    outlines = []
    for centre in centres:
        east = sphere.unit(np.cross(centre, [0., 0., 1.]))
        north = np.cross(centre, east)
        direction = np.cos(angles)[:, None] * east + np.sin(angles)[:, None] * north
        radius = np.ones(len(angles))
        for harmonic, amplitude in ((2, .15), (3, .13), (5, .08), (9, .045), (17, .025), (31, .012), (61, .005)):
            radius += amplitude * np.sin(harmonic * angles + rng.uniform(0., 2. * np.pi))
        outlines.append((centre, direction, radius * rng.uniform(.85, 1.15)))

    def build(scale):
        shapes = []
        for c, d, r in outlines:
            ring = np.cos(scale * r)[:, None] * c + np.sin(scale * r)[:, None] * d
            shapes.append(sphere.polygon(warp(ring) if warp else ring))
        return sphere.union(*shapes)
    lower, upper = .05, 1.3
    for _ in range(40):
        middle = .5 * (lower + upper)
        shape = build(middle)
        actual = sphere.area_km2(shape) / sphere.SPHERE_KM2
        if abs(actual - fraction) < 1e-7:
            return shape
        if actual < fraction:
            lower = middle
        else:
            upper = middle
    raise ValueError('Could not initialize the requested continental area.')


def make_world(seed=1, plates=10, land_fraction=.55, continent_fraction=.65, spacing_km=150.,
               layout='varied'):
    """Dispersed continents with inherited relief, shelves and an explicit water inventory.

    The inherited structures are chosen starting conditions, not simulated ancient history.
    Continents cross plate boundaries. Fine terrain follows material when ownership changes.
    """
    if not 0. < land_fraction < continent_fraction:
        raise ValueError('Initial land must be positive and smaller than continental coverage.')
    if layout not in ('simple', 'varied'):
        raise ValueError('Unknown starting layout: %s' % layout)
    network, rotations = voronoi_world(count=plates, seed=seed, spacing_km=spacing_km)
    if layout == 'varied':
        scenario.curve_network(network, seed, spacing_km)
    world = from_network(network)
    land = _land_shape(seed, continent_fraction, layout)
    world.crust = []
    for plate, region in world.regions.items():
        continent = sphere.intersection(region, land)
        ocean = sphere.difference(region, land)
        if sphere.area_km2(ocean) > 1.:
            world.crust.append(Crust(plate, 'ocean', ocean, born_myr=-60.))
        if sphere.area_km2(continent) > 1.:
            world.crust.append(Crust(plate, 'continent', continent))
    geology.activate(world, seed)
    geology.separate_bodies(world)
    coasts = np.vstack([densify(r, 30., closed=True) for rings in sphere.polygons(land) for r in rings])
    coast_tree = cKDTree(coasts)
    rng = np.random.default_rng(seed + 2749)
    inherited_axes = sphere.unit(rng.normal(size=(3, 3)))
    for piece in world.crust:
        if piece.kind != 'continent':
            continue

        def thickness(points):
            distance = coast_tree.query(points)[0] * sphere.RADIUS_KM
            shelf = 1. - np.exp(-(distance / 210.) ** 1.5)
            ancient_belts = np.exp(-.5 * ((points @ inherited_axes.T) / .045) ** 2).max(axis=1)
            return 28. + 11. * shelf + 24. * ancient_belts * shelf + 2.5 * texture(points, seed, .35)

        piece.mesh = ContinentMesh.build(piece.shape, 35., field=thickness)
        if piece.mesh is not None:
            piece.thickness_km = piece.mesh.volume_km3(piece.area_km2()) / piece.area_km2()
        else:
            centre = spherely.centroid(piece.shape)
            point = sphere.from_lonlat(float(spherely.get_x(centre)), float(spherely.get_y(centre)))
            piece.thickness_km = float(thickness(point[None])[0])
    from poly.coarsen import consolidate
    consolidate(world)
    scenario.initialize_contacts(world, rotations)
    initialize_water(world, land_fraction)
    return world, rotations


def sample(world, points, sea_level_km=None):
    """Geological surface plus bounded, material-attached sub-mesh relief."""
    if not geology.enabled(world):
        raise ValueError('Artist terrain requires a geological scenario.')
    xyz = np.asarray(points).reshape(-1, 3)
    level = world.geology['sea_level_km'] if sea_level_km is None else sea_level_km
    result = surface.sample(world, xyz, level)
    for piece in world.crust:
        if piece.kind != 'continent':
            continue
        candidates = np.flatnonzero((result.plate == piece.plate) & result.continental_crust)
        if not len(candidates):
            continue
        local = sphere.rotate(xyz[candidates], world.frame(piece.plate).T)
        lon, lat = sphere.to_lonlat(local)
        hit = np.asarray(spherely.contains(piece.shape, spherely.points(lon, lat)))
        ids = candidates[hit]
        reference = geology.reference_points(piece, local[hit])
        # One global material texture remains continuous across an initially cut continent.
        noise = texture(reference, world.geology['seed'])
        height = result.elevation_km[ids]
        amplitude = .09 + .45 * np.maximum(height, 0.)
        result.elevation_km[ids] += amplitude * noise
        result.elevation_km[ids] += .9 * np.maximum(height, 0.) * mountain_detail(reference, world.geology['seed'])
    return result


def initialize_water(world, land_fraction, count=32768):
    """Set the initial emerged area, then retain equivalent global water depth in km."""
    bed = sample(world, equal_area_points(count), sea_level_km=0.).elevation_km
    if not np.isfinite(bed).all():
        raise ValueError('Initial surface has unresolved crust gaps.')
    level = float(np.quantile(bed, 1. - land_fraction))
    world.geology['sea_level_km'] = level
    world.geology['water_depth_km'] = float(np.maximum(level - bed, 0.).mean())
    return level


def _water_bed(world, points, bed, diagnostics=None):
    """Move quadrature points off tolerated numerical slits; never alter crust geometry.

    A single quadrature point represents ~15,500 km2, far more than the full accepted
    geometry-defect budget. Treating a point on a metre-scale slit as an empty entire
    quadrature cell is inappropriate. Audit the actual geometry first, then sample a
    nearby valid surface, recording the displacement. Surface masks retain their gaps.
    """
    bed = np.asarray(bed).copy()
    missing = np.flatnonzero(~np.isfinite(bed))
    if not len(missing):
        return bed
    from poly import acceptance
    from poly.regions import FULL
    acceptance.geometry(world, FULL, {})  # raises on invalid coverage; no budget is changed
    original_count, maximum_shift = len(missing), 0.
    for distance in (.001, .01, .1, 1., 10., 50.):
        if not len(missing):
            break
        probes = np.array([sphere.disc(points[i], distance, count=8) for i in missing])
        heights = sample(world, probes.reshape(-1, 3), sea_level_km=0.).elevation_km.reshape(-1, 8)
        valid = np.isfinite(heights)
        hit = valid.any(axis=1)
        if hit.any():
            rows = np.flatnonzero(hit)
            bed[missing[hit]] = heights[rows, np.argmax(valid[hit], axis=1)]
            maximum_shift = distance
        missing = missing[~hit]
    if len(missing):
        raise ValueError('Surface gaps prevent a resolved water quadrature.')
    if diagnostics is not None:
        diagnostics['water_quadrature_shifted_points'] = original_count
        diagnostics['water_quadrature_max_shift_km'] = maximum_shift
    return bed


def update_water(world, count=32768, diagnostics=None):
    """Solve sea level from the saved water volume as topography and basins change."""
    water = world.geology['water_depth_km']
    if water is None:
        raise ValueError('Water inventory has not been initialized.')
    points = equal_area_points(count)
    bed = sample(world, points, sea_level_km=0.).elevation_km
    bed = _water_bed(world, points, bed, diagnostics)
    lower, upper = float(bed.min()), float(bed.max() + water + 1.)
    for _ in range(60):
        level = .5 * (lower + upper)
        if np.maximum(level - bed, 0.).mean() < water:
            lower = level
        else:
            upper = level
    world.geology['sea_level_km'] = .5 * (lower + upper)
    if diagnostics is not None:
        level = world.geology['sea_level_km']
        diagnostics['land_fraction'] = float(np.mean(bed >= level))
        diagnostics['sea_level_km'] = level
        diagnostics['water_residual_km'] = float(np.maximum(level - bed, 0.).mean() - water)
    return world.geology['sea_level_km']


def metrics(world, count=32768):
    from poly.coarsen import MINIMUM_KM2, volume_km3
    points = equal_area_points(count)
    sampled = sample(world, points)
    valid = np.isfinite(sampled.elevation_km)
    height = sampled.height_above_water_km
    peak_thickness = geology.maximum_thickness(world)
    measured = dict(time_myr=world.time_myr, land_fraction=float(sampled.land.mean()),
                continental_fraction=float(sampled.continental_crust.mean()),
                sea_level_km=sampled.sea_level_km,
                maximum_height_km=float(height[valid].max()),
                minimum_height_km=float(height[valid].min()),
                maximum_continental_thickness_km=peak_thickness,
                maximum_resolved_height_km=ISOSTASY * (peak_thickness - REFERENCE_KM) - sampled.sea_level_km,
                minimum_continental_body_km2=MINIMUM_KM2,
                unresolved_continent_volume_km3=volume_km3(world),
                unresolved_fraction=float((~valid).mean()), plate_count=len(world.plates()),
                collision_belts=sum(f['kind'] == 'collision' for f in world.geology['features']),
                rift_belts=sum(f['kind'] == 'rift' for f in world.geology['features']))
    bed = _water_bed(world, points, sampled.elevation_km, measured)
    measured['water_depth_km'] = float(np.maximum(sampled.sea_level_km - bed, 0.).mean())
    return measured


def draw(world, path, width=1600, boundaries=False):
    """Shaded physical relief. Returns the metrics of equal-area samples, not map pixels."""
    from PIL import Image, ImageDraw
    from poly.render import _tint, COLOURS
    height = width // 2
    longitudes = np.linspace(-180., 180., width, endpoint=False) + 180. / width
    latitudes = np.linspace(90., -90., height, endpoint=False) - 90. / height
    # Keep full-resolution height and final RGB only. S2 objects, strings and sampling
    # temporaries are bounded by a stripe, so 8K exports do not allocate a whole globe
    # of Python/native point objects at once. This changes output memory, not physics.
    z = np.empty((height, width), np.float32)
    stripe = max(1, min(64, 262144 // width))
    invalid_count = 0
    for start in range(0, height, stripe):
        stop = min(start + stripe, height)
        lon, lat = np.meshgrid(longitudes, latitudes[start:stop])
        points = sphere.from_lonlat(lon.ravel(), lat.ravel())
        sampled = sample(world, points)
        block = sampled.height_above_water_km.reshape(stop - start, width)
        invalid_count += int((~np.isfinite(block)).sum())
        z[start:stop] = block
    land_palette = [(0., (169, 183, 133)), (.4, (172, 172, 115)), (1.2, (167, 146, 103)),
                    (2.5, (139, 121, 98)), (4., (182, 178, 166)), (6., (240, 240, 233))]
    ocean_palette = [(0., (115, 182, 190)), (.25, (68, 142, 160)), (1., (35, 101, 128)),
                     (3., (20, 65, 94)), (6., (12, 37, 63))]
    pixels = np.empty((height, width, 3), np.uint8)
    dy = np.pi * sphere.RADIUS_KM / height
    for start in range(0, height, stripe):
        stop = min(start + stripe, height)
        block = z[start:stop]
        valid = np.isfinite(block)
        land, water = (block >= 0.) & valid, (block < 0.) & valid
        colours = np.full((*block.shape, 3), (225., 40., 160.), np.float32)
        colours[land] = _tint(block[land], land_palette)
        colours[water] = _tint(-block[water], ocean_palette)
        dx = 2. * np.pi * sphere.RADIUS_KM / width * np.maximum(np.cos(np.radians(latitudes[start:stop])), .08)
        safe = np.where(valid, block, 0.)
        east = (np.roll(safe, -1, axis=1) - np.roll(safe, 1, axis=1)) / (2. * dx[:, None])
        rows = np.arange(start, stop)
        before, after = np.maximum(rows - 1, 0), np.minimum(rows + 1, height - 1)
        up, down = z[before], z[after]
        north = (np.where(np.isfinite(up), up, 0.) - np.where(np.isfinite(down), down, 0.)) / ((after - before)[:, None] * dy)
        shade = np.clip(1. + 8. * (-.65 * east + .65 * north), .55, 1.3)
        colours *= shade[..., None]
        colours[~valid] = (225., 40., 160.)
        pixels[start:stop] = np.uint8(np.clip(colours, 0., 255.))
    picture = Image.fromarray(pixels)
    if boundaries:
        pen = ImageDraw.Draw(picture)
        for points, kind in world.boundary_lines():
            plon, plat = sphere.to_lonlat(points)
            x, y = (plon + 180.) / 360. * width, (90. - plat) / 180. * height
            for k in range(len(x) - 1):
                if abs(x[k + 1] - x[k]) < width / 2:
                    pen.line([(x[k], y[k]), (x[k + 1], y[k + 1])], fill=COLOURS[kind], width=1)
    picture.save(path)
    measured = metrics(world)
    measured['unresolved_map_pixels'] = invalid_count
    return measured
