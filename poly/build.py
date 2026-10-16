"""Starting worlds: boundary networks with kinds assigned from the plates' relative motion.

Along a boundary, the right plate's motion relative to the left one is resolved into opening across
the boundary and sliding along it. Opening makes a ridge, closing a trench (the overriding plate is
chosen by a rule) and mostly sliding a transform. A boundary is cut into runs of one kind; a cut adds
a junction between the runs.
"""
from __future__ import annotations

import math

import numpy as np

from poly import sphere
from poly.topology import Network
from poly.world import Crust, World

TRANSFORM_RATIO = math.tan(math.radians(15.))  # slide more than ~3.7x the opening or closing: transform
MINIMUM_RUN = 3                                 # segments; shorter runs take their neighbours' kind
# Absolute fraction of the sphere (about 5 km2 on Earth), not a fraction of the requested area.
CONTINENT_FRACTION_TOLERANCE = 1e-8


def densify(points, spacing_km=100., closed=False):
    """Great-circle points no more than `spacing_km` apart (end point kept unless closed)."""
    points = np.asarray(points, float)
    pairs = list(zip(points, np.roll(points, -1, axis=0))) if closed else list(zip(points[:-1], points[1:]))
    out = []
    for a, b in pairs:
        count = max(1, int(math.ceil(float(sphere.arc_km(a, b)) / spacing_km)))
        angle = math.atan2(np.linalg.norm(np.cross(a, b)), float(a @ b))
        for k in range(count):
            f = k / count
            out.append(sphere.unit(math.sin((1 - f) * angle) * a + math.sin(f * angle) * b) if angle > 0 else a)
    if not closed:
        out.append(points[-1])
    return np.array(out)


def kinds_along(points, left_omega, right_omega):
    """Kind of each segment of a polyline from the right plate's motion relative to the left."""
    a, b = points[:-1], points[1:]
    mid = sphere.unit(a + b)
    tangent = sphere.unit(b - a)
    toward_right = np.cross(tangent, mid)
    velocity = np.cross(np.asarray(right_omega, float) - np.asarray(left_omega, float), mid)
    opening = np.einsum('ij,ij->i', velocity, toward_right)
    sliding = np.abs(np.einsum('ij,ij->i', velocity, tangent))
    kinds = np.where(np.abs(opening) >= TRANSFORM_RATIO * sliding, np.where(opening > 0, 'ridge', 'trench'), 'transform')
    return _smooth(list(kinds))


def _smooth(kinds):
    """Give runs shorter than MINIMUM_RUN the kind of their longer neighbour."""
    kinds = list(kinds)
    changed = True
    while changed and len(set(kinds)) > 1:
        changed = False
        runs, start = [], 0
        for i in range(1, len(kinds) + 1):
            if i == len(kinds) or kinds[i] != kinds[start]:
                runs.append((start, i))
                start = i
        for r, (lo, hi) in enumerate(runs):
            if hi - lo < MINIMUM_RUN and len(runs) > 1:
                before = runs[r - 1] if r > 0 else None
                after = runs[r + 1] if r + 1 < len(runs) else None
                pick = max((x for x in (before, after) if x), key=lambda x: x[1] - x[0])
                kinds[lo:hi] = [kinds[pick[0]]] * (hi - lo)
                changed = True
                break
    return kinds


def add_boundary(network, start, end, points, left, right, omegas, overriding_rule, closed=False):
    """Add a boundary from junction `start` to `end` (the same for a closed loop), cut into runs by kind."""
    kinds = kinds_along(points, omegas[left], omegas[right])
    cuts = [i for i in range(1, len(kinds)) if kinds[i] != kinds[i - 1]]
    if closed and len(set(kinds)) > 1:
        # Rotate the loop so it starts at a change of kind; the loop's own junction is then a cut.
        shift = cuts[0]
        points = np.vstack((points[shift:-1], points[:shift + 1]))
        kinds = kinds[shift:] + kinds[:shift]
        network.junctions[start] = sphere.unit(points[0])
        cuts = [i for i in range(1, len(kinds)) if kinds[i] != kinds[i - 1]]
    bounds = [0] + cuts + [len(kinds)]
    ends = [start] + [network.add_junction(points[c]) for c in cuts] + [end]
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        kind = str(kinds[lo])
        over = overriding_rule(left, right) if kind == 'trench' else None
        network.add_section(ends[k], ends[k + 1], points[lo + 1:hi], left, right, kind, over)


def cap_world(centre_lonlat=(0., 0.), radius_deg=40., cap_omega=(0., 0., .01), rest_omega=(0., 0., 0.),
              spacing_km=100., overriding='rest'):
    """A cap plate (1) inside a world plate (2); boundary kinds follow their relative motion."""
    centre = sphere.from_lonlat(*centre_lonlat)
    pole = sphere.unit(np.cross(centre, [0., 0., 1.]) if abs(centre[2]) < .99 else np.cross(centre, [1., 0., 0.]))
    start = sphere.rotate(centre, sphere.rotation(pole * math.radians(radius_deg)))
    count = max(12, int(2 * math.pi * math.sin(math.radians(radius_deg)) * sphere.RADIUS_KM / spacing_km))
    ring = np.array([sphere.rotate(start, sphere.rotation(centre * 2 * math.pi * k / count)) for k in range(count)])
    omegas = {1: np.asarray(cap_omega, float), 2: np.asarray(rest_omega, float)}
    network = Network()
    junction = network.add_junction(ring[0])
    loop = np.vstack((ring, ring[:1]))
    rule = (lambda left, right: 2) if overriding == 'rest' else (lambda left, right: 1)
    add_boundary(network, junction, junction, loop, 1, 2, omegas, rule, closed=True)
    world = World(network, [], names={1: 'Cap', 2: 'World'})
    polygons = network.polygons()
    world.crust = [Crust(p, 'ocean', shape, born_myr=-50.) for p, shape in polygons.items()]
    return world, omegas


def voronoi_world(count=8, seed=1, spacing_km=100., speed_km_myr=30., continents=0, overriding=None,
                  continent_fraction=0.):
    """`count` Voronoi plates with seeded geometry and random rotations.

    `continents` places that many circular continents, as before. Alternatively, a positive
    `continent_fraction` requests a total continental *crust* area fraction in [0, 1], within
    CONTINENT_FRACTION_TOLERANCE of the sphere. It is not an emerged-land fraction. Fraction mode
    shrinks plate-shaped continents towards their Voronoi seeds, leaving ocean margins; the
    outlines need not be circular. Do not combine the count and fraction modes. Zero fraction
    leaves the count mode in control (and gives an all-ocean world when `continents` is zero).
    """
    from scipy.spatial import SphericalVoronoi
    try:
        continent_fraction = float(continent_fraction)
    except (TypeError, ValueError) as error:
        raise ValueError('continent_fraction must be a finite number in [0, 1].') from error
    if not math.isfinite(continent_fraction) or not 0. <= continent_fraction <= 1.:
        raise ValueError('continent_fraction must be a finite number in [0, 1].')
    if continent_fraction > 0. and continents:
        raise ValueError('Choose continents or continent_fraction, not both.')
    rng = np.random.default_rng(seed)
    seeds = sphere.unit(rng.normal(size=(count, 3)))
    voronoi = SphericalVoronoi(seeds, radius=1.)
    voronoi.sort_vertices_of_regions()
    # scipy sorts each region's vertices around it but not always the same way round; turn every
    # region counter-clockwise, which is when its own seed lies inside the region left of the ring.
    from poly.topology import _contains
    for p, region in enumerate(voronoi.regions):
        if not _contains(sphere.polygon(voronoi.vertices[region]), seeds[p]):
            voronoi.regions[p] = list(region)[::-1]
    omegas = {p + 1: sphere.unit(rng.normal(size=3)) * speed_km_myr / sphere.RADIUS_KM for p in range(count)}
    network = Network()
    corner = {i: network.add_junction(v) for i, v in enumerate(voronoi.vertices)}
    done = set()
    rule = overriding or (lambda left, right: max(left, right))
    for p, region in enumerate(voronoi.regions):
        for k in range(len(region)):
            a, b = region[k], region[(k + 1) % len(region)]
            if (b, a) in done:
                continue
            done.add((a, b))
            # The region ring is counter-clockwise, so plate p lies to the left of a -> b.
            q = next(r for r, other in enumerate(voronoi.regions) if r != p and a in other and b in other)
            points = densify([voronoi.vertices[a], voronoi.vertices[b]], spacing_km)
            add_boundary(network, corner[a], corner[b], points, p + 1, q + 1, omegas, rule)
    world = World(network, [], names={p: 'Plate %02d' % p for p in omegas})
    polygons = network.polygons()
    world.crust = [Crust(p, 'ocean', shape, born_myr=-50.) for p, shape in polygons.items()]
    if continents:
        _add_continents(world, polygons, continents, rng)
    if continent_fraction:
        _add_continental_fraction(world, polygons, seeds, continent_fraction)
    return world, omegas


def _add_continents(world, polygons, count, rng, fill=.45):
    """Put a circular continent inside `count` random plates (cut out of their ocean).

    Each disc's radius is `fill` of the distance from the plate's centre to its edge.
    """
    order = [int(p) for p in rng.choice(sorted(polygons), size=min(count, len(polygons)), replace=False)]
    for plate in order:
        plate = int(plate)
        ring = sphere.polygons(polygons[plate])[0][0]
        centre = sphere.unit(ring.mean(axis=0))
        radius = fill * float(np.min(sphere.arc_km(ring, centre[None]))) / sphere.RADIUS_KM
        pole = sphere.unit(np.cross(centre, ring[0]))
        start = sphere.rotate(centre, sphere.rotation(pole * radius))
        disc = np.array([sphere.rotate(start, sphere.rotation(centre * 2 * math.pi * k / 48)) for k in range(48)])
        continent = sphere.polygon(disc)
        for piece in world.crust:
            if piece.plate == plate:
                piece.shape = sphere.difference(piece.shape, continent)
        world.crust.append(Crust(plate, 'continent', continent))


def _add_continental_fraction(world, polygons, seeds, fraction):
    """Solve for seeded, plate-contained continents with the requested global crust area.

    A common scale moves each plate's outline along great circles towards its generating seed.
    This retains the irregular plate shapes at high coverage, where inscribed discs cannot fill
    enough area. Clipping every candidate to its source plate prevents crust overlap. The two
    endpoints are explicitly empty and the complete plate, avoiding degenerate rings.
    """
    if fraction == 1.:
        world.crust = [Crust(p, 'continent', shape) for p, shape in polygons.items()]
        return
    outlines = {}
    for plate, shape in polygons.items():
        parts = sphere.polygons(shape)
        if len(parts) != 1 or len(parts[0]) != 1:
            raise ValueError('Continental fraction initialization requires simple Voronoi plates.')
        ring, centre = parts[0][0], seeds[plate - 1]
        cosine = np.clip(ring @ centre, -1., 1.)
        radial = ring - cosine[:, None] * centre
        sine = np.linalg.norm(radial, axis=1)
        if np.any(sine < 1e-12):
            raise ValueError('Continental fraction initialization has a degenerate radial outline; try another seed.')
        outlines[plate] = (centre, np.arctan2(sine, cosine), radial / sine[:, None])
    lower, upper = 0., 1.
    for _ in range(50):
        scale = .5 * (lower + upper)
        continents = {}
        try:
            for plate, (centre, angle, tangent) in outlines.items():
                points = (np.cos(scale * angle)[:, None] * centre
                          + np.sin(scale * angle)[:, None] * tangent)
                continents[plate] = sphere.intersection(polygons[plate], sphere.polygon(points))
        except (ValueError, RuntimeError) as error:
            raise ValueError('Continental fraction initialization could not build valid plate-contained '
                             'outlines; try another seed.') from error
        achieved = sum(sphere.area_km2(shape) for shape in continents.values()) / sphere.SPHERE_KM2
        if abs(achieved - fraction) <= CONTINENT_FRACTION_TOLERANCE:
            break
        if achieved < fraction:
            lower = scale
        else:
            upper = scale
    else:
        raise ValueError('Continental crust fraction %.9g could not be reached within %.1g; achieved %.9g.'
                         % (fraction, CONTINENT_FRACTION_TOLERANCE, achieved))
    crust = []
    for plate, continent in continents.items():
        ocean = sphere.difference(polygons[plate], continent)
        if sphere.area_km2(ocean) > 0.:
            crust.append(Crust(plate, 'ocean', ocean, born_myr=-50.))
        if sphere.area_km2(continent) > 0.:
            crust.append(Crust(plate, 'continent', continent))
    world.crust = crust
