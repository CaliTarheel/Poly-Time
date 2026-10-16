"""Plates as regions: overlaps and gaps resolved by boundary type, boundaries derived every step.

The plates are the primary objects. Each plate is its crust (continental and ocean polygons stored in
the plate's own frame, as GPlates stores features) and its region is where that crust is. Each step:

1. every plate's region is carried by its rotation;
2. where carried plates overlap, the boundary that was there decides: at a trench the downgoing plate
   loses the overlap (subducted, or continent arriving); anywhere else the overlap is split along the
   pair's half-stage line, so the boundary stays midway, and each side's loss is booked (compression,
   collision, sliver);
3. where they leave a gap, the gap is split the same way among the plates around it and becomes new
   ocean (ridge, opening behind a trench, sliver, junction);
4. a plate whose region has come apart becomes separate plates, a piece too small to be a plate goes
   to the neighbour it borders most, and a plate with no region is gone;
5. the boundaries are derived again from where the new regions touch. Each boundary segment takes
   the type, trench polarity and closing-so-far of the nearest old segment of the same plates; a new
   contact is typed from the motion. Then the same rules as before apply: types follow the motion,
   a converging boundary that has closed 100 km becomes a trench, a trench a continent reaches becomes
   a collision.

Nothing is normalized away: the step checks that the plates tile the sphere, that each plate's crust
tiles it, and that the area created equals the area destroyed, term by term.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import spherely

from poly import sphere
from poly.build import TRANSFORM_RATIO, _smooth
from poly.mesh import ContinentMesh
from poly.sphere import DETAIL_KM2
from poly.topology import segment_motion
from poly.world import (AGE_BAND_MYR, COLLISION_KM2, INITIATION_KM, MINIMUM_PLATE_KM2, SAMPLE_KM, TOLERANCE,
                        Crust, _merge_bands)

GAINS = ('ridge', 'trench_opening', 'sliver', 'junction', 'escape', 'arc', 'accreted')
LOSSES = ('subducted', 'continent_at_trench', 'collision', 'compression', 'sliver', 'junction', 'escape', 'arc',
          'accreted', 'overridden')
INHERIT_KM = 300.   # a boundary segment takes the type of the nearest old one of the same plates within this
PROBE_KM = .5       # how far outside a plate's outline its neighbour is looked for
MARGIN = .2         # hysteresis around the transform threshold when re-typing
TINY_KM2 = 1e-6     # areas below this are numerical
CHECK_TOLERANCE = 1e-4  # of the sphere's area, per step (0.01%)
CRUMB_KM2 = 100.        # detached bits of a plate smaller than this are left with it
SUTURE_KM = 100.        # a collision that has closed this far welds its two plates into one
SIMPLIFY_KM2 = 100.     # a boundary vertex cutting off less than this with its neighbours is dropped
SIMPLIFY_EVERY_MYR = 1. # outlines are simplified once this much time has passed, not every sub-step
SIMPLIFY_GROWTH = 1.25  # or sooner, once the outlines have this many times the vertices they had after
# Lateral escape: this share of the continent shortened at a collision comes back as continent
# along the piece's coast within ESCAPE_REACH_KM of the collision, over the plate's own ocean
# (Indochina out of the India-Asia collision). 0 keeps it all as thickness.
ESCAPE_FRACTION = .25
ESCAPE_REACH_KM = 1500.
ESCAPE_STEP_KM = 2.     # escape is delivered as a band of the coast this wide, once that much is owed
# Volcanic arcs: magma rises into the overriding plate ARC_KM behind every trench, at
# ARC_KM3_PER_KM_MYR per km of trench per Myr for ARC_REFERENCE_KM_MYR of convergence (Earth's
# arcs add 20-40 km3/km/Myr). Into a continent it thickens the crust there; over ocean it is owed
# on the trench segments and, once there is enough for a strip ARC_WIDTH_KM wide and ARC_THICK_KM
# thick along the run, built as an island arc: continental crust over the plate's ocean.
ARC_KM = 150.
ARC_KM3_PER_KM_MYR = 30.
ARC_REFERENCE_KM_MYR = 50.
ARC_WIDTH_KM = 60.
ARC_THICK_KM = 30.  # ARC_WIDTH_KM x ARC_THICK_KM of root per km of trench: 60 Myr of reference convergence
# Before the ribbon, the arc shows as islands. ISLAND_SHARE of the magma owed over ocean becomes
# land straight away (Earth's arcs carry 0.5-10 km2 of land per km of arc; the rest is the
# submarine ridge and root, carried on the segments as owed until the ribbon), in chunks of
# ISLAND_KM2 at ISLAND_THICK_KM. A chunk goes to the free site with the most banked (one site per
# trench segment, fixed along it by the segment's identity, free while no island lies within
# ISLAND_SPACING_KM) or, with no site free, beside the island nearest the most banked site, where
# it welds with every island it touches. Islands take no magma themselves: they grow by chunks, so
# none thickens without limit.
ISLAND_SHARE = .05
ISLAND_KM2 = 50.
ISLAND_THICK_KM = 20.
ISLAND_SPACING_KM = 70.
# Nothing emerges until the arc has built its submarine ridge: a run of trench segments banks for
# islands only once its root holds this much per km of trench (a ridge 60 km wide and 10 km thick;
# 20 Myr at the reference convergence, 11 Myr at 90 km/Myr). Earth's arcs take 5-15 Myr to break
# the surface. A third of the ribbon's threshold, so islands always come first.
ARC_EMERGE_KM3_PER_KM = 600.
# Terranes: a continent piece smaller than this reaching a trench is not shortened into itself but
# docks onto the overriding plate with its volume (islands, arcs, microcontinents: Kohistan,
# Wrangellia); larger ones collide and suture as before.
TERRANE_KM2 = 1e6


def _full():
    ring = sphere.from_lonlat([0., 90., 180., 270.], [0., 0., 0., 0.])
    return spherely.union(sphere.polygon(ring), sphere.polygon(ring[::-1]))


FULL = _full()


@dataclass
class Boundary:
    """A boundary between two plates, the lower-numbered on its left, with per-segment state."""
    left: int
    right: int
    points: np.ndarray
    kinds: list
    overriding: list
    closed_km: np.ndarray
    idents: object = None   # one int per segment, kept as the segment is carried from step to step
    arc_km3: object = None  # per segment: magma owed as arc root, and magma banked for islands

    def arcs(self):
        if self.arc_km3 is None:
            self.arc_km3 = np.zeros((len(self.kinds), 2))
        elif np.ndim(self.arc_km3) == 1:  # saved before islands were banked
            self.arc_km3 = np.c_[np.asarray(self.arc_km3, float), np.zeros(len(self.arc_km3))]
        return self.arc_km3

    def mids(self):
        return sphere.unit(self.points[:-1] + self.points[1:])


@dataclass
class RegionWorld:
    crust: list
    frames: dict
    regions: dict                      # plate -> region in the world frame, as of the last step
    boundaries: list
    names: dict = field(default_factory=dict)
    parents: dict = field(default_factory=dict)
    time_myr: float = 0.
    ledger: list = field(default_factory=list)
    next_ident: int = 0
    # Where plates welded: each a dict(plate, points, time_myr) with the suture line in that plate's
    # frame, carried with the plate like its crust. Sutures are where continents break again.
    sutures: list = field(default_factory=list)

    # ------------------------------------------------------------------ views
    def plates(self) -> list:
        return sorted(self.regions)

    def frame(self, plate):
        return self.frames.setdefault(plate, np.eye(3))

    def plate_polygons(self) -> dict:
        return dict(self.regions)

    def world_regions(self) -> dict:
        return dict(self.regions)

    def segments(self):
        """(left, right, a, b, kind, overriding, age_myr) for every boundary segment, for the force balance.

        `age_myr` is the age of the crust going down at a trench segment (None elsewhere), so slab
        pull can depend on it.
        """
        ages = self._slab_ages()
        for b in self.boundaries:
            for k in range(len(b.points) - 1):
                yield (b.left, b.right, b.points[k], b.points[k + 1], b.kinds[k], b.overriding[k],
                       ages.get((id(b), k)))

    def _slab_ages(self) -> dict:
        """(boundary id, segment) -> age (Myr) of the crust SAMPLE_KM down the downgoing side of each trench."""
        by_plate = {}
        for b in self.boundaries:
            if len(b.points) < 2:
                continue
            trench = [k for k, kind in enumerate(b.kinds) if kind == 'trench' and b.overriding[k] in (b.left, b.right)]
            if not trench:
                continue
            mids = b.mids()
            poles = sphere.unit(np.cross(b.points[:-1], b.points[1:]))
            angle = SAMPLE_KM / sphere.RADIUS_KM
            for k in trench:
                down = b.right if b.overriding[k] == b.left else b.left
                sign = 1. if down == b.left else -1.
                point = sphere.unit(mids[k] * np.cos(angle) + sign * poles[k] * np.sin(angle))
                by_plate.setdefault(down, []).append(((id(b), k), point))
        ages = {}
        for plate, items in by_plate.items():
            keys = [key for key, _ in items]
            local = sphere.rotate(np.array([p for _, p in items]), self.frame(plate).T)
            lon, lat = sphere.to_lonlat(local)
            where = spherely.points(lon, lat)
            found = np.zeros(len(keys), bool)
            for c in self.crust:
                if c.plate != plate:
                    continue
                spherely.prepare(c.shape)
                hit = np.asarray(spherely.contains(c.shape, where)) & ~found
                if hit.any():
                    age = 0. if c.born_myr is None else max(self.time_myr - c.born_myr, 0.)
                    for k in np.flatnonzero(hit):
                        ages[keys[k]] = age
                    found |= hit
        return ages

    def boundary_lines(self):
        """(points, kind) for every run of one kind, for maps."""
        for b in self.boundaries:
            start = 0
            for k in range(1, len(b.kinds) + 1):
                if k == len(b.kinds) or b.kinds[k] != b.kinds[start]:
                    yield b.points[start:k + 1], b.kinds[start]
                    start = k

    def suture_lines(self):
        """(points, time_myr) of every suture, in the world frame, for maps and for rifting."""
        for s in getattr(self, 'sutures', []):
            if s['plate'] in self.regions:
                yield sphere.rotate(s['points'], self.frame(s['plate'])), s['time_myr']

    def areas(self) -> dict:
        result = {}
        for piece in self.crust:
            result[piece.kind] = result.get(piece.kind, 0.) + piece.area_km2()
        return result

    # ------------------------------------------------------------------- step
    def step(self, rotations_per_myr: dict, dt: float = 1.) -> dict:
        rotations_per_myr = self._complete(rotations_per_myr)
        self._mesh_continents()
        crust_before = sum(c.area_km2() for c in self.crust)
        old_regions = dict(self.regions)
        old_index = self._index(rotations_per_myr=rotations_per_myr, dt=dt)
        motion = {p: sphere.rotation(np.asarray(rotations_per_myr[p], float) * dt) for p in self.plates()}
        for plate, matrix in motion.items():
            self.frames[plate] = matrix @ self.frame(plate)
        # The crust is the plate: each region was rebuilt from its crust at the end of the last step,
        # so carrying the region is carrying the crust, and the rebuild below keeps them together.
        moved = {p: sphere.rotate_geography(r, motion[p]) for p, r in self.regions.items()}
        entry = dict(time_myr=self.time_myr + dt, gained={k: 0. for k in GAINS}, lost={k: 0. for k in LOSSES},
                     events=[], continent_arrived=[], thickened_km3=0., arc_km3=0., eroded_km3=0.,
                     sediment_to_sea_km3=0.)
        lost = {p: [] for p in moved}      # (world geography, boundary kind, overriding plate or None)
        gained = {p: [] for p in moved}    # (world geography, gain category)
        stage_cache = {}

        def stage(group, plate):
            """`plate`'s old region carried by the mean rotation of `group` (its share of their gap or overlap)."""
            key = (tuple(sorted(group)), plate)
            if key not in stage_cache:
                mean = np.mean([np.asarray(rotations_per_myr[p], float) for p in group], axis=0) * dt
                # Only used to draw the line an overlap or gap is split along: a looser allowance
                # moves which side gets a sliver, never how much crust there is.
                stage_cache[key] = sphere.rotate_geography(old_regions[plate], sphere.rotation(mean), tolerance=1e-4)
            return stage_cache[key]

        plates = self.plates()
        reach = {}
        for p in plates:
            reach[p] = float(np.linalg.norm(rotations_per_myr[p])) * sphere.RADIUS_KM * dt
        runs = self._runs(rotations_per_myr, dt, reach)   # pair -> [(zone, kind, overriding)]
        # Overlaps: within a run's zone, by that run's type; elsewhere split along the half-stage line.
        # Pairs are resolved one after another, each plate's losses taken out before its next pair:
        # where three plates overlap, the pairs alone could each give the ground away (a gap) or
        # each keep it (an overlap); this way exactly one plate keeps it.
        holding = dict(moved)
        for i, p in enumerate(plates):
            for q in plates[i + 1:]:
                overlap = sphere.intersection(holding[p], holding[q])
                if sphere.area_km2(overlap) < TINY_KM2:
                    continue
                before = len(lost[p]), len(lost[q])
                remaining = overlap
                for zone, kind, over in runs.get((p, q), []):
                    local = sphere.intersection(remaining, zone)
                    if sphere.area_km2(local) < TINY_KM2:
                        continue
                    remaining = sphere.difference(remaining, local)
                    self._split_overlap(local, kind, over, p, q, stage, lost)
                if sphere.area_km2(remaining) >= TINY_KM2:
                    self._split_overlap(remaining, 'junction', None, p, q, stage, lost)
                for plate, start in zip((p, q), before):
                    given = [shape for shape, *_ in lost[plate][start:]]
                    if given:
                        holding[plate] = sphere.difference(holding[plate], sphere.union(*given))
        # Gaps: within a run's zone, split along that pair's half-stage line; the rest to the nearest plate.
        cover = sphere.union(*moved.values())
        remaining = sphere.difference(FULL, cover)
        for (p, q), items in sorted(runs.items(), key=lambda kv: -len(kv[1])):
            for zone, kind, _ in items:
                if sphere.area_km2(remaining) < TINY_KM2:
                    break
                local = sphere.intersection(remaining, zone)
                if sphere.area_km2(local) < TINY_KM2:
                    continue
                remaining = sphere.difference(remaining, local)
                category = {'ridge': 'ridge', 'trench': 'trench_opening'}.get(kind, 'sliver')
                mine = sphere.intersection(local, stage((p, q), p))
                gained[p].append((mine, category))
                gained[q].append((sphere.difference(local, mine), category))
        if sphere.area_km2(remaining) >= TINY_KM2:
            try:
                pieces, _ = sphere.parts(remaining)
            except (ValueError, RuntimeError):
                pieces = [remaining]
            for piece in pieces:
                nearest = min(plates, key=lambda p: spherely.distance(piece, moved[p]))
                gained[nearest].append((piece, 'junction'))
        for p in plates:
            self._update_crust(p, lost[p], gained[p], entry, dt)
        self.crust = _merge_bands(self.crust)
        self._spread(dt)  # thick crust flows outward under its own weight
        self._erode(dt, entry)  # and is worn down where it stands high
        # Pieces still below the level of detail after merging (a gain with no band of its own to
        # join, a sliver left on its own) are not tracked: booked as a sliver and let go. The ground
        # comes back to a neighbour as a junction gain when the regions are rebuilt.
        kept = []
        for piece in self.crust:
            area = piece.area_km2()
            if area < DETAIL_KM2 and piece.kind == 'ocean':
                entry['lost']['sliver'] += area
                continue
            kept.append(piece)
        self.crust = kept
        # Joining pieces whose edges coincide only to rounding leaves slit holes, and resolving
        # overlaps leaves crumbs below the level of detail; left in, both multiply with every later
        # step. Cleaned once here, after every change to the crust this step, and the area change
        # booked: debris let go is a sliver lost, a slit filled is ground gained at a junction.
        for piece in self.crust:
            area = piece.area_km2()
            piece.shape, _, _ = sphere.clean(piece.shape, minimum_km2=DETAIL_KM2)
            change = piece.area_km2() - area
            if change < 0.:
                entry['lost']['sliver'] -= change
                if piece.kind == 'continent':  # continent let go as debris still keeps its volume
                    entry['thickened_km3'] += self._thicken(piece, -change, [c for c in self.crust if c.plate == piece.plate])
            elif change > 0.:
                entry['gained']['junction'] += change
        self.crust = [c for c in self.crust if c.area_km2() >= TINY_KM2]
        # Simplifying costs about as much as the rest of a sub-step, so it runs once a Myr, or
        # sooner when the outlines have grown by SIMPLIFY_GROWTH since the last time.
        vertices = sum(len(r) for c in self.crust for poly in sphere.polygons(c.shape) for r in poly)
        due = (self.time_myr + dt - getattr(self, 'simplified_myr', -np.inf) >= SIMPLIFY_EVERY_MYR - 1e-9
               or vertices > SIMPLIFY_GROWTH * getattr(self, 'simplified_count', np.inf))
        entry['simplified_vertices'] = self._simplify() if due else 0
        if due:
            self.simplified_myr = self.time_myr + dt
            self.simplified_count = sum(len(r) for c in self.crust for poly in sphere.polygons(c.shape) for r in poly)
        self.regions = self._from_crust()
        entry['events'] += self._regroup(entry)  # keeps regions and crust together as it moves pieces
        rotations_per_myr = self._complete(rotations_per_myr)  # a plate that came apart moves with its parent
        self.time_myr += dt
        self.boundaries = self._derive(old_index, rotations_per_myr)
        entry['events'] += self._evolve(rotations_per_myr, dt, entry)
        entry['plates_km2'] = {p: sphere.area_km2(r) for p, r in self.regions.items()}
        crust_after = sum(c.area_km2() for c in self.crust)
        entry['rebuild_drift_km2'] = crust_after - crust_before - sum(entry['gained'].values()) + sum(entry['lost'].values())
        self._check(entry)
        self.ledger.append(entry)
        return entry

    # -------------------------------------------------------------- pieces
    def _from_crust(self) -> dict:
        """Every plate's region in the world frame: its crust carried by its frame."""
        regions = {}
        for plate in sorted({c.plate for c in self.crust}):
            shapes = [c.shape for c in self.crust if c.plate == plate]
            region = sphere.rotate_geography(sphere.union(*shapes), self.frame(plate))
            if sphere.area_km2(region) >= TINY_KM2:
                regions[plate] = region
        return regions

    def _simplify(self) -> int:
        """Take out the zig-zags and points on straight edges that splitting along stage lines leaves.

        Every crust ring is simplified together in the world frame, so a vertex leaves both plates'
        outlines at once and the plates still tile the sphere (`sphere.simplify_shared`).
        """
        rings, where = [], []
        shapes = [sphere.polygons(c.shape) for c in self.crust]
        for i, (c, polys) in enumerate(zip(self.crust, shapes)):
            frame = self.frame(c.plate)
            for j, poly in enumerate(polys):
                for k, ring in enumerate(poly):
                    rings.append(sphere.rotate(ring, frame))
                    where.append((i, j, k))
        groups = {}
        polygon_of = [groups.setdefault((i, j), len(groups)) for i, j, _ in where]
        new, removed, dropped = sphere.simplify_shared(rings, area_km2=SIMPLIFY_KM2, groups=polygon_of)
        changed = {}
        for (i, j, k), before, after, count in zip(where, rings, new, dropped):
            # A ring that only gained points shared with a neighbour (on its own edges) is the
            # same region; only rings that lost vertices need their piece rebuilt.
            if count:
                changed[i] = changed.get(i, 0) + int(count)
        for i, count in changed.items():
            c = self.crust[i]
            back = self.frame(c.plate).T
            polys = [list(poly) for poly in shapes[i]]
            for (ii, j, k), ring in zip(where, new):
                if ii == i:
                    polys[j][k] = sphere.rotate(ring, back)
            area = c.area_km2()
            try:
                pieces = [sphere.part(poly, whole_km2=area)[0] for poly in polys]
                shape = sphere.union(*pieces)
            except (ValueError, RuntimeError):
                continue
            if abs(sphere.area_km2(shape) - area) <= count * SIMPLIFY_KM2 + 1.:
                c.shape = shape
                if c.kind == 'continent':
                    # The outline moved a little; the crust keeps its volume on the new area.
                    grown = c.area_km2()
                    mesh = getattr(c, 'mesh', None)
                    if mesh is not None and grown > 0.:
                        mesh.thickness = mesh.thickness * (area / grown)
                    if grown > 0.:
                        c.thickness_km = float(c.thickness_km) * area / grown
        return removed

    def _complete(self, rotations_per_myr):
        rotations_per_myr = dict(rotations_per_myr)
        for plate in self.plates():
            ancestor = plate
            while ancestor not in rotations_per_myr and ancestor in self.parents:
                ancestor = self.parents[ancestor]
            rotations_per_myr.setdefault(plate, rotations_per_myr.get(ancestor, np.zeros(3)))
        return rotations_per_myr

    def _update_crust(self, plate, losses, gains, entry, dt):
        back = self.frame(plate).T
        mine = [c for c in self.crust if c.plate == plate]
        others = [c for c in self.crust if c.plate != plate]
        # The losses are disjoint pieces of the plate's overlaps; joined by boundary kind, each crust
        # piece meets at most one geography per kind and is cut once, instead of once per loss.
        by_kind = {}
        for shape, kind, over in losses:
            if sphere.area_km2(shape) >= TINY_KM2:
                by_kind.setdefault((kind, over), []).append(shape)
        local_by_kind = {key: sphere.rotate_geography(sphere.union(*shapes), back) for key, shapes in by_kind.items()}
        if local_by_kind:
            all_local = sphere.union(*local_by_kind.values())
            for piece in mine:
                whole = sphere.intersection(piece.shape, all_local)
                if sphere.area_km2(whole) < TINY_KM2:
                    continue
                if piece.kind == 'continent' and piece.area_km2() < TERRANE_KM2:
                    # A terrane reaching a trench docks whole onto the overriding plate.
                    to = next((over for (kind, over), local in local_by_kind.items()
                               if kind == 'trench' and over in self.regions
                               and sphere.area_km2(sphere.intersection(whole, local)) >= DETAIL_KM2), None)
                    if to is not None and self._dock(piece, plate, to, others, entry):
                        continue
                continental = 0.
                before_km2 = piece.area_km2()
                before_km3 = piece.volume_km3() if piece.kind == 'continent' else 0.
                for (kind, over), local in local_by_kind.items():
                    taken = sphere.intersection(whole, local)
                    area = sphere.area_km2(taken)
                    if area < TINY_KM2:
                        continue
                    category = _loss_category(kind, piece.kind)
                    entry['lost'][category] += area
                    if piece.kind == 'continent':
                        continental += area
                    if category == 'continent_at_trench':
                        # Where the continent went down, arrival by arrival, not the middle of them all.
                        for shape in by_kind[(kind, over)]:
                            part = sphere.intersection(whole, sphere.rotate_geography(shape, back))
                            if sphere.area_km2(part) >= TINY_KM2:
                                entry['continent_arrived'].append(
                                    (sphere.rotate(_point_in(part), self.frame(plate)), sphere.area_km2(part), plate))
                piece.shape = sphere.difference(piece.shape, whole)
                if piece.kind == 'continent' and continental >= TINY_KM2:
                    # Continent is too light to go away: it shortens and thickens, keeping its volume,
                    # and some of it escapes sideways over the plate's own ocean.
                    left = piece.area_km2()
                    if left >= TINY_KM2:
                        if piece.mesh is not None:
                            recut = piece.mesh.recut(piece.shape, left, before_km3)
                            if recut is not None:
                                piece.mesh = recut
                        piece.thickness_km = before_km3 / left
                    else:
                        self._bequeath(piece, before_km3, mine)
                    entry['thickened_km3'] += continental * before_km3 / max(before_km2, TINY_KM2)
                    self._escape(piece, continental, mine, entry)
        for shape, category in gains:
            area = sphere.area_km2(shape)
            if area < TINY_KM2:
                continue
            entry['gained'][category] += area
            mine.append(Crust(plate, 'ocean', sphere.rotate_geography(shape, back), self.time_myr + .5 * dt))
        self.crust = others + [c for c in mine if sphere.area_km2(c.shape) >= TINY_KM2]

    def _bequeath(self, piece, volume, mine):
        """A continent piece taken away whole hands its volume to its plate's largest other continent."""
        others = [c for c in mine if c is not piece and c.kind == 'continent' and c.area_km2() >= TINY_KM2]
        if not others:
            return
        heir = max(others, key=lambda c: c.area_km2())
        if heir.mesh is not None:
            heir.mesh.add_volume(volume, heir.area_km2(), near=None if piece.mesh is None else piece.mesh.nodes)
        heir.thickness_km += volume / heir.area_km2()

    def _dock(self, piece, plate, to, others, entry) -> bool:
        """Dock the continent `piece` of `plate`, a terrane that has reached a trench, onto plate `to`.

        The whole piece crosses as a body, with its volume and its thickness field, and the trench
        then runs round it (the regions are rebuilt from the crust). It replaces the ocean under it
        (an `accreted` gain and loss, and the ocean `overridden`); where it lands on continent, a
        terrane docked a step earlier included, that much stacks onto it as thickness where it
        lands. `others` is the crust of the other plates, which the caller writes back. Returns
        whether the piece was dealt with here.
        """
        area = piece.area_km2()
        volume = piece.volume_km3()
        if area < DETAIL_KM2:
            return False
        carry = self.frame(to).T @ self.frame(plate)
        shape = sphere.rotate_geography(piece.shape, carry)
        theirs = [c for c in others if c.plate == to]
        for c in [c for c in theirs if c.kind == 'continent']:
            part = sphere.intersection(shape, c.shape)
            a = sphere.area_km2(part)
            if a < TINY_KM2:
                continue
            share = volume * a / area
            if getattr(c, 'mesh', None) is not None:
                near = np.vstack([r for poly in sphere.polygons(part) for r in poly])
                c.mesh.add_volume(share, c.area_km2(), near=near)
            c.thickness_km += share / c.area_km2()
            entry['thickened_km3'] += share
            entry['lost']['continent_at_trench'] += a
            shape = sphere.difference(shape, part)
        left = sphere.area_km2(shape)
        if left < DETAIL_KM2:
            entry['lost']['continent_at_trench'] += left  # a crumb: gone
            piece.shape = sphere.EMPTY  # all of it stacked: the caller drops the empty piece
            return True
        docked_km3 = volume * left / area
        overridden = 0.
        for c in [c for c in theirs if c.kind == 'ocean']:
            part = sphere.intersection(c.shape, shape)
            a = sphere.area_km2(part)
            if a >= TINY_KM2:
                c.shape = sphere.difference(c.shape, part)
                overridden += a
        piece.plate, piece.shape = to, shape
        piece.thickness_km = docked_km3 / left
        if getattr(piece, 'mesh', None) is not None:
            rotated = piece.mesh.rotated(carry)
            recut = rotated.recut(shape, left, docked_km3) if left < .999 * area else None
            piece.mesh = recut if recut is not None else rotated
        # Books: the terrane leaves one plate and joins the other (a loss and a gain of the same
        # continent), and the ocean it covers on the plate it joins is gone.
        entry['gained']['accreted'] += left
        entry['lost']['accreted'] += left
        entry['lost']['overridden'] += overridden
        entry['events'].append(dict(kind='accreted', plates=(plate, to), area_km2=left, volume_km3=docked_km3))
        return True

    def _thicken(self, piece, area, mine):
        """Keep the volume of `area` km2 of continent just taken from `piece`; returns that volume.

        The piece keeps it, thicker: on its mesh, near the ground that was lost; with one
        thickness, everywhere. One taken away whole hands it to its plate's largest other
        continental piece.
        """
        volume = area * piece.thickness_km
        left = piece.area_km2()
        if left >= TINY_KM2:
            total = volume + left * piece.thickness_km
            if piece.mesh is not None:
                piece.mesh = piece.mesh.recut(piece.shape, left, total)
            piece.thickness_km = total / left
            return volume
        others = [c for c in mine if c is not piece and c.kind == 'continent' and c.area_km2() >= TINY_KM2]
        if others:
            heir = max(others, key=lambda c: c.area_km2())
            if heir.mesh is not None:
                heir.mesh.add_volume(volume, heir.area_km2(), near=None if piece.mesh is None else piece.mesh.nodes)
            heir.thickness_km += volume / heir.area_km2()
        return volume

    def _escape(self, piece, shortened_km2, mine, entry):
        """Push the piece's coast out over its plate's ocean within reach of the collision, by
        ESCAPE_FRACTION of the area shortened; the crust there spreads out (same volume).

        A sub-step's share would be a ribbon metres wide, below the level of detail, so it is owed
        and delivered as a band ESCAPE_STEP_KM wide along the coast within reach when enough is owed.
        """
        if ESCAPE_FRACTION <= 0.:
            return
        piece.escape_km2 = getattr(piece, 'escape_km2', 0.) + ESCAPE_FRACTION * shortened_km2
        mesh = getattr(piece, 'mesh', None)
        lost = getattr(mesh, 'lost_nodes', None)
        if mesh is None or lost is None or not len(lost):
            return
        ocean = [c for c in mine if c.kind == 'ocean' and c is not piece]
        if not ocean:
            return
        rings = [r for poly in sphere.polygons(piece.shape) for r in poly]
        near = sphere.strip(lost, ESCAPE_REACH_KM) if len(lost) > 1 else sphere.EMPTY
        if sphere.is_empty(near):
            return
        # The coast within reach of the collision that fronts the plate's own ocean is what moves
        # (the rest fronts the colliding plate); the width is set from its length.
        from scipy.spatial import cKDTree
        tree = cKDTree(np.asarray(lost, float))
        sea = sphere.union(*[c.shape for c in ocean])
        spherely.prepare(sea)
        probe_angle = 1. / sphere.RADIUS_KM
        coast = 0.
        for r in rings:
            nxt = np.roll(r, -1, axis=0)
            lengths = sphere.arc_km(r, nxt)
            mids = sphere.unit(r + nxt)
            within = (tree.query(mids, k=1)[0] * sphere.RADIUS_KM <= ESCAPE_REACH_KM) & (lengths > 0.)
            if not within.any():
                continue
            pole = sphere.unit(np.cross(r[within], nxt[within]))  # a zero-length edge has no pole
            outward = sphere.unit(mids[within] * np.cos(probe_angle) - pole * np.sin(probe_angle))  # right of the ring
            lon, lat = sphere.to_lonlat(sphere.rotate(outward, self.frame(piece.plate).T))
            fronts = np.asarray(spherely.contains(sea, spherely.points(lon, lat)))
            coast += float(lengths[within][fronts].sum())
        if coast <= 0.:
            return
        width = piece.escape_km2 / coast  # km outward along the coast within reach
        if width < ESCAPE_STEP_KM:
            return

        def banded(w):
            b = sphere.union(*[sphere.strip(np.vstack((r, r[:1])), w) for r in rings])
            b = sphere.intersection(sphere.difference(b, piece.shape), near)
            return sphere.intersection(b, sea)
        band = banded(width)
        area = sphere.area_km2(band)
        if area > 1.2 * piece.escape_km2:  # the band is not exactly width times coast: narrow it
            band = banded(width * piece.escape_km2 / area)
            area = sphere.area_km2(band)
        if area < TINY_KM2:
            return
        volume = piece.volume_km3()
        for c in ocean:
            if sphere.area_km2(sphere.intersection(c.shape, band)) >= TINY_KM2:
                c.shape = sphere.difference(c.shape, band)
        piece.shape = sphere.union(piece.shape, band)
        grown = piece.area_km2()
        recut = mesh.recut(piece.shape, grown, volume)
        if recut is not None:
            piece.mesh = recut
        piece.thickness_km = volume / grown
        piece.escape_km2 = max(piece.escape_km2 - area, 0.)
        entry['gained']['escape'] += area
        entry['lost']['escape'] += area

    def _spread(self, dt):
        for c in self.crust:
            if c.kind == 'continent' and getattr(c, 'mesh', None) is not None:
                c.mesh.spread(dt)

    def _erode(self, dt, entry):
        """Wear the highlands down; sediment settles on the piece's lowlands or goes to the sea."""
        for c in self.crust:
            if c.kind != 'continent' or getattr(c, 'mesh', None) is None:
                continue
            area = c.area_km2()
            removed, settled, to_sea = c.mesh.erode(dt, area)
            if removed > 0.:
                entry['eroded_km3'] += removed
                entry['sediment_to_sea_km3'] += to_sea
                c.thickness_km = c.mesh.volume_km3(area) / area

    def _mesh_continents(self):
        """Give every continental piece without a mesh one (a world built or loaded without them)."""
        for c in self.crust:
            if c.kind == 'continent' and getattr(c, 'mesh', None) is None:
                c.mesh = ContinentMesh.build(c.shape, c.thickness_km)

    def _continent_km3(self) -> float:
        return sum(c.volume_km3() for c in self.crust if c.kind == 'continent')

    def _regroup(self, entry=None) -> list:
        """Plates that came apart become separate plates; slivers and vanished plates go to neighbours."""
        events = []
        for plate in list(self.plates()):
            region = self.regions[plate]
            area = sphere.area_km2(region)
            pieces, _ = sphere.parts(region)
            pieces.sort(key=lambda g: -sphere.area_km2(g))
            if area < MINIMUM_PLATE_KM2:
                pieces = [region] if area >= TINY_KM2 else []
                keep = []
            else:
                keep, pieces = pieces[:1], pieces[1:]
            # Crumbs left where overlaps and gaps were split, often enclosed by another plate, go to
            # the plate beside them in one hand-over per neighbour; left with their own plate they
            # would sit inside the other plate's crust as holes and be split again every step.
            crumbs = [g for g in pieces if sphere.area_km2(g) < CRUMB_KM2] if keep else []
            pieces = [g for g in pieces if sphere.area_km2(g) >= CRUMB_KM2 or not keep]
            if crumbs:
                into = self._main_neighbours(crumbs, plate)
                for other in sorted(set(i for i in into if i is not None)):
                    batch = [g for g, i in zip(crumbs, into) if i == other]
                    self._give(plate, sphere.union(*batch), other, entry)
                    events.append(dict(kind='crumbs_given', plate=plate, into=other, count=len(batch),
                                       area_km2=sum(sphere.area_km2(g) for g in batch)))
            for piece in pieces:
                if sphere.area_km2(piece) >= MINIMUM_PLATE_KM2:
                    new = max(list(self.regions) + list(self.frames) + list(self.parents)) + 1
                    self.frames[new] = self.frame(plate).copy()
                    self.parents[new] = plate
                    self.names[new] = '%s (part)' % self.names.get(plate, 'Plate %d' % plate)
                    self._give(plate, piece, new)
                    events.append(dict(kind='plate_split', plate=plate, new=new, area_km2=sphere.area_km2(piece)))
                else:
                    into = self._main_neighbour(piece, plate)
                    if into is None:
                        continue
                    self._give(plate, piece, into, entry)
                    events.append(dict(kind='absorbed', plate=plate, into=into, area_km2=sphere.area_km2(piece)))
            if not keep and plate in self.regions and sphere.area_km2(self.regions[plate]) < MINIMUM_PLATE_KM2:
                rest = self.regions[plate]
                if sphere.area_km2(rest) >= TINY_KM2:
                    into = self._main_neighbour(rest, plate)
                    if into is not None:
                        self._give(plate, rest, into, entry)
                if sphere.area_km2(self.regions.get(plate, sphere.EMPTY)) < TINY_KM2:
                    self.regions.pop(plate, None)
                    self.crust = [c for c in self.crust if c.plate != plate]
                    events.append(dict(kind='plate_gone', plate=plate))
        return events

    def _give(self, plate, piece, to, entry=None):
        """Move the world-frame region `piece` (and the crust in it) from `plate` to `to`.

        Where `to` already has crust (the plates overlapped there by rounding), `plate`'s copy is
        destroyed and booked as a sliver, so no plate holds the same ground twice.
        """
        doubled = sphere.intersection(piece, self.regions.get(to, sphere.EMPTY))
        if sphere.area_km2(doubled) >= TINY_KM2:
            local = sphere.rotate_geography(doubled, self.frame(plate).T)
            for c in [c for c in self.crust if c.plate == plate]:
                area = sphere.area_km2(sphere.intersection(c.shape, local))
                if area < TINY_KM2:
                    continue
                c.shape = sphere.difference(c.shape, local)
                if entry is not None:
                    entry['lost']['sliver'] += area
            self.regions[plate] = sphere.difference(self.regions[plate], doubled)
            piece = sphere.difference(piece, doubled)
        local = sphere.rotate_geography(piece, self.frame(plate).T)
        carry = self.frame(to).T @ self.frame(plate)
        moved = []
        for c in [c for c in self.crust if c.plate == plate]:
            part = sphere.intersection(c.shape, local)
            part_km2 = sphere.area_km2(part)
            if part_km2 < TINY_KM2:
                continue
            whole_km3 = c.volume_km3()
            c.shape = sphere.difference(c.shape, local)
            shape = part if np.allclose(carry, np.eye(3)) else sphere.rotate_geography(part, carry)
            given = Crust(to, c.kind, shape, c.born_myr, c.thickness_km)
            owed = getattr(c, 'escape_km2', 0.)
            if owed and whole_km3 > 0.:
                share = owed * part_km2 / (part_km2 + max(c.area_km2(), 0.))
                given.escape_km2, c.escape_km2 = share, owed - share
            if c.kind == 'continent' and getattr(c, 'mesh', None) is not None:
                # The part takes the field as it is there; the rest keeps the remainder exactly.
                taken = ContinentMesh.build(part, c.thickness_km, field=c.mesh.sample)
                part_km3 = taken.volume_km3(part_km2) if taken is not None else part_km2 * c.thickness_km
                given.thickness_km = part_km3 / part_km2
                given.mesh = None if taken is None else taken.rotated(carry)
                left_km2 = c.area_km2()
                if left_km2 >= TINY_KM2:
                    c.mesh = c.mesh.recut(c.shape, left_km2, whole_km3 - part_km3)
                    c.thickness_km = (whole_km3 - part_km3) / left_km2
                elif part_km3 > 0.:
                    # Nothing stays behind: the part takes the whole volume exactly (sampling the
                    # field onto new nodes changes its integral a little, and the rest is dropped).
                    if given.mesh is not None:
                        given.mesh.thickness = given.mesh.thickness * (whole_km3 / part_km3)
                    given.thickness_km = whole_km3 / part_km2
            moved.append(given)
        self.crust = [c for c in self.crust if sphere.area_km2(c.shape) >= TINY_KM2] + moved
        self.regions[plate] = sphere.difference(self.regions[plate], piece)
        self.regions[to] = sphere.union(self.regions.get(to, sphere.EMPTY), piece)

    def _main_neighbours(self, pieces, plate):
        """The nearest other plate to each of `pieces` (None if there is no other plate)."""
        array = np.array(pieces, dtype=object)
        best, best_km = [None] * len(pieces), np.full(len(pieces), np.inf)
        for other, region in self.regions.items():
            if other == plate:
                continue
            km = np.asarray(spherely.distance(array, region), float)
            closer = km < best_km
            best_km[closer] = km[closer]
            for k in np.flatnonzero(closer):
                best[k] = other
        return best

    def _main_neighbour(self, piece, plate):
        best, best_km = None, np.inf
        for other, region in self.regions.items():
            if other == plate:
                continue
            km = spherely.distance(piece, region)
            if km < best_km:
                best, best_km = other, km
        return best

    # ---------------------------------------------------------- boundaries
    def _fresh(self, count) -> np.ndarray:
        """`count` identities never used before."""
        first = getattr(self, 'next_ident', 0)
        self.next_ident = first + count
        return np.arange(first, first + count)

    def _index(self, boundaries=None, rotations_per_myr=None, dt=0.) -> dict:
        """Per plate pair, each old boundary as (midpoints, kinds, overriding, closed_km, idents).

        With `rotations_per_myr` and `dt`, the midpoints are carried by the pair's mean rotation, which
        is where a boundary between two plates goes, so the new boundary is matched where it is now.
        """
        index = {}
        for b in (self.boundaries if boundaries is None else boundaries):
            if len(b.points) < 2:
                continue
            mids = b.mids()
            if rotations_per_myr is not None and dt:
                mean = .5 * (np.asarray(rotations_per_myr[b.left], float) + np.asarray(rotations_per_myr[b.right], float)) * dt
                mids = sphere.rotate(mids, sphere.rotation(mean))
            idents = b.idents if getattr(b, 'idents', None) is not None else self._fresh(len(b.kinds))
            arcs = b.arcs()
            index.setdefault((b.left, b.right), []).append(
                (mids, list(b.kinds), list(b.overriding), np.asarray(b.closed_km, float), np.asarray(idents), arcs))
        return index

    def _split_overlap(self, local, kind, over, p, q, stage, lost):
        if kind == 'trench' and over in (p, q):
            lost[q if over == p else p].append((local, 'trench', over))
            return
        mine = sphere.intersection(local, stage((p, q), p))  # what q loses
        theirs = sphere.difference(local, mine)              # what p loses
        # In the leftovers no zone covers, continent yields last: ocean goes before continent on
        # either side (a terrane just docked sticks out past the coarse zone of its trench). Along a
        # collision or a converging boundary the continent shortens as it should.
        for a, b, their_continent in ((p, q, self._continent_world(p)), (q, p, self._continent_world(q))) if kind == 'junction' else ():
            loses = theirs if a == p else mine
            keep = sphere.intersection(loses, their_continent)
            if sphere.area_km2(keep) >= TINY_KM2:
                if a == p:
                    theirs, mine = sphere.difference(theirs, keep), sphere.union(mine, keep)
                else:
                    mine, theirs = sphere.difference(mine, keep), sphere.union(theirs, keep)
        lost[p].append((theirs, kind, None))
        lost[q].append((mine, kind, None))

    def _continent_world(self, plate):
        """The plate's continent in the world frame, as it stood at the start of this step (cached)."""
        cache = getattr(self, '_continent_cache', None)
        if cache is None or cache.get('time') != self.time_myr:
            cache = self._continent_cache = {'time': self.time_myr}
        if plate not in cache:
            shapes = [c.shape for c in self.crust if c.plate == plate and c.kind == 'continent']
            frame = self.frame(plate)
            cache[plate] = sphere.rotate_geography(sphere.union(*shapes), frame) if shapes else sphere.EMPTY
        return cache[plate]

    def _runs(self, rotations_per_myr, dt, reach) -> dict:
        """Per plate pair, the old boundary's runs of one type as zones carried by the pair's mean motion."""
        result = {}
        for b in self.boundaries:
            if len(b.points) < 2:
                continue
            mean = .5 * (np.asarray(rotations_per_myr[b.left], float) + np.asarray(rotations_per_myr[b.right], float)) * dt
            points = sphere.rotate(b.points, sphere.rotation(mean))
            width = reach[b.left] + reach[b.right] + 25.
            start = 0
            for k in range(1, len(b.kinds) + 1):
                if k == len(b.kinds) or (b.kinds[k], b.overriding[k]) != (b.kinds[start], b.overriding[start]):
                    zone = sphere.strip(points[start:k + 1], width)
                    result.setdefault((b.left, b.right), []).append((zone, b.kinds[start], b.overriding[start]))
                    start = k
        return result

    def _inherit(self, index, b, rotations_per_myr):
        """Fill `b`'s segment state from the old boundaries of its plates.

        Each segment takes the state and identity of the nearest old segment of the same plates
        within INHERIT_KM, with one rule for continuity: a segment whose nearest old boundary differs
        from its neighbours' is next to another contact of the same plates, not part of it, and takes
        its neighbours' boundary when that is in reach. A segment with no old boundary in reach is a
        new contact: typed from the motion, with a new identity.
        """
        pair = (b.left, b.right)
        mids = b.mids()
        n = len(mids)
        candidates = index.get(pair, [])
        choice, where, best = np.full(n, -1), np.zeros(n, int), np.full(n, np.inf)
        distances = []
        for c, (old_mids, *_) in enumerate(candidates):
            km = sphere.RADIUS_KM * np.arccos(np.clip(mids @ old_mids.T, -1., 1.))
            distances.append(km)
            k = np.argmin(km, axis=1)
            d = km[np.arange(n), k]
            better = d < best
            best[better], choice[better], where[better] = d[better], c, k[better]
        choice[best > INHERIT_KM] = -1
        for k in range(n):
            neighbours = [choice[j] for j in (k - 1, k + 1) if 0 <= j < n]
            if not neighbours or choice[k] in neighbours or len(set(neighbours)) != 1 or neighbours[0] < 0:
                continue
            c = neighbours[0]
            j = int(np.argmin(distances[c][k]))
            if distances[c][k, j] <= INHERIT_KM:
                choice[k], where[k] = c, j
        idents = np.zeros(n, int)
        b.arc_km3 = np.zeros((n, 2))
        fresh = iter(self._fresh(int((choice < 0).sum())))
        for k in range(n):
            if choice[k] >= 0:
                _, kinds, over, closed, old_idents, arcs = candidates[choice[k]]
                j = where[k]
                b.kinds.append(kinds[j])
                b.overriding.append(over[j])
                b.closed_km[k] = closed[j]
                b.arc_km3[k] = arcs[j]
                idents[k] = old_idents[j]
            else:
                b.kinds.append(_kind_from_motion(mids[k], pair, rotations_per_myr))
                b.overriding.append(None)
                b.closed_km[k] = 0.
                idents[k] = next(fresh)
        b.idents = idents

    def _derive(self, old_index, rotations_per_myr) -> list:
        """Boundaries where the new regions touch, carrying the old state by position."""
        plates = self.plates()
        for region in self.regions.values():
            spherely.prepare(region)
        boundaries = []
        for p in plates:
            for rings in sphere.polygons(self.regions[p]):
                for ring in rings:
                    boundaries += self._trace(p, ring, plates, old_index, rotations_per_myr)
        return boundaries

    def _trace(self, plate, ring, plates, old_index, rotations_per_myr) -> list:
        nxt = np.roll(ring, -1, axis=0)
        keep = sphere.arc_km(ring, nxt) > 1e-6
        if keep.sum() < 3:
            return []
        a, b = ring[keep], np.roll(ring, -1, axis=0)[keep]
        mid = sphere.unit(a + b)
        pole = sphere.unit(np.cross(a, b))
        angle = PROBE_KM / sphere.RADIUS_KM
        probe = sphere.unit(mid * np.cos(angle) - pole * np.sin(angle))  # just to the right: outside
        lon, lat = sphere.to_lonlat(probe)
        points = spherely.points(lon, lat)
        neighbour = np.full(len(probe), -1)
        for other in plates:
            if other == plate:
                continue
            inside = np.asarray(spherely.contains(self.regions[other], points))
            neighbour[inside & (neighbour < 0)] = other
        # A probe that lands in a gap (debris below the level of detail let go, filled again next
        # step) finds no neighbour; it does not end the run. Each such stretch takes the neighbour
        # of the segment before it, round the ring.
        missing = neighbour < 0
        if missing.any() and not missing.all():
            order = np.roll(np.arange(len(neighbour)), -int(np.argmin(missing)))  # start on a known one
            last = neighbour[order[0]]
            for k in order:
                if neighbour[k] < 0:
                    neighbour[k] = last
                last = neighbour[k]
        # Runs of one neighbour, joined round the ring's start.
        runs, start = [], 0
        for k in range(1, len(neighbour) + 1):
            if k == len(neighbour) or neighbour[k] != neighbour[start]:
                runs.append((start, k))
                start = k
        if len(runs) > 1 and neighbour[0] == neighbour[-1]:
            first, last = runs[0], runs[-1]
            runs = [(last[0], first[1] + len(neighbour))] + runs[1:-1]
        result = []
        for lo, hi in runs:
            other = int(neighbour[lo % len(neighbour)])
            if other < 0 or other < plate:
                continue  # no neighbour found (numerical) or traced from the other side
            index = [k % len(neighbour) for k in range(lo, hi)]
            line = np.vstack((a[index], b[index[-1]][None]))
            if len(line) < 2:
                continue
            boundary = Boundary(plate, other, line, [], [], np.zeros(len(line) - 1))
            self._inherit(old_index, boundary, rotations_per_myr)
            result.append(boundary)
        return result

    def _evolve(self, rotations_per_myr, dt, entry) -> list:
        """Types follow the motion; converging boundaries initiate; trenches continents reach collide."""
        events = []
        for b in self.boundaries:
            if len(b.points) < 2:
                continue
            opening, sliding = segment_motion(b.points, rotations_per_myr[b.left], rotations_per_myr[b.right])
            ratio = np.abs(opening) / np.maximum(sliding, 1e-12)
            for k, kind in enumerate(b.kinds):
                if kind in ('trench', 'collision'):
                    continue
                threshold = TRANSFORM_RATIO * (1. + MARGIN if kind == 'transform' else 1. - MARGIN)
                b.kinds[k] = ('ridge' if opening[k] > 0. else 'converging') if ratio[k] > threshold else 'transform'
            b.kinds = [str(k) for k in _smooth(b.kinds)]
            _fill_overriding(b)
            for k in range(len(b.kinds)):
                if b.kinds[k] in ('converging', 'collision'):
                    b.closed_km[k] += max(-opening[k], 0.) * dt
                elif b.kinds[k] != 'trench':
                    b.closed_km[k] = 0.
                    b.overriding[k] = None
            events += self._initiate(b)
            events += self._arcs(b, opening, dt, entry)
        for point, area, plate in entry['continent_arrived']:
            events += self._collide(point, area, plate)
        events += self._sutures(rotations_per_myr)
        return events

    def _arcs(self, b, opening, dt, entry) -> list:
        """Magma behind each trench segment of `b`: into the overriding continent, or owed over its
        ocean, where it shows as islands first and as a ribbon once the arc is mature."""
        events = []
        if len(b.points) < 2:
            return events
        owed = b.arcs()
        root, bank = owed[:, 0], owed[:, 1]
        mids = b.mids()
        poles = sphere.unit(np.cross(b.points[:-1], b.points[1:]))
        lengths = sphere.arc_km(b.points[:-1], b.points[1:])
        angle = ARC_KM / sphere.RADIUS_KM
        idents = np.asarray(b.idents) if getattr(b, 'idents', None) is not None else np.arange(len(b.kinds))
        # A run of trench segments under one overriding plate matures together (the ribbon is
        # judged on the run's root too): its segments bank for islands once the run holds a
        # ridge's worth per km, however the root is spread among them.
        emerged = np.zeros(len(b.kinds), bool)
        for lo, hi in _runs_of(b.kinds, 'trench'):
            for over in set(b.overriding[lo:hi]):
                ks = [k for k in range(lo, hi) if b.overriding[k] == over]
                if float(root[ks].sum()) >= ARC_EMERGE_KM3_PER_KM * float(lengths[ks].sum()):
                    emerged[ks] = True
        arc_points, sites = {}, {}
        for k, kind in enumerate(b.kinds):
            over = b.overriding[k]
            if kind != 'trench' or over not in (b.left, b.right) or over not in self.regions:
                continue
            rate = max(-opening[k], 0.) / ARC_REFERENCE_KM_MYR
            magma = ARC_KM3_PER_KM_MYR * float(lengths[k]) * dt * rate
            if magma <= 0.:
                continue
            sign = 1. if over == b.left else -1.
            point = sphere.unit(mids[k] * np.cos(angle) + sign * poles[k] * np.sin(angle))
            arc_points[k] = point
            # The segment's island site: along the segment by a fraction fixed by its identity.
            f = (int(idents[k]) * 0.6180339887498949) % 1.
            along = sphere.unit(b.points[k] * (1. - f) + b.points[k + 1] * f)
            sites[k] = sphere.unit(along * np.cos(angle) + sign * poles[k] * np.sin(angle))
            piece = self._meshed_at(over, point)
            if piece is not None:
                # Into a continent: the mesh thickens near the arc.
                area = piece.area_km2()
                local = sphere.rotate(point, self.frame(over).T)
                piece.mesh.add_volume(magma, area, near=local[None])
                piece.thickness_km += magma / area
                entry['arc_km3'] += magma
            elif emerged[k]:
                # Over ocean, or an island already built (islands grow by chunks, not by thickening).
                root[k] += (1. - ISLAND_SHARE) * magma
                bank[k] += ISLAND_SHARE * magma
            else:
                root[k] += magma  # the ridge first
        # A run of trench segments under one overriding plate builds islands as their bank allows,
        # and a ribbon along the run once the root owed is enough for a strip along it.
        for lo, hi in _runs_of(b.kinds, 'trench'):
            for over in sorted({o for o in b.overriding[lo:hi] if o in (b.left, b.right)}):
                ks = [k for k in range(lo, hi) if b.overriding[k] == over and k in arc_points]
                if not ks:
                    continue
                events += self._islands(b, ks, over, sites, bank, entry)
                if len(ks) < 2:
                    continue
                total = float(root[ks].sum())
                length = float(lengths[ks].sum())
                needed = length * ARC_WIDTH_KM * ARC_THICK_KM
                if total < needed:
                    continue
                strip = sphere.strip(np.array([arc_points[k] for k in ks]), ARC_WIDTH_KM / 2.)
                built, used, _ = self._island_arc(over, strip, entry, total)
                if built > 0.:
                    root[ks] = root[ks] * (total - used) / total  # what the strip did not take stays owed
                    events.append(dict(kind='island_arc', plate=over, plates=(b.left, b.right), area_km2=built,
                                       magma_km3=used))
                elif total > 2. * needed:
                    root[ks] = 0.  # nowhere to put it: lost to the mantle
        return events

    def _islands(self, b, ks, plate, sites, bank, entry) -> list:
        """Chunks of ISLAND_KM2 paid from the island bank of the run `ks` (see ISLAND_SHARE)."""
        events = []
        chunk = ISLAND_KM2 * ISLAND_THICK_KM
        if float(bank[ks].sum()) < chunk:
            return events
        back, forth = self.frame(plate).T, self.frame(plate)
        radius = np.sqrt(ISLAND_KM2 / np.pi)
        step = .8 * radius / sphere.RADIUS_KM  # a growth chunk overlaps its island by a fifth of its width
        islands = {id(c): c for c in self.crust if c.plate == plate and c.kind == 'continent'
                   and getattr(c, 'mesh', None) is None}
        about = {i: _about(c) for i, c in islands.items()}  # centre and reach of each island
        local = {k: sphere.rotate(sites[k], back) for k in ks}
        # Only sites over the plate's own ocean can take a chunk; the rest are not tried at all.
        ocean = [c.shape for c in self.crust if c.plate == plate and c.kind == 'ocean']
        for shape in ocean:
            spherely.prepare(shape)
        wet = [k for k in ks if any(spherely.contains(shape, _point(local[k])) for shape in ocean)]
        order = sorted(wet, key=lambda k: (-float(bank[k]), k))

        def lower(point):
            """A lower bound on the distance from `point` (plate frame) to each island, km, and their ids."""
            ids = list(about)
            if not ids:
                return np.zeros(0), ids
            centres = np.array([about[i][0] for i in ids])
            reach = np.array([about[i][1] for i in ids])
            return sphere.arc_km(centres, point) - reach, ids

        for _ in range(50):
            total = float(bank[ks].sum())
            if total < chunk:
                break
            placed, last = None, False
            for k in order:
                bound, ids = lower(local[k])
                here, free = None, True
                for i in np.flatnonzero(bound < ISLAND_SPACING_KM):
                    here = _point(local[k]) if here is None else here
                    if spherely.distance(islands[ids[i]].shape, here) < ISLAND_SPACING_KM * 1e3:
                        free = False
                        break
                if free:
                    placed = self._chunk(plate, sites[k], entry)
                    if placed[0] > 0.:
                        break
                    placed = None  # no ocean under this site
            if placed is None and about and ks:
                # No site free: the chunk extends the island nearest the most banked site, at the
                # tip of it along the arc, on the side clearer of the other islands.
                k = order[0] if order else max(ks, key=lambda k: (float(bank[k]), -k))
                bound, ids = lower(local[k])
                target = ids[int(np.argmin(bound))]
                centre = about[target][0]
                along = sphere.rotate(b.points[k + 1] - b.points[k], back)
                along = along - centre * float(along @ centre)
                if np.linalg.norm(along) < 1e-12:
                    along = np.cross(centre, [0., 0., 1.])
                along = sphere.unit(along)
                verts = np.vstack([r for poly in sphere.polygons(islands[target].shape) for r in poly])
                proj = (verts - centre) @ along
                others = [i for i in ids if i != target]
                choices = []
                for side in (1., -1.):
                    tip = verts[int(np.argmax(side * proj))]
                    at = sphere.unit(tip * np.cos(step) + side * along * np.sin(step))
                    clear = min((float(sphere.arc_km(about[i][0], at)) - about[i][1] for i in others), default=np.inf)
                    choices.append((clear, side, at))
                for _, _, at in sorted(choices, key=lambda s: -s[0]):
                    placed = self._chunk(plate, sphere.rotate(at, forth), entry)
                    if placed[0] > 0.:
                        last = placed[0] < .5 * ISLAND_KM2  # hemmed in: take what there was and stop
                        break
                    placed = None
            if placed is None:
                break  # nowhere this step: the bank waits
            built, used, piece = placed
            bank[ks] = bank[ks] * (total - used) / total
            events.append(dict(kind='island', plate=plate, plates=(b.left, b.right), area_km2=built, magma_km3=used))
            alive = {id(c) for c in self.crust}
            islands = {i: c for i, c in islands.items() if i in alive}
            about = {i: a for i, a in about.items() if i in alive}
            islands[id(piece)] = piece
            about[id(piece)] = _about(piece)
            if last:
                break
        return events

    def _chunk(self, plate, centre, entry):
        """A chunk of ISLAND_KM2 of island at the world-frame `centre`, over `plate`'s ocean."""
        disc = sphere.polygon(sphere.disc(centre, np.sqrt(ISLAND_KM2 / np.pi)))
        return self._island_arc(plate, disc, entry, ISLAND_KM2 * ISLAND_THICK_KM, ISLAND_THICK_KM)

    def _island_arc(self, plate, strip, entry, owed_km3, thick_km=ARC_THICK_KM):
        """Make the part of `strip` (world frame) over `plate`'s ocean continental crust, as thick as
        the magma owed allows up to `thick_km`; returns (km2 built, km3 of magma used, the piece)."""
        back = self.frame(plate).T
        local = sphere.rotate_geography(strip, back)
        ocean = [c for c in self.crust if c.plate == plate and c.kind == 'ocean' and spherely.intersects(c.shape, local)]
        pieces, built = [], 0.
        for c in ocean:
            part = sphere.intersection(c.shape, local)
            area = sphere.area_km2(part)
            if area < TINY_KM2:
                continue
            c.shape = sphere.difference(c.shape, part)  # exactly what the arc takes, every bit of it
            pieces.append(part)
            built += area
        if built < DETAIL_KM2:
            return 0., 0., None
        thickness = min(thick_km, owed_km3 / built)
        shape = sphere.union(*pieces)
        # New arc crust beside arcs already built extends them, and welds every one it touches
        # into one piece, rather than seeding another.
        arcs = [c for c in self.crust if c.plate == plate and c.kind == 'continent'
                and getattr(c, 'mesh', None) is None and spherely.distance(c.shape, shape) < 1e3]
        if arcs:
            arc = max(arcs, key=lambda c: c.area_km2())
            volume = sum(c.volume_km3() for c in arcs) + built * thickness
            arc.shape = sphere.union(arc.shape, shape, *[c.shape for c in arcs if c is not arc])
            arc.thickness_km = volume / arc.area_km2()
            arc.born_myr = min(c.born_myr for c in arcs)
            self.crust = [c for c in self.crust if c is arc or c not in arcs]
        else:
            arc = Crust(plate, 'continent', shape, self.time_myr, thickness)
            self.crust.append(arc)
        entry['gained']['arc'] += built
        entry['lost']['arc'] += built
        entry['arc_km3'] += built * thickness
        return built, built * thickness, arc

    def _meshed_at(self, plate, point):
        """The meshed continent piece of `plate` under a world-frame `point`, or None."""
        here = _point(sphere.rotate(point, self.frame(plate).T))
        for c in self.crust:
            if c.plate == plate and c.kind == 'continent' and getattr(c, 'mesh', None) is not None                     and spherely.contains(c.shape, here):
                return c
        return None

    def rift(self, cut) -> dict:
        """Split `cut.plate` along the great circle normal to `cut.pole`; the + side becomes a new plate.

        The new plate starts in its parent's frame (so it moves with it until told otherwise) and takes
        the crust on its side; boundaries keep their state, renamed by side. The cut itself is a new
        boundary, typed from the motion at the next step.
        """
        plate = cut.plate
        centre, pole = sphere.unit(cut.centre), sphere.unit(cut.pole)
        u = np.cross(pole, centre)
        t = np.linspace(0., 2. * np.pi, 72, endpoint=False)
        half = sphere.polygon(np.outer(np.cos(t), centre) + np.outer(np.sin(t), u))  # the + side
        piece = sphere.intersection(self.regions[plate], half)
        new = max(list(self.regions) + list(self.frames) + list(self.parents)) + 1
        self.frames[new] = self.frame(plate).copy()
        self.parents[new] = plate
        self.names[new] = '%s (rift)' % self.names.get(plate, 'Plate %d' % plate)
        area = sphere.area_km2(piece)
        self._give(plate, piece, new)
        for s in getattr(self, 'sutures', []):
            if s['plate'] == plate and np.mean(sphere.rotate(s['points'], self.frame(plate)) @ pole > 0.) > .5:
                s['plate'] = new  # same frame as the parent for now
        self.boundaries = self._derive(self._index(_split_by_side(self.boundaries, plate, new, pole)), {})
        return dict(kind='rift', plate=plate, new=new, area_km2=area, length_km=cut.length_km,
                    tension_n_per_m=cut.tension_n_per_m, strength_n_per_m=cut.strength_n_per_m,
                    continent_fraction=cut.continent_fraction)

    def _sutures(self, rotations_per_myr) -> list:
        """Weld the plates on either side of every collision that has closed SUTURE_KM.

        The smaller plate's crust passes into the larger's frame unchanged and the boundaries between
        them go; its other boundaries become the larger plate's, keeping their state.
        """
        events, into = [], {}
        for b in self.boundaries:
            for lo, hi in _runs_of(b.kinds, 'collision'):
                closed = float(np.mean(b.closed_km[lo:hi]))
                if closed < SUTURE_KM:
                    continue
                left, right = into.get(b.left, b.left), into.get(b.right, b.right)
                if left == right or left not in self.regions or right not in self.regions:
                    continue
                keep, gone = sorted((left, right), key=lambda p: -sphere.area_km2(self.regions[p]))
                area = sphere.area_km2(self.regions[gone])
                sutures = getattr(self, 'sutures', [])
                sutures.append(dict(plate=keep, points=sphere.rotate(b.points[lo:hi + 1], self.frame(keep).T),
                                    time_myr=self.time_myr))
                carry = self.frame(keep).T @ self.frame(gone)
                for s in sutures:
                    if s['plate'] == gone:
                        s['plate'], s['points'] = keep, sphere.rotate(s['points'], carry)
                self.sutures = sutures
                self._give(gone, self.regions[gone], keep)
                self.regions.pop(gone, None)
                self.crust = [c for c in self.crust if c.plate != gone]
                self.parents[gone] = keep
                into = {p: (keep if q == gone else q) for p, q in into.items()}
                into[gone] = keep
                events.append(dict(kind='suture', plates=(b.left, b.right), into=keep, gone=gone,
                                   area_km2=area, closed_km=closed))
        if into:
            self.boundaries = self._derive(self._index(_relabel(self.boundaries, into)), rotations_per_myr)
        return events

    def _initiate(self, b) -> list:
        events = []
        start = 0
        kinds = b.kinds + [None]
        for k in range(1, len(kinds)):
            if kinds[k] == kinds[start]:
                continue
            run = range(start, k)
            if kinds[start] == 'converging' and np.mean(b.closed_km[start:k]) >= INITIATION_KM:
                middle = (start + k) // 2
                left, right = self._sides(b, middle)
                if left == 'continent' and right == 'continent':
                    for j in run:
                        b.kinds[j], b.overriding[j] = 'collision', None
                    events.append(dict(kind='collision', plates=(b.left, b.right), closed_km=float(np.mean(b.closed_km[start:k]))))
                else:
                    if left != right:
                        over = b.left if left == 'continent' else b.right
                    else:
                        over = b.left if self._age(b, middle, b.left) < self._age(b, middle, b.right) else b.right
                    for j in run:
                        b.kinds[j], b.overriding[j] = 'trench', over
                    events.append(dict(kind='subduction_initiated', plates=(b.left, b.right), overriding=over,
                                       closed_km=float(np.mean(b.closed_km[start:k]))))
                b.closed_km[start:k] = 0.
            start = k
        return events

    def _collide(self, point, area, plate) -> list:
        """Trench segments where continent arrived (>= COLLISION_KM2 this step) become collision."""
        if area < COLLISION_KM2:
            return []
        for b in self.boundaries:
            if plate not in (b.left, b.right):
                continue
            mids = b.mids()
            near = (sphere.arc_km(mids, point[None]) < 500.) & np.array([k == 'trench' for k in b.kinds])
            if near.any():
                for k in np.flatnonzero(near):
                    b.kinds[k], b.overriding[k] = 'collision', None
                    b.closed_km[k] = 0.  # closing across the collision counts from here
                return [dict(kind='collision', plates=(b.left, b.right), arrived_km2=area)]
        return []

    def _sides(self, b, k):
        """Crust kind on each side of segment k, SAMPLE_KM away."""
        m = b.mids()[k]
        pole = sphere.unit(np.cross(b.points[k], b.points[k + 1]))
        angle = SAMPLE_KM / sphere.RADIUS_KM
        result = []
        for plate, sign in ((b.left, 1.), (b.right, -1.)):
            point = sphere.unit(m * np.cos(angle) + sign * pole * np.sin(angle))
            result.append(self._crust_kind(plate, point))
        return result

    def _age(self, b, k, plate):
        m = b.mids()[k]
        pole = sphere.unit(np.cross(b.points[k], b.points[k + 1]))
        sign = 1. if plate == b.left else -1.
        point = sphere.unit(m * np.cos(SAMPLE_KM / sphere.RADIUS_KM) + sign * pole * np.sin(SAMPLE_KM / sphere.RADIUS_KM))
        local = sphere.rotate(point, self.frame(plate).T)
        here = spherely.create_point(*map(float, sphere.to_lonlat(local)))
        for c in self.crust:
            if c.plate == plate and spherely.contains(c.shape, here):
                return 0. if c.born_myr is None else self.time_myr - c.born_myr
        return 0.

    def _crust_kind(self, plate, point):
        local = sphere.rotate(point, self.frame(plate).T)
        here = spherely.create_point(*map(float, sphere.to_lonlat(local)))
        for c in self.crust:
            if c.plate == plate and spherely.contains(c.shape, here):
                return c.kind
        return 'ocean'

    # ---------------------------------------------------------------- check
    def _check(self, entry):
        total = sum(entry['plates_km2'].values())
        entry['tiling_error_km2'] = total - sphere.SPHERE_KM2
        crust = {}
        for piece in self.crust:
            crust[piece.plate] = crust.get(piece.plate, 0.) + piece.area_km2()
        entry['crust_vs_plate_error_km2'] = max((abs(crust.get(p, 0.) - a) for p, a in entry['plates_km2'].items()), default=0.)
        entry['balance_error_km2'] = sum(entry['gained'].values()) - sum(entry['lost'].values())
        entry['continent_km3'] = self._continent_km3()
        # Regions are rebuilt from the crust every step, so rebuilding error cannot accumulate; a step
        # is refused beyond 1e-4 of the sphere (0.01%, ~51,000 km2), ten times inside the 0.1% standard.
        scale = CHECK_TOLERANCE * sphere.SPHERE_KM2
        for key, message in (('tiling_error_km2', 'Plates no longer tile the sphere'),
                             ('crust_vs_plate_error_km2', 'Crust no longer tiles its plate'),
                             ('balance_error_km2', 'Crust created and destroyed do not balance')):
            if abs(entry[key]) > scale:
                raise ValueError('%s: %.3g km2 off.' % (message, entry[key]))


def _loss_category(boundary_kind, crust_kind):
    if boundary_kind == 'trench':
        return 'subducted' if crust_kind == 'ocean' else 'continent_at_trench'
    if boundary_kind in ('converging', 'collision'):
        return 'collision' if crust_kind == 'continent' else 'compression'
    return 'sliver'


def _point(xyz):
    """A spherely point for a unit vector."""
    return spherely.create_point(*map(float, sphere.to_lonlat(xyz)))


def _centroid(shape):
    """The centroid of a geography as a unit vector."""
    p = spherely.centroid(shape)
    return sphere.from_lonlat(float(spherely.get_x(p)), float(spherely.get_y(p)))


def _about(piece):
    """A piece's centre (unit vector, its plate's frame) and reach (km to its farthest vertex)."""
    centre = _centroid(piece.shape)
    verts = [r for poly in sphere.polygons(piece.shape) for r in poly]
    reach = float(sphere.arc_km(np.vstack(verts), centre).max()) if verts else 0.
    return centre, reach


def _runs_of(kinds, kind):
    """(start, end) of every run of `kind` in a list of segment kinds."""
    runs, start = [], None
    for k, here in enumerate(list(kinds) + [None]):
        if here == kind and start is None:
            start = k
        elif here != kind and start is not None:
            runs.append((start, k))
            start = None
    return runs


def _relabel(boundaries, into):
    """The boundaries with plates renamed by `into` (gone -> kept); those now inside one plate dropped."""
    result = []
    for b in boundaries:
        left, right = into.get(b.left, b.left), into.get(b.right, b.right)
        if left == right:
            continue
        over = [into.get(o, o) for o in b.overriding]
        idents = None if getattr(b, 'idents', None) is None else np.asarray(b.idents)
        arcs = b.arcs().copy()
        if left < right:
            result.append(Boundary(left, right, b.points, list(b.kinds), over, b.closed_km.copy(), idents, arcs))
        else:
            result.append(Boundary(right, left, b.points[::-1], list(b.kinds)[::-1], over[::-1],
                                   b.closed_km[::-1].copy(), None if idents is None else idents[::-1], arcs[::-1]))
    return result


def _split_by_side(boundaries, plate, new, pole):
    """The boundaries with `plate`'s segments on the + side of `pole` given to `new`."""
    result = []
    for b in boundaries:
        if plate not in (b.left, b.right):
            result.append(b)
            continue
        side = b.mids() @ pole > 0.
        start = 0
        for k in range(1, len(side) + 1):
            if k < len(side) and side[k] == side[start]:
                continue
            into = {plate: new} if side[start] else {}
            idents = None if getattr(b, 'idents', None) is None else np.asarray(b.idents)[start:k]
            part = Boundary(b.left, b.right, b.points[start:k + 1], b.kinds[start:k], b.overriding[start:k],
                            b.closed_km[start:k].copy(), idents, b.arcs()[start:k].copy())
            result += _relabel([part], into)
            start = k
    return result


def _fill_overriding(b):
    """Trench segments smoothing gave no overriding plate take that of the nearest trench segment."""
    known = [k for k, (kind, over) in enumerate(zip(b.kinds, b.overriding)) if kind == 'trench' and over is not None]
    for k, kind in enumerate(b.kinds):
        if kind == 'trench' and b.overriding[k] is None:
            if not known:
                b.kinds[k] = 'converging'  # a trench nobody overrides is not a trench yet
                continue
            b.overriding[k] = b.overriding[min(known, key=lambda j: abs(j - k))]


def _kind_from_motion(point, pair, rotations_per_myr):
    """Type of a new contact at `point` (left plate pair[0]) from the plates' relative motion."""
    left, right = pair
    # Neutral until the motion re-typing that follows the derivation types it.
    return 'transform'


def _point_in(geography):
    """A point of (the largest polygon of) a geography, as a unit vector."""
    c = spherely.centroid(geography)
    return sphere.from_lonlat(spherely.get_x(c), spherely.get_y(c))


def from_network(world) -> RegionWorld:
    """A regions-first world from a network-based one (same plates, crust and boundary types)."""
    regions = world.network.polygons()
    boundaries, next_ident = [], 0
    for ident, s in world.network.sections.items():
        points = world.network.points(ident)
        left, right = s.left, s.right
        if left > right:
            points, left, right = points[::-1], right, left
        count = len(points) - 1
        boundaries.append(Boundary(left, right, points, [s.kind] * count, [s.overriding] * count, np.zeros(count),
                                   np.arange(next_ident, next_ident + count)))
        next_ident += count
    return RegionWorld([Crust(c.plate, c.kind, c.shape, c.born_myr, c.thickness_km) for c in world.crust],
                       {p: world.frame(p).copy() for p in regions}, regions, boundaries, dict(world.names),
                       next_ident=next_ident)
