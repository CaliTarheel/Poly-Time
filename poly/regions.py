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
from poly.sphere import DETAIL_KM2
from poly.topology import segment_motion
from poly.world import (AGE_BAND_MYR, COLLISION_KM2, INITIATION_KM, MINIMUM_PLATE_KM2, SAMPLE_KM, TOLERANCE,
                        Crust, _merge_bands)

GAINS = ('ridge', 'trench_opening', 'sliver', 'junction')
LOSSES = ('subducted', 'continent_at_trench', 'collision', 'compression', 'sliver', 'junction')
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
                     events=[], continent_arrived=[], thickened_km3=0.)
        lost = {p: [] for p in moved}      # (world geography, boundary kind)
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
                    given = [shape for shape, _ in lost[plate][start:]]
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
        anchors = _vertices(mine)  # the plate's own outline, which every loss and gain was cut from
        # The losses are disjoint pieces of the plate's overlaps; joined by boundary kind, each crust
        # piece meets at most one geography per kind and is cut once, instead of once per loss.
        by_kind = {}
        for shape, kind in losses:
            if sphere.area_km2(shape) >= TINY_KM2:
                by_kind.setdefault(kind, []).append(shape)
        local_by_kind = {kind: sphere.rotate_geography(sphere.union(*shapes), back, anchors=anchors)
                         for kind, shapes in by_kind.items()}
        if local_by_kind:
            all_local = sphere.union(*local_by_kind.values())
            for piece in mine:
                whole = sphere.intersection(piece.shape, all_local)
                if sphere.area_km2(whole) < TINY_KM2:
                    continue
                continental = 0.
                for kind, local in local_by_kind.items():
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
                        for shape in by_kind[kind]:
                            part = sphere.intersection(whole, sphere.rotate_geography(shape, back, anchors=anchors))
                            if sphere.area_km2(part) >= TINY_KM2:
                                entry['continent_arrived'].append(
                                    (sphere.rotate(_point_in(part), self.frame(plate)), sphere.area_km2(part), plate))
                piece.shape = sphere.difference(piece.shape, whole)
                if continental >= TINY_KM2:
                    # Continent is too light to go away: it shortens and thickens, keeping its volume.
                    entry['thickened_km3'] += self._thicken(piece, continental, mine)
        for shape, category in gains:
            area = sphere.area_km2(shape)
            if area < TINY_KM2:
                continue
            entry['gained'][category] += area
            mine.append(Crust(plate, 'ocean', sphere.rotate_geography(shape, back, anchors=anchors), self.time_myr + .5 * dt))
        self.crust = others + [c for c in mine if sphere.area_km2(c.shape) >= TINY_KM2]

    def _thicken(self, piece, area, mine):
        """Keep the volume of `area` km2 of continent just taken from `piece`; returns that volume.

        The piece keeps it, thicker; one taken away whole hands it to its plate's largest other
        continental piece. (Thickness is one number per piece until continents carry a mesh.)
        """
        volume = area * piece.thickness_km
        left = piece.area_km2()
        if left >= TINY_KM2:
            piece.thickness_km += volume / left
            return volume
        others = [c for c in mine if c is not piece and c.kind == 'continent' and c.area_km2() >= TINY_KM2]
        if others:
            heir = max(others, key=lambda c: c.area_km2())
            heir.thickness_km += volume / heir.area_km2()
        return volume

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
        anchors = _vertices([c for c in self.crust if c.plate == plate])
        anchors_to = _vertices([c for c in self.crust if c.plate == to])
        if sphere.area_km2(doubled) >= TINY_KM2:
            local = sphere.rotate_geography(doubled, self.frame(plate).T, anchors=anchors)
            for c in [c for c in self.crust if c.plate == plate]:
                area = sphere.area_km2(sphere.intersection(c.shape, local))
                if area < TINY_KM2:
                    continue
                c.shape = sphere.difference(c.shape, local)
                if entry is not None:
                    entry['lost']['sliver'] += area
            self.regions[plate] = sphere.difference(self.regions[plate], doubled)
            piece = sphere.difference(piece, doubled)
        local = sphere.rotate_geography(piece, self.frame(plate).T, anchors=anchors)
        carry = self.frame(to).T @ self.frame(plate)
        moved = []
        for c in [c for c in self.crust if c.plate == plate]:
            part = sphere.intersection(c.shape, local)
            if sphere.area_km2(part) < TINY_KM2:
                continue
            c.shape = sphere.difference(c.shape, local)
            shape = part if np.allclose(carry, np.eye(3)) else sphere.rotate_geography(part, carry, anchors=anchors_to)
            moved.append(Crust(to, c.kind, shape, c.born_myr, c.thickness_km))
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
            index.setdefault((b.left, b.right), []).append(
                (mids, list(b.kinds), list(b.overriding), np.asarray(b.closed_km, float), np.asarray(idents)))
        return index

    def _split_overlap(self, local, kind, over, p, q, stage, lost):
        if kind == 'trench' and over in (p, q):
            lost[q if over == p else p].append((local, 'trench'))
            return
        mine = sphere.intersection(local, stage((p, q), p))
        lost[p].append((sphere.difference(local, mine), kind))
        lost[q].append((mine, kind))

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
        fresh = iter(self._fresh(int((choice < 0).sum())))
        for k in range(n):
            if choice[k] >= 0:
                _, kinds, over, closed, old_idents = candidates[choice[k]]
                j = where[k]
                b.kinds.append(kinds[j])
                b.overriding.append(over[j])
                b.closed_km[k] = closed[j]
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
        for point, area, plate in entry['continent_arrived']:
            events += self._collide(point, area, plate)
        events += self._sutures(rotations_per_myr)
        return events

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


def _vertices(pieces):
    """Every outline vertex of `pieces` (crust in one frame), for welding shapes onto."""
    rings = [r for c in pieces for poly in sphere.polygons(c.shape) for r in poly]
    return np.vstack(rings) if rings else np.zeros((0, 3))


def _loss_category(boundary_kind, crust_kind):
    if boundary_kind == 'trench':
        return 'subducted' if crust_kind == 'ocean' else 'continent_at_trench'
    if boundary_kind in ('converging', 'collision'):
        return 'collision' if crust_kind == 'continent' else 'compression'
    return 'sliver'


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
        if left < right:
            result.append(Boundary(left, right, b.points, list(b.kinds), over, b.closed_km.copy(), idents))
        else:
            result.append(Boundary(right, left, b.points[::-1], list(b.kinds)[::-1], over[::-1],
                                   b.closed_km[::-1].copy(), None if idents is None else idents[::-1]))
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
                            b.closed_km[start:k].copy(), idents)
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
