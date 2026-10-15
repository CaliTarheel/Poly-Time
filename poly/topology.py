"""The plate-boundary network: typed boundary sections joined at junctions, and the plates they bound.

A section is a polyline from one junction to another (or back to the same junction, for a closed
loop) with a plate on its left and a plate on its right, seen walking from start to end. Its kind sets
how it moves: a trench moves with its overriding plate; every other kind moves with the half-stage
rotation of its two plates, so a ridge stays midway between the plates it separates. A junction moves
to the mean of where its sections' moved ends would put it.

Plates are not stored: a plate's polygon is rebuilt from the sections it lies to the left of (sections
it lies to the right of, walked backwards). Because every section has exactly one plate on each side,
the rebuilt plates tile the sphere with no gap and no overlap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re

import numpy as np

from poly import sphere

KINDS = ('ridge', 'trench', 'transform', 'converging', 'collision')


class BoundaryCollision(ValueError):
    """Two boundaries that share no junction cross: the plate between them is pinched."""

    def __init__(self, first, second, plates):
        super().__init__('Boundaries %d and %d collide away from any junction.' % (first, second))
        self.first, self.second, self.plates = first, second, plates


@dataclass
class Section:
    start: int
    end: int
    interior: np.ndarray
    left: int
    right: int
    kind: str
    overriding: int | None = None
    closed_km: float = 0.  # closing accumulated while converging without a trench

    def __post_init__(self):
        self.interior = np.asarray(self.interior, float).reshape(-1, 3)
        if self.kind not in KINDS:
            raise ValueError('Unknown boundary kind: %r' % self.kind)
        if self.left == self.right:
            raise ValueError('A boundary needs a different plate on each side.')
        if self.kind == 'trench' and self.overriding not in (self.left, self.right):
            raise ValueError('A trench names one of its two plates as overriding.')
        if self.kind != 'trench' and self.overriding is not None:
            raise ValueError('Only a trench has an overriding plate.')


@dataclass
class Network:
    junctions: dict = field(default_factory=dict)
    sections: dict = field(default_factory=dict)
    next_section: int = 1
    next_junction: int = 1

    def add_junction(self, xyz) -> int:
        ident = self.next_junction
        self.next_junction += 1
        self.junctions[ident] = sphere.unit(xyz)
        return ident

    def add_section(self, start, end, interior, left, right, kind, overriding=None) -> int:
        ident = self.next_section
        self.next_section += 1
        self.sections[ident] = Section(start, end, interior, left, right, kind, overriding)
        return ident

    def points(self, ident) -> np.ndarray:
        """All points of a section from its start junction to its end junction."""
        s = self.sections[ident]
        return np.vstack((self.junctions[s.start][None], s.interior, self.junctions[s.end][None]))

    def plates(self) -> list:
        return sorted({p for s in self.sections.values() for p in (s.left, s.right)})

    def motion(self, section: Section, rotations: dict) -> np.ndarray:
        """The rotation matrix a section moves by this step."""
        if section.kind == 'trench':
            return sphere.rotation(rotations[section.overriding])
        return sphere.rotation(.5 * (np.asarray(rotations[section.left]) + np.asarray(rotations[section.right])))

    def move(self, rotations: dict) -> dict:
        """Move every section and junction; `rotations` maps plate -> rotation vector for this step.

        Each section moves as a whole by its own motion. A junction then goes to the point nearest
        all its sections' moved end lines (as in GPlates' resolved topologies, where boundaries are
        cut where they meet), and each section is cut back or extended to it. If two sections still
        cross near the junction, the junction moves to that crossing. Returns, per junction, how far
        (km) the moved section ends were from where it went.
        """
        moved, ends = {}, {ident: [] for ident in self.junctions}
        for ident, s in self.sections.items():
            matrix = self.motion(s, rotations)
            moved[ident] = sphere.rotate(self.points(ident), matrix)
            ends[s.start].append((ident, 0))
            ends[s.end].append((ident, -1))
        misfit = {}
        for junction, incident in ends.items():
            if not incident:
                continue
            tips = np.array([moved[ident][index] for ident, index in incident])
            nexts = np.array([moved[ident][1 if index == 0 else -2] for ident, index in incident])
            # A trench is fixed to its overriding plate, so a junction on it slides along it.
            weights = np.array([100. if self.sections[ident].kind == 'trench' else 1. for ident, _ in incident])
            centre = _junction_point(tips, nexts, weights)
            misfit[junction] = float(sphere.arc_km(tips, centre[None]).max())
            self.junctions[junction] = centre
        for ident, s in self.sections.items():
            line = moved[ident]
            line = _trim(line, self.junctions[s.start], from_start=True)
            line = _trim(line, self.junctions[s.end], from_start=False)
            s.interior = line[1:-1]
        for junction, incident in ends.items():
            if len(incident) > 1:
                misfit[junction] = max(misfit.get(junction, 0.), self._uncross(junction))
        return misfit

    def _uncross(self, junction, rounds=4) -> float:
        """Move a junction to where two of its sections cross nearby, and trim them all to it.

        Where the motions at a junction are inconsistent (its velocity triangle does not close), the
        junction migrates along one of its boundaries; the sections then cross a little way out, and
        that crossing is where the junction now is. Returns how far it moved, km.
        """
        start = self.junctions[junction].copy()
        for _ in range(rounds):
            incident = self.incident(junction)
            lines = {}
            for ident in incident:
                s = self.sections[ident]
                points = self.points(ident)
                lines[ident] = points if s.start == junction else points[::-1]
            best, best_km = None, np.inf
            for i in range(len(incident)):
                for j in range(i + 1, len(incident)):
                    point = _first_crossing(lines[incident[i]], lines[incident[j]])
                    if point is not None:
                        km = float(sphere.arc_km(point, self.junctions[junction]))
                        if km < best_km:
                            best, best_km = point, km
            if best is None:
                break
            self.junctions[junction] = best
            for ident in incident:
                s = self.sections[ident]
                line = self.points(ident)
                if s.start == junction:
                    line = _trim(line, best, from_start=True)
                if s.end == junction:
                    line = _trim(line, best, from_start=False)
                s.interior = line[1:-1]
        return float(sphere.arc_km(start, self.junctions[junction]))

    def reclassify(self, rotations_per_myr, margin=.2) -> list:
        """Re-type ridges, transforms and converging boundaries from the plates' current motion.

        Each segment is typed from the right plate's motion relative to the left (see
        `segment_motion`), with a margin around the threshold so a boundary near it keeps its
        type. Closing makes a converging boundary, never a trench: a trench is made by initiation.
        Trenches and collisions keep their type here. A boundary whose type changes along it is
        split where it changes; afterwards neighbours of one plate pair and type are joined.
        """
        from poly.build import TRANSFORM_RATIO, _smooth
        events = []
        for ident in list(self.sections):
            s = self.sections[ident]
            if s.kind in ('trench', 'collision'):
                continue
            points = self.points(ident)
            opening, sliding = segment_motion(points, rotations_per_myr[s.left], rotations_per_myr[s.right])
            ratio = np.abs(opening) / np.maximum(sliding, 1e-12)
            if s.kind == 'transform':
                moving = ratio > TRANSFORM_RATIO * (1. + margin)
            else:
                moving = ratio > TRANSFORM_RATIO * (1. - margin)
            kinds = np.where(moving, np.where(opening > 0., 'ridge', 'converging'), 'transform')
            kinds = [str(k) for k in _smooth(list(kinds))]
            if set(kinds) == {s.kind}:
                continue
            events.append(dict(kind='retyped', section=ident, plates=(s.left, s.right), was=s.kind,
                               now=sorted(set(kinds))))
            cuts = [i for i in range(1, len(kinds)) if kinds[i] != kinds[i - 1]]
            bounds = [0] + cuts + [len(kinds)]
            ends = [s.start] + [self.add_junction(points[c]) for c in cuts] + [s.end]
            closed = s.closed_km
            del self.sections[ident]
            for k in range(len(bounds) - 1):
                lo, hi = bounds[k], bounds[k + 1]
                kind = kinds[lo]
                new = self.add_section(ends[k], ends[k + 1], points[lo + 1:hi], s.left, s.right, kind)
                self.sections[new].closed_km = closed if kind == 'converging' else 0.
        events += self.join_alike()
        return events

    def join_alike(self) -> list:
        """Join the two boundaries at a junction of two when they have one plate pair and type."""
        events = []
        changed = True
        while changed:
            changed = False
            for junction in list(self.junctions):
                incident = self.incident(junction)
                if len(incident) != 2:
                    continue
                a, b = (self.sections[i] for i in incident)
                if a is b or a.start == a.end or b.start == b.end or a.kind != b.kind or a.overriding != b.overriding:
                    continue
                if {a.left, a.right} != {b.left, b.right}:
                    continue
                keep, drop = incident
                if self._merge_run(drop, events, only=junction):
                    self.sections[keep].closed_km = max(a.closed_km, b.closed_km)
                    changed = True
                    break
        return events

    def absorb(self, plate, into) -> list:
        """Hand `plate` to its neighbour `into`: their shared boundaries go, its others pass to `into`."""
        if not any({s.left, s.right} == {plate, into} for s in self.sections.values()):
            raise ValueError('Plate %d does not touch plate %d.' % (plate, into))
        for ident in [i for i, s in self.sections.items() if {s.left, s.right} == {plate, into}]:
            del self.sections[ident]
        for s in self.sections.values():
            if s.left == plate:
                s.left = into
            if s.right == plate:
                s.right = into
            if s.overriding == plate:
                s.overriding = into
        for junction in [j for j in self.junctions if not self.incident(j)]:
            del self.junctions[junction]
        return self.join_alike()

    def split_section(self, ident, point) -> int:
        """Put a junction on a section at (the point of it nearest) `point`; returns the junction."""
        s = self.sections[ident]
        line = self.points(ident)
        nearest = int(np.argmin(sphere.arc_km(line, point[None])))
        if float(sphere.arc_km(line[nearest], point)) < 1.:
            # On (or within a km of) an existing point: cut there, or use the junction it is.
            if nearest == 0:
                return s.start
            if nearest == len(line) - 1:
                return s.end
            junction = self.add_junction(line[nearest])
            end = s.end
            first, second = line[1:nearest], line[nearest + 1:-1]
            s.interior, s.end = np.array(first).reshape(-1, 3), junction
            new = self.add_section(junction, end, np.array(second).reshape(-1, 3), s.left, s.right, s.kind, s.overriding)
            self.sections[new].closed_km = s.closed_km
            return junction
        k = min(range(len(line) - 1), key=lambda i: _segment_km(point, line[i], line[i + 1]))
        a, b = line[k], line[k + 1]
        pole = np.cross(a, b)
        norm = float(np.linalg.norm(pole))
        foot = sphere.unit(point - (point @ (pole / norm)) * (pole / norm)) if norm > 1e-15 else a
        if not _on_arc(foot, a, b):
            foot = a if float(sphere.arc_km(point, a)) < float(sphere.arc_km(point, b)) else b
        junction = self.add_junction(foot)
        first, second = line[1:k + 1], line[k + 1:-1]
        end = s.end
        s.interior, s.end = np.array(first).reshape(-1, 3), junction
        new = self.add_section(junction, end, np.array(second).reshape(-1, 3), s.left, s.right, s.kind, s.overriding)
        self.sections[new].closed_km = s.closed_km
        return junction

    def necks(self, plate, within_km, apart_km=1000., ratio=5.) -> list:
        """Places where a plate's outline comes back within `within_km` of itself: necks.

        Two points of the outline count only if, going round the outline, they are more than
        `apart_km` and more than `ratio` times their distance apart; points near each other along
        the outline (either side of a junction, round a bend) are not a neck. Returns
        (km, section, point, section, point), nearest first.
        """
        found = []
        for ring, mapping in self.rings(plate, with_map=True):
            if len(ring) < 4:
                continue
            step = sphere.arc_km(ring, np.roll(ring, -1, axis=0))
            along = np.r_[0., np.cumsum(step)[:-1]]
            total = float(step.sum())
            distance = sphere.RADIUS_KM * np.arccos(np.clip(ring @ ring.T, -1., 1.))
            gap = np.abs(along[:, None] - along[None, :])
            gap = np.minimum(gap, total - gap)
            ok = (gap > apart_km) & (gap > ratio * distance) & (distance < within_km)
            if not ok.any():
                continue
            masked = np.where(ok, distance, np.inf)
            i, j = np.unravel_index(int(np.argmin(masked)), masked.shape)
            found.append((float(distance[i, j]), mapping[i][0], ring[i], mapping[j][0], ring[j]))
        return sorted(found, key=lambda f: f[0])

    def split_plate(self, plate, new, first, point_a, second, point_b) -> int:
        """Cut `plate` from a point on one of its boundaries to a point on another.

        The new boundary runs from the first point to the second with `plate` on its left; the
        part of the plate on its right becomes `new`. Returns the new boundary's identity.
        """
        start = self.split_section(first, point_a)
        # The second section may have been the first one, now split in two: find the piece that
        # holds the second point.
        candidates = [i for i, s in self.sections.items() if plate in (s.left, s.right)]
        second = min(candidates, key=lambda i: min(_segment_km(point_b, q, r) for q, r in
                                                    zip(self.points(i)[:-1], self.points(i)[1:])))
        end = self.split_section(second, point_b)
        # Walk the plate's boundary (plate on the left) from the first cut to the second: those
        # boundaries face the new boundary's right side and go to `new`.
        walks = {}
        for ident, s in self.sections.items():
            if s.left == plate:
                walks.setdefault(s.start, []).append((ident, s.end))
            elif s.right == plate:
                walks.setdefault(s.end, []).append((ident, s.start))
        chain, here, seen = [], start, set()
        while here != end:
            options = [w for w in walks.get(here, []) if w[0] not in seen]
            if not options:
                raise ValueError('Cannot walk plate %d from the first cut to the second.' % plate)
            ident, here = options[0]
            seen.add(ident)
            chain.append(ident)
        for ident in chain:
            s = self.sections[ident]
            if s.left == plate:
                s.left = new
            if s.right == plate:
                s.right = new
            if s.overriding == plate:
                s.overriding = new
        from poly.build import densify
        line = densify(np.array([self.junctions[start], self.junctions[end]]), 100.)
        return self.add_section(start, end, line[1:-1], plate, new, 'transform')

    def shared_km(self, a, b) -> float:
        return sum(self.length_km(i) for i, s in self.sections.items() if {s.left, s.right} == {a, b})

    def length_km(self, ident) -> float:
        points = self.points(ident)
        return float(sphere.arc_km(points[:-1], points[1:]).sum())

    def incident(self, junction) -> list:
        return [ident for ident, s in self.sections.items() if junction in (s.start, s.end)]

    def resolve_short(self, rotations_per_myr, minimum_km, overriding_rule) -> list:
        """Remove sections about to shrink through zero, before they invert. Returns the events.

        A short run between two changes of kind along one plate pair merges into its neighbour. A
        short boundary between two triple junctions is swapped (a T1 transition): the plates at its
        ends, which it kept apart, now meet along a new short boundary across it, of the kind their
        relative motion makes.
        """
        events = []
        changed = True
        while changed:
            changed = False
            for ident in sorted(self.sections, key=self.length_km):
                if self.length_km(ident) >= minimum_km:
                    break
                if self.sections[ident].start == self.sections[ident].end:
                    continue
                if self._merge_run(ident, events) or self._swap(ident, rotations_per_myr, minimum_km, overriding_rule, events):
                    changed = True
                    break
        return events

    def _merge_run(self, ident, events, only=None) -> bool:
        """Fold a section into the neighbour of the same plate pair across a junction of two
        (across `only`, if given)."""
        s = self.sections[ident]
        for junction in (s.start, s.end):
            if only is not None and junction != only:
                continue
            others = [o for o in self.incident(junction) if o != ident]
            if len(others) != 1:
                continue
            other = self.sections[others[0]]
            if {other.left, other.right} != {s.left, s.right} or other.start == other.end:
                continue
            run = self.points(ident)
            run = run if s.start == junction else run[::-1]  # from the junction to the far end
            far = s.end if s.start == junction else s.start
            line = self.points(others[0])
            if other.end == junction:
                merged = np.vstack((line[:-1], run))
                other.end = far
            else:
                merged = np.vstack((run[::-1][:-1], line))
                other.start = far
            other.interior = merged[1:-1]
            del self.sections[ident]
            del self.junctions[junction]
            events.append(dict(kind='run_merged', section=ident, into=others[0], plates=(s.left, s.right)))
            return True
        return False

    def _swap(self, ident, rotations_per_myr, minimum_km, overriding_rule, events) -> bool:
        """T1 transition of a short boundary between two triple junctions."""
        s = self.sections[ident]
        a, b = s.left, s.right
        sides = {}
        for junction in (s.start, s.end):
            others = [o for o in self.incident(junction) if o != ident]
            if len(others) != 2:
                return False
            plates = [{self.sections[o].left, self.sections[o].right} for o in others]
            third = (plates[0] | plates[1]) - {a, b}
            if len(third) != 1:
                return False
            sides[junction] = (third.pop(), others)
        c, d = sides[s.start][0], sides[s.end][0]
        if c == d:
            return False
        points = self.points(ident)
        middle = sphere.unit(points[0] + points[-1])
        along = sphere.unit(points[-1] - points[0])
        toward_a = sphere.unit(np.cross(middle, along))  # a lies left of start -> end
        half = 1.5 * minimum_km / sphere.RADIUS_KM
        near_a = sphere.unit(middle * np.cos(half) + toward_a * np.sin(half))
        near_b = sphere.unit(middle * np.cos(half) - toward_a * np.sin(half))
        new_a, new_b = self.add_junction(near_a), self.add_junction(near_b)
        for junction in (s.start, s.end):
            for other in sides[junction][1]:
                o = self.sections[other]
                target = new_a if a in (o.left, o.right) else new_b
                if o.start == junction:
                    o.start = target
                if o.end == junction:
                    o.end = target
        del self.sections[ident]
        del self.junctions[s.start]
        del self.junctions[s.end]
        # Walking from a's side to b's side, d (beyond the old end) lies on the left, c on the right.
        from poly.build import kinds_along
        kind = kinds_along(np.array([near_a, near_b]), rotations_per_myr[d], rotations_per_myr[c])[0]
        over = overriding_rule(d, c) if kind == 'trench' else None
        new = self.add_section(new_a, new_b, np.zeros((0, 3)), d, c, str(kind), over)
        events.append(dict(kind='boundary_swapped', removed=ident, removed_plates=(a, b), added=new,
                           added_plates=(d, c), added_kind=str(kind)))
        return True

    def rings(self, plate, with_map=False) -> list:
        """The closed boundary rings of a plate, each with the plate on its left.

        With `with_map`, each ring comes with, per point, (section, index of that point in the
        section's own points from its start).
        """
        walks = []
        for ident, s in self.sections.items():
            count = len(s.interior) + 2
            if s.left == plate:
                walks.append((s.start, s.end, self.points(ident)[:-1], ident, [(ident, k) for k in range(count - 1)]))
            elif s.right == plate:
                points = self.points(ident)[::-1]
                walks.append((s.end, s.start, points[:-1], ident, [(ident, count - 1 - k) for k in range(count - 1)]))
        by_start = {}
        for walk in walks:
            by_start.setdefault(walk[0], []).append(walk)
        used, rings = set(), []
        for walk in walks:
            if walk[3] in used:
                continue
            ring, mapping, current = [], [], walk
            while current[3] not in used:
                used.add(current[3])
                ring.append(current[2])
                mapping += current[4]
                options = [w for w in by_start.get(current[1], []) if w[3] not in used]
                if not options:
                    break
                if len(options) > 1:
                    # Several boundaries of this plate leave the same junction: take the one that
                    # turns furthest to the left, which keeps the plate's interior on the left.
                    options.sort(key=lambda w: -_turn(current[2][-1] if len(current[2]) else self.junctions[current[0]],
                                                      self.junctions[current[1]], w[2][1] if len(w[2]) > 1 else self.junctions[w[1]]))
                current = options[0]
            if current[1] != walk[0]:
                raise ValueError('The boundary of plate %d does not close.' % plate)
            rings.append((np.vstack(ring), mapping) if with_map else np.vstack(ring))
        return rings

    def repair(self, rounds=20) -> list:
        """Make every plate outline valid again after a move; returns what was repaired.

        S2 names the two edges of an outline that cross. Inside one section, the small loop between
        them is cut out at the crossing. Between two sections that meet at a junction, the junction
        moves to the crossing and its sections are cut back to it. A crossing between sections that
        share no junction is a collision of boundaries, a topology event this cannot repair.
        """
        repairs = []
        for _ in range(rounds):
            problem = None
            for plate in self.plates():
                for ring, mapping in self.rings(plate, with_map=True):
                    try:
                        sphere.polygon(ring)
                    except ValueError as error:
                        found = re.findall(r'Edge (\d+) crosses edge (\d+)', str(error))
                        if not found:
                            raise
                        problem = (plate, ring, mapping, int(found[0][0]), int(found[0][1]))
                        break
                if problem:
                    break
            if problem is None:
                return repairs
            plate, ring, mapping, i, j = problem
            n = len(ring)
            a, b, c, d = ring[i], ring[(i + 1) % n], ring[j], ring[(j + 1) % n]
            meet = sphere.unit(np.cross(np.cross(a, b), np.cross(c, d)))
            if float(meet @ (a + b + c + d)) < 0.:
                meet = -meet
            first, second = mapping[i][0], mapping[j][0]
            if first == second:
                ident = first
                s = self.sections[ident]
                points = self.points(ident)
                lo, hi = sorted((mapping[i][1], mapping[j][1]))
                hi = min(hi + 1, len(points) - 1)
                if lo == 0 or hi == len(points) - 1:
                    # The loop reaches the section's end: the crossing becomes the junction there.
                    junction = s.start if lo == 0 else s.end
                    self.junctions[junction] = meet
                    self._retrim(junction)
                else:
                    line = np.vstack((points[:lo + 1], meet[None], points[hi:]))
                    s.interior = line[1:-1]
                repairs.append(dict(kind='loop_cut', section=ident, plate=plate))
            else:
                ends = ({self.sections[first].start, self.sections[first].end}
                        & {self.sections[second].start, self.sections[second].end})
                if not ends:
                    shared = ({self.sections[first].left, self.sections[first].right}
                              & {self.sections[second].left, self.sections[second].right})
                    raise BoundaryCollision(first, second, shared)
                junction = min(ends, key=lambda k: float(sphere.arc_km(self.junctions[k], meet)))
                self.junctions[junction] = meet
                self._retrim(junction)
                repairs.append(dict(kind='junction_moved', junction=junction, sections=(first, second), plate=plate))
        raise ValueError('Plate outlines still cross after %d repairs.' % rounds)

    def _retrim(self, junction):
        for ident in self.incident(junction):
            s = self.sections[ident]
            line = self.points(ident)
            if s.start == junction:
                line = _trim(line, self.junctions[junction], from_start=True)
            if s.end == junction:
                line = _trim(line, self.junctions[junction], from_start=False)
            s.interior = line[1:-1]

    def zone(self, ident, half_width_km, matrix=None):
        """A strip either side of a section (optionally in the frame `matrix` carries to the world).

        Area a boundary created or destroyed in one step lies within the plates' displacement of it,
        so a strip a little wider than that attributes it to this section. The strip is the union of
        one quadrilateral per segment, so a kink in the section cannot make it invalid.
        """
        points = self.points(ident)
        if matrix is not None:
            points = sphere.rotate(points, np.asarray(matrix).T)
        angle = half_width_km / sphere.RADIUS_KM
        length = np.r_[0., np.cumsum(sphere.arc_km(points[:-1], points[1:]))]
        # The strip only has to cover the ground near the section, so it follows a coarsened copy
        # of it (points about two strip widths apart): kinks at junctions would make a faithful
        # offset cross itself.
        for spacing in (2. * half_width_km, 4. * half_width_km, 8. * half_width_km):
            marks = np.unique(np.r_[np.searchsorted(length, np.arange(0., length[-1], spacing)), len(points) - 1])
            coarse = points[marks]
            if len(coarse) < 2:
                continue
            segment = sphere.unit(np.cross(coarse[:-1], coarse[1:]))
            normal = sphere.unit(np.vstack((segment[:1], segment[:-1] + segment[1:], segment[-1:])))
            left = sphere.unit(coarse * np.cos(angle) + normal * np.sin(angle))
            right = sphere.unit(coarse * np.cos(angle) - normal * np.sin(angle))
            ring = np.vstack((right, left[::-1]))
            try:
                strip = sphere.polygon(ring)
            except (ValueError, RuntimeError):
                continue
            if sphere.area_km2(strip) > .5 * sphere.SPHERE_KM2:
                strip = sphere.polygon(ring[::-1])
            return strip
        quads = []
        for a, b in zip(points[:-1], points[1:]):
            if float(sphere.arc_km(a, b)) < 1e-6:
                continue
            normal = sphere.unit(np.cross(a, b))
            offsets = [sphere.unit(p * np.cos(angle) + side * normal * np.sin(angle)) for p in (a, b) for side in (-1., 1.)]
            # a-right, b-right, b-left, a-left: counter-clockwise, the segment's strip on its left.
            quads.append(sphere.polygon(np.array([offsets[0], offsets[2], offsets[3], offsets[1]])))
        return sphere.union(*quads)

    def tidy(self, spacing_km=100., minimum_km=10.):
        """Keep sections well formed after a move: no doubling back or crowding at their ends, and
        no segment longer than `spacing_km` (great-circle points are inserted)."""
        from poly.build import densify
        for ident, s in self.sections.items():
            points = self.points(ident)
            keep = [points[0]]
            for k in range(1, len(points) - 1):
                p = points[k]
                if float(sphere.arc_km(p, keep[-1])) < minimum_km or float(sphere.arc_km(p, points[-1])) < minimum_km:
                    continue
                if len(keep) >= 2:
                    before = sphere.unit(np.cross(np.cross(keep[-2], keep[-1]), keep[-1]))
                    after = sphere.unit(np.cross(np.cross(keep[-1], p), keep[-1]))
                    if float(before @ after) < 0.:
                        continue
                keep.append(p)
            keep.append(points[-1])
            line = _cut_loops(np.array(keep))
            line = densify(line, spacing_km) if len(line) > 1 else line
            s.interior = line[1:-1]

    def polygon(self, plate, matrix=None):
        """The plate's region: rings that enclose each other form one face, faces are joined.

        With `matrix`, the region is returned in the frame that `matrix` carries to the world (the
        rings are rotated by its transpose): a plate's own frame, where its crust is stored.
        """
        rings = self.rings(plate)
        if matrix is not None:
            rings = [sphere.rotate(r, np.asarray(matrix).T) for r in rings]
        regions = [sphere.polygon(r) for r in rings]
        inside = [_interior_point(r) for r in rings]
        faces, assigned = [], set()
        for i in range(len(rings)):
            if i in assigned:
                continue
            group = [i] + [j for j in range(len(rings)) if j != i and j not in assigned
                           and _contains(regions[i], inside[j]) and _contains(regions[j], inside[i])]
            assigned.update(group)
            face = regions[group[0]]
            for j in group[1:]:
                face = sphere.intersection(face, regions[j])
            faces.append(face)
        return sphere.union(*faces)

    def polygons(self) -> dict:
        return {plate: self.polygon(plate) for plate in self.plates()}

    def kind_between(self, a, b) -> list:
        return [ident for ident, s in self.sections.items() if {s.left, s.right} == {a, b}]


def _turn(a, b, c) -> float:
    """Signed turn at b from a->b to b->c (positive = left), radians."""
    t1 = sphere.unit(np.cross(np.cross(a, b), b))
    t2 = sphere.unit(np.cross(b, np.cross(b, c)))
    return float(np.arctan2(np.dot(np.cross(t1, t2), b), np.dot(t1, t2)))


def _interior_point(ring) -> np.ndarray:
    """A point just to the left of the ring's longest segment."""
    nxt = np.roll(ring, -1, axis=0)
    k = int(np.argmax(sphere.arc_km(ring, nxt)))
    a, b = ring[k], nxt[k]
    mid = sphere.unit(a + b)
    # Walking a->b along a great circle, the left side is towards that circle's pole a x b.
    offset = min(1e-3, .25 * float(sphere.arc_km(a, b)) / sphere.RADIUS_KM)
    return sphere.unit(mid * np.cos(offset) + sphere.unit(np.cross(a, b)) * np.sin(offset))


def _contains(region, point) -> bool:
    import spherely
    lon, lat = sphere.to_lonlat(point)
    return bool(spherely.contains(region, spherely.create_point(float(lon), float(lat))))


def _on_arc(p, a, b, tolerance=1e-12) -> bool:
    pole = np.cross(a, b)
    return float(np.dot(np.cross(a, p), pole)) >= -tolerance and float(np.dot(np.cross(p, b), pole)) >= -tolerance


def _trim(line, junction, from_start, window=15):
    """Make `junction` the polyline's end: project it onto the line near that end and drop the
    points before the projection. A line that overshoots the junction is cut back; one that falls
    short gains the segment to it."""
    line = line if from_start else line[::-1]
    best_k, best_km = 0, np.inf
    for k in range(min(window, len(line) - 1)):
        a, b = line[k], line[k + 1]
        km = _segment_km(junction, a, b)
        if km < best_km:
            best_k, best_km = k, km
    rest = line[best_k + 1:]
    if len(rest) == 0:
        rest = line[-1:]
    result = np.vstack((junction[None], rest))
    return result if from_start else result[::-1]


def _segment_km(point, a, b) -> float:
    """Distance from a point to a great-circle segment, km."""
    pole = np.cross(a, b)
    norm = float(np.linalg.norm(pole))
    if norm > 1e-15:
        pole = pole / norm
        foot = sphere.unit(point - (point @ pole) * pole)
        if _on_arc(foot, a, b):
            return float(sphere.arc_km(point, foot))
    return float(min(sphere.arc_km(point, a), sphere.arc_km(point, b)))


def _junction_point(tips, nexts, weights=None, hold=1e-2):
    """The point nearest every section's moved end line, held towards the ends' mean.

    Each section near the junction is the great circle through its moved end and the next point;
    the junction is the least-squares point on all of them. Where they barely constrain it (nearly
    parallel lines, as at a change of kind along one boundary) `hold` keeps it at the mean.
    """
    centre = sphere.unit(tips.mean(axis=0))
    east = sphere.unit(np.cross([0., 0., 1.], centre) if abs(centre[2]) < .99 else np.cross([1., 0., 0.], centre))
    north = np.cross(centre, east)
    rows, rhs = [], []
    weights = np.ones(len(tips)) if weights is None else np.asarray(weights, float)
    for a, b, w in zip(tips, nexts, weights):
        pole = np.cross(a, b)
        norm = float(np.linalg.norm(pole))
        if norm < 1e-15:
            continue
        pole /= norm
        rows.append([w * (pole @ east), w * (pole @ north)])
        rhs.append(-w * (pole @ centre))
    rows += [[hold ** .5, 0.], [0., hold ** .5]]
    rhs += [0., 0.]
    shift = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    return sphere.unit(centre + shift[0] * east + shift[1] * north)


def _first_crossing(line_a, line_b, segments=12, tolerance_km=1e-3):
    """Where two polylines leaving the same junction first cross (away from the junction), or None."""
    junction = line_a[0]
    best, best_km = None, np.inf
    for a, b in zip(line_a[:segments], line_a[1:segments + 1]):
        for c, d in zip(line_b[:segments], line_b[1:segments + 1]):
            line = np.cross(np.cross(a, b), np.cross(c, d))
            norm = float(np.linalg.norm(line))
            if norm < 1e-12:
                continue
            for point in (line / norm, -line / norm):
                if not (_on_arc(point, a, b) and _on_arc(point, c, d)):
                    continue
                km = float(sphere.arc_km(point, junction))
                if tolerance_km < km < best_km:
                    best, best_km = point, km
    return best


def _cut_loops(line, window=24):
    """Remove small loops where a polyline crosses itself: the loop is replaced by the crossing."""
    points = [p for p in line]
    i = 0
    while i < len(points) - 3:
        cut = False
        for j in range(i + 2, min(len(points) - 1, i + window)):
            a, b, c, d = points[i], points[i + 1], points[j], points[j + 1]
            pole_ab, pole_cd = np.cross(a, b), np.cross(c, d)
            meet = np.cross(pole_ab, pole_cd)
            norm = float(np.linalg.norm(meet))
            if norm < 1e-15:
                continue
            for x in (meet / norm, -meet / norm):
                if _on_arc(x, a, b, 0.) and _on_arc(x, c, d, 0.) and                         min(float(sphere.arc_km(x, q)) for q in (a, b, c, d)) > 1e-6:
                    points = points[:i + 1] + [x] + points[j + 1:]
                    cut = True
                    break
            if cut:
                break
        if not cut:
            i += 1
    return np.array(points)


def segment_motion(points, left_omega, right_omega):
    """Per segment of a polyline: opening across it (positive apart) and sliding along it, km/Myr.

    The right plate's velocity relative to the left one, at each segment's middle, resolved across
    and along the segment. Walking the polyline, the right plate is on the right.
    """
    points = np.asarray(points, float)
    a, b = points[:-1], points[1:]
    mid = sphere.unit(a + b)
    tangent = sphere.unit(b - a)
    toward_right = np.cross(tangent, mid)
    velocity = np.cross(np.asarray(right_omega, float) - np.asarray(left_omega, float), mid) * sphere.RADIUS_KM
    return np.einsum('ij,ij->i', velocity, toward_right), np.abs(np.einsum('ij,ij->i', velocity, tangent))
