"""Plates, crust polygons and the closed crust ledger, stepped under given plate rotations.

Every plate is tiled by crust pieces: continental polygons and ocean polygons, the latter with the
time they were made. As in GPlates, a piece is stored in its plate's own frame together with the
plate's accumulated rotation, so crust is never rotated or rebuilt; only the boundary network, which
is held as plain arrays, is rotated into a plate's frame to clip against. Each step:

1. the boundary network moves (see topology) and each plate's region is rebuilt from it;
2. every crust piece is clipped to its plate's new region (in the plate's frame);
3. crust left outside its plate is booked by the boundary it crossed: ocean going down a trench is
   subducted; continental crust at a trench, overlap at a collision and overlap at a converging
   boundary are booked as such, never deleted unrecorded;
4. plate area not covered by its crust is new ocean, booked by the boundary it opened at: a ridge, a
   trench (opening behind it), or a sliver at a transform; anything near no boundary is a junction.

The step checks that the plates tile the sphere, that each plate's crust tiles the plate, and that
the area created equals the area destroyed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from poly import sphere
from poly.topology import Network

GAINS = ('ridge', 'trench_opening', 'sliver', 'junction')
LOSSES = ('subducted', 'continent_at_trench', 'overriding_edge', 'collision', 'compression', 'sliver', 'junction')
TOLERANCE = 1e-9  # of the sphere's area
ZONE_MARGIN_KM = 25.
# A converging boundary becomes a trench once it has closed this far: about 100 km of convergence
# starts induced subduction (Gurnis et al. 2004 and later models).
INITIATION_KM = 100.
# A trench where at least this much continental crust arrives in one step becomes a collision.
COLLISION_KM2 = 500.
# How far either side of a boundary its crust is sampled to choose which plate goes down.
SAMPLE_KM = 150.


@dataclass
class Crust:
    plate: int
    kind: str      # 'ocean' or 'continent'
    shape: object  # in the plate's own frame
    born_myr: float | None = None

    def area_km2(self) -> float:
        return sphere.area_km2(self.shape)


@dataclass
class World:
    network: Network
    crust: list
    frames: dict = field(default_factory=dict)  # plate -> rotation from its own frame to the world
    names: dict = field(default_factory=dict)
    time_myr: float = 0.
    ledger: list = field(default_factory=list)
    overriding_rule: object = staticmethod(lambda left, right: max(left, right))

    def frame(self, plate):
        return self.frames.setdefault(plate, np.eye(3))

    def world_shape(self, piece):
        """A crust piece's polygons in the world frame, as rings (for maps and export)."""
        matrix = self.frame(piece.plate)
        return [[sphere.rotate(ring, matrix) for ring in rings] for rings in sphere.polygons(piece.shape)]

    def plate_polygons(self) -> dict:
        return self.network.polygons()

    def step(self, rotations_per_myr: dict, dt: float = 1.) -> dict:
        """Advance by `dt` Myr; `rotations_per_myr` maps plate -> rotation vector (rad/Myr)."""
        rotations = {p: np.asarray(v, float) * dt for p, v in rotations_per_myr.items()}
        for plate, vector in rotations.items():
            self.frames[plate] = sphere.rotation(vector) @ self.frame(plate)
        speed = max(float(np.linalg.norm(v)) for v in rotations.values()) * sphere.RADIUS_KM
        half_width = 2. * speed + ZONE_MARGIN_KM
        misfit = self.network.move(rotations)
        self.network.tidy()
        events = self.network.reclassify(rotations_per_myr)
        # A section can shrink by both its ends' displacement in a step; catch it well before zero.
        events += self.network.resolve_short(rotations_per_myr, 2. * speed + 50., self.overriding_rule)
        repairs = self.network.repair()
        plates = self.network.plates()
        entry = dict(time_myr=self.time_myr + dt, gained={k: 0. for k in GAINS}, lost={k: 0. for k in LOSSES},
                     subducted_by_section={}, continent_at_trench_by_section={},
                     junction_misfit_km=max(misfit.values(), default=0.), events=events, repairs=repairs)
        world_regions = {p: self.network.polygon(p) for p in plates}
        self._zones, self._regions = {}, {}
        crust = []
        for plate in plates:
            matrix = self.frame(plate)
            region = self.network.polygon(plate, matrix)
            mine = [piece for piece in self.crust if piece.plate == plate]
            kept_shapes = []
            for piece in mine:
                kept = sphere.intersection(piece.shape, region)
                lost = sphere.difference(piece.shape, region)
                if sphere.area_km2(kept) > 0.:
                    crust.append(Crust(plate, piece.kind, kept, piece.born_myr))
                    kept_shapes.append(kept)
                if sphere.area_km2(lost) > 0.:
                    self._book_loss(entry, plate, piece.kind, lost, matrix, half_width)
            new = sphere.difference(region, sphere.union(*kept_shapes)) if kept_shapes else region
            if sphere.area_km2(new) > 0.:
                self._book_gain(entry, plate, new, matrix, half_width)
                crust.append(Crust(plate, 'ocean', new, self.time_myr + .5 * dt))
        self.crust = _merge_bands(crust)
        # Keep stored crust to polygons only: S2 leaves zero-width lines and debris where edges
        # coincide, which would otherwise ride along and pile up in every later union.
        entry['debris_km2'], entry['clean_failures'] = 0., 0
        for piece in self.crust:
            piece.shape, debris, rebuilt = sphere.clean(piece.shape)
            entry['debris_km2'] += debris
            entry['clean_failures'] += not rebuilt
        self.time_myr += dt
        entry['plates_km2'] = {p: sphere.area_km2(shape) for p, shape in world_regions.items()}
        self._check(entry)
        entry['events'] += self._evolve(rotations_per_myr, dt, entry)
        self.ledger.append(entry)
        return entry

    def _evolve(self, rotations_per_myr, dt, entry) -> list:
        """Boundary changes that take effect from the next step: initiation and collision."""
        from poly.topology import segment_motion
        events = []
        regions = None
        for ident, s in list(self.network.sections.items()):
            if s.kind == 'trench' and entry['continent_at_trench_by_section'].get(ident, 0.) > COLLISION_KM2:
                s.kind, s.overriding = 'collision', None
                events.append(dict(kind='collision', section=ident, plates=(s.left, s.right),
                                   arrived_km2=entry['continent_at_trench_by_section'][ident]))
                continue
            if s.kind != 'converging':
                continue
            points = self.network.points(ident)
            opening, _ = segment_motion(points, rotations_per_myr[s.left], rotations_per_myr[s.right])
            lengths = sphere.arc_km(points[:-1], points[1:])
            closing = float(np.sum(np.maximum(-opening, 0.) * lengths) / max(float(lengths.sum()), 1e-12))
            s.closed_km += closing * dt
            if s.closed_km < INITIATION_KM:
                continue
            if regions is None:
                regions = {p: self.network.polygon(p) for p in self.network.plates()}
            sides = self._sides(ident, regions)
            kinds = {plate: max(set(k for k, _ in samples), key=[k for k, _ in samples].count) if samples else 'ocean'
                     for plate, samples in sides.items()}
            if kinds[s.left] == 'continent' and kinds[s.right] == 'continent':
                s.kind = 'collision'
                events.append(dict(kind='collision', section=ident, plates=(s.left, s.right), closed_km=s.closed_km))
            else:
                if kinds[s.left] != kinds[s.right]:
                    over = s.left if kinds[s.left] == 'continent' else s.right
                    reason = 'ocean under continent'
                else:
                    ages = {plate: np.mean([age for _, age in samples]) if samples else 0. for plate, samples in sides.items()}
                    over = s.left if ages[s.left] < ages[s.right] else s.right
                    reason = 'older ocean under younger'
                s.kind, s.overriding = 'trench', over
                events.append(dict(kind='subduction_initiated', section=ident, plates=(s.left, s.right),
                                   overriding=over, closed_km=s.closed_km, reason=reason))
            s.closed_km = 0.
        return events

    def _sides(self, ident, regions) -> dict:
        """Crust (kind, age) sampled SAMPLE_KM either side of a section, at a quarter, half and three quarters."""
        s = self.network.sections[ident]
        points = self.network.points(ident)
        lengths = np.r_[0., np.cumsum(sphere.arc_km(points[:-1], points[1:]))]
        samples = {s.left: [], s.right: []}
        angle = SAMPLE_KM / sphere.RADIUS_KM
        for fraction in (.25, .5, .75):
            k = min(int(np.searchsorted(lengths, fraction * lengths[-1])), len(points) - 1)
            a, b = points[max(k - 1, 0)], points[min(k, len(points) - 1)]
            if np.allclose(a, b):
                continue
            middle = sphere.unit(a + b)
            left = sphere.unit(np.cross(a, b))  # walking a -> b, the left plate is towards this pole
            for plate, sign in ((s.left, 1.), (s.right, -1.)):
                point = sphere.unit(middle * np.cos(angle) + sign * left * np.sin(angle))
                found = self._crust_at(point, regions)
                if found is not None and found[0] == plate:
                    samples[plate].append(found[1:])
        return samples

    def _crust_at(self, point, regions):
        """(plate, kind, age in Myr) of the crust at a world point, or None."""
        import spherely
        lon, lat = sphere.to_lonlat(point)
        for plate, region in regions.items():
            if not spherely.contains(region, spherely.create_point(float(lon), float(lat))):
                continue
            local = sphere.rotate(point, self.frame(plate).T)
            llon, llat = sphere.to_lonlat(local)
            here = spherely.create_point(float(llon), float(llat))
            for piece in self.crust:
                if piece.plate == plate and spherely.contains(piece.shape, here):
                    age = 0. if piece.born_myr is None else self.time_myr - piece.born_myr
                    return plate, piece.kind, age
            return plate, 'ocean', 0.
        return None

    def _sections_of(self, plate, other=None):
        return [ident for ident, s in self.network.sections.items()
                if plate in (s.left, s.right) and (other is None or other in (s.left, s.right))]

    def _book_gain(self, entry, plate, new, matrix, half_width):
        remaining = new
        for ident in self._sections_of(plate):
            part = sphere.intersection(remaining, self._zone(ident, plate, half_width, matrix))
            area = sphere.area_km2(part)
            if area > 0.:
                kind = self.network.sections[ident].kind
                entry['gained'][{'ridge': 'ridge', 'trench': 'trench_opening'}.get(kind, 'sliver')] += area
                remaining = sphere.difference(remaining, part)
        entry['gained']['junction'] += sphere.area_km2(remaining)

    def _book_loss(self, entry, plate, kind, lost, matrix, half_width):
        """Book crust of `plate` (of `kind`) that left the plate, by the plate it is now inside and the boundary."""
        remaining = lost
        for other in self.network.plates():
            if other == plate:
                continue
            key = (other, plate)
            if key not in self._regions:
                self._regions[key] = self.network.polygon(other, matrix)
            inside = sphere.intersection(remaining, self._regions[key])
            if sphere.area_km2(inside) <= 0.:
                continue
            remaining = sphere.difference(remaining, inside)
            for ident in self._sections_of(plate, other):
                part = sphere.intersection(inside, self._zone(ident, plate, half_width, matrix))
                area = sphere.area_km2(part)
                if area <= 0.:
                    continue
                inside = sphere.difference(inside, part)
                entry['lost'][self._category(entry, plate, kind, ident, area)] += area
            entry['lost']['junction'] += sphere.area_km2(inside)
        entry['lost']['junction'] += sphere.area_km2(remaining)

    def _zone(self, ident, plate, half_width, matrix):
        """A section's booking strip in a plate's frame, built once per step."""
        key = (ident, plate)
        if key not in self._zones:
            self._zones[key] = self.network.zone(ident, half_width, matrix)
        return self._zones[key]

    def _category(self, entry, plate, kind, ident, area):
        section = self.network.sections[ident]
        if section.kind == 'trench':
            if section.overriding == plate:
                return 'overriding_edge'
            if kind == 'ocean':
                entry['subducted_by_section'][ident] = entry['subducted_by_section'].get(ident, 0.) + area
                return 'subducted'
            entry['continent_at_trench_by_section'][ident] = entry['continent_at_trench_by_section'].get(ident, 0.) + area
            return 'continent_at_trench'
        if section.kind in ('collision', 'converging'):
            return 'collision' if kind == 'continent' else 'compression'
        return 'sliver'

    def _check(self, entry):
        entry['tiling_error_km2'] = sum(entry['plates_km2'].values()) - sphere.SPHERE_KM2
        crust = {}
        for piece in self.crust:
            crust[piece.plate] = crust.get(piece.plate, 0.) + piece.area_km2()
        entry['crust_vs_plate_error_km2'] = max((abs(crust.get(p, 0.) - a) for p, a in entry['plates_km2'].items()), default=0.)
        entry['balance_error_km2'] = sum(entry['gained'].values()) - sum(entry['lost'].values())
        scale = TOLERANCE * sphere.SPHERE_KM2
        for key, message in (('tiling_error_km2', 'Plates no longer tile the sphere'),
                             ('crust_vs_plate_error_km2', 'Crust no longer tiles its plate'),
                             ('balance_error_km2', 'Crust created and destroyed do not balance')):
            if abs(entry[key]) > scale:
                raise ValueError('%s: %.3g km2 off.' % (message, entry[key]))

    def areas(self) -> dict:
        """Crust area by kind, km2."""
        result = {}
        for piece in self.crust:
            result[piece.kind] = result.get(piece.kind, 0.) + piece.area_km2()
        return result


AGE_BAND_MYR = 10.


def _merge_bands(crust):
    """Join each plate's ocean pieces made in the same AGE_BAND_MYR band into one piece.

    A piece per step would grow without bound; GPlates likewise keeps ocean floor as polygons between
    isochrons. A merged piece keeps the area-weighted mean of its parts' times of making.
    """
    bands, result = {}, []
    for piece in crust:
        if piece.kind != 'ocean' or piece.born_myr is None:
            result.append(piece)
            continue
        bands.setdefault((piece.plate, int(np.floor(piece.born_myr / AGE_BAND_MYR))), []).append(piece)
    for (plate, _), pieces in sorted(bands.items()):
        if len(pieces) == 1:
            result.append(pieces[0])
            continue
        areas = np.array([p.area_km2() for p in pieces])
        born = float(np.average([p.born_myr for p in pieces], weights=areas)) if areas.sum() > 0 else pieces[0].born_myr
        result.append(Crust(plate, 'ocean', sphere.union(*[p.shape for p in pieces]), born))
    return result
