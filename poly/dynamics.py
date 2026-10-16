"""Plate motions from forces on the polygons and boundary lines (a linear torque balance).

Plates have no inertia: each step, every plate's rotation is the one at which the torques on it
balance. The torques are

- slab pull along trenches, on the downgoing plate, towards the trench (the slab's weight,
  transmitted to the plate at the trench);
- ridge push along ridges, on both plates, away from the ridge;
- basal drag over the plate's area, opposing its motion over the mantle;
- resistance along every boundary, opposing the plates' relative motion across it and along it,
  with a strength set by the boundary's kind: weak at ridges, moderate at trenches and transforms,
  strong where plates converge without a trench, strongest in collisions.

Drag and resistance are linear in the rotations, so the balance is one symmetric linear system.
Units are SI inside (N, m, s); rotations go in and out in rad/Myr.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import spherely

from poly import sphere

SECONDS_PER_MYR = 3.15576e13
R_M = sphere.RADIUS_KM * 1e3


@dataclass
class Parameters:
    slab_pull_n_per_m: float = 3e13   # per metre of trench, for a slab of `slab_reference_myr`
    # A slab's pull grows with the thickness of the lithosphere going down, which grows with the
    # square root of its age: pull = slab_pull_n_per_m * sqrt(age / reference), capped. Ocean that
    # is young when it reaches a trench (back-arc, a trench that has only just started) pulls little.
    slab_reference_myr: float = 80.
    slab_pull_cap: float = 1.25
    ridge_push_n_per_m: float = 3e12
    # Set so the Earth-like world moves like Earth: with 1e15 the rms plate speed settled at 65-80
    # km/Myr and the fastest plate at 100-190 (Earth: about 40 and 100); speed goes as drag to the
    # -0.55, and 3e15 gives 41-49 rms and 83-106 for the fastest plate from 10 Myr on.
    basal_drag_pa_s_per_m: float = 3e15
    # Boundary resistance, N per m of boundary per m/s of relative motion (across, along). Across a
    # trench it stands for the slab's own resistance to sinking: a slab-pulled plate then moves at
    # about pull / resistance (3e13 / 1.2e22 m/s = 8 cm/yr) whatever its size, as on Earth.
    resistance: dict = field(default_factory=lambda: dict(
        ridge=(1e19, 1e19), transform=(1e21, 1e21), trench=(1.2e22, 1e21),
        converging=(2e22, 1e21), collision=(2e23, 2e22)))
    # No plate moves further than this in one sub-step; a step is split to keep to it.
    step_limit_km: float = 50.
    sample_points: int = 20000


def _fibonacci(count):
    k = np.arange(count) + .5
    z = 1. - 2. * k / count
    phi = math.pi * (1. + math.sqrt(5.)) * k
    r = np.sqrt(1. - z * z)
    return np.column_stack((r * np.cos(phi), r * np.sin(phi), z))


class Balance:
    """The torque balance of a world's plates under `Parameters`."""

    def __init__(self, parameters: Parameters | None = None):
        self.parameters = parameters or Parameters()
        self.points = _fibonacci(self.parameters.sample_points)
        self.point_area_m2 = 4. * math.pi * R_M ** 2 / len(self.points)
        lon, lat = sphere.to_lonlat(self.points)
        self.geographies = spherely.points(lon, lat)

    def solve(self, world) -> dict:
        """Rotation of every plate (rad/Myr) at which the torques balance; also stores the parts."""
        p = self.parameters
        plates = world.plates() if hasattr(world, 'segments') else world.network.plates()
        slot = {plate: k for k, plate in enumerate(plates)}
        size = 3 * len(plates)
        matrix = np.zeros((size, size))
        torque = np.zeros(size)
        # Basal drag: F = -c v over the area, v = w x r R, torque = R r x F = -c R^2 (I - r r^T) w.
        regions = world.world_regions() if hasattr(world, 'world_regions') else world.network.polygons()
        drag = {}
        for plate, region in regions.items():
            spherely.prepare(region)
            inside = np.asarray(spherely.contains(region, self.geographies))
            r = self.points[inside]
            moment = np.eye(3) * len(r) - r.T @ r
            block = p.basal_drag_pa_s_per_m * R_M ** 2 * self.point_area_m2 * moment
            k = 3 * slot[plate]
            matrix[k:k + 3, k:k + 3] += block
            drag[plate] = block
        # Boundaries: every segment at once.
        seg = _segment_arrays(world, slot, p)
        count = len(plates)
        big = np.zeros((count, count, 3, 3))
        for i, j, sign in ((seg.right, seg.right, 1.), (seg.left, seg.left, 1.),
                           (seg.right, seg.left, -1.), (seg.left, seg.right, -1.)):
            np.add.at(big, (i, j), sign * seg.block)
        matrix += big.transpose(0, 2, 1, 3).reshape(size, size)
        pull = np.zeros((count, 3))
        np.add.at(pull, seg.down[seg.trench], seg.slab_pull[seg.trench])
        np.add.at(pull, seg.right[seg.ridge], seg.ridge_push[seg.ridge])
        np.add.at(pull, seg.left[seg.ridge], -seg.ridge_push[seg.ridge])
        torque += pull.ravel()
        pulled_km = np.bincount(seg.down[seg.trench], seg.length_m[seg.trench] / 1e3, minlength=count)
        pulled = {plate: float(pulled_km[k]) for plate, k in slot.items() if pulled_km[k] > 0.}
        omega = np.linalg.solve(matrix, torque)  # rad/s
        rotations = {plate: omega[3 * slot[plate]:3 * slot[plate] + 3] * SECONDS_PER_MYR for plate in plates}
        self.last = dict(pulled_km=pulled, speed_km_myr={
            plate: float(np.linalg.norm(w)) * sphere.RADIUS_KM for plate, w in rotations.items()})
        return rotations


@dataclass
class Rifting:
    """When a plate splits: where the pull across a straight cut beats the strength along it."""
    # Strength of the lithosphere against being pulled apart, N per m of cut. Continental
    # lithosphere is the weaker, so continents are where plates rift.
    ocean_strength_n_per_m: float = 1.6e13
    # On the Earth-like world (30% continent) 1e12 gave a continental rift every 5 Myr, two to
    # four times Earth's rate; 3e12 keeps the strongest-pulled ones and loses the marginal ones.
    continent_strength_n_per_m: float = 3e12
    suture_strength_n_per_m: float = 3e11   # an old suture is the weakest lithosphere there is
    suture_reach_km: float = 150.           # a cut this close to a suture line follows it
    minimum_plate_km2: float = 20e6   # only plates this large are tested
    minimum_part_km2: float = 3e6     # and only cuts leaving both parts this large
    azimuths: int = 12                # cut directions tried through each centre
    spacing_km: float = 50.           # sampling along a cut


@dataclass
class Cut:
    plate: int
    centre: np.ndarray        # a point the cut passes through
    pole: np.ndarray          # the cut is the great circle normal to this; the new plate is on its + side
    length_km: float
    tension_n_per_m: float    # mean pull across the cut (positive: being pulled apart)
    strength_n_per_m: float   # mean strength along it
    continent_fraction: float

    @property
    def ratio(self) -> float:
        return self.tension_n_per_m / self.strength_n_per_m


@dataclass
class Segments:
    """Every boundary segment of a world as arrays, with the resistance block and drivers of each."""
    left: np.ndarray        # plate slots
    right: np.ndarray
    mid: np.ndarray         # (N, 3) unit midpoints
    normal: np.ndarray      # (N, 3) towards the right plate
    length_m: np.ndarray
    block: np.ndarray       # (N, 3, 3): torque on the right plate per unit relative rotation
    trench: np.ndarray      # bool
    ridge: np.ndarray       # bool
    down: np.ndarray        # the downgoing plate's slot (trench segments; others arbitrary)
    slab_pull: np.ndarray   # (N, 3) torque from slab pull on the downgoing plate (trench segments)
    ridge_push: np.ndarray  # (N, 3) torque from ridge push on the right plate (ridge segments)


def _segment_arrays(world, slot, p) -> Segments:
    rows = [r for r in _segments(world)]
    if not rows:
        empty = np.zeros((0, 3))
        return Segments(np.zeros(0, int), np.zeros(0, int), empty, empty, np.zeros(0), np.zeros((0, 3, 3)),
                        np.zeros(0, bool), np.zeros(0, bool), np.zeros(0, int), empty, empty)
    left = np.array([slot[r[0]] for r in rows])
    right = np.array([slot[r[1]] for r in rows])
    a = np.array([r[2] for r in rows], float)
    b = np.array([r[3] for r in rows], float)
    kind = np.array([r[4] for r in rows])
    over = [r[5] for r in rows]
    age = np.array([p.slab_reference_myr if len(r) < 7 or r[6] is None else r[6] for r in rows], float)
    length_m = sphere.arc_km(a, b) * 1e3
    keep = length_m > 0.
    left, right, a, b, kind, length_m = left[keep], right[keep], a[keep], b[keep], kind[keep], length_m[keep]
    over = [o for o, k in zip(over, keep) if k]
    age = age[keep]
    m = sphere.unit(a + b)
    t_ = sphere.unit(b - a)
    n = np.cross(t_, m)
    normal_c = np.array([p.resistance[k][0] for k in kind])
    along_c = np.array([p.resistance[k][1] for k in kind])
    # Relative velocity of right over left at m: (w_R - w_L) x m R. Resistance on the right plate
    # F = -L (c_n n n^T + c_t t t^T) v, equal and opposite on the left; torque R m x F.
    g = length_m[:, None, None] * (normal_c[:, None, None] * n[:, :, None] * n[:, None, :]
                                   + along_c[:, None, None] * t_[:, :, None] * t_[:, None, :])
    cross = np.zeros((len(m), 3, 3))
    cross[:, 0, 1], cross[:, 0, 2] = -m[:, 2], m[:, 1]
    cross[:, 1, 0], cross[:, 1, 2] = m[:, 2], -m[:, 0]
    cross[:, 2, 0], cross[:, 2, 1] = -m[:, 1], m[:, 0]
    block = R_M ** 2 * cross @ g @ cross.transpose(0, 2, 1)
    trench = kind == 'trench'
    ridge = kind == 'ridge'
    # The overriding plate is named by id; the downgoing one is the other.
    over_is_left = np.array([o is not None and slot.get(o) == l for o, l in zip(over, left)])
    down = np.where(over_is_left, right, left)
    towards_overriding = np.where((down == right)[:, None], -n, n)
    strength = np.minimum(np.sqrt(np.maximum(age, 0.) / p.slab_reference_myr), p.slab_pull_cap)
    slab_pull = R_M * np.cross(m, (p.slab_pull_n_per_m * strength * length_m)[:, None] * towards_overriding)
    ridge_push = R_M * np.cross(m, p.ridge_push_n_per_m * length_m[:, None] * n)
    return Segments(left, right, m, n, length_m, block, trench, ridge, down, slab_pull, ridge_push)


def plate_torques(balance, world, rotations, plate):
    """The external torques (N m) on `plate`'s boundary segments under `rotations` (rad/Myr).

    Returns (segment midpoints, torques): drivers (slab pull, ridge push) and the resistance of its
    neighbours, as in `Balance.solve`, segment by segment.
    """
    plates = world.plates()
    slot = {q: k for k, q in enumerate(plates)}
    seg = _segment_arrays(world, slot, balance.parameters)
    w = np.array([np.asarray(rotations[q], float) / SECONDS_PER_MYR for q in plates])
    me = slot[plate]
    mine = (seg.left == me) | (seg.right == me)
    relative = np.einsum('nij,nj->ni', seg.block[mine], w[seg.right[mine]] - w[seg.left[mine]])
    torque = np.where((seg.right[mine] == me)[:, None], -relative, relative)
    pulled = mine & seg.trench & (seg.down == me)
    torque[pulled[mine]] += seg.slab_pull[pulled]
    pushed = mine & seg.ridge
    sign = np.where(seg.right[pushed] == me, 1., -1.)
    torque[pushed[mine]] += sign[:, None] * seg.ridge_push[pushed]
    return seg.mid[mine], torque


CUT_CENTRE_KM2 = 1e5  # a continent piece this large seeds cuts through its centre; islands do not


def cuts(balance, world, rotations, plate, rifting=None):
    """Straight cuts across `plate` through its centre and its continents, with the pull across each.

    The plate moves rigidly, so each side of a cut needs the other to pull or push it along: the
    external torque on one side (drivers, neighbours' resistance, basal drag) is what the cut must
    carry. That torque, fitted as an even pull across the cut, is its tension.
    """
    rifting = rifting or Rifting()
    p = balance.parameters
    region = world.world_regions()[plate]
    if sphere.area_km2(region) < rifting.minimum_plate_km2:
        return []
    spherely.prepare(region)
    inside = np.asarray(spherely.contains(region, balance.geographies))
    points = balance.points[inside]
    omega = np.asarray(rotations[plate], float) / SECONDS_PER_MYR
    mids, torques = plate_torques(balance, world, rotations, plate)
    drag = p.basal_drag_pa_s_per_m * R_M ** 2 * balance.point_area_m2
    frame = world.frame(plate)
    continents = [c for c in world.crust if c.plate == plate and c.kind == 'continent']
    shapes = [sphere.rotate_geography(c.shape, frame) for c in continents]  # once, not once per cut
    for shape in shapes:
        spherely.prepare(shape)
    centres = [_centre(region)] + [sphere.rotate(_centre(c.shape), frame) for c in continents
                                   if c.area_km2() >= CUT_CENTRE_KM2]
    # Old sutures on the plate: a cut is tried along the great circle that fits each, and any cut
    # within reach of a suture is as weak as the suture there.
    sutures = [line for line, _ in getattr(world, 'suture_lines', lambda: [])()
               if spherely.contains(region, spherely.create_point(*map(float, sphere.to_lonlat(sphere.unit(line.mean(axis=0))))))]
    along = []
    for line in sutures:
        if len(line) < 2:
            continue
        _, vectors = np.linalg.eigh(line.T @ line)
        along.append((sphere.unit(line.mean(axis=0)), sphere.unit(vectors[:, 0])))  # centre, pole of the fit
    suture_points = np.vstack(sutures) if sutures else np.zeros((0, 3))
    step = rifting.spacing_km / sphere.RADIUS_KM
    point_km2 = sphere.SPHERE_KM2 / len(balance.points)
    result = []
    tried = []
    for centre in centres:
        e1 = sphere.unit(np.cross(centre, [0., 0., 1.]) if abs(centre[2]) < .9 else np.cross(centre, [1., 0., 0.]))
        e2 = np.cross(centre, e1)
        for k in range(rifting.azimuths):
            angle = math.pi * k / rifting.azimuths
            tried.append((centre, math.cos(angle) * e1 + math.sin(angle) * e2))
    tried += along
    for centre, pole in tried:
        if True:
            side = points @ pole > 0.
            if min(side.sum(), (~side).sum()) * point_km2 < rifting.minimum_part_km2:
                continue
            # The cut: the great circle through the centre normal to the pole, where it is inside the plate.
            u = np.cross(pole, centre)
            t = np.arange(0., 2. * math.pi, step)
            line = np.outer(np.cos(t), centre) + np.outer(np.sin(t), u)
            lon, lat = sphere.to_lonlat(line)
            on = np.asarray(spherely.contains(region, spherely.points(lon, lat)))
            if on.sum() < 2:
                continue
            line = line[on]
            length_m = len(line) * rifting.spacing_km * 1e3
            # External torque on the minus side (the side that stays the old plate).
            a = points[~side]
            block = drag * (np.eye(3) * len(a) - a.T @ a)
            torque = torques[mids @ pole < 0.].sum(axis=0) - block @ omega
            arm = np.cross(line.sum(axis=0) * rifting.spacing_km * 1e3, pole)
            # The plus side holds the minus side with an even pull f along +pole: R f (S x pole) = -torque.
            tension = float(-torque @ arm / (R_M * (arm @ arm))) if arm @ arm > 0. else 0.
            continental = _in_any(line, shapes)
            fraction = float(continental.mean())
            each = np.where(continental, rifting.continent_strength_n_per_m, rifting.ocean_strength_n_per_m)
            if len(suture_points):
                near = sphere.RADIUS_KM * np.arccos(np.clip(line @ suture_points.T, -1., 1.)).min(axis=1)
                each = np.where(near <= rifting.suture_reach_km, rifting.suture_strength_n_per_m, each)
            strength = float(each.mean())
            result.append(Cut(plate, centre, pole, length_m / 1e3, tension, strength, fraction))
    return result


def _centre(geography):
    c = spherely.centroid(geography)
    x, y = spherely.get_x(c), spherely.get_y(c)
    if not (np.isfinite(x) and np.isfinite(y)):
        return np.array([1., 0., 0.])  # the whole sphere has no centre; any point will do
    return sphere.from_lonlat(x, y)


def _in_any(points, geographies):
    lon, lat = sphere.to_lonlat(points)
    where = spherely.points(lon, lat)
    hit = np.zeros(len(points), bool)
    for g in geographies:
        hit |= np.asarray(spherely.contains(g, where))
    return hit


def _segments(world):
    """(left, right, a, b, kind, overriding) for every boundary segment of either kind of world."""
    if hasattr(world, 'segments'):
        yield from world.segments()
        return
    for ident, s in world.network.sections.items():
        points = world.network.points(ident)
        for a, b in zip(points[:-1], points[1:]):
            yield s.left, s.right, a, b, s.kind, s.overriding


def run(world, steps, parameters=None, dt=1., report=None, rifting=None):
    """Step `world` `steps` times with motions from the force balance; returns the balance.

    Each step is split into sub-steps short enough that no plate moves further than
    `step_limit_km`, and the forces are solved again before each. At the start of each step a world
    that can rift (`world.rift`) splits the plate whose best cut is pulled hardest beyond its
    strength (`rifting`, default `Rifting()`).
    """
    balance = Balance(parameters)
    if rifting is None and hasattr(world, 'rift'):
        rifting = Rifting()
    for _ in range(steps):
        remaining = dt
        rifted = []
        if rifting is not None:
            rotations = balance.solve(world)
            found = [c for plate in world.plates() for c in cuts(balance, world, rotations, plate, rifting)]
            best = max(found, key=lambda c: c.ratio, default=None)
            if best is not None and best.ratio > 1.:
                rifted.append(world.rift(best))
        while remaining > 1e-9:
            rotations = balance.solve(world)
            fastest = max(balance.last['speed_km_myr'].values())
            sub = min(remaining, balance.parameters.step_limit_km / max(fastest, 1e-9))
            entry = world.step(rotations, sub)
            entry['events'] = rifted + entry['events']
            rifted = []
            entry['speed_km_myr'] = balance.last['speed_km_myr']
            entry['substep_myr'] = sub
            remaining -= sub
            if report:
                report(world, entry)
    return balance
