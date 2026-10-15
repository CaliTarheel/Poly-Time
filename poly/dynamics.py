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
    slab_pull_n_per_m: float = 3e13
    ridge_push_n_per_m: float = 3e12
    basal_drag_pa_s_per_m: float = 1e15
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
        # Boundaries, segment by segment.
        pulled = {}
        for left_plate, right_plate, a, b, kind, overriding in _segments(world):
            mid = sphere.unit(a + b)
            length_m = float(sphere.arc_km(a, b)) * 1e3
            if length_m <= 0.:
                continue
            t_ = sphere.unit(b - a)
            n = np.cross(t_, mid)
            normal_c, along_c = p.resistance[kind]
            left, right = 3 * slot[left_plate], 3 * slot[right_plate]
            # Relative velocity of right over left at m: (w_R - w_L) x m R. Resistance on the right
            # plate F = -L (c_n n n^T + c_t t t^T) v, equal and opposite on the left.
            g = length_m * (normal_c * np.outer(n, n) + along_c * np.outer(t_, t_))
            m = mid
            cross = np.array([[0., -m[2], m[1]], [m[2], 0., -m[0]], [-m[1], m[0], 0.]])
            block = R_M ** 2 * cross @ g @ cross.T
            for i, j, sign in ((right, right, 1.), (left, left, 1.), (right, left, -1.), (left, right, -1.)):
                matrix[i:i + 3, j:j + 3] += sign * block
            if kind == 'trench':
                down = right_plate if overriding == left_plate else left_plate
                direction = -n if down == right_plate else n  # towards the overriding plate
                k = 3 * slot[down]
                torque[k:k + 3] += R_M * np.cross(m, p.slab_pull_n_per_m * length_m * direction)
                pulled[down] = pulled.get(down, 0.) + length_m / 1e3
            elif kind == 'ridge':
                for plate, direction in ((right_plate, n), (left_plate, -n)):
                    k = 3 * slot[plate]
                    torque[k:k + 3] += R_M * np.cross(m, p.ridge_push_n_per_m * length_m * direction)
        omega = np.linalg.solve(matrix, torque)  # rad/s
        rotations = {plate: omega[3 * slot[plate]:3 * slot[plate] + 3] * SECONDS_PER_MYR for plate in plates}
        self.last = dict(pulled_km=pulled, speed_km_myr={
            plate: float(np.linalg.norm(w)) * sphere.RADIUS_KM for plate, w in rotations.items()})
        return rotations


def _segments(world):
    """(left, right, a, b, kind, overriding) for every boundary segment of either kind of world."""
    if hasattr(world, 'segments'):
        yield from world.segments()
        return
    for ident, s in world.network.sections.items():
        points = world.network.points(ident)
        for a, b in zip(points[:-1], points[1:]):
            yield s.left, s.right, a, b, s.kind, s.overriding


def run(world, steps, parameters=None, dt=1., report=None):
    """Step `world` `steps` times with motions from the force balance; returns the balance.

    Each step is split into sub-steps short enough that no plate moves further than
    `step_limit_km`, and the forces are solved again before each.
    """
    balance = Balance(parameters)
    for _ in range(steps):
        remaining = dt
        while remaining > 1e-9:
            rotations = balance.solve(world)
            fastest = max(balance.last['speed_km_myr'].values())
            sub = min(remaining, balance.parameters.step_limit_km / max(fastest, 1e-9))
            entry = world.step(rotations, sub)
            entry['speed_km_myr'] = balance.last['speed_km_myr']
            entry['substep_myr'] = sub
            remaining -= sub
            if report:
                report(world, entry)
    return balance
