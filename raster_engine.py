"""A reproducible, reduced-complexity tectonic planet, using only NumPy.

This is an exploratory geological model, not a reconstruction or a mantle
convection solver. Rigid plates rotate on a sphere. Oceanic lithosphere is an
advected age field; buoyant crust is carried by persistent, area-weighted
Lagrangian parcels, so repeated raster interpolation cannot erase continents.
Boundary velocities drive heuristic subduction, arc addition, continental
shortening, rifting and erosion. Forces set Euler poles with an overdamped
traction balance, rather than solving the three-dimensional Stokes equations.

Distances are km, time Myr, angular velocity rad/Myr, exported relief metres.
One km/Myr is 0.1 cm/yr. The raster is north-to-south, longitude -180 to +180.
Longitude wraps; the north and south poles are never connected to one another.
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Callable

import numpy as np

from fracture import barriers_from_grid, component_labels, make_fracture

from crust_transport import deposit as deposit_crust, footprints as crust_footprints
from plate_topology import plan_domain_repairs, assign_material_components
from material_geometry import patch_centres, splits_protected_groups
from basement_relief import initial_basement_relief
from plate_domains import PersistentDomainTracker
from trench_dynamics import extend_forearcs, retreat_speed
from boundary_geometry import refine_boundary_geometry
from ridge_interaction import update_ridge_interactions, sample_ridge_effects, reassign_ridge_episodes
from rift_inversion import inversion_increment
import backarc
import ridge_spreading
import trench_history
import slab_memory
import normal_partition
import transform_coupling
import local_accretion
import structure_engine
import geology_snapshot
import surface_exposure
import progressive_rifting
import initial_plate_topology

RADIUS_KM = 6371.0
DEFAULT_CONFIG = dict(seed=12, width=512, height=256, duration_myr=1000.0,
                      dt_myr=2.0, snapshot_myr=2.0, plate_count=12,
                      slab_pull=1.0, ridge_push=0.25, erosion=1.0,
                      rift_strength=1.0, mechanics_nodes=1024)
BOUNDARY_NAMES = {0: "none", 1: "spreading ridge", 2: "subduction",
                  3: "transform", 4: "continental collision", 5: "continental rift"}
# Bound coordinate/rotation temporaries independently of the simulation grid.
# This changes allocation size only; every cell follows the same arithmetic.
ADVECTION_CHUNK_CELLS = 65_536


def validate_config(config=None):
    """Return validated, JSON-friendly settings; reject invalid/unknown fields."""
    out = dict(DEFAULT_CONFIG)
    if config is not None:
        if not isinstance(config, dict):
            raise ValueError("Configuration must be an object.")
        unknown = set(config) - set(out)
        if unknown:
            raise ValueError("Unknown configuration fields: " + ", ".join(sorted(unknown)))
        out.update(config)
    bounds = {"seed": (0, 2**32 - 1), "width": (48, 2048), "height": (24, 1024),
              "duration_myr": (1, 1000), "dt_myr": (0.5, 5),
              "snapshot_myr": (0.5, 100), "plate_count": (4, 24),
              "slab_pull": (0, 3), "ridge_push": (0, 3),
              "erosion": (0, 3), "rift_strength": (0, 3), "mechanics_nodes": (128, 8192)}
    ints = {"seed", "width", "height", "plate_count", "mechanics_nodes"}
    for key, (lo, hi) in bounds.items():
        try:
            if isinstance(out[key], bool):
                raise ValueError()
            val = float(out[key])
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f"{key} must be a finite number.") from None
        if not math.isfinite(val) or not lo <= val <= hi:
            raise ValueError(f"{key} must be between {lo} and {hi}.")
        if key in ints and val != int(val):
            raise ValueError(f"{key} must be an integer.")
        out[key] = int(val) if key in ints else val
    if out["snapshot_myr"] < out["dt_myr"]:
        raise ValueError("snapshot_myr must be at least dt_myr.")
    return out


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _xyz(lon, lat):
    c = np.cos(lat)
    return np.stack((c * np.cos(lon), c * np.sin(lon), np.sin(lat)), axis=-1)


def _rotate(points, rotation):
    """Rodrigues rotation, accepting a single or one-per-point rotation vector."""
    angle = np.linalg.norm(rotation, axis=-1, keepdims=True)
    axis = rotation / np.maximum(angle, 1e-20)
    c, s = np.cos(angle), np.sin(angle)
    return points * c + np.cross(axis, points) * s + axis * np.sum(axis * points, axis=-1, keepdims=True) * (1 - c)


def make_initial(config=None, preset='pangaea'):
    """Construct a connected, lobed Pangaea-like landmass with strong cratons.

    This is a synthetic initial condition, not a geographic reconstruction.
    Users may replace its flat raster with any continental/cratonic painting.
    """
    c = validate_config(config)
    if preset == 'land65':
        from initial_worlds import make_land65
        return make_land65(c)
    if preset == 'highland65':
        from initial_worlds import make_highland65
        return make_highland65(c)
    if preset != 'pangaea':
        raise ValueError('Unknown initial preset; choose pangaea, land65 or highland65.')
    w, h = c["width"], c["height"]
    lon, lat = np.meshgrid((np.arange(w) + .5) * 2 * np.pi / w - np.pi,
                           np.pi / 2 - (np.arange(h) + .5) * np.pi / h)
    phase = np.random.default_rng(c["seed"]).uniform(-.3, .3)
    warped_lon = lon + .17 * np.sin(2.4 * lat + phase)
    radius = (warped_lon / 1.31)**2 + (lat / 1.13)**2
    edge = 1 + .13 * np.sin(5 * lon + .4) * np.cos(3 * lat) + .12 * np.cos(7 * lat - 2 * lon)
    crust = (radius < edge).astype(np.uint8)
    for x, y, rx, ry in [(-.69, .41, .23, .24), (.28, .65, .26, .20),
                          (.58, -.09, .27, .28), (-.32, -.57, .31, .24),
                          (-.16, .09, .22, .23)]:
        craton = ((warped_lon - x) / rx)**2 + ((lat - y) / ry)**2 < 1
        crust[craton & (crust > 0)] = 2
    return dict(width=w, height=h, crust=crust.ravel())


class Simulation:
    """An evolving spherical lithosphere, with streamed, independent snapshots.

    ``initial`` is {width, height, crust}, with flat codes 0 ocean, 1 continent,
    2 craton (3 island arc is also accepted). Different input dimensions are
    sampled to the simulation grid. Boundary and plate fields represent both
    oceanic and continental lithosphere; they are not continent outlines.
    """

    def __init__(self, config=None, initial=None):
        self.config = validate_config(config)
        c = self.config
        self.rng = np.random.default_rng(c["seed"])
        self.w, self.h = c["width"], c["height"]
        self.n = self.w * self.h
        self.t = 0.0
        self.steps = 0
        self.lon, self.lat = np.meshgrid((np.arange(self.w) + .5) * 2 * np.pi / self.w - np.pi,
                                       np.pi / 2 - (np.arange(self.h) + .5) * np.pi / self.h)
        self.xyz = _xyz(self.lon.ravel(), self.lat.ravel())
        lat_edges = np.pi / 2 - np.arange(self.h + 1) * np.pi / self.h
        row_area = (np.sin(lat_edges[:-1]) - np.sin(lat_edges[1:])) * 2 * np.pi / self.w * RADIUS_KM**2
        self.cell_area = np.repeat(row_area, self.w)
        self.earth_area = float(self.cell_area.sum())
        idx = np.arange(self.n).reshape(self.h, self.w)
        self.east = np.roll(idx, -1, axis=1).ravel()
        self.west = np.roll(idx, 1, axis=1).ravel()
        self.north = np.concatenate((idx[:1], idx[:-1]), axis=0).ravel()
        self.south = np.concatenate((idx[1:], idx[-1:]), axis=0).ravel()
        self.edge_a = np.concatenate((idx.ravel(), idx[:-1].ravel()))
        self.edge_b = np.concatenate((self.east, idx[1:].ravel()))
        mid = _unit(self.xyz[self.edge_a] + self.xyz[self.edge_b])
        self.edge_mid = mid
        delta = self.xyz[self.edge_b] - self.xyz[self.edge_a]
        self.edge_normal = _unit(delta - mid * np.sum(delta * mid, axis=1)[:, None])
        # Length of the shared cell boundary: latitude extent for E/W, longitude
        # extent at the interface for N/S. This avoids polar over-weighting.
        ew_length = np.full(self.n, np.pi * RADIUS_KM / self.h)
        ns_length = np.repeat(2 * np.pi * RADIUS_KM / self.w * np.cos(lat_edges[1:-1]), self.w)
        self.edge_length = np.concatenate((ew_length, ns_length))
        self.capacity = max(32, c["plate_count"] + 12)
        self.count = c["plate_count"]
        self.split_serial = 0
        self.active = np.zeros(self.capacity, bool)
        self.active[:self.count] = True
        self.omega = np.zeros((self.capacity, 3))
        self.mantle = np.zeros_like(self.omega)
        self.centres = np.zeros_like(self.omega)
        self.polarity = np.full((self.capacity, self.capacity), -1, np.int16)
        self.rift_clock = np.zeros(self.capacity)
        self.ocean_rift_clock = np.zeros(self.capacity)
        self.collision_clock = np.zeros((self.capacity, self.capacity))
        self.born = np.zeros(self.capacity)
        self.names = [f"Plate {p + 1:02d}" for p in range(self.capacity)]
        # Array slots can be reused after an ocean plate vanishes. A UID never
        # is, so historical events and material tracks retain their identity.
        self.plate_uid = np.zeros(self.capacity, np.int64)
        self.plate_uid[:self.count] = np.arange(1, self.count + 1)
        self.plate_generation = self.active.astype(np.int32)
        self.plate_created = np.zeros(self.capacity)
        self.plate_parent_uid = np.zeros(self.capacity, np.int64)
        self.next_plate_uid = self.count + 1
        self.events = []
        self.event_keys = set()
        self.ridge_episodes = []
        self.next_ridge_episode_id = 1
        self.rift_records = []
        self.next_rift_id = 1
        self.rift_pair_ids = {}
        self.backarc_basins = []
        self.next_backarc_id = 1
        self.seed_rift_id = np.full(self.n, -1, np.int64)
        self.seed_rift_tangent = np.zeros((self.n, 3))
        self.process_totals = {"ocean_created_km2": 0., "ocean_consumed_km2": 0.,
                               "arc_added_km2": 0., "accreted_km2": 0.,
                               "rift_events": 0., "collision_events": 0.,
                               "trench_swept_km2": 0., "trench_arc_sweep_km2": 0.,
                               "ridge_subduction_events": 0., "rift_inversion_events": 0.,
                               "backarc_ruptures": 0.}
        self.initial_crust = self._read_initial(initial)
        self._seed_plates()
        # Continuous membership is essential: repeated nearest-cell label
        # sampling would permanently freeze motion slower than half a grid cell.
        self.support = np.zeros((self.capacity, self.n), np.float32)
        self.support[self.plate, np.arange(self.n)] = 1
        self.age = self.rng.uniform(15, 180, self.n).astype(np.float64)
        # Smooth large-scale ocean ages, initially youngest beside the supercontinent.
        self.age = np.clip(30 + 135 * (1 - self.xyz[:, 0]) / 2 + 18 * np.sin(3 * self.lon.ravel()) * np.cos(self.lat.ravel()), 0, 200)
        self.age[self.initial_crust > 0] = 0
        self.ocean_relief = np.zeros(self.n)
        self.arc_budget = np.zeros(self.n)
        self._make_parcels()
        self._make_traces()
        self.original_mass = float(self.mass.sum())
        self.original_craton_mass = float(self.mass[self.kind == 2].sum())
        self._rasterize()
        # A supplied painting is the exact reviewable t=0 state. Subsequent
        # deposition may smooth features below the chosen grid resolution.
        self.crust = self.initial_crust.copy()
        self.age[self.crust > 0] = 0
        self._boundaries()
        mixed = bool(self.initial_plate_topology_diagnostics.get('enabled'))
        ocean_note = (" The independent ocean reservoir begins as one plate; selected oceanic aprons share continental plate motion."
                      if mixed else " The surrounding ocean begins as one plate.") if self.initial_ocean_plate_uid is not None else ""
        self._record("initial", "Initial continental blocks follow curved, irregular fractures around connected cratons."
                     + ocean_note,
                     details={"partition": "recursive curved fractures", "initial_plates": int(self.count),
                              "initial_ocean_plate_uid": self.initial_ocean_plate_uid,
                              "initial_plate_topology": deepcopy(self.initial_plate_topology_diagnostics),
                              "fractures": self.initial_fractures})
        if np.any(self.crust > 0):
            self._record("breakup", "Initial extensional Euler motions load weak continental margins and sutures; continental rifts can open into new ocean.")
        initial_ocean_slots = np.flatnonzero(self.active & (self.plate_uid == self.initial_ocean_plate_uid)) \
            if self.initial_ocean_plate_uid is not None else ()
        self._repair_plate_domains(protected_slots=initial_ocean_slots)
        self._boundaries()
        self.domain_tracker = PersistentDomainTracker(self.w, self.h)
        self._update_surface_domains()
        trench_history.initialize(self)
        progressive_rifting.initialize(self)

    def _read_initial(self, initial):
        if initial is None:
            initial = make_initial(self.config)
        if not isinstance(initial, dict) or "crust" not in initial:
            raise ValueError("Initial condition must contain a flat crust raster.")
        self.initial_plate_topology = initial_plate_topology.normalize(initial.get('initial_plate_topology'))
        try:
            w, h = int(initial.get("width", self.w)), int(initial.get("height", self.h))
            a = np.asarray(initial["crust"])
            if w < 1 or h < 1 or w > 4096 or h > 2048 or a.size != w * h:
                raise ValueError()
            if not np.all(np.isfinite(a)) or not np.all(np.isin(a, [0, 1, 2, 3])):
                raise ValueError()
            a = a.astype(np.uint8).reshape(h, w)
        except (ValueError, TypeError, OverflowError):
            raise ValueError("Initial crust must have width × height finite codes 0, 1, 2 or 3.") from None
        rows = np.minimum(((np.arange(self.h) + .5) * h / self.h).astype(int), h - 1)
        cols = np.minimum(((np.arange(self.w) + .5) * w / self.w).astype(int), w - 1)
        return a[rows[:, None], cols[None, :]].ravel().copy()

    def _seed_plates(self):
        land_mask = self.initial_crust > 0
        land = np.flatnonzero(land_mask)
        ocean = np.flatnonzero(~land_mask)
        requested = self.count
        self.plate = np.zeros(self.n, dtype=np.int16)
        self.initial_fractures = []
        offset = int(bool(len(ocean)))
        target = requested - offset if len(land) else 0
        next_slot = offset
        if len(land):
            # Separate pre-existing landmasses first. Small offshore islands may
            # share a block; no nearest-seed tessellation cuts continental crust.
            labels, count = component_labels(land_mask, self.w, self.h)
            areas = np.bincount(labels[land], weights=self.cell_area[land], minlength=count)
            ordered = np.argsort(-areas, kind='stable')
            owners, centers = {}, []
            for label in ordered[:target]:
                selected = labels == label
                owners[int(label)] = next_slot
                center = _unit(np.sum(self.xyz[selected] * self.cell_area[selected, None], axis=0))
                centers.append(center)
                self.plate[selected] = next_slot
                next_slot += 1
            for label in ordered[target:]:
                selected = labels == label
                center = _unit(np.sum(self.xyz[selected] * self.cell_area[selected, None], axis=0))
                owner = offset + int(np.argmax(np.asarray(centers) @ center))
                self.plate[selected] = owner
            craton_labels, craton_count = component_labels(self.initial_crust == 2, self.w, self.h)
            barriers = barriers_from_grid(self.xyz, self.initial_crust, self.w, self.h, area=self.cell_area)
            exhausted = set()
            while next_slot < requested:
                candidates = [p for p in range(offset, next_slot) if p not in exhausted]
                if not candidates:
                    break
                p = max(candidates, key=lambda p: self.cell_area[(self.plate == p) & land_mask].sum())
                mask = (self.plate == p) & land_mask
                cells = np.flatnonzero(mask)
                weights = self.cell_area[cells]
                center = _unit(np.sum(self.xyz[cells] * weights[:, None], axis=0))
                if np.linalg.norm(center) < .5:
                    center = self.xyz[cells[len(cells)//2]]
                _, old_components = component_labels(mask, self.w, self.h)
                best = None
                for attempt in range(18):
                    v = self.rng.normal(size=3)
                    normal = _unit(v - center * np.dot(v, center))
                    cut = make_fracture(center, normal, int(self.rng.integers(2**32)), barriers)
                    distance = cut.signed_distance(self.xyz[cells])
                    positive = distance > 0
                    fraction = float(weights[positive].sum() / weights.sum())
                    if not .20 < fraction < .80:
                        continue
                    cl = craton_labels[cells]
                    a = np.bincount(cl[positive & (cl >= 0)], minlength=craton_count)
                    b = np.bincount(cl[~positive & (cl >= 0)], minlength=craton_count)
                    if np.any((a > 0) & (b > 0)):
                        continue
                    child = np.zeros(self.n, bool)
                    child[cells[positive]] = True
                    _, count_a = component_labels(child, self.w, self.h)
                    _, count_b = component_labels(mask & ~child, self.w, self.h)
                    if count_a + count_b > old_components + 1:
                        continue
                    score = min(fraction, 1-fraction)
                    if best is None or score > best[0]:
                        best = (score, child, cut)
                if best is None:
                    exhausted.add(p)
                    continue
                _, child, cut = best
                # An accepted starting cut supplies structural inheritance only.
                # Its reservoir stays zero until actual extension is recorded.
                belt = np.zeros(self.n, bool)
                belt[cells[np.abs(cut.signed_distance(self.xyz[cells])) < .09]] = True
                belt &= (self.seed_rift_id < 0) & np.isin(self.initial_crust, (1, 2))
                if np.any(belt):
                    rid = self._new_rift_record('initial fracture', (p, next_slot), cut.center)
                    self.seed_rift_id[belt] = rid
                    self.seed_rift_tangent[belt] = self._rift_tangents(cut, self.xyz[belt])
                self.plate[child] = next_slot
                self.initial_fractures.append({"parent_slot": int(p), "new_slot": int(next_slot),
                                               "center": cut.center.tolist(), "normal": cut.normal.tolist(),
                                               "knots_rad": cut.knots.tolist(), "offsets_rad": cut.offsets.tolist(),
                                               "interpolation": "periodic shape-preserving cubic", "geometry_version": 2})
                next_slot += 1
        self.initial_plate_topology_diagnostics = initial_plate_topology.apply(
            self, self.initial_plate_topology)
        self.count = max(1, next_slot)
        self.active[:] = False
        self.active[:self.count] = True
        self.plate_uid[:] = 0
        self.plate_uid[:self.count] = np.arange(1, self.count + 1)
        self.plate_generation[:] = self.active.astype(np.int32)
        self.next_plate_uid = self.count + 1
        self.initial_ocean_plate_uid = int(self.plate_uid[0]) if len(ocean) else None
        if self.initial_plate_topology_diagnostics.get('enabled'):
            # Passive margins change ownership, not crust: each plate carrying
            # represented water is an explicit initial slab carrier.
            water_slots = np.unique(self.plate[self.initial_crust == 0])
            self.primordial_ocean_carrier_uids = self.plate_uid[water_slots].copy()
            self.primordial_connectivity_version = 1
        for p in range(self.count):
            selected = self.plate == p
            center = _unit(np.sum(self.xyz[selected] * self.cell_area[selected, None], axis=0))
            self.centres[p] = center if np.linalg.norm(center) > .5 else self.xyz[np.flatnonzero(selected)[0]]
            self.names[p] = 'Primordial ocean' if p == 0 and len(ocean) else f'Continental block {p-offset+1:02d}'
        land_centre = _unit(np.sum(self.xyz[land] * self.cell_area[land, None], axis=0)) if len(land) else np.array([1., 0., 0.])
        for p in range(self.count):
            centre = self.centres[p]
            away = centre * np.dot(centre, land_centre) - land_centre
            away = _unit(away)
            random = _unit(self.rng.normal(size=3))
            # An initial outward traction represents supercontinent insulation and
            # distributed extension. Later motion responds to boundary tractions.
            drift = random * .0024 + np.cross(centre, away) * .0035 * self.config["rift_strength"]
            self.mantle[p] = drift
            self.omega[p] = drift

    def _make_parcels(self):
        cells = np.flatnonzero(self.initial_crust > 0)
        # Four subcell samples reduce grid aliasing; each has an exact area weight.
        row, col = cells // self.w, cells % self.w
        xyzs, masses = [], []
        for oy, ox in [(.25, .25), (.25, .75), (.75, .25), (.75, .75)]:
            lon = (col + ox) * 2 * np.pi / self.w - np.pi
            lat = np.pi / 2 - (row + oy) * np.pi / self.h
            xyzs.append(_xyz(lon, lat))
            # Equal latitude subdivisions have different areas near the poles.
            top = np.pi / 2 - (row + (0 if oy < .5 else .5)) * np.pi / self.h
            bottom = top - .5 * np.pi / self.h
            masses.append((np.sin(top) - np.sin(bottom)) * np.pi / self.w * RADIUS_KM**2)
        self.pos = np.concatenate(xyzs, axis=0)
        self.mass = np.concatenate(masses)
        self.kind = np.tile(self.initial_crust[cells], 4)
        self.parcel_plate = np.tile(self.plate[cells], 4)
        # Preserve the old draw so unrelated stochastic motion stays on the
        # same stream. Broad basement relief is sampled once, then advected
        # with these parcels and their reference columns, never world-fixed.
        self.rng.uniform(0, 220, len(self.mass))
        self.relief = initial_basement_relief(self.pos, self.config['seed'])
        self.suture = np.where(self.kind == 2, .02, .2).astype(float)
        self.parcel_birth = np.zeros(len(self.mass))
        # Finite material footprints keep their physical orientation while
        # crossing a pole; longitude/latitude point splats cannot do this.
        self.parcel_east, self.parcel_extent = crust_footprints(self.pos, self.w, self.h)
        self.parcel_patch = np.tile(cells, 4).astype(np.int64)
        craton_groups, _ = component_labels(self.initial_crust == 2, self.w, self.h)
        self.parcel_craton = np.tile(craton_groups[cells], 4)
        self.next_patch_uid = self.n
        self.rift_id = np.tile(self.seed_rift_id[cells], 4)
        self.rift_birth_myr = np.where(self.rift_id > 0, 0., -1.)
        tangent = np.tile(self.seed_rift_tangent[cells], (4, 1))
        self.rift_tangent = _unit(tangent-self.pos*np.sum(tangent*self.pos, axis=1)[:, None])
        self.rift_extension_m = np.zeros(len(self.mass))
        self.inversion_uplift_m = np.zeros(len(self.mass))
        structure_engine.initialize_parcels(self)

    def _make_traces(self):
        """Sample passive material histories without changing the solver RNG.

        One subcell parcel per original land cell is eligible. These bounded,
        independent markers obey the same Euler motion and local relief rules,
        but are never coalesced with neighbouring arc parcels. Their relief is
        a representative material history, not the complete raster elevation.
        """
        eligible = np.arange(np.count_nonzero(self.initial_crust > 0))
        # Preserve the original sampling at legacy grid sizes; denser runs keep
        # more inspectable material histories without storing every parcel.
        scale = max(1, math.ceil(self.n / (256 * 128)))
        self.initial_marker_limit = min(16_384, 4096 * scale)
        self.arc_marker_limit = min(4096, 2048 * scale)
        chosen = eligible[np.linspace(0, len(eligible)-1, min(len(eligible), self.initial_marker_limit), dtype=int)] if len(eligible) else eligible
        self.trace_id = np.arange(1, len(chosen) + 1, dtype=np.int64)
        self.trace_xyz = self.pos[chosen].copy()
        self.trace_plate = self.parcel_plate[chosen].copy()
        self.trace_kind = self.kind[chosen].copy()
        self.trace_origin_kind = self.kind[chosen].copy()
        self.trace_patch = self.parcel_patch[chosen].copy()
        self.trace_craton = self.parcel_craton[chosen].copy()
        self.trace_birth_myr = np.zeros(len(chosen))
        self.trace_relief_m = self.relief[chosen].copy()
        self.trace_suture = self.suture[chosen].copy()
        for name in ('rift_id', 'rift_birth_myr', 'rift_tangent', 'rift_extension_m', 'inversion_uplift_m'):
            setattr(self, 'trace_'+name, getattr(self, name)[chosen].copy())
        for name in ("uplift", "extension", "erosion", "adjustment", "ridge_uplift"):
            setattr(self, f"trace_{name}_m", np.zeros(len(chosen)))
        self.next_trace_id = len(chosen) + 1
        self.arc_trace_count = 0
        self.arc_trace_buckets = set()
        structure_engine.initialize_traces(self, chosen)

    def _new_arc_traces(self, cells, owners, patches=None):
        """Retain a bounded selection of actual new arc-material birth sites."""
        if not len(cells) or self.arc_trace_count >= self.arc_marker_limit:
            return
        chosen = []
        bucket = int(self.t // 50)
        for p in np.unique(owners):
            key = (int(self.plate_uid[p]), bucket)
            if key in self.arc_trace_buckets:
                continue
            eligible = np.flatnonzero(owners == p)
            count = min(2, len(eligible), self.arc_marker_limit-self.arc_trace_count-len(chosen))
            if count <= 0:
                break
            chosen.extend(eligible[np.linspace(0, len(eligible)-1, count, dtype=int)])
            self.arc_trace_buckets.add(key)
        chosen = np.asarray(chosen, dtype=int)
        count = len(chosen)
        if not count:
            return
        values = {"id": np.arange(self.next_trace_id, self.next_trace_id+count, dtype=np.int64),
                  "xyz": self.xyz[cells[chosen]], "plate": owners[chosen],
                  "patch": patches[chosen] if patches is not None else np.full(count, -1, np.int64),
                  "craton": np.full(count, -1, np.int32),
                  "kind": np.full(count, 3, np.uint8), "origin_kind": np.full(count, 3, np.uint8),
                  "birth_myr": np.full(count, self.t), "relief_m": np.full(count, 850.),
                  "suture": np.full(count, .55),
                  "rift_id": np.full(count, -1, np.int64), "rift_birth_myr": np.full(count, -1.),
                  "rift_tangent": np.zeros((count, 3)), "rift_extension_m": np.zeros(count),
                  "inversion_uplift_m": np.zeros(count),
                  **{f"{name}_m": np.zeros(count) for name in ("uplift", "extension", "erosion", "adjustment", "ridge_uplift")}}
        for name, value in values.items():
            field = f"trace_{name}"
            setattr(self, field, np.concatenate((getattr(self, field), value)))
        self.next_trace_id += count
        self.arc_trace_count += count
        structure_engine.append_traces(self, count)

    @staticmethod
    def _empty_rift_material(count):
        return dict(rift_id=np.full(count, -1, np.int64), rift_birth_myr=np.full(count, -1.),
                    rift_tangent=np.zeros((count, 3)), rift_extension_m=np.zeros(count),
                    inversion_uplift_m=np.zeros(count))

    def _new_rift_record(self, origin, plates, center):
        rid = self.next_rift_id
        self.next_rift_id += 1
        uids = sorted(int(self.plate_uid[p]) for p in plates)
        self.rift_records.append(dict(id=rid, started_myr=float(self.t), origin=origin,
                                      center=np.asarray(center).tolist(), parent_plate_uids=uids,
                                      inversion_active=False, inversion_episode=0,
                                      last_inversion_myr=None))
        self.rift_pair_ids[tuple(uids)] = rid
        return rid

    @staticmethod
    def _rift_tangents(crack, points):
        """Finite spherical gradient follows the accepted curved rift strike."""
        if not len(points):
            return np.empty((0, 3))
        reference = np.where((np.abs(points[:, 2]) < .9)[:, None], [0., 0., 1.], [1., 0., 0.])
        east = _unit(np.cross(reference, points))
        north = np.cross(points, east)
        delta = 1./RADIUS_KM
        gradient = np.zeros_like(points)
        for direction in (east, north):
            plus = np.cos(delta)*points+np.sin(delta)*direction
            minus = np.cos(delta)*points-np.sin(delta)*direction
            gradient += direction*(crack.signed_distance(plus)-crack.signed_distance(minus))[:, None]
        return _unit(np.cross(points, gradient))

    def _prepare_rift_memory(self):
        self._rift_cell_id = np.full(self.n, -1, np.int64)
        self._rift_cell_tangent = np.zeros((self.n, 3))
        edges = np.flatnonzero((self.bcode == 5) & (self.normal_speed > 0))
        if not len(edges):
            return
        pairs = np.sort(np.column_stack((self.bp[edges], self.bq[edges])), axis=1)
        for p, q in np.unique(pairs, axis=0):
            ii = edges[(pairs[:, 0] == p) & (pairs[:, 1] == q)]
            key = tuple(sorted((int(self.plate_uid[p]), int(self.plate_uid[q]))))
            rid = self.rift_pair_ids.get(key)
            if rid is None:
                center = _unit(np.sum(self.bmid[ii]*self.bl[ii, None], axis=0))
                rid = self._new_rift_record('observed continental extension', (p, q), center)
            for cells in (self.ba[ii], self.bb[ii]):
                self._rift_cell_id[cells] = rid
                self._rift_cell_tangent[cells] = _unit(np.cross(self.xyz[cells], self.bn[ii]))

    def _remember_rift_extension(self, prefix, extension):
        points = self.trace_xyz if prefix else self.pos
        kinds = self.trace_kind if prefix else self.kind
        ids = getattr(self, prefix+'rift_id')
        cells = self._indices(points) if prefix else self.parcel_cell
        eligible = (extension > 0) & ((kinds == 1) | (kinds == 2))
        if not np.any(eligible):
            return
        if not hasattr(self, '_rift_cell_id'):
            self._prepare_rift_memory()
        new = eligible & (ids < 0) & (self._rift_cell_id[cells] > 0)
        if np.any(new):
            ids[new] = self._rift_cell_id[cells[new]]
            births = {row['id']: row['started_myr'] for row in self.rift_records}
            getattr(self, prefix+'rift_birth_myr')[new] = [births[int(rid)] for rid in ids[new]]
            tangent = self._rift_cell_tangent[cells[new]]
            local = points[new]
            getattr(self, prefix+'rift_tangent')[new] = _unit(tangent-local*np.sum(tangent*local, axis=1)[:, None])
        getattr(self, prefix+'rift_extension_m')[eligible & (ids > 0)] += extension[eligible & (ids > 0)]

    def _rift_inversion_gain(self, prefix, stretch, dt, *, sampled_extension=None):
        points = self.trace_xyz if prefix else self.pos
        kinds = self.trace_kind if prefix else self.kind
        owners = self.trace_plate if prefix else self.parcel_plate
        relief = self.trace_relief_m if prefix else self.relief
        cells = self._indices(points) if prefix else self.parcel_cell
        opening = stretch[cells] if sampled_extension is None else np.asarray(sampled_extension)
        select = ((getattr(self, prefix+'rift_id') > 0)
                  & (getattr(self, prefix+'rift_extension_m') > 0)
                  & ((kinds == 1) | (kinds == 2)) & (opening == 0))
        gain = np.zeros(len(points))
        if np.any(select):
            raw = inversion_increment(self, points[select], getattr(self, prefix+'rift_tangent')[select],
                                      owners[select], getattr(self, prefix+'rift_extension_m')[select],
                                      getattr(self, prefix+'inversion_uplift_m')[select], dt)
            gain[select] = raw*np.where(kinds[select] == 2, .35, 1.)*np.clip(1-relief[select]/7000, 0, 1)
        return gain

    def _record_rift_inversion(self, gain, dt):
        # Hysteresis avoids a new event for tiny numerical contact fluctuations.
        affected = gain/dt >= .1
        affected_ids = set(self.rift_id[affected].tolist())
        for row in self.rift_records:
            if row['id'] in affected_ids:
                row['last_inversion_myr'] = float(self.t)
                if row['inversion_active']:
                    continue
                selected = affected & (self.rift_id == row['id'])
                row['inversion_active'] = True
                row['inversion_episode'] += 1
                plates = tuple(int(p) for p in np.unique(self.parcel_plate[selected]))
                self.process_totals['rift_inversion_events'] += 1
                self._record('rift_inversion',
                             f"Compression reactivated continental rift {row['id']}, formed at {row['started_myr']:g} Myr, into an uplifted belt.",
                             ('rift_inversion', row['id'], row['inversion_episode']), plates=plates,
                             xyz=np.sum(self.pos[selected]*self.mass[selected, None], axis=0),
                             details={'source_rift_id': row['id'], 'rift_started_myr': row['started_myr'],
                                      'episode': row['inversion_episode'],
                                      'mean_uplift_rate_m_myr': float(np.average(gain[selected]/dt, weights=self.mass[selected])),
                                      'mechanism': 'compression localized in recorded extended continental crust'})
            elif row['inversion_active'] and self.t-row['last_inversion_myr'] >= 10:
                row['inversion_active'] = False
                selected = self.rift_id == row['id']
                self._record('rift_inversion_quiet',
                             f"Inversion of continental rift {row['id']} has become quiet; its relief continues to erode.",
                             ('rift_inversion_quiet', row['id'], row['inversion_episode']),
                             plates=tuple(int(p) for p in np.unique(self.parcel_plate[selected])),
                             xyz=np.sum(self.pos[selected]*self.mass[selected, None], axis=0) if np.any(selected) else None,
                             details={'source_rift_id': row['id'], 'rift_started_myr': row['started_myr'],
                                      'episode': row['inversion_episode'], 'quiet_interval_myr': 10})

    def _new_plate_identity(self, slot, parent):
        self.plate_uid[slot] = self.next_plate_uid
        self.next_plate_uid += 1
        self.plate_generation[slot] += 1
        self.plate_created[slot] = self.t
        self.plate_parent_uid[slot] = self.plate_uid[parent]

    def _ensure_plate_capacity(self, requested):
        """Grow only as fragments need slots, with a bounded support-field budget."""
        limit = max(self.capacity, min(256, (768 * 1024**2) // (4 * self.n)))
        target = min(limit, max(self.capacity, ((int(requested) + 7) // 8) * 8))
        if target <= self.capacity:
            return
        old = self.capacity
        for name in ('active', 'rift_clock', 'ocean_rift_clock', 'born', 'plate_uid',
                     'plate_generation', 'plate_created', 'plate_parent_uid'):
            setattr(self, name, np.pad(getattr(self, name), (0, target-old)))
        for name in ('omega', 'mantle', 'centres'):
            setattr(self, name, np.pad(getattr(self, name), ((0, target-old), (0, 0))))
        self.polarity = np.pad(self.polarity, ((0, target-old), (0, target-old)), constant_values=-1)
        self.collision_clock = np.pad(self.collision_clock, ((0, target-old), (0, target-old)))
        self.support = np.pad(self.support, ((0, target-old), (0, 0)))
        self.names.extend(f'Plate {p+1:02d}' for p in range(old, target))
        self.capacity = target

    def _material_cut(self, crack):
        """Apply one fracture side to all samples of each enduring source patch."""
        ids, centers, inverse = patch_centres(self.pos, self.mass, self.parcel_patch)
        patch_side = crack.signed_distance(centers) > 0
        parcel_side = patch_side[inverse]
        trace_side = crack.signed_distance(self.trace_xyz) > 0
        if len(ids) and len(self.trace_patch):
            at = np.searchsorted(ids, self.trace_patch)
            valid = (at < len(ids)) & (ids[np.minimum(at, len(ids)-1)] == self.trace_patch)
            trace_side[valid] = patch_side[at[valid]]
        return parcel_side, trace_side

    def _repair_plate_domains(self, protected_slots=()):
        """Separate disconnected domains in the supplied initial painting.

        This is an initialization operation. After motion begins, overlapping
        material can appear disconnected in the dominant-owner raster without
        a physical fracture. Those visible remnants receive tracked surface
        identities, while only accepted fractures change their actual motion.
        """
        _, centers, inverse = patch_centres(self.pos, self.mass, self.parcel_patch)
        attribution = centers[inverse]
        plan = plan_domain_repairs(self.plate, self.crust, self.w, self.h, self.cell_area,
                                  parcel_cells=self._indices(attribution), parcel_plate=self.parcel_plate,
                                  parcel_mass=self.mass, parcel_xyz=attribution, xyz=self.xyz,
                                  parcel_groups=self.parcel_craton, protected_slots=protected_slots)
        # Reclaim completely vanished, material-free slots before allocating.
        occupied = set(plan.component_owner.tolist()) | set(self.parcel_plate.tolist())
        occupied |= set(self.trace_plate.tolist())
        for p in np.flatnonzero(self.active):
            if int(p) not in occupied:
                self.active[p] = False
                self.support[p] = 0
        free = np.count_nonzero(~self.active)
        self._ensure_plate_capacity(self.capacity + max(0, len(plan.fragments)-free))
        free_slots = iter(np.flatnonzero(~self.active).tolist())
        component_slot = plan.component_owner.copy()
        fragment_records = []
        deferred = []
        for fragment in plan.fragments:
            q = next(free_slots, None)
            if q is None:
                deferred.append(fragment)
                continue
            p, component = fragment['parent_slot'], fragment['component']
            self.active[q] = True
            self.count = max(self.count, q + 1)
            self._new_plate_identity(q, p)
            self.names[q] = f'Block {int(self.plate_uid[q]):03d}'
            self.omega[q] = self.omega[p]
            self.mantle[q] = self.mantle[p]
            self.born[q] = self.born[p]
            self.rift_clock[q] = self.ocean_rift_clock[q] = 0
            self.polarity[q] = np.where(self.polarity[p] == p, q, self.polarity[p])
            self.polarity[:, q] = np.where(self.polarity[:, p] == p, q, self.polarity[:, p])
            self.polarity[p, q] = self.polarity[q, p] = -1
            self.polarity[q, q] = -1
            self.collision_clock[q] = self.collision_clock[:, q] = 0
            self.support[q] = 0
            self.event_keys = {key for key in self.event_keys if not (isinstance(key, tuple) and key[0] == 'collision' and q in key[1:])}
            region = plan.labels == component
            self.centres[q] = _unit(np.sum(self.xyz[region]*self.cell_area[region, None], axis=0))
            component_slot[component] = q
            fragment_records.append((p, q, fragment))
        targets = {row['component']: row['target_component'] for row in plan.absorptions}
        for component in targets:
            target = targets[component]
            while target in targets:
                target = targets[target]
            component_slot[component] = component_slot[target]
        if fragment_records or targets:
            trace_components = assign_material_components(
                plan.labels, plan.component_owner, self.w, self.h,
                self._indices(self.trace_xyz), self.trace_plate, self.trace_xyz, self.xyz)
            old_plate = self.plate.copy()
            self.plate = component_slot[plan.labels].astype(np.int16)
            moved = np.flatnonzero(self.plate != old_plate)
            # Move the parent's continuous membership only on the actual local
            # domain. Other plates' competing fractional claims remain intact.
            claim = self.support[old_plate[moved], moved].copy()
            self.support[old_plate[moved], moved] = 0
            self.support[self.plate[moved], moved] += claim
            valid = plan.parcel_components >= 0
            self.parcel_plate[valid] = component_slot[plan.parcel_components[valid]]
            valid_trace = trace_components >= 0
            self.trace_plate[valid_trace] = component_slot[trace_components[valid_trace]]
            # Passive history markers follow their actual source patch, including
            # coast-overlap cases where the marker lies outside the visible owner.
            patch_ids, first = np.unique(self.parcel_patch, return_index=True)
            if len(patch_ids):
                at = np.searchsorted(patch_ids, self.trace_patch)
                follows = (at < len(patch_ids)) & (patch_ids[np.minimum(at, len(patch_ids)-1)] == self.trace_patch)
                self.trace_plate[follows] = self.parcel_plate[first[at[follows]]]
            for p, q, fragment in fragment_records:
                self._record('plate_fragment',
                             f'A detached domain of {self.names[p]} became {self.names[q]}, retaining its inherited motion.',
                             plates=(p, q), xyz=self.centres[q],
                             details={**fragment, 'parent_plate_uid': int(self.plate_uid[p]),
                                      'new_plate_uid': int(self.plate_uid[q]),
                                      'cause': 'disconnected lithosphere', 'opening_impulse': False})
        self.domain_diagnostics = dict(components=len(plan.component_owner),
                                      detached_created=len(fragment_records),
                                      empty_ocean_remnants_merged=len(targets),
                                      deferred_components=len(deferred),
                                      unresolved_material_mass_km2=float(sum(plan.unresolved_mass_by_owner.values())))

    def _update_surface_domains(self, dt=0., previous_omega=None, previous_plate_uid=None):
        """Name connected visible remnants without fracturing hidden material."""
        previous_cells = None
        if dt and previous_omega is not None and previous_plate_uid is not None:
            # Slots can be added or reused after advection. A newborn plate's
            # material moved with its parent at the start of this step, not
            # with its new opening velocity or the slot's retired occupant.
            prior_motion = {int(uid): omega for uid, omega in
                            zip(previous_plate_uid, previous_omega) if uid > 0}
            parents = {int(uid): int(parent) for uid, parent in
                       zip(self.plate_uid, self.plate_parent_uid) if uid > 0}
            motion = np.zeros((self.capacity, 3))
            matched = np.zeros(self.capacity, bool)
            for p in np.unique(self.plate):
                uid, seen = int(self.plate_uid[p]), set()
                while uid > 0 and uid not in prior_motion and uid not in seen:
                    seen.add(uid)
                    uid = parents.get(uid, 0)
                if uid in prior_motion:
                    motion[p] = prior_motion[uid]
                    matched[p] = True
            previous_cells = np.full(self.n, -1, np.int32)
            for start in range(0, self.n, ADVECTION_CHUNK_CELLS):
                section = slice(start, start + ADVECTION_CHUNK_CELLS)
                owners = self.plate[section]
                back = _rotate(self.xyz[section], -motion[owners] * dt)
                previous_cells[section] = np.where(matched[owners], self._indices(back), -1)
        state = self.domain_tracker.update(self.plate, self.cell_area, self.plate_uid,
                                           self.names, self.t, previous_cells=previous_cells)
        self.domain = state.domain
        self.domains = state.domains
        for domain in state.created:
            if domain['primary']:
                continue
            p = domain['plate_id']
            self._record('surface_fragment',
                         f"{domain['name']} is a separate visible remnant sharing {self.names[p]}'s motion.",
                         plates=(p,), details={**domain, 'independent_motion': False,
                                               'cause': 'disconnected visible surface domain'})
        self.domain_diagnostics = dict(connected_surface_domains=len(state.domains),
                                       new_surface_domains=len(state.created),
                                       retired_surface_domains=len(state.retired_uids),
                                       independent_plate_creation=False)

    def _indices(self, xyz):
        lon = np.arctan2(xyz[:, 1], xyz[:, 0])
        lat = np.arcsin(np.clip(xyz[:, 2], -1, 1))
        x = np.floor((lon + np.pi) * self.w / (2 * np.pi)).astype(np.int32) % self.w
        y = np.clip(np.floor((np.pi / 2 - lat) * self.h / np.pi).astype(np.int32), 0, self.h - 1)
        return y * self.w + x

    def _sample_coordinates(self, xyz):
        """Periodic bilinear lookup continued across each pole by a half-turn.

        The row beyond the north cap is the north row at longitude +180°;
        likewise at the south cap. The two poles never connect to each other.
        """
        x = (np.arctan2(xyz[:, 1], xyz[:, 0]) + np.pi) * self.w / (2 * np.pi) - .5
        y = (np.pi / 2 - np.arcsin(np.clip(xyz[:, 2], -1, 1))) * self.h / np.pi - .5
        y0 = np.floor(y).astype(np.int32)
        fy = y - y0
        lookup = []
        for row, vertical in ((y0, 1-fy), (y0+1, fy)):
            crosses = (row < 0) | (row >= self.h)
            reflected = np.where(row < 0, -row-1, np.where(row >= self.h, 2*self.h-row-1, row))
            shifted = x + crosses * (self.w / 2)
            col = np.floor(shifted).astype(np.int32)
            fx = shifted-col
            lookup.extend(((reflected*self.w + col % self.w, (1-fx)*vertical),
                           (reflected*self.w + (col+1) % self.w, fx*vertical)))
        return tuple(lookup)

    @staticmethod
    def _sample(field, lookup):
        return sum(field[idx] * weight for idx, weight in lookup)

    def _rasterize(self):
        """Conservative finite-footprint deposition on the sphere."""
        self.parcel_cell = self._indices(self.pos)
        # Sparse occupancy is an exact by-product of this deposition. A
        # content signature lets local accretion reuse it only while every
        # geometry, area, owner and grid input still matches the projection.
        self._owner_occupancy = {}
        self._owner_occupancy_signature = None
        (self.land_mass, height_sum, suture_sum, craton_mass, arc_mass,
         best_mass, dominant, structure_sums) = deposit_crust(
            self.pos, self.parcel_east, self.parcel_extent, self.mass, self.kind,
            structure_engine.material_height(self.kind, self.relief), self.suture,
            self.parcel_plate, self.w, self.h,
            self.xyz, self.cell_area, self._sample_coordinates,
            extra_fields=geology_snapshot.deposit_fields(self),
            owner_occupancy=self._owner_occupancy)
        density = self.land_mass / self.cell_area
        self.crust = np.zeros(self.n, np.uint8)
        is_land = density >= .36
        self.crust[is_land] = 1
        self.crust[is_land & (arc_mass > .55 * self.land_mass)] = 3
        self.crust[is_land & (craton_mass > .45 * self.land_mass)] = 2
        self.plate = surface_exposure.resolve(self.plate, dominant,
                                              is_land & (best_mass > 0),
                                              self._owner_occupancy).astype(np.int16)
        # Persistent parcels anchor the continuous plate field inside continents.
        anchor = np.flatnonzero(is_land & (best_mass > 0))
        self.support[:, anchor] = 0
        self.support[self.plate[anchor], anchor] = 1
        denom = np.maximum(self.land_mass, 1e-10)
        self.land_height = height_sum / denom
        self.land_relief = (self.land_height-220.-180.*(self.crust == 2)
                            + 100.*(self.crust == 3))
        self.grid_suture = suture_sum / denom
        self.age[is_land] = 0
        self.density = density
        geology_snapshot.finish_grid(self, structure_sums)
        self._owner_occupancy_signature = local_accretion.occupancy_signature(self)

    def _boundaries(self):
        pa, pb = self.plate[self.edge_a], self.plate[self.edge_b]
        use = pa != pb
        self.ba, self.bb = self.edge_a[use], self.edge_b[use]
        self.bp, self.bq = pa[use], pb[use]
        self.bn = self.edge_normal[use]
        self.bmid = self.edge_mid[use]
        self.bl = self.edge_length[use]
        geometry = refine_boundary_geometry(self.plate, self.support, self.xyz, self.w, self.h,
                                             self.ba, self.bb, self.bmid, self.bn, self.bl)
        self.bn, self.bmid, self.bl = geometry['normals'], geometry['midpoints'], geometry['lengths']
        self.boundary_geometry_diagnostics = geometry['diagnostics']
        dv = np.cross(self.omega[self.bq] - self.omega[self.bp], self.bmid) * RADIUS_KM
        self.normal_speed = np.sum(dv * self.bn, axis=1)
        tangential = np.linalg.norm(dv - self.normal_speed[:, None] * self.bn, axis=1)
        ext = self.normal_speed > np.maximum(2.0, tangential * .35)
        con = self.normal_speed < -np.maximum(2.0, tangential * .35)
        la, lb = self.crust[self.ba] > 0, self.crust[self.bb] > 0
        self.bcode = np.full(len(self.ba), 3, np.uint8)
        self.bcode[ext & ~(la | lb)] = 1
        self.bcode[ext & (la | lb)] = 5
        self.bcode[con & ~(la & lb)] = 2
        self.bcode[con & la & lb] = 4
        self.boundary = np.zeros(self.n, np.uint8)
        # Explicit priority makes collisions visible above transforms.
        priority = np.array([0, 2, 4, 1, 6, 5], np.uint8)
        ranks = np.zeros(self.n, np.uint8)
        np.maximum.at(ranks, self.ba, priority[self.bcode])
        np.maximum.at(ranks, self.bb, priority[self.bcode])
        self.boundary = np.array([0, 3, 1, 0, 2, 5, 4], np.uint8)[ranks]
        self.down = np.full(len(self.ba), -1, np.int16)
        sub = np.flatnonzero(self.bcode == 2)
        # Established subduction polarity is retained for each plate pair. A
        # buoyant continent arriving at the trench can reverse that heuristic.
        for p, q in np.unique(np.sort(np.stack((self.bp[sub], self.bq[sub]), axis=1), axis=1), axis=0):
            select = sub[((self.bp[sub] == p) & (self.bq[sub] == q)) | ((self.bp[sub] == q) & (self.bq[sub] == p))]
            cells_p = np.where(self.bp[select] == p, self.ba[select], self.bb[select])
            cells_q = np.where(self.bp[select] == q, self.ba[select], self.bb[select])
            land_p = np.mean(self.crust[cells_p] > 0)
            land_q = np.mean(self.crust[cells_q] > 0)
            down = int(self.polarity[p, q])
            if down < 0:
                down = int(p if np.mean(self.age[cells_p]) >= np.mean(self.age[cells_q]) else q)
            if land_p > .5 and land_q < .25:
                down = int(q)
            elif land_q > .5 and land_p < .25:
                down = int(p)
            self.polarity[p, q] = self.polarity[q, p] = down
            self.down[select] = down
        # At a mixed boundary, ocean always goes beneath buoyant continental crust.
        self.down[sub[la[sub] & ~lb[sub]]] = self.bq[sub[la[sub] & ~lb[sub]]]
        self.down[sub[lb[sub] & ~la[sub]]] = self.bp[sub[lb[sub] & ~la[sub]]]
        self.trench_retreat_speed = np.zeros(len(self.ba))
        trench_history.prepare(self)

    def _migrate_trenches(self, dt):
        """Move oceanic hinges independently of the coastline, then refresh.

        The extra arc flux uses realized membership transfer, restricted to
        boundary segments next to actual motion. This is a local kinematic
        closure; it adds no continental parcels or prescribed mantle torque.
        """
        transfers = extend_forearcs(self, dt, exclude_pairs=backarc.driven_pairs(self))
        if transfers:
            self._boundaries()
        self.trench_retreat_speed = np.zeros(len(self.ba))
        plate_area = np.bincount(self.plate, weights=self.cell_area, minlength=self.capacity)
        craton_area = np.bincount(self.parcel_plate[self.kind == 2], weights=self.mass[self.kind == 2], minlength=self.capacity)
        for (p, q), result in transfers.items():
            area = result['area_km2']
            if area <= 0:
                continue
            self.process_totals['trench_swept_km2'] += area
            ii = np.flatnonzero(normal_partition.subduction(self) & (self.down == p)
                                & ((self.bp == q) | (self.bq == q)))
            if not len(ii):
                continue
            moved = np.zeros(self.n)
            moved[result['cells']] = result['gain']
            local = np.maximum(moved[self.ba[ii]], moved[self.bb[ii]])
            below = np.where(self.down[ii] == self.bp[ii], self.ba[ii], self.bb[ii])
            local[self.density[below] > 1e-10] = 0.
            weights = local*self.bl[ii]
            if weights.sum() > 0:
                # Allocate the measured swept area to nearby active segments;
                # no extra supply is assigned across a blocked buoyant margin.
                # A shrinking/fragmented contact must not concentrate a whole
                # pair's transfer into an arbitrarily fast volcanic segment.
                limit = retreat_speed(-self.normal_speed[ii], self.age[below],
                                      craton_area[q]/max(plate_area[q], 1.),
                    slab_buoyancy=trench_history.slab_pull_weights(self)[ii] if slab_memory.enabled(self) else None)
                self.trench_retreat_speed[ii] = np.minimum(area/dt*local/weights.sum(), limit)
            arc_sweep = float(np.sum(self.trench_retreat_speed[ii]*self.bl[ii])*dt)
            self.process_totals['trench_arc_sweep_km2'] += arc_sweep
            self._record('trench_migration',
                         f'The subduction hinge between {self.names[p]} and {self.names[q]} retreats as the overriding forearc extends.',
                         ('trench_migration', int(self.plate_uid[p]), int(self.plate_uid[q]), int(self.t//100)),
                         plates=(p, q), xyz=np.sum(self.xyz[result['cells']]*result['gain'][:, None], axis=0),
                         details={'downgoing_plate_uid': int(self.plate_uid[p]), 'overriding_plate_uid': int(self.plate_uid[q]),
                                  'swept_membership_area_km2': area, 'arc_flux_sweep_km2': arc_sweep,
                                  'reporting_interval_myr': 100,
                                  'mechanism': 'parameterized hinge retreat and oceanic forearc extension'})

    def _update_ridge_windows(self, dt):
        created = update_ridge_interactions(self, dt)
        slots = {int(self.plate_uid[p]): int(p) for p in np.flatnonzero(self.active)}
        for episode in created:
            self.process_totals['ridge_subduction_events'] += 1
            uids = episode['incoming_plate_uids']+[episode['overriding_plate_uid']]
            self._record('ridge_subduction',
                         'An active spreading ridge reached a trench, opening a modeled slab window beneath the overriding plate.',
                         ('ridge_subduction', episode['id']),
                         plates=tuple(slots[uid] for uid in uids if uid in slots), xyz=episode['center'],
                         details=deepcopy(episode))
        for episode in self.ridge_episodes:
            owner = slots.get(int(episode['overriding_plate_uid']))
            for milestone, typ, description in (
                    ('peak_myr', 'slab_window_peak', 'The slab-window thermal bulge and volcanic pulse reached their modeled peak.'),
                    ('end_myr', 'slab_window_cooling', 'The slab-window thermal support has faded; volcanic relief remains subject to erosion.')):
                flag = typ+'_recorded'
                if self.t >= episode[milestone] and not episode.get(flag, False):
                    episode[flag] = True
                    self._record(typ, description, (typ, episode['id']),
                                 plates=(owner,) if owner is not None else (), xyz=episode['center'],
                                 details={'ridge_episode_id': episode['id'],
                                          'overriding_plate_uid': episode['overriding_plate_uid'],
                                          'scheduled_time_myr': episode[milestone],
                                          'radius_km': episode['radius_km']})

    def _transfer_ridge_hosts(self, source, target):
        """Follow a physically reassigned window center, preserving its budget."""
        if not self.ridge_episodes:
            return
        cells = self._indices(np.asarray([episode['center'] for episode in self.ridge_episodes]))
        reassign_ridge_episodes(self, int(self.plate_uid[source]), int(self.plate_uid[target]),
                               self.plate[cells] == target)

    def _record(self, typ, description, key=None, *, plates=(), xyz=None, details=None):
        if key is not None:
            if key in self.event_keys:
                return
            self.event_keys.add(key)
        plate_ids = [int(p) for p in plates]
        event = dict(id=len(self.events)+1, time_myr=round(float(self.t), 4),
                     type=typ, description=description, plate_ids=plate_ids,
                     plate_uids=[int(self.plate_uid[p]) for p in plate_ids],
                     details=deepcopy(details or {}))
        if xyz is not None:
            location = np.asarray(xyz, float)
            if np.linalg.norm(location) > 1e-8:
                location = _unit(location)
                event.update(lon=float(np.degrees(np.arctan2(location[1], location[0]))),
                             lat=float(np.degrees(np.arcsin(np.clip(location[2], -1, 1)))))
        self.events.append(event)

    def _advect(self, dt):
        old_plate = self.plate.copy()
        transported = np.zeros_like(self.support)
        arrivals = np.zeros_like(self.support)
        for p in np.flatnonzero(self.active):
            for start in range(0, self.n, ADVECTION_CHUNK_CELLS):
                section = slice(start, start + ADVECTION_CHUNK_CELLS)
                lookup = self._sample_coordinates(_rotate(self.xyz[section], -self.omega[p] * dt))
                transported[p, section] = self._sample(self.support[p], lookup)
                # True arrivals protect an incoming third plate from spreading
                # reconstruction. Existing fractional trench motion is retained.
                arrivals[p, section] = sum(weight*(old_plate[index] == p) for index, weight in lookup)
        total = transported.sum(axis=0)
        consumed = np.maximum(total - 1, 0)
        ocean_mask = self.crust == 0
        self.process_totals["ocean_consumed_km2"] += float(np.sum(self.cell_area[ocean_mask] * consumed[ocean_mask]))
        # Where convergent supports overlap, the established down-going plate
        # loses its fractional claim. This allows slow subcell trench migration.
        compression = np.minimum(consumed, .6)
        for p in np.flatnonzero(self.active):
            down_weight = trench_history.downgoing_weight(self, p)
            transported[p] *= (1 - compression*down_weight)
        newborn = ridge_spreading.advance(self, transported, dt, arrivals=arrivals)
        total_after = transported.sum(axis=0)
        gap = total_after < 1e-8
        transported[old_plate[gap], np.flatnonzero(gap)] = 1
        total_after[gap] = 1
        self.support = transported / total_after[None, :]
        owner = np.argmax(self.support, axis=0).astype(np.int16)
        # Sample continuous ocean properties only once with the destination
        # plate's inverse Euler rotation, instead of nearest-neighbour resampling.
        new_age = np.empty_like(self.age)
        new_relief = np.empty_like(self.ocean_relief)
        for start in range(0, self.n, ADVECTION_CHUNK_CELLS):
            section = slice(start, start + ADVECTION_CHUNK_CELLS)
            back = _rotate(self.xyz[section], -self.omega[owner[section]] * dt)
            lookup = self._sample_coordinates(back)
            # Existing ocean ages normally; only the explicitly generated
            # paired strips contribute juvenile (mean dt/2) material.
            new_age[section] = ((self._sample(self.age, lookup) + dt)*(1-newborn[section])
                                + dt*.5*newborn[section])
            new_relief[section] = self._sample(self.ocean_relief, lookup)
        self.age = new_age
        self.ocean_relief = new_relief
        self.plate = owner
        if len(self.pos):
            for start in range(0, len(self.pos), ADVECTION_CHUNK_CELLS):
                section = slice(start, start + ADVECTION_CHUNK_CELLS)
                self.pos[section] = _unit(_rotate(self.pos[section], self.omega[self.parcel_plate[section]] * dt))
                self.parcel_east[section] = _unit(_rotate(self.parcel_east[section], self.omega[self.parcel_plate[section]] * dt))
                self.rift_tangent[section] = _unit(_rotate(self.rift_tangent[section], self.omega[self.parcel_plate[section]] * dt))
        if len(self.trace_id):
            self.trace_xyz = _unit(_rotate(self.trace_xyz, self.omega[self.trace_plate] * dt))
            self.trace_rift_tangent = _unit(_rotate(self.trace_rift_tangent, self.omega[self.trace_plate] * dt))
        for p in np.flatnonzero(self.active):
            self.centres[p] = _rotate(self.centres[p], self.omega[p] * dt)
        backarc.advect_records(self, dt, _rotate)
        trench_history.advect(self, dt)
        progressive_rifting.advect(self, dt, _rotate)

    def _deform_and_accrete(self, dt):
        collision = self.bcode == 4
        subduction = self.bcode == 2
        rifting = self.bcode == 5
        convergence = np.maximum(-self.normal_speed, 0)
        retreat = getattr(self, 'trench_retreat_speed', np.empty(0))
        if len(retreat) == len(convergence):
            convergence = convergence + np.where(subduction, retreat, 0.)
        extension = np.maximum(self.normal_speed, 0)
        uplift = np.zeros(self.n)
        stretch = np.zeros(self.n)
        volcanic = np.zeros(self.n)
        # Shortening is converted to topographic growth, with saturation and
        # erosion below. Cratons shorten less readily than mobile belts.
        np.maximum.at(uplift, self.ba[collision], convergence[collision] * 2.8)
        np.maximum.at(uplift, self.bb[collision], convergence[collision] * 2.8)
        np.maximum.at(stretch, self.ba[rifting], extension[rifting] * 1.3)
        np.maximum.at(stretch, self.bb[rifting], extension[rifting] * 1.3)
        self.ocean_relief *= np.exp(-dt / 45)
        arc_candidates = np.empty(0, int)
        arc_addition = np.empty(0)
        if np.any(subduction):
            k = np.flatnonzero(subduction)
            maturity = trench_history.weights(self)[k]
            down_a = self.down[k] == self.bp[k]
            below = np.where(down_a, self.ba[k], self.bb[k])
            above = np.where(down_a, self.bb[k], self.ba[k])
            # The trench lies on the down-going side; the volcanic chain is on
            # the overriding side, 180 km from the reconstructed interface.
            self.ocean_relief[below] = np.minimum(self.ocean_relief[below], -np.clip(convergence[k] * 25, 400, 1900))
            direction = np.where(down_a[:, None], self.bn[k], -self.bn[k])
            arc_angle = 180/RADIUS_KM
            arc_xyz = np.cos(arc_angle)*self.bmid[k]+np.sin(arc_angle)*direction
            arc_cell = self._indices(arc_xyz)
            valid = self.plate[arc_cell] == self.plate[above]
            arc_cell = np.where(valid, arc_cell, above)
            np.maximum.at(uplift, arc_cell, convergence[k] * 1.25 * maturity)
            np.maximum.at(volcanic, arc_cell, convergence[k] * 1.25 * maturity)
            # Emergence does not end arc construction. Otherwise finer grids
            # cross the visibility threshold sooner and lose subsequent flux.
            juvenile = (self.crust[arc_cell] == 0) | (self.crust[arc_cell] == 3)
            arc_candidates = arc_cell[juvenile]
            # A small fraction of trench swept area becomes juvenile buoyant
            # crust. Deposit this directly into persistent Lagrangian arc parcels:
            # a diffuse raster magma budget would lose mass during transport and
            # make island-arc formation depend catastrophically on resolution.
            arc_addition = (convergence[k][juvenile] + 8) * self.bl[k][juvenile] * dt * .015 * maturity[juvenile]
        # Spread deformation over a resolvable belt without smoothing across poles.
        uplift = np.maximum.reduce((uplift, uplift[self.east] * .55, uplift[self.west] * .55,
                                    uplift[self.north] * .55, uplift[self.south] * .55))
        shortening = np.zeros(self.n)
        np.maximum.at(shortening, self.ba[collision], convergence[collision])
        np.maximum.at(shortening, self.bb[collision], convergence[collision])
        shortening = np.maximum.reduce((shortening, shortening[self.east]*.55,
            shortening[self.west]*.55, shortening[self.north]*.55, shortening[self.south]*.55))
        volcanic = np.maximum.reduce((volcanic, volcanic[self.east]*.55,
            volcanic[self.west]*.55, volcanic[self.north]*.55, volcanic[self.south]*.55))
        deformation = structure_engine.deform(self, shortening, stretch/1.3, volcanic, dt,
                                               spherical_boundaries=True)
        progressive_rifting.update(self, dt, stretch/1.3, realized_extension=(
            deformation['parcel']['extension_strain'], deformation['trace']['extension_strain']))
        # Sustained contact belongs to the actual approaching terranes. A
        # remote island sharing the source plate cannot inherit this contact's
        # loading or move into another plate because its neighbor collided.
        for plan in local_accretion.plan_accretions(self, dt):
            local_accretion.apply_accretion(self, plan)
        self._add_arc_crust(arc_candidates, arc_addition)
        if np.any(collision):
            pairs = np.unique(np.sort(np.stack((self.bp[collision], self.bq[collision]), axis=1), axis=1), axis=0)
            for p, q in pairs:
                key = ("collision", int(p), int(q))
                if key not in self.event_keys:
                    self.process_totals["collision_events"] += 1
                    pair = collision & (((self.bp == p) & (self.bq == q)) | ((self.bp == q) & (self.bq == p)))
                    self._record("collision", f"{self.names[p]} and {self.names[q]} began continental shortening and mountain building.", key,
                                 plates=(p, q), xyz=np.sum(self.bmid[pair]*self.bl[pair, None], axis=0),
                                 details={"boundary_length_km": float(self.bl[pair].sum()),
                                          "mean_convergence_km_myr": float(np.average(convergence[pair], weights=self.bl[pair]))})

    def _add_arc_crust(self, cells, additions):
        """Accumulate trench flux in enduring parcels, including subgrid arcs."""
        positive = np.asarray(additions) > 0
        cells, additions = np.asarray(cells)[positive], np.asarray(additions)[positive]
        if not len(cells):
            return
        owner = self.plate[cells]
        key = owner.astype(np.int64) * self.n + cells
        unique, inverse = np.unique(key, return_inverse=True)
        amount = np.bincount(inverse, weights=additions)
        existing = np.flatnonzero(self.kind == 3)
        if len(existing):
            existing_key = self.parcel_plate[existing].astype(np.int64) * self.n + self._indices(self.pos[existing])
            old_key, first = np.unique(existing_key, return_index=True)
            at = np.searchsorted(old_key, unique)
            match = (at < len(old_key)) & (old_key[np.minimum(at, len(old_key)-1)] == unique)
            dest = existing[first[at[match]]]
            np.add.at(self.mass, dest, amount[match])
            trace_keys = self.trace_plate.astype(np.int64)*self.n + self._indices(self.trace_xyz)
            rejuvenated = (self.trace_kind == 3) & np.isin(trace_keys, unique[match])
            structure_engine.replenish(self, dest, np.flatnonzero(rejuvenated), 700.)
        else:
            match = np.zeros(len(unique), bool)
        new = unique[~match]
        if len(new):
            cells = (new % self.n).astype(int)
            owner = (new // self.n).astype(np.int16)
            self.pos = np.concatenate((self.pos, self.xyz[cells]))
            self.mass = np.concatenate((self.mass, amount[~match]))
            self.parcel_plate = np.concatenate((self.parcel_plate, owner))
            self.kind = np.concatenate((self.kind, np.full(len(new), 3, np.uint8)))
            self.relief = np.concatenate((self.relief, np.full(len(new), 850.)))
            self.suture = np.concatenate((self.suture, np.full(len(new), .55)))
            self.parcel_birth = np.concatenate((self.parcel_birth, np.full(len(new), self.t)))
            arc_east, arc_extent = crust_footprints(self.xyz[cells], self.w, self.h)
            self.parcel_east = np.concatenate((self.parcel_east, arc_east))
            self.parcel_extent = np.concatenate((self.parcel_extent, arc_extent))
            patches = np.arange(self.next_patch_uid, self.next_patch_uid + len(new), dtype=np.int64)
            self.next_patch_uid += len(new)
            self.parcel_patch = np.concatenate((self.parcel_patch, patches))
            self.parcel_craton = np.concatenate((self.parcel_craton, np.full(len(new), -1, np.int32)))
            for name, value in self._empty_rift_material(len(new)).items():
                setattr(self, name, np.concatenate((getattr(self, name), value)))
            structure_engine.append_parcels(self, len(new))
            self._new_arc_traces(cells, owner, patches)
        self.process_totals["arc_added_km2"] += float(amount.sum())
        for p in np.unique(unique // self.n):
            select = unique // self.n == p
            arc_cells = (unique[select] % self.n).astype(int)
            self._record("island_arc", f"Subduction-generated magma added persistent buoyant island-arc crust on {self.names[p]}.",
                         ("arc", int(self.plate_uid[p]), int(self.t // 100)), plates=(p,),
                         xyz=np.sum(self.xyz[arc_cells]*amount[select, None], axis=0),
                         details={"added_area_this_step_km2": float(amount[select].sum()),
                                  "reporting_interval_myr": 100})
        if self.steps % 20 == 0:
            self._coalesce_arcs()

    def _coalesce_arcs(self):
        """Combine unresolved same-cell arc samples without losing their mass."""
        arc = np.flatnonzero(self.kind == 3)
        if not len(arc):
            return
        keys = self.parcel_plate[arc].astype(np.int64) * self.n + self._indices(self.pos[arc])
        unique, inverse = np.unique(keys, return_inverse=True)
        if len(unique) == len(arc):
            return
        m = np.bincount(inverse, weights=self.mass[arc])
        xyz = np.stack([np.bincount(inverse, weights=self.mass[arc]*self.pos[arc, k]) / m for k in range(3)], axis=1)
        keep = self.kind != 3
        structure_engine.coalesce(self, arc, keep, inverse)
        # Coalesced unresolved arcs obtain a local cell footprint at the new
        # mass centroid. Continental/cratonic footprints are never coalesced.
        arc_east, arc_extent = crust_footprints(_unit(xyz), self.w, self.h)
        patches = np.arange(self.next_patch_uid, self.next_patch_uid + len(unique), dtype=np.int64)
        self.next_patch_uid += len(unique)
        old_patches = self.parcel_patch[arc]
        local_accretion.remap_contact_patches(self, old_patches, patches[inverse])
        patch_order = np.argsort(old_patches)
        at = np.searchsorted(old_patches[patch_order], self.trace_patch)
        valid = (at < len(old_patches)) & (old_patches[patch_order][np.minimum(at, len(old_patches)-1)] == self.trace_patch)
        self.trace_patch[valid] = patches[inverse[patch_order[at[valid]]]]
        self.parcel_patch = np.concatenate((self.parcel_patch[keep], patches))
        self.parcel_craton = np.concatenate((self.parcel_craton[keep], np.full(len(unique), -1, np.int32)))
        self.parcel_east = np.concatenate((self.parcel_east[keep], arc_east))
        self.parcel_extent = np.concatenate((self.parcel_extent[keep], arc_extent))
        self.pos = np.concatenate((self.pos[keep], _unit(xyz)))
        self.relief = np.concatenate((self.relief[keep], np.bincount(inverse, weights=self.mass[arc]*self.relief[arc]) / m))
        self.suture = np.concatenate((self.suture[keep], np.bincount(inverse, weights=self.mass[arc]*self.suture[arc]) / m))
        self.parcel_birth = np.concatenate((self.parcel_birth[keep], np.bincount(inverse, weights=self.mass[arc]*self.parcel_birth[arc]) / m))
        # Only continental/cratonic material can acquire these scars. Juvenile
        # arc coalescence must preserve the continental arrays and append zeros.
        for name, value in self._empty_rift_material(len(unique)).items():
            setattr(self, name, np.concatenate((getattr(self, name)[keep], value)))
        self.mass = np.concatenate((self.mass[keep], m))
        self.parcel_plate = np.concatenate((self.parcel_plate[keep], (unique // self.n).astype(np.int16)))
        self.kind = np.concatenate((self.kind[keep], np.full(len(unique), 3, np.uint8)))

    def _weld(self, source, target):
        """Explicit whole-source join for controlled mechanics fixtures.

        Automatic evolution uses local_accretion plans instead. This private
        forced operation retains ocean lithosphere and all column state.
        """
        moving = self.parcel_plate == source
        m = float(self.mass[moving].sum())
        source_area = m
        target_area = float(self.mass[self.parcel_plate == target].sum())
        total_area = max(source_area + target_area, 1)
        self.omega[target] = (self.omega[source] * source_area + self.omega[target] * target_area) / total_area
        self.mantle[target] = (self.mantle[source] * source_area + self.mantle[target] * target_area) / total_area
        # Suture memory is confined to parcels that experienced actual uplift.
        at_belt = moving & (self.relief > 400)
        self.suture[at_belt] = np.maximum(self.suture[at_belt], .85)
        converted = moving & (self.kind == 3)
        self.relief[converted] -= 100.
        self.kind[converted] = 1
        self.parcel_plate[moving] = target
        traced = self.trace_plate == source
        traced_belt = traced & (self.trace_relief_m > 400)
        self.trace_suture[traced_belt] = np.maximum(self.trace_suture[traced_belt], .85)
        converted_trace = traced & (self.trace_kind == 3)
        self.trace_relief_m[converted_trace] -= 100.
        self.trace_adjustment_m[converted_trace] -= 100.
        self.trace_kind[converted_trace] = 1
        self.trace_plate[traced] = target
        region = (self.plate == source) & (self.crust > 0)
        self.plate[region] = target
        self.support[target, region] += self.support[source, region]
        self.support[source, region] = 0
        self._transfer_ridge_hosts(source, target)
        self.collision_clock[source, target] = self.collision_clock[target, source] = 0
        self.process_totals["accreted_km2"] += m
        self.rift_clock[target] *= .6
        self.born[target] = self.t - 45
        self._record("accretion", f"Sustained convergence welded the buoyant crust of {self.names[source]} to {self.names[target]}; its remaining oceanic lithosphere persists and the collision belt becomes a suture.",
                     plates=(source, target), xyz=np.sum(self.pos[moving]*self.mass[moving, None], axis=0),
                     details={"accreted_area_km2": m, "source_plate_uid": int(self.plate_uid[source]),
                              "target_plate_uid": int(self.plate_uid[target])})

    def _forces(self, dt):
        import plate_balance
        if plate_balance.enabled(self):
            # The lithosphere's Reynolds number is ~1e-20, so velocity is not integrated
            # from an acceleration: it is whatever makes this instant's forces sum to
            # zero. omega is solved from scratch, never relaxed toward a target, and a
            # warm start changes only the iteration count. Everything the law below does
            # AFTER computing its target -- the 24 Myr e-fold, the seeded basal traction,
            # the area-mean net-rotation subtraction and the .014 speed cap -- is absent
            # by construction here, not merely disabled. Net rotation is reported as a
            # diagnostic and checked against a bound; it is never subtracted by fiat.
            omega, diagnostics = plate_balance.solve(self, dt)
            self.omega = omega
            self.omega[~self.active] = 0.
            self.plate_balance_diagnostics = diagnostics
            return
        slab = np.zeros_like(self.omega)
        ridge = np.zeros_like(self.omega)
        resistance = np.zeros_like(self.omega)
        count_s = np.zeros(self.capacity)
        count_r = np.zeros(self.capacity)
        count_c = np.zeros(self.capacity)
        pull_owner, pull_weight = trench_history.slab_pull_state(self)
        sub = (pull_owner >= 0) & (pull_weight > 0) if slab_memory.enabled(self) else self.bcode == 2
        rid = (self.bcode == 1) | (self.bcode == 5)
        col = self.bcode == 4
        for mask, target, weights in [(sub, slab, count_s), (rid, ridge, count_r), (col, resistance, count_c)]:
            ii = np.flatnonzero(mask)
            if not len(ii):
                continue
            torque = np.cross(self.bmid[ii], self.bn[ii])
            if target is slab:
                down_a = pull_owner[ii] == self.bp[ii]
                sign = np.where(down_a, 1., -1.)
                traction = pull_weight[ii]
                value = torque * (self.bl[ii] * sign * traction)[:, None]
                np.add.at(target, pull_owner[ii], value)
                np.add.at(weights, pull_owner[ii], self.bl[ii])
            else:
                sign = -1. if target is ridge else 1.
                value = torque * (self.bl[ii] * sign)[:, None]
                np.add.at(target, self.bp[ii], value)
                np.add.at(target, self.bq[ii], -value)
                np.add.at(weights, self.bp[ii], self.bl[ii])
                np.add.at(weights, self.bq[ii], self.bl[ii])
        slab /= np.maximum(count_s[:, None], 1)
        ridge /= np.maximum(count_r[:, None], 1)
        resistance /= np.maximum(count_c[:, None], 1)
        area = np.bincount(self.plate, weights=self.cell_area, minlength=self.capacity)
        ca = np.bincount(self.parcel_plate[self.kind == 2], weights=self.mass[self.kind == 2], minlength=self.capacity)
        drag = 1 + 2.5 * np.minimum(ca / np.maximum(area, 1), 1)
        target = (self.mantle + .0055 * self.config["slab_pull"] * slab
                  + .003 * self.config["ridge_push"] * ridge - .003 * resistance) / drag[:, None]
        # Slowly evolving basal traction supplies unresolved mantle forcing. The
        # noise is seeded, time-correlated, and does not prescribe continent paths.
        self.mantle[self.active] += self.rng.normal(0, .000015 * np.sqrt(dt), (np.count_nonzero(self.active), 3))
        self.mantle *= np.exp(-dt / 3000)
        self.omega += (target - self.omega) * (1 - np.exp(-dt / 24))
        self.omega[~self.active] = 0
        spin = np.sum(self.omega * area[:, None], axis=0) / self.earth_area
        self.omega[self.active] -= spin
        speed = np.linalg.norm(self.omega, axis=1)
        self.omega *= np.minimum(1, .014 / np.maximum(speed, 1e-12))[:, None]

    def _valid_loading_edges(self):
        # Welding and absorption happen after the boundary cache is built.
        # A transferred cell must not keep applying its former owner's load.
        return ((self.plate[self.ba] == self.bp) & (self.plate[self.bb] == self.bq)
                & self.active[self.bp] & self.active[self.bq])

    def _internal_loading(self, p, area, parcel_area, perimeter, valid_edges=None):
        """Return a dimensionless boundary-load proxy, not a stress in pascals.

        Opposing loads can cancel a plate's net torque while still deforming
        its interior. Fit the best rigid Euler response to the boundary
        traction field and retain its incompatible residual. Shared rigid
        rotation produces exactly zero load; size and enclosed neighbours
        amplify real loading but never initiate it by themselves.
        """
        if valid_edges is None:
            valid_edges = self._valid_loading_edges()
        edges = ((self.bp == p) | (self.bq == p)) & valid_edges
        weak = cratonic = 0.
        select = self.parcel_plate == p
        if np.any(select):
            weak = float(np.average(self.suture[select], weights=self.mass[select]))
            cratonic = float(self.mass[select & (self.kind == 2)].sum()) / max(parcel_area[p], 1)
        result = dict(load_proxy=0., tensile_speed_km_myr=0., compressive_speed_km_myr=0.,
                      incompatible_speed_km_myr=0., surrounded_neighbors=0,
                      continental_weakness=weak, craton_fraction=cratonic)
        if not np.any(edges):
            return result, None
        mid, length = self.bmid[edges], self.bl[edges]
        weights = length / max(float(length.sum()), 1.)
        other = np.where(self.bp[edges] == p, self.bq[edges], self.bp[edges])
        outward = self.bn[edges] * np.where(self.bp[edges] == p, 1., -1.)[:, None]
        velocity = np.cross(self.omega[other] - self.omega[p], mid) * RADIUS_KM
        normal = np.sum(velocity * outward, axis=1)
        extension, compression = np.maximum(normal, 0), np.maximum(-normal, 0)
        # A kinematic loading closure: tensile traction, somewhat weaker
        # compressive transmission, and a smaller contribution from shear.
        traction = outward * (extension - .6 * compression)[:, None]
        traction += .25 * (velocity - normal[:, None] * outward)
        matrix = np.eye(3) - np.einsum('n,ni,nj->ij', weights, mid, mid)
        torque = np.sum(np.cross(mid, traction) * weights[:, None], axis=0)
        fitted = np.linalg.solve(matrix + np.eye(3) * 1e-9, torque)
        residual = traction - np.cross(fitted, mid)
        incompatible = float(np.sqrt(np.sum(weights * np.sum(residual**2, axis=1))))
        tensile = float(np.sum(weights * extension))
        compressive = float(np.sum(weights * compression))
        shared = np.bincount(other, weights=length, minlength=self.capacity)
        surrounded = ((shared >= .9 * np.maximum(perimeter, 1)) & (area < area[p] * .4)
                      & (area > self.earth_area * .0005))
        surrounded[p] = False
        count = int(np.count_nonzero(surrounded))
        fraction = float(area[p] / self.earth_area)
        size_factor = .65 + math.sqrt(min(fraction / .15, 3.))
        load = (incompatible / 15 + .6 * tensile / 20 + .35 * compressive / 20)
        load *= size_factor * (1 + .3 * min(count, 3)) * (1 + .6 * weak) / (1 + 1.5 * cratonic)
        result.update(load_proxy=float(load), tensile_speed_km_myr=tensile,
                      compressive_speed_km_myr=compressive, incompatible_speed_km_myr=incompatible,
                      surrounded_neighbors=count)
        # Orient candidate fractures across the strongest differential loading.
        covariance = np.einsum('n,ni,nj->ij', weights, residual, residual)
        preferred = np.linalg.eigh(covariance)[1][:, -1] if incompatible > 1e-8 else None
        return result, preferred

    def _topology(self, dt):
        # Tiny ocean-only plates are absorbed into their largest neighbouring
        # plate. Continental parcels are never deleted during a topology change.
        area = np.bincount(self.plate, weights=self.cell_area, minlength=self.capacity)
        parcel_area = np.bincount(self.parcel_plate, weights=self.mass, minlength=self.capacity)
        if self.steps % 20 == 0:
            for p in np.flatnonzero(self.active):
                if (area[p] > self.earth_area * .001 or parcel_area[p] > 0
                        or np.any((self.plate == p) & (self.crust > 0)) or np.any(self.trace_plate == p)):
                    continue
                a, b = self.plate[self.edge_a], self.plate[self.edge_b]
                valid = ((a == p) & (b != p)) | ((b == p) & (a != p))
                neighbours = np.where(a[valid] == p, b[valid], a[valid])
                if not len(neighbours):
                    if area[p] == 0:
                        self.active[p] = False
                        self.support[p] = 0
                    continue
                q = int(np.bincount(neighbours, weights=self.edge_length[valid], minlength=self.capacity).argmax())
                self.plate[self.plate == p] = q
                self.parcel_plate[self.parcel_plate == p] = q
                self.trace_plate[self.trace_plate == p] = q
                self.support[q] += self.support[p]
                self.support[p] = 0
                self.active[p] = False
                reassign_ridge_episodes(self, int(self.plate_uid[p]), int(self.plate_uid[q]))
                self._record("plate_absorption", f"The remnant of {self.names[p]} was incorporated into {self.names[q]}.",
                             plates=(p, q), xyz=self.centres[p],
                             details={"source_plate_uid": int(self.plate_uid[p]), "target_plate_uid": int(self.plate_uid[q])})
        # Recompute areas after absorption and discard stale boundary owners.
        # Newly relocated boundaries enter the load calculation after the
        # normal end-of-step boundary refresh, avoiding a phantom load from
        # transferred crust or a consumed plate during this step.
        area = np.bincount(self.plate, weights=self.cell_area, minlength=self.capacity)
        parcel_area = np.bincount(self.parcel_plate, weights=self.mass, minlength=self.capacity)
        valid_edges = self._valid_loading_edges()
        perimeter = np.bincount(np.concatenate((self.bp[valid_edges], self.bq[valid_edges])),
                                weights=np.tile(self.bl[valid_edges], 2), minlength=self.capacity)
        self.stress_diagnostics = {}
        candidates = []
        backarc_hosts = backarc.protected_hosts(self)
        continental_area = np.bincount(self.parcel_plate[self.kind != 3],
                                       weights=self.mass[self.kind != 3], minlength=self.capacity)
        decay = math.exp(-dt / 180.)
        initial_ocean_uids = set(getattr(self, 'initial_ocean_plate_uids',
                                        (getattr(self, 'initial_ocean_plate_uid', None),)))
        for p in np.flatnonzero(self.active):
            diagnostic, preferred = self._internal_loading(int(p), area, parcel_area, perimeter, valid_edges)
            # A large ocean skirt cannot let a continental block bypass its
            # local material-damage requirement through the old ocean timer.
            oceanic = continental_area[p] == 0
            threshold, cooldown = (95., 140.) if oceanic else (75., 110.)
            age = self.t - self.born[p]
            initial_ocean = self.plate_uid[p] in initial_ocean_uids
            eligible = (not initial_ocean and int(self.plate_uid[p]) not in backarc_hosts
                        and parcel_area[p] > self.earth_area * .012
                        and area[p] > self.earth_area * .035)
            # Crust-free lithosphere can remain one ocean plate until consumed.
            # No convection-cell size rule or nonzero background rift timer.
            self.rift_clock[p] *= decay
            self.ocean_rift_clock[p] *= decay
            clock = self.ocean_rift_clock if oceanic else self.rift_clock
            if oceanic and eligible and age >= cooldown and self.config["rift_strength"] > 0:
                clock[p] += diagnostic["load_proxy"] * self.config["rift_strength"] * 180 * (1 - decay)
            diagnostic.update(accumulated_load_myr=float(clock[p]), rupture_threshold_myr=threshold,
                              cooldown_remaining_myr=float(max(cooldown - age, 0)),
                              eligible=bool(eligible), initial_ocean_protected=bool(initial_ocean))
            self.stress_diagnostics[int(p)] = diagnostic
            if not oceanic:
                diagnostic.update(model=progressive_rifting.MODEL)
            if oceanic and eligible and age >= cooldown and clock[p] > threshold and self.config["rift_strength"] > 0:
                candidates.append((float(clock[p] / threshold), int(p), oceanic, preferred, diagnostic))
        # At most one fracture per integration step keeps slot reuse and plate
        # identity recording unambiguous; loaded candidates persist to later steps.
        if progressive_rifting.commit(self):
            return
        if candidates and np.any(~self.active):
            _, p, oceanic, preferred, diagnostic = max(candidates, key=lambda row: row[0])
            if oceanic:
                self._split_ocean(p, preferred=preferred, loading=diagnostic)
            else:
                self._split(p, preferred=preferred, loading=diagnostic)

    def _choose_fracture(self, p, continental, preferred=None):
        """Choose a bent, craton-avoiding cut without detached new plate crumbs."""
        from fracture import barriers_from_grid, component_labels, make_fracture
        mask = self.plate == p
        select = self.parcel_plate == p
        if not np.any(mask) or (continental and np.count_nonzero(select) < 20):
            return None
        if continental:
            center = _unit(np.sum(self.pos[select] * self.mass[select, None], axis=0))
        else:
            center = _unit(np.sum(self.xyz[mask] * self.cell_area[mask, None], axis=0))
        if np.linalg.norm(center) < .5:
            center = self.xyz[np.flatnonzero(mask)[0]]
        barriers = barriers_from_grid(self.xyz, self.crust, self.w, self.h,
                                      mask=mask, area=self.cell_area)
        craton = mask & (self.crust == 2)
        craton_labels, craton_count = component_labels(craton, self.w, self.h)
        parent_labels, parent_count = component_labels(mask, self.w, self.h)
        candidates = []
        for attempt in range(12):
            v = self.rng.normal(size=3)
            if preferred is not None and attempt < 5:
                v = preferred + v * (.2 * attempt)
            normal = _unit(v - center * np.dot(v, center))
            if np.linalg.norm(normal) < .5:
                continue
            crack = make_fracture(center, normal, int(self.rng.integers(2**31)), barriers=barriers)
            grid_dist = crack.signed_distance(self.xyz)
            region = mask & (grid_dist > 0)
            area_fraction = float(self.cell_area[region].sum() / self.cell_area[mask].sum())
            if not .12 < area_fraction < .88:
                continue
            if craton_count:
                positive = np.bincount(craton_labels[craton], weights=(grid_dist[craton] > 0), minlength=craton_count)
                sizes = np.bincount(craton_labels[craton], minlength=craton_count)
                if np.any((positive > 0) & (positive < sizes)):
                    continue
            # Raster craton cells can disappear below the display threshold.
            # Enduring material groups still prohibit cutting that strong body.
            parcel_side, _ = self._material_cut(crack)
            if splits_protected_groups(self.parcel_craton[select], parcel_side[select]):
                continue
            if continental:
                dist = crack.signed_distance(self.pos[select])
                frac = float(self.mass[select][parcel_side[select]].sum() / self.mass[select].sum())
                if not .12 < frac < .88:
                    continue
                belt = np.abs(dist) < .10
                weak = float(np.mean(self.suture[select][belt])) if np.any(belt) else 0.
                crat = float(np.mean(self.kind[select][belt] == 2)) if np.any(belt) else 0.
                score = min(frac, 1-frac) + .35 * weak - .8 * crat
            else:
                frac = area_fraction
                belt = mask & (np.abs(grid_dist) < .08)
                ocean_belt = belt & (self.crust == 0)
                if not np.any(ocean_belt):
                    continue
                ocean_share = float(self.cell_area[ocean_belt].sum() / max(self.cell_area[belt].sum(), 1.))
                if ocean_share < .75:
                    continue
                young = float(np.average(np.exp(-self.age[ocean_belt] / 90), weights=self.cell_area[ocean_belt]))
                score = min(frac, 1-frac) + .25 * young + .1 * ocean_share
            # Retain only the compact curve, not twelve whole-world distance
            # rasters at the largest laptop grid. Re-evaluate finalists below.
            candidates.append((score, crack, frac))
        # Connectivity is checked only for the strongest candidates. Separate
        # inherited islands are allowed; severing new substantial crumbs is not.
        for _, crack, frac in sorted(candidates, key=lambda row: row[0], reverse=True)[:4]:
            grid_dist = crack.signed_distance(self.xyz)
            disconnected = 0.
            acceptable = True
            for half in (mask & (grid_dist > 0), mask & (grid_dist <= 0)):
                labels, count = component_labels(half, self.w, self.h)
                weights = np.bincount(labels[half], weights=self.cell_area[half], minlength=count)
                inherited = np.full(count, parent_count, dtype=np.int32)
                np.minimum.at(inherited, labels[half], parent_labels[half])
                largest = np.zeros(parent_count)
                np.maximum.at(largest, inherited, weights)
                total = np.bincount(parent_labels[half], weights=self.cell_area[half], minlength=parent_count)
                crumbs = float(np.sum(total - largest) / max(total.sum(), 1))
                disconnected = max(disconnected, crumbs)
                if crumbs > .025:
                    acceptable = False
                    break
            if acceptable:
                return dict(crack=crack, fraction=frac, grid_distance=grid_dist,
                            disconnected_fraction=disconnected)
        return None

    def _split_ocean(self, p, preferred=None, loading=None):
        mask = self.plate == p
        ocean = mask & (self.crust == 0)
        if np.count_nonzero(ocean) < 20 or not np.any(~self.active):
            return False
        chosen = self._choose_fracture(p, False, preferred)
        if chosen is None:
            self.ocean_rift_clock[p] *= .7
            return False
        crack, frac = chosen["crack"], chosen["fraction"]
        center, normal = crack.center, crack.normal
        region = mask & (chosen["grid_distance"] > 0)
        q = int(np.flatnonzero(~self.active)[0])
        self.count = max(self.count, q + 1)
        self.split_serial += 1
        self.active[q] = True
        self._new_plate_identity(q, p)
        self.names[q] = f"Ocean rift {self.split_serial:02d}"
        self.polarity[q, :] = self.polarity[:, q] = -1
        self.collision_clock[q, :] = self.collision_clock[:, q] = 0
        self.support[q] = 0
        self.support[q, region] = self.support[p, region]
        self.support[p, region] = 0
        self.plate[region] = q
        self._transfer_ridge_hosts(p, q)
        parcel_side, trace_side = self._material_cut(crack)
        select = (self.parcel_plate == p) & parcel_side
        self.parcel_plate[select] = q
        trace_select = (self.trace_plate == p) & trace_side
        self.trace_plate[trace_select] = q
        opening = np.cross(center, normal) * .0035 * self.config["rift_strength"]
        self.omega[q] = self.omega[p] + opening * (1-frac)
        self.omega[p] -= opening * frac
        self.mantle[q] = self.mantle[p] + opening * (1-frac)
        self.mantle[p] -= opening * frac
        self.centres[p] = _unit(center - normal * .3)
        self.centres[q] = _unit(center + normal * .3)
        self.born[p] = self.born[q] = self.t
        self.rift_clock[p] = self.rift_clock[q] = 0
        self.ocean_rift_clock[p] = self.ocean_rift_clock[q] = 0
        self.event_keys = {key for key in self.event_keys if not (isinstance(key, tuple) and key[0] == "collision" and q in key[1:])}
        self.process_totals["rift_events"] += 1
        self._record("rift", f"Incompatible boundary loading fractured the oceanic part of {self.names[p]} along a bent weak belt, opening a ridge and creating {self.names[q]}.",
                     plates=(p, q), xyz=center,
                     details={"setting": "oceanic", "parent_plate_uid": int(self.plate_uid[p]),
                              "new_plate_uid": int(self.plate_uid[q]), "new_plate_area_fraction": frac,
                              "fracture_geometry": "curved spherical crack with rounded craton detours",
                              "new_disconnected_area_fraction": chosen["disconnected_fraction"],
                              "loading_model": "leaky kinematic boundary-load proxy; not physical stress in pascals",
                              "loading": deepcopy(loading or {})})
        return True

    def _split(self, p, preferred=None, loading=None, *, chosen=None, progressive=False):
        from coastal_partition import bounded_continental_partition

        select = self.parcel_plate == p
        if np.count_nonzero(select) < 20:
            return
        if chosen is None:
            chosen = self._choose_fracture(p, True, preferred)
        if chosen is None:
            self.rift_clock[p] *= .7
            return False
        crack, frac = chosen["crack"], chosen["fraction"]
        center, normal = crack.center, crack.normal
        available = np.flatnonzero(~self.active)
        if not len(available):
            return
        q = int(available[0])
        self.count = max(self.count, q + 1)
        self.split_serial += 1
        self.active[q] = True
        self._new_plate_identity(q, p)
        self.names[q] = f"Rift plate {self.split_serial:02d}"
        self.polarity[q, :] = self.polarity[:, q] = -1
        self.collision_clock[q, :] = self.collision_clock[:, q] = 0
        self.support[q] = 0
        self.event_keys = {key for key in self.event_keys if not (isinstance(key, tuple) and key[0] == "collision" and q in key[1:])}
        # A continental crack can terminate at an existing plate edge instead
        # of continuing lengthwise through every remote ocean strip of its
        # parent. Preserve its continental sides and coherent material cut;
        # only shorten the new ocean connection where shore geometry permits.
        region, ocean_continuation = bounded_continental_partition(
            self.plate == p, self.crust > 0, chosen["grid_distance"] > 0,
            self.w, self.h, self.cell_area, self.edge_a, self.edge_b, self.edge_length)
        self.plate[region] = q
        self.support[q, region] = self.support[p, region]
        self.support[p, region] = 0
        self._transfer_ridge_hosts(p, q)
        parcel_distance = crack.signed_distance(self.pos)
        if progressive:
            parcel_side, trace_side = chosen['parcel_side'], chosen['trace_side']
        else:
            parcel_side, trace_side = self._material_cut(crack)
        parcel_region = select & parcel_side
        self.parcel_plate[parcel_region] = q
        trace_select = self.trace_plate == p
        trace_distance = crack.signed_distance(self.trace_xyz)
        trace_region = trace_select & trace_side
        self.trace_plate[trace_region] = q
        # Progressive daughters fit the local response; the explicit legacy
        # fracture helper retains its original prescribed opening for fixtures.
        old = self.omega[p].copy()
        if progressive:
            self.omega[p] = old + chosen['rotations'][0]
            self.omega[q] = old + chosen['rotations'][1]
            self.mantle[q] = self.mantle[p].copy()
        else:
            opening = np.cross(center, normal) * .0035 * self.config["rift_strength"]
            self.omega[p] = old - opening * frac
            self.omega[q] = old + opening * (1-frac)
            self.mantle[p] -= opening * frac
            self.mantle[q] = self.mantle[p] + opening
        self.centres[p] = _unit(center - normal * .3)
        self.centres[q] = _unit(center + normal * .3)
        self.rift_clock[p] = self.rift_clock[q] = 0
        self.ocean_rift_clock[p] = self.ocean_rift_clock[q] = 0
        self.born[p] = self.born[q] = self.t
        weak_belt = select & (np.abs(parcel_distance) < .09)
        system = next((row for row in self.rift_systems
                       if progressive and row['id'] == (loading or {}).get('rift_system_id')), None)
        rid = system.get('source_rift_id') if system else None
        if rid is None:
            rid = self._new_rift_record('continental breakup', (p, q), center)
        new_scar = weak_belt & (self.rift_id < 0) & (self.kind != 3)
        self.rift_id[new_scar] = rid
        self.rift_birth_myr[new_scar] = self.t
        self.rift_tangent[new_scar] = self._rift_tangents(crack, self.pos[new_scar])
        self.suture[weak_belt] = np.maximum(self.suture[weak_belt], .8)
        trace_belt = trace_select & (np.abs(trace_distance) < .09)
        new_scar = trace_belt & (self.trace_rift_id < 0) & (self.trace_kind != 3)
        self.trace_rift_id[new_scar] = rid
        self.trace_rift_birth_myr[new_scar] = self.t
        self.trace_rift_tangent[new_scar] = self._rift_tangents(crack, self.trace_xyz[new_scar])
        self.trace_suture[trace_belt] = np.maximum(self.trace_suture[trace_belt], .8)
        if not progressive:
            structure_engine.cut_rift(self, weak_belt, trace_belt, 250.)
        self.process_totals["rift_events"] += 1
        self._record("rift", (f"A developed belt of local extension and damage split {self.names[p]}, creating {self.names[q]}." if progressive else
                             f"Accumulated incompatible boundary loading split {self.names[p]}, creating {self.names[q]} along a bent, preferentially weak continental belt."),
                     plates=(p, q), xyz=center,
                     details={"setting": "continental", "parent_plate_uid": int(self.plate_uid[p]),
                              "source_rift_id": rid,
                              "new_plate_uid": int(self.plate_uid[q]), "new_plate_mass_fraction": frac,
                              "rift_belt_relief_reduction_m": 0 if progressive else 250,
                              "fracture_geometry": "developed material damage corridor" if progressive else "curved spherical crack with rounded craton detours",
                              "new_disconnected_area_fraction": chosen["disconnected_fraction"],
                              "ocean_continuation": ocean_continuation,
                              "loading_model": progressive_rifting.MODEL if progressive else "leaky kinematic boundary-load proxy; not physical stress in pascals",
                              "loading": deepcopy(loading or {})})
        return True

    def step(self, dt=None):
        """Advance one integration step (normally called by snapshots)."""
        if dt is None:
            dt = self.config["dt_myr"]
        if not math.isfinite(float(dt)) or dt <= 0 or dt > 5:
            raise ValueError("Step length must be finite, positive and at most 5 Myr.")
        self._forces(dt)
        backarc.apply_motion(self, dt)
        transform_coupling.apply(self, dt)
        previous_omega = self.omega.copy()
        previous_plate_uid = self.plate_uid.copy()
        self._advect(dt)
        self.t += dt
        self.steps += 1
        self._rasterize()
        self._boundaries()
        self._migrate_trenches(dt)
        trench_history.update(self, dt)
        backarc_pending = backarc.update(self, dt)
        self._update_ridge_windows(dt)
        self._deform_and_accrete(dt)
        if not backarc.commit_pending(self, backarc_pending):
            self._topology(dt)
        self._rasterize()
        self._boundaries()
        ridge_spreading.refresh(self)
        self._update_surface_domains(dt, previous_omega, previous_plate_uid)

    def _label_codes(self):
        """Labels for the map, statistics and exports. The raster edition shows its
        own kinematic classification; the native engine overrides this with the
        mechanism actually running on each boundary."""
        return np.asarray(self.bcode, np.uint8)

    def snapshot(self):
        """Copy export fields; arrays are flat and safe to retain or serialize."""
        # GDH1-style thermal subsidence: young half-space cooling gives way to
        # exponential plate cooling. Bathymetry is metres below nominal sea level.
        ocean = np.where(self.age < 20, -2600 - 365 * np.sqrt(self.age),
                         -5651 + 2473 * np.exp(-.0278 * self.age))
        ocean += self.ocean_relief
        land = self.land_height.copy()
        # Collision roots now come from the material columns. Raster overlap
        # cannot supply a second, unrelated isostatic elevation bonus.
        elevation = np.where(self.crust > 0, land, ocean)
        # The history records material elevation and modeled thermal support.
        # Fine cosmetic roughness belongs to the separate terrain export stage;
        # adding it here would create shorelines fixed to the world coordinates.
        thermal = sample_ridge_effects(self.ridge_episodes, self.xyz, self.plate_uid[self.plate], self.t)
        elevation += thermal['thermal_support_m']
        # The 9000 m ceiling predates the collision surface. Two sibling sites --
        # native_engine's snapshot elevation and native_frame_sampling's exposed
        # elevation -- already release it when collision_surface_version is 1,
        # because that model supplies real orogenic support from retained
        # overlapping columns. This one was missed, and it is the site that feeds
        # mesh_elevation_m, so recorded peaks flattened onto an exact 9000 m mesa:
        # 5 faces at 292 Myr, 192 by 318 Myr, while the underlying
        # material_collision_support_m already reached 10,931 m.
        elevation = (elevation if getattr(self, 'collision_surface_version', 0) == 1
                     else np.clip(elevation, -10000, 9000)).astype(np.float32)
        area = np.bincount(self.plate, weights=self.cell_area, minlength=self.capacity)
        speed = np.linalg.norm(np.cross(self.omega[self.plate], self.xyz), axis=1) * RADIUS_KM * .1
        ocean_mask = self.crust == 0
        # Arc source area is a 25-km magma-volume equivalent. Fixed-footprint
        # deposits thicken existing material without creating reference area.
        arc_reference_created = (self.process_totals["arc_added_km2"]
                                 - self.process_totals.get("arc_deposited_source_area_km2", 0.))
        stats = {"continental_area_km2": float(self.cell_area[~ocean_mask].sum()),
                 "continental_fraction": float(self.cell_area[~ocean_mask].sum() / self.earth_area),
                 "craton_area_km2": float(self.cell_area[self.crust == 2].sum()),
                 "buoyant_crust_mass_km2": float(self.mass.sum()),
                 "original_continental_mass_km2": self.original_mass,
                 "continental_mass_retained_fraction": float((self.mass.sum() - arc_reference_created) / self.original_mass) if self.original_mass else 1.,
                 "craton_mass_retained_fraction": float(self.mass[self.kind == 2].sum() / self.original_craton_mass) if self.original_craton_mass else 1.,
                 "mean_ocean_age_myr": float(np.average(self.age[ocean_mask], weights=self.cell_area[ocean_mask])) if np.any(ocean_mask) else 0.,
                 "mean_plate_speed_cm_yr": float(np.average(speed, weights=self.cell_area)),
                 "max_plate_speed_cm_yr": float(speed.max()),
                 "active_plates": int(np.count_nonzero(area > 0)),
                 "surface_domains": len(self.domains),
                 "parcel_count": int(len(self.mass)),
                 "max_elevation_m": float(elevation.max()),
                 "min_elevation_m": float(elevation.min()),
                 "mean_elevation_m": float(np.average(elevation, weights=self.cell_area)),
                 "land_fraction": float(self.cell_area[elevation > 0].sum() / self.earth_area),
                 "integration_steps": int(self.steps),
                 "active_slab_windows": int(sum(e['started_myr'] <= self.t < e['end_myr'] for e in self.ridge_episodes)),
                 **self.process_totals}
        labels = self._label_codes()
        for code, name in [(1, "ridge"), (2, "subduction"), (3, "transform"), (4, "collision"), (5, "rift")]:
            stats[f"{name}_length_km"] = float(self.bl[labels == code].sum())
        # collision_length_km above counts only bcode==4, which requires continental
        # crust on BOTH sides of a graph edge. The thrust stacking that actually builds
        # mountains is done by collision_contacts on overlapping material sheets and is
        # invisible to that number -- reading it alone once produced the conclusion that
        # "collision is not happening" while 5,000 km of suture was actively loading.
        # Carry the contact-based measures beside it so the two cannot be confused.
        _cd = getattr(self, 'collision_diagnostics', {}) or {}
        for _src, _dst in (('active_contacts', 'collision_active_contacts'),
                           ('sutured_contacts', 'collision_sutured_contacts'),
                           ('pair_overlap_area_km2', 'collision_overlap_area_km2')):
            if _src in _cd:
                stats[_dst] = float(_cd[_src])
        # Convergence the exclusive classification still discards: transform-labelled
        # edges that are actually closing. Measured offline at 344 Myr as 656 km
        # (2.15% of trench) at <= 4.27 km/Myr, which is why full strain partitioning
        # was not built. If this ever grows toward the trench length, revisit that.
        _oblique = (self.bcode == 3) & (self.normal_speed < -2.)
        stats["oblique_closing_length_km"] = float(self.bl[_oblique].sum())
        # Counters created lazily on first increment would otherwise be ABSENT until they
        # fire, which reads as "not instrumented" rather than zero. Report zero instead.
        for _key in ("rift_events_backarc", "rift_events_breakup", "plate_splits_isolated",
                     "collision_pair_occurrences"):
            stats.setdefault(_key, 0.)
        geological, column_stats = geology_snapshot.snapshot(self)
        stats.update(column_stats)
        # A surviving material marker can belong to a plate with no dominant
        # raster cell. Keep that owner's real name and generation in history,
        # rather than forcing inspectors to invent a name from a reusable slot.
        recorded_plates = np.union1d(np.flatnonzero(area > 0), np.unique(self.trace_plate))
        plates = [dict(id=int(p), name=self.names[p], angular_velocity=self.omega[p].tolist(),
                       uid=int(self.plate_uid[p]), generation=int(self.plate_generation[p]),
                       active=bool(self.active[p]),
                       parent_uid=int(self.plate_parent_uid[p]), created_myr=float(self.plate_created[p]),
                       area_km2=float(area[p]), born_myr=float(self.born[p]),
                       internal_loading=deepcopy(getattr(self, 'stress_diagnostics', {}).get(int(p))))
                  for p in recorded_plates]
        traces = {f"trace_{name}": getattr(self, f"trace_{name}").copy()
                  for name in ("id", "xyz", "plate", "kind", "origin_kind", "birth_myr", "relief_m",
                               "uplift_m", "extension_m", "erosion_m", "adjustment_m", "suture", "patch", "craton", "ridge_uplift_m",
                               "rift_id", "rift_birth_myr", "rift_extension_m", "inversion_uplift_m")}
        traces["trace_plate_uid"] = self.plate_uid[self.trace_plate].copy()
        traces["trace_ridge_thermal_m"] = sample_ridge_effects(self.ridge_episodes, self.trace_xyz,
                                                              self.plate_uid[self.trace_plate], self.t)['thermal_support_m']
        episodes = []
        for original in self.ridge_episodes:
            if original['started_myr'] <= self.t < original['end_myr']:
                episode = deepcopy(original)
                effect = sample_ridge_effects([episode], np.asarray([episode['center']]),
                                             np.asarray([episode['overriding_plate_uid']]), self.t)
                episode['heat'] = float(effect['heat'][0])
                episode['phase'] = 'heating' if self.t < episode['peak_myr'] else 'cooling'
                episodes.append(episode)
        return dict(time_myr=round(float(self.t), 6), width=self.w, height=self.h,
                    elevation=elevation, plate=self.plate.copy(), crust=self.crust.copy(),
                    age=self.age.astype(np.float32), boundary=self.boundary.copy(),
                    domain=self.domain.copy(), domains=deepcopy(self.domains),
                    stats=stats, events=deepcopy(self.events), plates=plates, ridge_episodes=episodes,
                    rift_records=deepcopy(self.rift_records),
                    backarc_basins=deepcopy(self.backarc_basins), **traces,
                    domain_diagnostics=deepcopy(getattr(self, 'domain_diagnostics', {})),
                    initial_plate_topology=deepcopy(getattr(self, 'initial_plate_topology', None)),
                    initial_plate_topology_diagnostics=deepcopy(getattr(self, 'initial_plate_topology_diagnostics', {})),
                    boundary_geometry_diagnostics=deepcopy(getattr(self, 'boundary_geometry_diagnostics', {})),
                    spreading_diagnostics=deepcopy(getattr(self, 'spreading_diagnostics', {})),
                    trench_systems=deepcopy(self.trench_systems),
                    trench_history_diagnostics=deepcopy(getattr(self, 'trench_history_diagnostics', {})),
                    **slab_memory.snapshot(self), **transform_coupling.snapshot(self),
                    local_accretion_contacts=deepcopy(getattr(self, 'local_accretion_contacts', [])),
                    **geological, **progressive_rifting.snapshot(self), history_version=1,
                    history={"kind": "representative buoyant material markers",
                             "initial_marker_limit": self.initial_marker_limit,
                             "arc_marker_limit": self.arc_marker_limit,
                             "coverage": "Continental, cratonic and sampled juvenile arc material; no ocean material tracking.",
                             "relief": "Marker relief is the local model component, not full terrain elevation or a GoSPL forcing field.",
                             "process_counters": {"uplift_m": "Cumulative positive gain from shortening, magmatic construction, thermal support and foreland unloading.",
                                                  "extension_m": "Cumulative tectonic lowering from thinning, rift cooling and foreland loading.",
                                                  "erosion_m": "Cumulative net denudation loss after associated Airy rebound; gross removal and rebound are separate diagnostics.",
                                                  "adjustment_m": "Cumulative signed gain from arc replenishment and material-class datum changes."}})

    def snapshots(self, cancel: Callable[[], bool] | None = None):
        """Yield t=0, requested history intervals, and the exact terminal time.

        Cancellation is checked at each small integration step. Snapshots do not
        share mutable simulation arrays. Integrations land exactly on requested
        history times even when dt does not divide the output interval.
        """
        if cancel is not None and cancel():
            return
        yield self.snapshot()
        stop = self.config["duration_myr"]
        cadence = self.config["snapshot_myr"]
        next_output = min(stop, (math.floor(self.t / cadence) + 1) * cadence)
        while self.t < stop - 1e-8:
            if cancel is not None and cancel():
                return
            dt = min(self.config["dt_myr"], next_output - self.t, stop - self.t)
            self.step(dt)
            if self.t >= next_output - 1e-8:
                yield self.snapshot()
                next_output = min(stop, next_output + cadence)
