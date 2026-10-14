"""Mesh-authoritative Deep Time engine and raster review adapter.

Ocean fields evolve on a closed unstructured spherical finite-volume mesh.
Buoyant crust is a connected moving triangular material surface. The latitude
and longitude grid is used for initial artwork import and output projection.
The retained raster engine supplies initial curved partitioning and reusable
process kernels; none of its raster transport runs after initialization.
"""
from __future__ import annotations
from copy import deepcopy
import math
import hashlib
import numpy as np
import raster_engine as legacy
import mesh_geometry as geometry
import material_surface as surface
import mesh_transport as transport
import mesh_coverage
import material_reconstruction
import native_boundary_geometry
import boundary_labels
import native_initial_ownership
import material_initialization
import native_material_evolution
import bounded_gravity
import native_material_adaptivity
import native_arc_material
import native_spreading
import native_subduction
import normal_partition
import arc_surface
import native_frame_sampling
import continental_margin
import collision_contacts
import collision_surface
import eclogite_sink
import plate_balance
import physics_profile
import evolution_policy
import continental_lifecycle
import production_restart_migration
import restart_boundary
import retained_phase_profile
import primordial_subduction
import supercontinent_ring
import primordial_ocean
import world_design
import slab_anchors
import ocean_rifting
import force_rifting
import structure_engine
import geology_snapshot
import progressive_rifting
import rift_traction
import enhanced_rifting
import lip_events
import local_accretion
import localized_accretion
import trench_history
import slab_memory
import effective_subduction
import coarse_history
import transform_coupling
import backarc
from basement_relief import initial_basement_relief
from ridge_interaction import sample_ridge_effects
from ridge_geometry import normal_motion_threshold

# Which plate goes down at a trench is a question about density. Oceanic lithosphere
# thickens as sqrt(age) and its column grows denser than the asthenosphere it displaces,
# saturating once the plate is thermally mature; continental crust is too light to be
# drawn down at any age. These express both as a buoyancy in effective metres so the two
# sides of a trench can be compared on one scale. The numbers are the observed relief
# contrasts they stand for -- continents ride 4-5 km above abyssal plains, and a mature
# slab's density excess is of the same order with the opposite sign -- not fitted values.
CONTINENTAL_BUOYANCY_M = 5000.
OCEANIC_NEGATIVE_BUOYANCY_M = 3500.
SUBDUCTION_AGE_SATURATION_MYR = 80.     # GDH1 flattens beyond roughly this age
# An established subduction zone does not reverse because the crust sampled at its trench
# drifted a little. On Earth a polarity reversal needs a buoyant block to arrive and jam
# the trench (Solomon Islands, Taiwan); it is rare and discrete. The margin is wide enough
# that composition noise cannot trip it and narrow enough that a real continental arrival
# does. Without this, polarity was re-elected from scratch every step: as the isolated-split
# fix spawned ocean slivers, southern trench ids climbed 218 -> 636 in 120 Myr and both
# polarities appeared on one boundary at once.
POLARITY_REVERSAL_MARGIN_M = 1500.




def _budget_stage(name, function, *args, **kwargs):
    import budget
    budget.require_physical_time(name)
    budget.record(name, event='entry')
    try:
        result = function(*args, **kwargs)
    except BaseException as error:
        budget.record(name, event='exception', error_type=type(error).__name__, message=str(error))
        raise
    budget.record(name, event='return')
    budget.require_physical_time(name + ':completed')
    return result

def _continental(crust):
    """The side is continental if most of the sampled contact carries continental crust."""
    return np.mean(crust > 0) > .5


def _oceanic(crust):
    """Clearly oceanic: too little continental crust at the contact to resist subduction."""
    return np.mean(crust > 0) < .25


def _subduction_buoyancy(crust, age):
    """Net buoyancy of one side of a trench, positive upward, in effective metres.

    The heavier side is the one that subducts. Where both sides are oceanic this
    reduces to "older goes down", because age is what makes ocean floor dense -- so
    the retired age rule is recovered exactly in the case it was right about, without
    letting a continent be assigned as the downgoing plate merely for being old.
    """
    continental = float(np.mean(np.asarray(crust) > 0))
    maturity = math.sqrt(min(float(np.mean(age)), SUBDUCTION_AGE_SATURATION_MYR)
                         / SUBDUCTION_AGE_SATURATION_MYR)
    return (CONTINENTAL_BUOYANCY_M * continental
            - OCEANIC_NEGATIVE_BUOYANCY_M * maturity * (1. - continental))

RADIUS_KM = legacy.RADIUS_KM
NATIVE_DEFAULTS = dict(mesh_level=4, deforming_regions=1, adaptive_refinement=1,
                      deformation_width_km=400., material_face_budget=0,
                      force_limit_rifting=None,
                      coast_geometry_level=4, continental_margin_width_km=150., continental_shelf_depth_m=180.,
                      world_design=None, enhanced_rifting=None, lip_events=None, physics_profile='legacy',
                      trench_persistence='kinematic', slab_allocation='uniform',
                      primordial_subduction=None, primordial_ocean=None, subduction_response='fixed_trench',
                      supercontinent_ring=None,
                      retained_phases='disabled', continental_lifecycle=None, rift_traction=None,
                      effective_subduction=None, history_mode=None)
DEFAULT_CONFIG = dict(legacy.DEFAULT_CONFIG, **NATIVE_DEFAULTS)


def validate_config(config=None):
    incoming = dict(config or {})
    history_mode = coarse_history.normalize(incoming.pop('history_mode', None))
    coarse_history.select_config(incoming, history_mode)
    native = dict(retained_phases=retained_phase_profile.normalize(incoming.pop('retained_phases', 'disabled')),
                  physics_profile=physics_profile.normalize(incoming.pop('physics_profile', 'legacy')),
                  subduction_response=physics_profile.normalize_subduction_response(
                      incoming.pop('subduction_response', 'fixed_trench')),
                  primordial_ocean=primordial_ocean.normalize(incoming.pop('primordial_ocean', None)),
                  primordial_subduction=primordial_subduction.normalize(incoming.pop('primordial_subduction', None)),
                  supercontinent_ring=supercontinent_ring.normalize(incoming.pop('supercontinent_ring', None)),
                  world_design=world_design.normalize(incoming.pop('world_design', None)),
                  trench_persistence=trench_history.normalize_persistence(incoming.pop('trench_persistence', 'kinematic')),
                  slab_allocation=slab_anchors.normalize(incoming.pop('slab_allocation', 'uniform')),
                  enhanced_rifting=enhanced_rifting.normalize(incoming.pop('enhanced_rifting', None)),
                  force_limit_rifting=force_rifting.normalize(incoming.pop('force_limit_rifting', None)),
                  lip_events=lip_events.normalize(incoming.pop('lip_events', None)),
                  continental_lifecycle=continental_lifecycle.normalize(incoming.pop('continental_lifecycle', None)),
                  rift_traction=rift_traction.normalize(incoming.pop('rift_traction', None)),
                  effective_subduction=effective_subduction.normalize(incoming.pop('effective_subduction', None)))
    if native['effective_subduction']['enabled']:
        if native['physics_profile'] != 'reviewed_v1' or not native['primordial_subduction']['enabled']:
            raise ValueError('Effective subduction requires reviewed_v1 and explicit primordial_subduction declarations.')
        if native['subduction_response'] != 'fixed_trench':
            raise ValueError('Effective subduction requires fixed_trench and its one-sided traction closure.')
        if native['continental_lifecycle']['enabled'] or native['rift_traction']['enabled']:
            raise ValueError('Effective subduction is incompatible with detailed continental_lifecycle and rift_traction.')
        if native['retained_phases'] != 'disabled':
            raise ValueError('Effective subduction cannot enable the detailed retained_phases entry closure.')
        if native['trench_persistence'] != 'kinematic' or native['slab_allocation'] != 'uniform':
            raise ValueError('Effective subduction cannot select detailed attached_slab persistence or located slab allocation.')
    if native['retained_phases'] != 'disabled' and native['physics_profile'] != 'reviewed_v1':
        raise ValueError('Retained phases require the reviewed_v1 physics profile.')
    if native['primordial_subduction']['enabled'] and native['physics_profile'] != 'reviewed_v1':
        raise ValueError('Primordial subduction requires the reviewed_v1 physics profile.')
    if native['supercontinent_ring']['enabled']:
        if native['physics_profile'] != 'reviewed_v1' or not native['primordial_subduction']['enabled']:
            raise ValueError('A supercontinent ring requires reviewed_v1 and primordial_subduction, which seeds its slabs.')
        if native['primordial_ocean']['enabled']:
            raise ValueError('supercontinent_ring and primordial_ocean are alternative ocean initializations.')
    if native['primordial_ocean']['enabled'] and native['physics_profile'] != 'reviewed_v1':
        raise ValueError('Primordial ocean splitting requires the reviewed_v1 physics profile.')
    if native['trench_persistence'] != 'kinematic' and native['physics_profile'] != 'reviewed_v1':
        raise ValueError('Attached-slab trench persistence requires the reviewed_v1 physics profile.')
    if native['slab_allocation'] != 'uniform' and native['physics_profile'] != 'reviewed_v1':
        raise ValueError('Located slab allocation requires the reviewed_v1 conservative slab inventory.')
    if native['subduction_response'] != 'fixed_trench' and native['physics_profile'] != 'reviewed_v1':
        raise ValueError('Moving-hinge subduction requires the reviewed_v1 physics profile.')
    if native['force_limit_rifting']['enabled'] and native['physics_profile'] != 'reviewed_v1':
        raise ValueError('Force-limit rifting requires the reviewed_v1 force balance.')
    if native['continental_lifecycle']['enabled'] and native['physics_profile'] != 'reviewed_v1':
        raise ValueError('Continental entry/breakoff lifecycle requires the reviewed_v1 physics profile.')
    if native['rift_traction']['enabled'] and native['physics_profile'] != 'reviewed_v1':
        raise ValueError('Force-derived rift traction requires the reviewed_v1 physics profile.')
    if native['rift_traction']['enabled'] and native['subduction_response'] != 'moving_hinge_v1':
        raise ValueError('Force-derived rift traction v1 requires moving_hinge_v1 reciprocal slab work.')
    if native['rift_traction']['enabled'] and native['enhanced_rifting']['enabled']:
        raise ValueError('Force-derived rift traction v1 cannot be combined with enhanced_rifting realized-material mode.')
    limits = dict(deforming_regions=(0, 1), adaptive_refinement=(0, 3),
                  deformation_width_km=(100, 1500), material_face_budget=(0, 2000000),
                  coast_geometry_level=(2, 6), continental_margin_width_km=(10, 500),
                  continental_shelf_depth_m=(20, 500))
    continuous = ('deformation_width_km', 'continental_margin_width_km', 'continental_shelf_depth_m')
    for key, (low, high) in limits.items():
        value = incoming.pop(key, NATIVE_DEFAULTS[key])
        try:
            number = float(value)
        except (ValueError, TypeError):
            raise ValueError(f'{key} must be a finite number from {low} to {high}.') from None
        if (isinstance(value, bool) or not np.isfinite(number) or not low <= number <= high
                or (key not in continuous and number != int(number))):
            raise ValueError(f'{key} must be a finite number from {low} to {high}.')
        native[key] = number if key in continuous else int(number)
    level = incoming.pop('mesh_level', DEFAULT_CONFIG['mesh_level'])
    if isinstance(level, bool) or not np.isscalar(level):
        raise ValueError('Mesh level must be an integer from 2 through 6.')
    try:
        value = float(level)
    except (TypeError, ValueError):
        raise ValueError('Mesh level must be an integer from 2 through 6.') from None
    if not np.isfinite(value) or value != int(value) or not 2 <= value <= 6:
        raise ValueError('Mesh level must be an integer from 2 through 6.')
    return dict(legacy.validate_config(incoming), mesh_level=int(value), history_mode=history_mode, **native)


def make_initial(config=None, preset='pangaea'):
    cfg = validate_config(config)
    initial = legacy.make_initial({k:v for k,v in cfg.items() if k not in NATIVE_DEFAULTS}, preset)
    initial['world_design'] = deepcopy(cfg['world_design'])
    return initial


class Simulation(legacy.Simulation):
    def __init__(self, config=None, initial=None):
        config = dict(config or {})
        if isinstance(initial, dict) and 'world_design' in initial:
            design = world_design.normalize(initial['world_design'])
            if 'world_design' in config and world_design.normalize(config['world_design']) != design:
                raise ValueError('The starting map and run configuration specify different world designs.')
            config['world_design'] = design
        initial_subduction = (primordial_subduction.normalize(initial.get('initial_subduction'))
                              if isinstance(initial, dict) and initial.get('initial_subduction') is not None else None)
        inherited_subduction_from_world = bool(initial_subduction is not None
            and config.get('physics_profile', 'legacy') == 'reviewed_v1'
            and 'primordial_subduction' not in config)
        if inherited_subduction_from_world:
            config['primordial_subduction'] = initial_subduction
        initial_lifecycle = (continental_lifecycle.normalize(initial.get('continental_lifecycle'))
                             if isinstance(initial, dict) and initial.get('continental_lifecycle') is not None else None)
        lifecycle_from_world = bool(initial_lifecycle is not None
            and config.get('physics_profile', 'legacy') == 'reviewed_v1'
            and 'continental_lifecycle' not in config)
        if lifecycle_from_world:
            config['continental_lifecycle'] = initial_lifecycle
        initial_rift_traction = (rift_traction.normalize(initial.get('rift_traction'))
                                 if isinstance(initial, dict) and initial.get('rift_traction') is not None else None)
        requested_response=config.get('subduction_response')
        rift_traction_from_world = bool(initial_rift_traction is not None
            and config.get('physics_profile', 'legacy') == 'reviewed_v1'
            and 'rift_traction' not in config
            and requested_response in (None,'moving_hinge_v1'))
        if rift_traction_from_world:
            config['rift_traction'] = initial_rift_traction
            if requested_response is None:
                config['subduction_response']='moving_hinge_v1'
        cfg = validate_config(config)
        # Initial artwork/curved partition is imported once at a resolution
        # determined independently for coast geometry, never by review size.
        coast_level = cfg['coast_geometry_level']
        seed_width = 2**(coast_level+4)
        seed_cfg = {k:v for k,v in cfg.items() if k not in NATIVE_DEFAULTS}
        seed_cfg.update(width=max(48, seed_width), height=max(24, seed_width//2))
        source_initial = deepcopy(initial) if isinstance(initial, dict) else initial
        topology_overridden = bool(cfg['primordial_ocean']['enabled'] and isinstance(source_initial, dict)
                                   and source_initial.get('initial_plate_topology') is not None)
        if topology_overridden:
            # Explicit primordial-ocean partitioning is a stronger authored
            # initial condition. Do not first attach preset ocean aprons and
            # then ask primordial_ocean to reinterpret the same water twice.
            source_initial.pop('initial_plate_topology', None)
        source = legacy.Simulation(seed_cfg, source_initial)
        self.__dict__.update(source.__dict__)
        self.initial_plate_topology_overridden_by_primordial_ocean = topology_overridden
        self.initial_subduction = deepcopy(initial_subduction)
        self.initial_subduction_adopted = inherited_subduction_from_world
        self.initial_continental_lifecycle = deepcopy(initial_lifecycle)
        self.initial_continental_lifecycle_adopted = lifecycle_from_world
        self.initial_rift_traction = deepcopy(initial_rift_traction)
        self.initial_rift_traction_adopted = rift_traction_from_world
        self.config = cfg
        self.w, self.h = cfg['width'], cfg['height']
        self.native_mesh = geometry.icosphere(cfg['mesh_level'])
        self.native_spreading_version = 1
        self.native_subduction_version = 1
        self.native_subduction_geometry_version = 1
        self.foreland_loading_version = 1
        self.foreland_loading_diagnostics = dict(state='not_yet_evaluated')
        self.slab_memory_version = 0 if cfg['effective_subduction']['enabled'] else slab_memory.VERSION
        self.transform_coupling_version = transform_coupling.VERSION
        self.native_arc_source_geometry_version = 1
        self.native_arc_emplacement_version = 1
        self.native_arc_birth_profile_version = 1
        self.native_arc_deposition_version = 1
        self.native_arc_footprint_version = 1
        self.native_arc_point_version = 1
        self.native_locator = geometry.build_locator(self.native_mesh['vertices'], self.native_mesh['faces'])
        self.native_stencil = transport.face_vertex_stencil(self.native_mesh)
        xyz = self.native_mesh['xyz']
        seed_cells = source._indices(xyz)
        self.n = len(xyz)
        self.xyz = xyz.copy()
        self.cell_area = self.native_mesh['area_km2'].copy()
        self.earth_area = float(self.cell_area.sum())
        self.edge_a, self.edge_b = self.native_mesh['edge_faces'].T.copy()
        self.edge_mid = self.native_mesh['edge_mid'].copy()
        self.edge_normal = self.native_mesh['edge_normal'].copy()
        self.edge_length = self.native_mesh['edge_length'].copy()
        self.plate = source.plate[seed_cells].copy()
        self.initial_crust = source.initial_crust[seed_cells].copy()
        self.crust = self.initial_crust.copy()
        self.age = source.age[seed_cells].copy()
        self.ocean_relief = np.zeros(self.n)
        self.arc_budget = np.zeros(self.n)
        self.support = np.zeros((self.capacity, self.n), float)
        self.support[self.plate, np.arange(self.n)] = 1.
        self.seed_rift_id = source.seed_rift_id[seed_cells].copy()
        self.seed_rift_tangent = source.seed_rift_tangent[seed_cells].copy()
        # The same coastline setting must create the same material and seeded
        # motion even when the ocean/control mesh changes. Starting from the
        # control mesh (or raising the artwork raster with it) changes the
        # physical initial condition and confounds a resolution comparison.
        material_seed_mesh=(self.native_mesh if cfg['mesh_level']==coast_level
                            else geometry.icosphere(coast_level))
        fitted = material_initialization.build_initial_material(source, material_seed_mesh,
            boundary_refinement_levels=1)
        fitted = primordial_ocean.initialize(self, fitted)
        fitted = supercontinent_ring.initialize(self, fitted)
        self.initial_geometry_diagnostics = fitted['diagnostics']
        self.initial_geometry_diagnostics.update(coast_geometry_level=coast_level,
            source_artwork_width=seed_cfg['width'], source_artwork_height=seed_cfg['height'],
            native_control_faces=len(self.native_mesh['faces']),
            material_seed_mesh_faces=len(material_seed_mesh['faces']),
            material_seed_independent_of_control=True,
            ocean_control_resolution_unchanged=True)
        initial_intersections = native_initial_ownership.initialize(self, fitted)
        primordial_ocean.initialize_controls(self, fitted, initial_intersections)
        supercontinent_ring.initialize_controls(self, fitted, initial_intersections)
        self.continental_margin_version = 1
        self.continental_margin_parameters = continental_margin.parameters(dict(
            width_km=cfg['continental_margin_width_km'], shelf_depth_m=cfg['continental_shelf_depth_m']))
        self.material_surface = surface.initialize_surface(
            fitted['vertices'], fitted['faces'], fitted['face_owner'],
            fitted['face_kind'], face_id=np.arange(len(fitted['faces']), dtype=np.int64))
        self.pos = surface.face_centres(self.material_surface)
        self.kind = self.material_surface['face_kind'].copy()
        self.parcel_plate = self.material_surface['face_owner'].copy()
        self.parcel_patch = self.material_surface['face_id'].copy()
        self.mass = self.material_surface['reference_area_km2'].copy()
        self.parcel_cell = self._indices(self.pos)
        self.relief = initial_basement_relief(self.pos, cfg['seed'])
        self.suture = np.where(self.kind == 2, .02, .2)
        self.parcel_birth = np.zeros(len(self.mass))
        self.parcel_craton = fitted['face_craton'][self.parcel_patch].copy()
        self.next_patch_uid = len(fitted['faces'])
        material_seeds = fitted['source_seed_cells'][self.parcel_patch]
        self.rift_id = source.seed_rift_id[material_seeds].copy()
        self.rift_birth_myr = np.where(self.rift_id > 0, 0., -1.)
        tangent = source.seed_rift_tangent[material_seeds]
        self.rift_tangent = legacy._unit(tangent-self.pos*np.sum(tangent*self.pos, axis=1)[:, None])
        self.rift_extension_m = np.zeros(len(self.mass))
        self.inversion_uplift_m = np.zeros(len(self.mass))
        self._update_footprint_diagnostics()
        structure_engine.initialize_parcels(self)
        self._make_traces()
        native_material_evolution.ensure_lineage(self)
        native_arc_material.ensure_fields(self)
        self.deformation_diagnostics = dict(model='rigid blocks and deforming material belts', initialized=True)
        self.adaptivity_diagnostics = dict(initialized=True, material_faces=len(self.mass))
        self.original_mass = float(self.mass.sum())
        self.original_craton_mass = float(self.mass[self.kind == 2].sum())
        self.rift_material = None
        self.rift_bonds, self.rift_systems = {}, []
        self.rift_pending = None
        self.rift_mechanics = {}
        self.trench_systems = []
        self.next_trench_id = 1
        self.local_accretion_contacts = []
        self.next_accretion_contact_id = 1
        self.ocean_history = ocean_rifting.initialize(self.n)
        self.primordial_fraction = (self.crust == 0).astype(float)
        self.ocean_diagnostics = {}
        self.native_domains = dict(next_id=1, previous=np.zeros(self.n, np.int64), records={})
        self.domain = np.zeros(self.n, np.int64)
        self.domains = []
        self.spreading_diagnostics = {}
        self.domain_tracker = None
        self.collision_surface_version = collision_surface.VERSION
        self._rasterize()
        self._boundaries()
        trench_history.initialize(self)
        progressive_rifting.initialize(self)
        self._update_surface_domains()
        self._record('mesh_initialized', 'A connected spherical material surface and unstructured ocean mesh now carry this world. The primordial ocean can develop new boundaries under sustained local loading.', details=dict(
            representation='connected moving continental triangles and finite-volume ocean cells',
            mesh_level=cfg['mesh_level'], cells=self.n, vertices=len(self.native_mesh['vertices']),
            initial_partition=deepcopy(self.initial_geometry_diagnostics), display_grid_is_output_only=True))
        world_design.initialize(self)
        localized_accretion.initialize(self)
        self.material_mechanics_version = 1
        self.native_gravity_constraint_version = bounded_gravity.VERSION
        self.native_contact_constraint_version = 1
        # Checkpoints made before this guard already contain same-sheet
        # intersections. Keep their recorded contact law on continuation;
        # only fresh states can promise a zero-overlap postcondition.
        self.same_sheet_nonpenetration_version = 1
        enhanced_rifting.initialize(self)
        lip_events.initialize(self)
        physics_profile.initialize(self)
        force_rifting.initialize(self)
        continental_lifecycle.initialize(self)

    def _indices(self, points):
        face, _ = geometry.locate_points(np.asarray(points), self.native_locator)
        if np.any(face < 0):
            raise ValueError('Closed native mesh failed to contain a spherical point.')
        return face

    def _sample_coordinates(self, points):
        return transport.sample_coordinates(self.native_mesh, self.native_locator, self.native_stencil, points)

    def _make_traces(self):
        """Bounded, area-stratified histories of actual moving material faces."""
        scale = max(1, math.ceil(self.n/(256*128)))
        self.initial_marker_limit = min(16384, 4096*scale)
        self.arc_marker_limit = min(4096, 2048*scale)
        chosen = np.arange(len(self.mass))
        if len(chosen) > self.initial_marker_limit:
            cumulative = np.cumsum(self.mass)
            targets = (np.arange(self.initial_marker_limit)+.5)*cumulative[-1]/self.initial_marker_limit
            chosen = np.unique(np.searchsorted(cumulative, targets))
        self.trace_id = np.arange(1, len(chosen)+1, dtype=np.int64)
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
        for name in ('uplift', 'extension', 'erosion', 'adjustment', 'ridge_uplift'):
            setattr(self, 'trace_'+name+'_m', np.zeros(len(chosen)))
        self.next_trace_id = len(chosen)+1
        self.arc_trace_count = 0
        self.arc_trace_buckets = set()
        structure_engine.initialize_traces(self, chosen)

    def _update_footprint_diagnostics(self):
        lon = np.arctan2(self.pos[:, 1], self.pos[:, 0])
        self.parcel_east = np.column_stack((-np.sin(lon), np.cos(lon), np.zeros(len(lon))))
        # Only a physical size allowance for legacy strip selection. It is
        # never passed to the former tent-deposition transport.
        extent = np.sqrt(np.maximum(self.mass, 0))/RADIUS_KM*.5
        self.parcel_extent = np.column_stack((extent, extent))

    def _sync_material(self):
        surface.reassign_owners(self.material_surface, self.parcel_plate)
        self.material_surface['face_kind'] = self.kind.copy()
        self.pos = surface.face_centres(self.material_surface)
        self.parcel_cell = self._indices(self.pos)
        self._update_footprint_diagnostics()
        if hasattr(self, 'structure'):
            native_material_evolution.ensure_lineage(self)
            # Arc birth can occur between column evolution and refinement.
            # Append collision fields while original face rows still align;
            # post-refinement padding cannot recover their material ancestry.
            collision_contacts.ensure_fields(self)
            collision_surface.ensure_fields(self)

    def _exposed_faces(self, points, background):
        hits = surface.sample_surface(self.material_surface, points)
        query, face = hits['query_index'], hits['face_index']
        winner = np.full(len(points), -1, np.int32)
        if len(query):
            # Preserve an existing exposed owner where it still has actual
            # triangular coverage. Otherwise choose a deterministic source.
            eligible = collision_contacts.eligible_hits(query, face, self.parcel_collision_sheet,
                self.collision_contacts) if hasattr(self, 'parcel_collision_sheet') else np.ones(len(query), bool)
            candidate = np.flatnonzero(eligible)
            preferred = self.parcel_plate[face[candidate]] == background[query[candidate]]
            order = candidate[np.lexsort((self.parcel_patch[face[candidate]], ~preferred, query[candidate]))]
            _, first = np.unique(query[order], return_index=True)
            chosen = order[first]
            winner[query[chosen]] = face[chosen]
        return winner, hits

    def _rasterize(self):
        """Refresh native control-cell exposure from actual material triangles."""
        self._sync_material()
        collision_contacts.refresh(self)
        collision_surface.refresh(self, self._collision_overlap)
        mesh = self.material_surface
        signature = hashlib.sha256(mesh['vertices'].tobytes()+mesh['faces'].tobytes()).hexdigest()
        if signature != getattr(self, '_coverage_signature', None):
            self._material_coverage = mesh_coverage.intersections(
                self.native_mesh, mesh['vertices'], mesh['faces'], locator=self.native_locator)
            self._coverage_signature = signature
        coverage = self._material_coverage
        self.land_mass = np.bincount(coverage['control_index'], weights=coverage['area_km2'], minlength=self.n)
        # Overlapping collision sheets remain separate material, so density
        # may exceed one; they are not silently deleted or counted as new crust.
        self.density = self.land_mass/self.cell_area
        # Stack area remains in the material ledger, while resolved surface
        # classification uses exposed area. Several buried slivers cannot add
        # their duplicate footprints to manufacture a full continental cell.
        exposed_mass = np.bincount(coverage['control_index'], weights=coverage['area_km2']*
            self.parcel_exposed_fraction[coverage['material_index']], minlength=self.n)
        self.exposed_density = np.clip(exposed_mass/self.cell_area, 0., 1.)
        winner, hits = self._exposed_faces(self.xyz, self.plate)
        self.exposed_material = winner
        # A tiny arc is a tiny triangle, even if its centre happens to coincide
        # with a control centre. Classify only resolved buoyant coverage here;
        # output maps always query the full material geometry directly.
        land = (winner >= 0) & (self.exposed_density >= .5)
        self.crust = np.zeros(self.n, np.uint8)
        self.crust[land] = self.kind[winner[land]]
        self.plate[land] = self.parcel_plate[winner[land]]
        dry=native_spreading.dry_control_cells(self,land) if getattr(self,'native_spreading_version',0)==1 else land
        if not native_initial_ownership.active(self):
            self.support[:, dry] = 0.
            self.support[self.plate[dry], np.flatnonzero(dry)] = 1.
        self.land_height = np.zeros(self.n)
        height = structure_engine.material_height(self.kind, self.relief)
        if getattr(self, 'collision_surface_version', 0) == 1:
            height = height+self.parcel_collision_support_m
        self.land_height[land] = height[winner[land]]
        self.land_relief = self.land_height-220.-180.*(self.crust == 2)+100.*(self.crust == 3)
        self.grid_suture = np.zeros(self.n)
        self.grid_suture[land] = self.suture[winner[land]]
        self.age[dry] = 0.
        fields = geology_snapshot.deposit_fields(self)
        self.structure_grid = {}
        for name in geology_snapshot.GRID_FIELDS:
            value = np.zeros(self.n)
            value[land] = fields[name][winner[land]]
            if name == 'rift_cooling_age_myr':
                value[~land] = -1.
                value[land] = np.where(fields['rift_cooling_valid'][winner[land]] > 0, value[land], -1.)
            self.structure_grid[name] = value
        self._owner_occupancy = {int(p): np.unique(hits['query_index'][hits['owner'] == p]).astype(np.int32)
                                 for p in np.unique(hits['owner'])}
        self._material_occupancy_hits = hits
        self._owner_occupancy_signature = local_accretion.occupancy_signature(self)

    def _boundaries(self):
        a, b = self.edge_a, self.edge_b
        use = self.plate[a] != self.plate[b]
        self.boundary_edge_indices = np.flatnonzero(use)
        self.ba, self.bb = a[use], b[use]
        self.bp, self.bq = self.plate[self.ba], self.plate[self.bb]
        if native_initial_ownership.active(self):
            self.native_boundary_geometry = native_initial_ownership.reconstruct(self, self.boundary_edge_indices)
            self.boundary_edge_indices = self.native_boundary_geometry['graph_edge_indices'].copy()
            self.ba, self.bb = self.native_mesh['edge_faces'][self.boundary_edge_indices].T
            self.bp, self.bq = self.plate[self.ba], self.plate[self.bb]
        else:
            self.native_boundary_geometry = native_boundary_geometry.reconstruct(
                self.native_mesh, self.plate, self.support, self.boundary_edge_indices)
        self.bn = self.native_boundary_geometry['normals'].copy()
        self.bmid = self.native_boundary_geometry['midpoints'].copy()
        self.bl = self.native_boundary_geometry['lengths'].copy()
        dv = np.cross(self.omega[self.bq]-self.omega[self.bp], self.bmid)*RADIUS_KM
        self.normal_speed = np.sum(dv*self.bn, axis=1)
        # A categorical graph edge without an initial physical interface must
        # not seed force, trench, rift or transform history from raw geometry.
        self.normal_speed[self.bl <= 1e-10] = 0.
        shear = np.linalg.norm(dv-self.normal_speed[:, None]*self.bn, axis=1)
        # Classification shares the finite-spreading threshold. Normal partition
        # version 1 independently admits closing motion to convergence processes,
        # allowing sliding and subduction to coexist without changing ridge policy.
        threshold = normal_motion_threshold(shear)
        ext = self.normal_speed > threshold
        con = self.normal_speed < -threshold
        la, lb = self.crust[self.ba] > 0, self.crust[self.bb] > 0
        self.bcode = np.where(self.bl > 1e-10, 3, 0).astype(np.uint8)
        self.bcode[ext & ~(la | lb)] = 1
        self.bcode[ext & (la | lb)] = 5
        self.bcode[con & ~(la & lb)] = 2
        self.bcode[con & la & lb] = 4
        if getattr(self,'native_spreading_version',0)==1:
            pieces=native_spreading.prepare(self)
            ocean=pieces['parent_ocean_length_km']; continental=pieces['parent_continental_length_km']
            self.bcode[ext & (ocean>1e-8) & (continental<=1e-8)]=1
            self.bcode[ext & (continental>1e-8)]=5
        self.down = np.full(len(self.ba), -1, np.int16)
        reviewed_polarity = physics_profile.polarity_enabled(self)
        if reviewed_polarity:
            self._subduction_stalled_mask = np.zeros(len(self.ba), bool)
        candidates = (normal_partition.convergence(self) & ~(la & lb)
                      if normal_partition.enabled(self) else self.bcode == 2)
        sub = np.flatnonzero(candidates)
        for p, q in np.unique(np.sort(np.column_stack((self.bp[sub], self.bq[sub])), axis=1), axis=0):
            edges = sub[((self.bp[sub] == p) & (self.bq[sub] == q)) | ((self.bp[sub] == q) & (self.bq[sub] == p))]
            cp = np.where(self.bp[edges] == p, self.ba[edges], self.bb[edges])
            cq = np.where(self.bp[edges] == q, self.ba[edges], self.bb[edges])
            previous = int(self.polarity[p, q])
            score_p = _subduction_buoyancy(self.crust[cp], self.age[cp])
            score_q = _subduction_buoyancy(self.crust[cq], self.age[cq])
            if previous < 0:
                # A pair with no history: the denser side goes down.
                down = int(p if score_p <= score_q else q)
            elif reviewed_polarity:
                # A buoyant arrival does not resolve a new trench on the other
                # side. Preserve identity; the incoming-water test below stalls it.
                down = previous
            else:
                # Otherwise the zone keeps the polarity it has. It reverses only when the
                # side currently descending has become the clearly buoyant one -- which is
                # a buoyant block arriving at the trench, not a shift in sampling. Re-electing
                # every step is what let the southern Primordial ocean flip back and forth.
                other = q if previous == p else p
                descending = score_p if previous == p else score_q
                overriding = score_q if previous == p else score_p
                down = int(other) if descending - overriding > POLARITY_REVERSAL_MARGIN_M else previous
            # A hard constraint rather than a preference, and so applied last: continental
            # lithosphere is too buoyant to be drawn beneath oceanic lithosphere, however
            # the pair's history reads.
            if not reviewed_polarity or previous < 0:
                if _continental(self.crust[cp]) and _oceanic(self.crust[cq]):
                    down = int(q)
                elif _continental(self.crust[cq]) and _oceanic(self.crust[cp]):
                    down = int(p)
            self.polarity[p, q] = self.polarity[q, p] = down
            self.down[edges] = down
        if not reviewed_polarity:
            self.down[sub[la[sub] & ~lb[sub]]] = self.bq[sub[la[sub] & ~lb[sub]]]
            self.down[sub[lb[sub] & ~la[sub]]] = self.bp[sub[lb[sub] & ~la[sub]]]
        self.trench_retreat_speed = np.zeros(len(self.ba))
        self.boundary_geometry_diagnostics = deepcopy(self.native_boundary_geometry['diagnostics'])
        trench_history.prepare(self)
        physics_profile.stall_buoyant_incoming(self)
        ranks = np.zeros(self.n, np.uint8)
        priority = np.array([0, 2, 4, 1, 6, 5], np.uint8)
        display = boundary_labels.codes(self)
        np.maximum.at(ranks, self.ba, priority[display])
        np.maximum.at(ranks, self.bb, priority[display])
        self.boundary = np.array([0, 3, 1, 0, 2, 5, 4], np.uint8)[ranks]

    def _migrate_trenches(self, dt):
        """The moving hinge follows its upper plate without a second remap."""
        if effective_subduction.enabled(self):
            # The declared trace is advected with its upper plate. Effective
            # incoming traction supplies no independent rollback velocity.
            self.trench_retreat_speed = np.zeros(len(self.ba))
            return
        if plate_balance.subduction_response_version(self) == 1:
            # _advect already carries the trace with the overriding plate.
            # The inherited forearc transfer would add heuristic retreat after
            # the force solve and contradict the slab-work kinematics. Its
            # temporary speed is hidden by the final boundary reconstruction,
            # so the force solver's nonzero-retreat guard cannot catch it.
            self.trench_retreat_speed = np.zeros(len(self.ba))
            return
        super()._migrate_trenches(dt)

    def _label_codes(self):
        """Display and statistics labels: the mechanism, not the kinematic gate."""
        return boundary_labels.codes(self)

    def _update_surface_domains(self, dt=0., previous_omega=None, previous_plate_uid=None):
        from native_domain_components import classify
        grouping = classify(self.native_mesh, self.plate, self.material_surface, self._material_coverage)
        labels, material_labels = grouping['control_labels'], grouping['material_labels']
        old = self.native_domains['previous']
        if dt and previous_omega is not None:
            moved = np.zeros_like(old)
            for p in np.unique(self.plate):
                select = self.plate == p
                source = self._indices(legacy._rotate(self.xyz[select], -previous_omega[p]*dt))
                moved[select] = old[source]
            old = moved
        # Material identity is exact and independent of display sampling, so a
        # tiny detached island keeps its name even without a control centre.
        old_ids = self.native_domains.get('material_ids', np.empty(0, np.int64))
        old_material = np.zeros(len(self.mass), np.int64)
        if len(old_ids):
            _, now, before = np.intersect1d(self.parcel_patch, old_ids, return_indices=True)
            old_material[now] = self.native_domains['material_domains'][before]
        previous_records = self.native_domains['records']
        records, claimed = {}, set()
        result = np.zeros(self.n, np.int64)
        material_domains = np.zeros(len(self.mass), np.int64)
        components = [(np.flatnonzero(labels == group), np.flatnonzero(material_labels == group))
                      for group in range(grouping['count'])]
        area = lambda pair: float(self.cell_area[pair[0]].sum()) if len(pair[0]) else float(self.mass[pair[1]].sum())
        for cells, faces in sorted(components, key=area, reverse=True):
            p = int(self.plate[cells[0]] if len(cells) else self.parcel_plate[faces[0]])
            uid = int(self.plate_uid[p])
            values, inverse = np.unique(np.concatenate((old[cells], old_material[faces])), return_inverse=True)
            weights = np.bincount(inverse, weights=np.concatenate((self.cell_area[cells], self.mass[faces])))
            candidates = [int(values[k]) for k in np.argsort(-weights) if values[k] > 0
                          and int(values[k]) not in claimed
                          and previous_records.get(int(values[k]), {}).get('plate_uid') == uid]
            domain = candidates[0] if candidates else self.native_domains['next_id']
            if not candidates:
                self.native_domains['next_id'] += 1
            claimed.add(domain)
            result[cells] = domain
            material_domains[faces] = domain
            records[domain] = dict(uid=domain, plate_uid=uid, plate_id=p,
                name=self.names[p] if not any(r['plate_uid'] == uid for r in records.values()) else f'{self.names[p]} · fragment {domain}',
                area_km2=area((cells, faces)), active=True, material_only=not len(cells),
                area_basis='native control area; material area for unresolved isolated surfaces')
        self.domain, self.material_domain = result, material_domains
        self.domains = list(records.values())
        self.native_domains.update(previous=result.copy(), records=records,
                                   material_ids=self.parcel_patch.copy(), material_domains=material_domains.copy())
        self.domain_diagnostics = dict(surface_components=grouping['count'],
            geometry='joint native control and connected material adjacency')

    def _advect(self, dt):
        from native_processes import advect_ocean
        # Both transports advance the same source epoch. Ocean transport
        # changes control-cell winners before the material mesh moves, so its
        # destination owners cannot invalidate source-epoch material loading.
        prepared_loading = None
        if (getattr(self, 'material_mechanics_version', 0) == 1
                and self.config.get('deforming_regions', 1)):
            prepared_loading = native_material_evolution.material_loading_boundaries(self, 1)
        advect_ocean(self, dt)
        if prepared_loading is None:
            native_material_evolution.advect(self, dt)
        else:
            native_material_evolution.advect(self, dt, prepared_loading=prepared_loading)
        import entry_regions
        if entry_regions.enabled(self):entry_regions.advance_hinges(self,dt)
        trench_history.advect(self, dt)
        progressive_rifting.advect(self, dt, legacy._rotate)
        if dt > 0 and native_initial_ownership.active(self):
            self.native_initial_ownership_consumed = True

    def _deform_and_accrete(self, dt):
        from native_processes import deform_and_accrete
        deform_and_accrete(self, dt)

    def _add_arc_crust(self, cells, additions):
        from native_processes import add_arc_crust
        return add_arc_crust(self, cells, additions)

    def _topology(self, dt, backarc_pending=None):
        from native_topology import update_topology
        return update_topology(self, dt, backarc_pending=backarc_pending)

    def _split(self, p, preferred=None, loading=None, *, chosen=None, progressive=False):
        from native_topology import split_continent
        return split_continent(self, p, chosen, loading) if progressive and chosen is not None else False

    def step(self, dt=None):
        coarse_history.validate(self)
        import budget
        requested = self.config['dt_myr'] if dt is None else float(dt)
        with budget.step_budget(requested):
            budget.require_physical_time('native_step')
            return self._budgeted_step(requested)

    def _budgeted_step(self, dt):
        import entry_regions
        effective_subduction.validate_compatibility(self)
        if continental_lifecycle.enabled(self):
            dt = self.config['dt_myr'] if dt is None else float(dt)
            if not np.isfinite(dt) or not 0 < dt <= 5:
                raise ValueError('Step length must be positive and at most 5 Myr.')
            return continental_lifecycle.advance(self, dt)
        if hasattr(self, 'channel_region_store'):
            raise ValueError('Regional channel histories need a coupled native source law before time can advance.')
        if entry_regions.enabled(self):
            raise ValueError('Persistent entry experiments require the atomic coupled slab_tether_native.advance path.')
        if trench_history._rupture_governed(self):
            raise ValueError('Rupture-governed trenches require the atomic coupled slab_tether_native.advance path.')
        policies = evolution_policy.versions(self)
        dt = self.config['dt_myr'] if dt is None else float(dt)
        if not np.isfinite(dt) or not 0 < dt <= 5:
            raise ValueError('Step length must be positive and at most 5 Myr.')
        if policies['adaptive_timestep_version'] == 1:
            from adaptive_timestepping import advance
            return advance(self, dt, self._step_once)
        # Keep adapter fixtures and historical states on their recorded path.
        return Simulation._step_once(self, dt)

    def _step_once(self, dt):
        if hasattr(self, 'channel_region_store'):
            raise ValueError('Regional channel histories need a coupled native source law before time can advance.')
        transitions = [v for v in (world_design.next_transition(self),lip_events.next_transition(self)) if v is not None]
        transition = min(transitions) if transitions else None
        if transition is not None and self.t+1e-10 < transition < self.t+dt-1e-10:
            remainder = self.t+dt-transition
            self.step(transition-self.t)
            self.step(remainder)
            return
        _budget_stage('prepare_step', Simulation._prepare_step, self)
        _budget_stage('forces', self._forces, dt)
        if not plate_balance.enabled(self):
            # The instantaneous balance already includes these resistances;
            # applying the legacy corrections as well would count them twice.
            backarc.apply_motion(self, dt)
            transform_coupling.apply(self, dt)
            collision_contacts.resist_motion(self, dt)
        else:
            collision_contacts.resist_motion(self, dt)   # diagnostics only; writes no motion
        _budget_stage('advance_step', Simulation._advance_step, self, dt)

    def _prepare_step(self):
        """Prepare source-epoch contacts before selecting instantaneous motion."""
        world_design.record_transitions(self)
        if collision_contacts.weld_enabled(self):
            # Consolidation reads the PREVIOUS step's exposure, suture and overlap, all of
            # which are current, and writes ownership. It belongs at the top of the step
            # because several things later in a step are checked against the ownership that
            # was in force when they were recorded -- the realized rift motion is captured
            # during advection and compared with its material links afterwards, so a
            # transfer in between invalidates it. Ownership changes must rebuild
            # contact geometry before the force solve uses the daughter sheets.
            self.consolidation_diagnostics = collision_contacts.consolidate(self)
            if self.consolidation_diagnostics.get('moved'):
                collision_contacts.refresh(self, 0.)
        import effective_subduction_initiation
        # New Lite trenches enter the next source solve, never a post-solve
        # history snapshot. Adaptive trial backup includes this activation.
        effective_subduction_initiation.activate_pending(self)

    def _advance_step(self,dt,*,after_slab_source=None):
        """Commit one accepted motion interval through native source/topology paths.

        The optional mechanical event callback runs after actual slab feeding
        and before trench identities can split, join or change owners.
        """
        if hasattr(self, 'channel_region_store'):
            raise ValueError('Regional channel histories need a coupled native source law before time can advance.')
        omega, uids = self.omega.copy(), self.plate_uid.copy()
        present = {int(u) for u, on in zip(self.plate_uid, self.active) if on}
        _budget_stage('advect', self._advect, dt)
        self.t += dt
        if not effective_subduction.enabled(self):
            slab_memory.advance(self, dt)
        if after_slab_source is not None:
            after_slab_source(self)
        world_design.record_transitions(self)
        self.steps += 1
        _budget_stage('rasterize', self._rasterize)
        _budget_stage('boundaries', self._boundaries)
        _budget_stage('migrate_trenches', self._migrate_trenches, dt)
        _budget_stage('trench_history', trench_history.update, self, dt)
        pending = _budget_stage('backarc', backarc.update, self, dt)
        _budget_stage('ridge_windows', self._update_ridge_windows, dt)
        _budget_stage('collision_refresh', collision_contacts.refresh, self, dt)
        _budget_stage('deform_and_accrete', self._deform_and_accrete, dt)
        _budget_stage('topology', self._topology, dt, pending)
        native_material_evolution.ensure_lineage(self)
        collision_contacts.ensure_fields(self)
        eclogite_sink.ensure_fields(self)
        _budget_stage('material_adaptivity', native_material_adaptivity.adapt, self, dt)
        _budget_stage('rasterize', self._rasterize)
        _budget_stage('boundaries', self._boundaries)
        if (plate_balance.enabled(self) and not trench_history._rupture_governed(self)
                and any(int(u) not in present for u, on in zip(self.plate_uid, self.active) if on)):
            # A split, basin isolation or arc separation gives the new plate a
            # guessed motion: a rigid fit to the pre-break velocities, checked at
            # one point. Plates carry no momentum, so their real motion is the
            # force balance of the new geometry. At SEP21T's 74 Myr breakup the
            # guess closed 76% of a 7,780 km rift at up to 26 km/Myr; the balance
            # opens it at a mean 2.8 km/Myr. Without this, that guess sets the
            # saved labels and speeds, and the next step's ocean transport reads
            # them before its boundaries are rebuilt. The solve is from scratch
            # (warm starts change only iterations) and takes seconds.
            # The rupture-governed coupled path (slab_tether_native.advance) is
            # excluded: it re-solves with slab-tether forces after this returns,
            # and an uncoupled solve here would drop those forces (and refuses
            # persistent entry regions outright).
            _budget_stage('forces', self._forces, dt)
            _budget_stage('boundaries', self._boundaries)
        import ridge_spreading
        _budget_stage('ridge_refresh', ridge_spreading.refresh, self)
        _budget_stage('surface_domains', self._update_surface_domains, dt, omega, uids)

    def snapshot(self):
        """Project for the existing UI, retaining native geometry in the epoch."""
        native = super().snapshot()
        lon, lat = np.meshgrid((np.arange(self.w)+.5)*2*np.pi/self.w-np.pi,
                               np.pi/2-(np.arange(self.h)+.5)*np.pi/self.h)
        points = legacy._xyz(lon.ravel(), lat.ravel())
        control = self._indices(points)
        result = dict(native)
        for name in ('elevation', 'plate', 'crust', 'age', 'boundary', 'domain', 'trench', *geology_snapshot.GRID_FIELDS):
            if name in native:
                result[name] = native[name][control].copy()
        owner_field = native_boundary_geometry.prepare_owner_sampling(
            self.native_mesh, self.plate, self.support, locator=self.native_locator)
        recorded = {row['id'] for row in result['plates']}
        for p in owner_field['owner_slots']:
            if int(p) not in recorded:
                result['plates'].append(dict(id=int(p), name=self.names[p], angular_velocity=self.omega[p].tolist(),
                    uid=int(self.plate_uid[p]), generation=int(self.plate_generation[p]), active=bool(self.active[p]),
                    parent_uid=int(self.plate_parent_uid[p]), created_myr=float(self.plate_created[p]),
                    area_km2=0., born_myr=float(self.born[p]),
                    internal_loading=deepcopy(getattr(self, 'stress_diagnostics', {}).get(int(p)))))
        result['plate'], owner_cells = native_boundary_geometry.sample_owners(points, owner_field, return_cells=True)
        ocean_owners = result['plate'].copy()
        resolved_domain = owner_cells >= 0
        result['domain'] = np.zeros(len(points), np.int64)
        result['domain'][resolved_domain] = self.domain[owner_cells[resolved_domain]]
        _, hits = self._exposed_faces(points, result['plate'])
        # Query the connected crust directly, rather than projecting a sampled
        # continental mask back out of the control mesh.
        heights = structure_engine.material_height(self.kind, self.relief)
        context = native_frame_sampling.prepare_live_surface(self)
        margin_version = getattr(self, 'continental_margin_version', 0)
        exposed = native_frame_sampling.select_exposed_material(context, points, control, ocean_owners,
            (hits['query_index'], hits['face_index'], hits['weights']), episodes=self.ridge_episodes, time_myr=self.t)
        material = exposed['material_face']
        land = material >= 0
        result['plate'], result['crust'] = exposed['owner'], exposed['crust']
        result['domain'][land] = self.material_domain[material[land]]
        result['age'] = exposed['ocean_age_myr']
        result['age'][land] = 0.
        physical_height = exposed['baseline_height_m']+exposed['display_thermal_support_m']
        result['elevation'] = (physical_height if getattr(self, 'collision_surface_version', 0) == 1
                              else np.clip(physical_height, -10000, 9000)).astype(np.float32)
        columns = geology_snapshot.material_fields(self.structure)
        columns.update(progressive_rifting.material_fields(self))
        for name in geology_snapshot.GRID_FIELDS:
            result[name][~land] = 7. if name == 'crustal_thickness_km' else -1. if name == 'rift_cooling_age_myr' else 1. if name == 'rift_strength_relative' else 0.
            result[name][land] = columns[name][material[land]]
        result.update(lip_events.project_grid(self, material))
        result.update(mesh_version=1, surface_reconstruction_version=1, owner_reconstruction_version=1,
            mesh_level=self.config['mesh_level'],
            mesh_owner_slots=owner_field['owner_slots'].copy(), mesh_vertex_support=owner_field['scores'].copy(),
            mesh_vertices=self.native_mesh['vertices'].copy(), mesh_faces=self.native_mesh['faces'].copy(),
            mesh_plate=self.plate.copy(), mesh_crust=self.crust.copy(), mesh_age_myr=self.age.copy(),
            mesh_elevation_m=native['elevation'].copy(), mesh_area_km2=self.cell_area.copy(),
            mesh_ocean_damage=self.ocean_history['damage'].copy(),
            mesh_ocean_strain=self.ocean_history['tensile_strain'].copy(),
            mesh_primordial_fraction=self.primordial_fraction.copy(),
            mesh_ocean_relief_m=self.ocean_relief.copy(),
            mesh_boundary=self.boundary.copy(),
            mesh_ocean_weakness=self.ocean_history['weakness'].copy(),
            mesh_ocean_tensile_exposure_myr=self.ocean_history['tensile_exposure_myr'].copy(),
            material_vertices=self.material_surface['vertices'].copy(), material_faces=self.material_surface['faces'].copy(),
            material_face_id=self.parcel_patch.copy(), material_owner=self.parcel_plate.copy(),
            material_kind=self.kind.copy(), material_height_m=heights.copy(), material_reference_area_km2=self.mass.copy(),
            material_domain=self.material_domain.copy(),
            native_boundary_edges=self.native_mesh['edge_vertices'][self.boundary_edge_indices].copy(),
            native_boundary_code=boundary_labels.codes(self), native_boundary_kinematic_code=self.bcode.copy(),
            boundary_label_version=int(getattr(self, 'boundary_label_version', 0)),
            boundary_label_diagnostics=boundary_labels.disagreement(self),
            native_boundary_owner_a=self.bp.copy(),
            native_boundary_owner_b=self.bq.copy(), native_boundary_down=self.down.copy(),
            native_boundary_normal_speed_km_myr=self.normal_speed.copy(),
            mesh_diagnostics=dict(native_cells=self.n, material_faces=len(self.mass),
                                  representation='finite-volume ocean plus connected Lagrangian crust',
                                  initial_geometry=deepcopy(self.initial_geometry_diagnostics),
                                  initial_plate_topology=deepcopy(getattr(self, 'initial_plate_topology_diagnostics', {})),
                                  initial_plate_topology_overridden_by_primordial_ocean=bool(
                                      getattr(self, 'initial_plate_topology_overridden_by_primordial_ocean', False)),
                                  initial_subduction=deepcopy(getattr(self, 'initial_subduction', None)),
                                  initial_subduction_adopted=bool(getattr(self, 'initial_subduction_adopted', False)),
                                  initial_continental_lifecycle=deepcopy(getattr(self, 'initial_continental_lifecycle', None)),
                                  initial_continental_lifecycle_adopted=bool(getattr(self, 'initial_continental_lifecycle_adopted', False)),
                                  initial_rift_traction=deepcopy(getattr(self, 'initial_rift_traction', None)),
                                  initial_rift_traction_adopted=bool(getattr(self, 'initial_rift_traction_adopted', False)),
                                  unresolved_ocean_domain_queries=int(np.count_nonzero(~resolved_domain & ~land)),
                                  display_grid_is_output_only=True),
            ocean_rifting_diagnostics=deepcopy(self.ocean_diagnostics),
            # Why a geographically detached region did NOT become its own plate. Empty
            # means either nothing was disconnected or an earlier stress-driven path
            # claimed the step; a populated list names the gate that declined.
            isolated_split_diagnostics=deepcopy(getattr(self, 'isolated_split_diagnostics', [])))
        result.update(native_material_evolution.snapshot_fields(self))
        result.update(enhanced_rifting.snapshot_fields(self))
        result.update(lip_events.snapshot(self))
        result.update(collision_contacts.snapshot_fields(self))
        result.update(collision_surface.snapshot_fields(self))
        result.update(eclogite_sink.snapshot_fields(self))
        result.update(physics_profile.snapshot(self))
        import surface_erosion
        result.update(surface_erosion.snapshot(self))
        result.update(evolution_policy.snapshot(self))
        result.update(continental_lifecycle.snapshot(self))
        result.update(production_restart_migration.snapshot(self))
        result.update(force_rifting.snapshot(self))
        result.update(primordial_ocean.snapshot(self))
        result.update(supercontinent_ring.snapshot(self))
        if plate_balance.enabled(self):
            result.update(plate_balance_version=plate_balance.VERSION,
                          plate_balance_diagnostics=getattr(self, 'plate_balance_diagnostics', None),
                          consolidation_diagnostics=getattr(self, 'consolidation_diagnostics', None))
        result['foreland_loading_version'] = getattr(self, 'foreland_loading_version', 0)
        result['foreland_loading_diagnostics'] = deepcopy(
            getattr(self, 'foreland_loading_diagnostics', {}))
        collision_offset = exposed.get('collision_surface_offset_m', np.zeros(len(points)))
        result['collision_surface_diagnostics'] = dict(query_count=len(points),
            offset_queries=int(np.count_nonzero(np.abs(collision_offset) > 1e-6)),
            maximum_absolute_offset_m=float(np.max(np.abs(collision_offset), initial=0.)),
            mean_absolute_offset_m=float(np.mean(np.abs(collision_offset))),
            meaning=('Selected sheet, explicit Airy support and thermal component residual.'
                     if getattr(self, 'collision_surface_version', 0) == 1 else
                     'Surface envelope minus the selected upper sheet reconstruction; not physical uplift.'),
            vertical_stack_mechanics_resolved=False)
        if getattr(self, 'collision_surface_version', 0) == 1:
            result['collision_surface_diagnostics'].update(
                surface_components_consistent=True, reduced_compensation_resolved=True,
                physical_height_clipped=False,
                maximum_raw_elevation_m=float(np.max(physical_height, initial=-np.inf)),
                minimum_raw_elevation_m=float(np.min(physical_height, initial=np.inf)))
        result.update(native_arc_material.snapshot_fields(self), arc_surface_version=2,
                      arc_material_diagnostics=deepcopy(getattr(self, 'arc_material_diagnostics', {})))
        if margin_version == 1:
            result.update(continental_margin_version=1,
                continental_margin_parameters=dict(self.continental_margin_parameters),
                continental_margin_diagnostics=dict(open_edges=context['continental_margin']['open_edge_count'],
                    model='finite inside-footprint slope, shelf and coast from exact native free edges',
                    physical_columns_unchanged=True, fills_ocean_gaps=False))
        for name, values in (
            ('deformation_weight', np.where(self.material_deformation['face_rigid'], 0., self.material_deformation['face_weight'])),
            ('geometric_strain_percent', 100.*np.expm1(self.material_deformation['face_strain'])),
            ('refinement_level', self.material_lineage['level'])):
            projected = np.full(len(points), -1. if name != 'geometric_strain_percent' else 0., np.float32)
            projected[land] = values[material[land]]
            result[name] = projected
        result['deformation_diagnostics'] = deepcopy(self.deformation_diagnostics)
        result['adaptivity_diagnostics'] = deepcopy(self.adaptivity_diagnostics)
        for name in geology_snapshot.GRID_FIELDS:
            result['material_'+name] = columns[name].copy()
        for name in ('rift_id', 'rift_birth_myr', 'rift_extension_m', 'inversion_uplift_m'):
            result['material_'+name] = getattr(self, name).copy()
        contour = self.native_boundary_geometry
        display = boundary_labels.codes(self)
        result['boundary_segments'] = [dict(
            geometry_xyz=[start.tolist(), end.tolist()], normal=normal.tolist(),
            code=int(display[contact]), kinematic_code=int(self.bcode[contact]),
            owner_a=int(self.bp[contact]),
            owner_b=int(self.bq[contact]), down=int(self.down[contact]), contact_index=int(contact))
            for start, end, normal, contact in zip(contour['segments_start'], contour['segments_end'],
                                                   contour['segment_normals'], contour['contact_index'])]
        if getattr(self,'native_spreading_version',0)==1:
            result['native_spreading_version']=1
            result['boundary_segments']=native_spreading.boundary_segments(self)
            exact=self.native_spreading_geometry
            for kind,name in (('ocean','ridge_length_km'),('continental','rift_length_km')):
                pieces=exact[kind]
                result['stats'][name]=float(pieces['length'][pieces['active']&pieces['divergent']].sum())
            result['native_spreading_diagnostics']=dict(
                ocean_pieces=len(exact['ocean']['parent']),continental_pieces=len(exact['continental']['parent']),
                classification=('Exact material union along '+contour['diagnostics']['representation']+
                                '; parent contact graph retained.'),
                force_equations_unchanged=True)
        if 'ocean_datum_diagnostics' in context:
            result['ocean_datum_diagnostics']=deepcopy(context['ocean_datum_diagnostics'])
        if getattr(self,'native_arc_source_geometry_version',0)==1:
            result['native_arc_source_geometry_version']=1
            result['arc_source_candidate_diagnostics']=deepcopy(getattr(self,'arc_source_candidate_diagnostics',{}))
        if getattr(self,'native_subduction_version',0)==1:
            result.update(native_subduction.snapshot_metadata(self))
        if getattr(self,'native_initial_ownership_version',0)==1:
            result['native_initial_ownership_version']=1
            result['native_initial_ownership_active']=native_initial_ownership.active(self)
        result.update(world_design.snapshot_metadata(self))
        result.update(localized_accretion.snapshot_fields(self))
        result.update(backarc.snapshot_fields(self))
        primordial_subduction.annotate_snapshot(self, result, control)
        coarse_history.annotate_snapshot(self, result)
        return restart_boundary.project_snapshot(self, result)
