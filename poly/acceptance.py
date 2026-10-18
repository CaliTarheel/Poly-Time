"""Acceptance checks for polygon worlds, without repairing or renormalizing them.

The existing cap/collision/arc/erosion/rift fixtures show up to 60 km2 of uncovered
area and 53 km2 of unbooked area change per step. A 1 ppm sphere allowance (510
km2 on Earth) leaves room for the model's 4 km2 detail cutoff and S2 rebuilding,
but cannot hide continent-sized defects. It is not a convergence claim.

Signed and absolute cumulative budgets prevent a small error from being forgiven
anew on every step. Numerical cleaning/simplification is also reported separately
from geological fluxes; its deliberately larger budget reflects the model's 100
km2 vertex simplification scale. Long runs exhausting a budget must stop for
review, not reset the counter or silently rescale crust.
"""
from __future__ import annotations

import math

import numpy as np

from poly import sphere

GEOMETRY_KM2 = 1e-6 * sphere.SPHERE_KM2
CUMULATIVE_DRIFT_KM2 = 1e-5 * sphere.SPHERE_KM2
ABSOLUTE_DRIFT_KM2 = 1e-4 * sphere.SPHERE_KM2
REPAIR_STEP_KM2 = 1e-4 * sphere.SPHERE_KM2
REPAIR_TOTAL_KM2 = 1e-3 * sphere.SPHERE_KM2
# Existing collision/weld fixtures lose ~3.5e-5 of their continental volume in
# geometric reconstruction. Keep that visible, with a 1e-4 acceptance ceiling.
VOLUME_RELATIVE = 1e-4
ABSOLUTE_VOLUME_RELATIVE = 1e-3
VOLUME_FLOOR_KM3 = 1.


class GeometryBudgetError(ValueError):
    """A finite spatial or per-step area defect eligible for timestep refinement."""
    def __init__(self, metric, value, limit):
        self.metric, self.value, self.limit = metric, float(value), float(limit)
        super().__init__('%s: %.6g km2 (budget %.6g).' % (metric, value, limit))


def _bounded(value, limit, name, units):
    if not math.isfinite(value) or abs(value) > limit:
        raise ValueError('%s: %.6g %s (budget %.6g).' % (name, value, units, limit))


def _nonnegative(value, name):
    if not math.isfinite(value) or value < 0.:
        raise ValueError('%s must be finite and nonnegative: %r.' % (name, value))


def geometry(world, full, entry):
    """Measure real coverage, not just sums that let gaps cancel overlaps."""
    covered = sphere.union(*world.regions.values())
    union_area = sphere.area_km2(covered)
    entry['plate_gap_km2'] = sphere.area_km2(sphere.difference(full, covered))
    # Sum minus union counts extra ownership (including multiplicity in triple
    # overlaps). Unlike sum minus sphere, an uncovered patch cannot cancel it.
    entry['plate_overlap_km2'] = max(0., sum(sphere.area_km2(r) for r in world.regions.values()) - union_area)
    entry['tiling_error_km2'] = sum(sphere.area_km2(r) for r in world.regions.values()) - sphere.SPHERE_KM2
    crust_by_plate = {}
    for piece in world.crust:
        if piece.plate not in world.regions:
            raise ValueError('Crust belongs to absent plate %s.' % piece.plate)
        crust_by_plate.setdefault(piece.plate, []).append(piece)
        if piece.kind == 'continent':
            _nonnegative(piece.thickness_km, 'Continental thickness')
            if piece.mesh is not None:
                if not np.isfinite(piece.mesh.thickness).all() or (piece.mesh.thickness < 0.).any():
                    raise ValueError('Continental mesh thickness must be finite and nonnegative.')
                _bounded(piece.mesh.volume_km3(piece.area_km2()) - piece.volume_km3(),
                         max(VOLUME_FLOOR_KM3, VOLUME_RELATIVE * piece.volume_km3()),
                         'Continental mesh and mean thickness disagree', 'km3')
    defects = {}
    for plate, region in world.regions.items():
        pieces = crust_by_plate.get(plate, [])
        local = sphere.union(*(c.shape for c in pieces))
        local_area = sphere.area_km2(local)
        # Validation must not use rotation's ordinary near-full-sphere shortcut
        # (which fills holes up to 51,000 km2). Constrain any reconstruction to
        # the same smaller allowance this check actually promises.
        carried = sphere.rotate_geography(local, world.frame(plate), tolerance=GEOMETRY_KM2 / sphere.SPHERE_KM2)
        defects[plate] = dict(
            gap_km2=sphere.area_km2(sphere.difference(region, carried)),
            outside_km2=sphere.area_km2(sphere.difference(carried, region)),
            overlap_km2=max(0., sum(c.area_km2() for c in pieces) - local_area),
            area_error_km2=abs(sum(c.area_km2() for c in pieces) - sphere.area_km2(region)))
    entry['crust_coverage_km2'] = defects
    entry['crust_gap_km2'] = sum(d['gap_km2'] for d in defects.values())
    entry['crust_outside_km2'] = sum(d['outside_km2'] for d in defects.values())
    entry['crust_overlap_km2'] = sum(d['overlap_km2'] for d in defects.values())
    entry['crust_vs_plate_error_km2'] = max((d['area_error_km2'] for d in defects.values()), default=0.)
    for key in ('plate_gap_km2', 'plate_overlap_km2', 'crust_gap_km2', 'crust_outside_km2',
                'crust_overlap_km2', 'crust_vs_plate_error_km2'):
        if math.isfinite(entry[key]) and abs(entry[key]) > GEOMETRY_KM2:
            raise GeometryBudgetError(key, entry[key], GEOMETRY_KM2)
        _bounded(entry[key], GEOMETRY_KM2, key, 'km2')


def budgets(world, entry):
    """Close area and realized continental-volume books across accepted steps.

    Arc banks are pending magma, not continental crust. Only ``arc_km3`` actually
    emplaced into crust is a source; erosion deposited on the same continent is
    internal redistribution. ``sediment_to_sea_km3`` is the current model's sink.
    Future sources/sinks (e.g. LIPs) must explicitly extend this accounting.
    """
    previous = world.ledger[-1] if world.ledger else {}
    for label in ('gained', 'lost'):
        for kind, value in entry[label].items():
            _nonnegative(value, '%s %s area' % (label, kind))
    entry['balance_error_km2'] = sum(entry['gained'].values()) - sum(entry['lost'].values())
    drift = entry['rebuild_drift_km2']
    entry['cumulative_rebuild_drift_km2'] = previous.get('cumulative_rebuild_drift_km2', 0.) + drift
    entry['absolute_rebuild_drift_km2'] = previous.get('absolute_rebuild_drift_km2', 0.) + abs(drift)
    for key in ('numerical_area_added_km2', 'numerical_area_removed_km2'):
        _nonnegative(entry[key], key)
    repair = entry['numerical_area_added_km2'] + entry['numerical_area_removed_km2']
    entry['numerical_area_repair_km2'] = repair
    entry['cumulative_numerical_area_repair_km2'] = previous.get('cumulative_numerical_area_repair_km2', 0.) + repair
    for key, limit in (
            ('balance_error_km2', GEOMETRY_KM2), ('rebuild_drift_km2', GEOMETRY_KM2),
            ('cumulative_rebuild_drift_km2', CUMULATIVE_DRIFT_KM2),
            ('absolute_rebuild_drift_km2', ABSOLUTE_DRIFT_KM2),
            ('numerical_area_repair_km2', REPAIR_STEP_KM2),
            ('cumulative_numerical_area_repair_km2', REPAIR_TOTAL_KM2)):
        if (key in ('balance_error_km2', 'rebuild_drift_km2')
                and math.isfinite(entry[key]) and abs(entry[key]) > limit):
            raise GeometryBudgetError(key, entry[key], limit)
        _bounded(entry[key], limit, key, 'km2')
    source, sink = entry['arc_km3'], entry['sediment_to_sea_km3']
    _nonnegative(source, 'Continental source')
    _nonnegative(sink, 'Continental sink')
    before = entry['continent_before_km3']
    after = world._continent_km3()
    _nonnegative(before, 'Continental volume before step')
    _nonnegative(after, 'Continental volume after step')
    initial = previous.get('continent_initial_km3', before)
    sources = previous.get('continent_sources_km3', 0.) + source
    sinks = previous.get('continent_sinks_km3', 0.) + sink
    residual = after - before - source + sink
    cumulative = after - initial - sources + sinks
    # If an accepted world is edited between steps, the long-term residual still
    # detects it. On an old checkpoint lacking these keys, accounting starts at
    # its first newly checked step; no historic conservation is claimed.
    entry.update(continent_km3=after, continent_initial_km3=initial,
                 continent_sources_km3=sources, continent_sinks_km3=sinks,
                 continent_balance_error_km3=residual,
                 cumulative_continent_error_km3=cumulative,
                 absolute_continent_error_km3=previous.get('absolute_continent_error_km3', 0.) + abs(residual))
    reference = max(initial + sources, before + source, after)
    limit = max(VOLUME_FLOOR_KM3, VOLUME_RELATIVE * reference)
    entry['continent_budget_km3'] = limit
    for key in ('continent_balance_error_km3', 'cumulative_continent_error_km3'):
        _bounded(entry[key], limit, key, 'km3')
    _bounded(entry['absolute_continent_error_km3'], max(VOLUME_FLOOR_KM3, ABSOLUTE_VOLUME_RELATIVE * reference),
             'absolute_continent_error_km3', 'km3')
