"""Standalone Poly-Time runs and repeatable timestep qualification.

This drives RegionWorld, not the inherited root production worker. Checkpoints
commit after every accepted substep and preserve the outer rifting cadence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import shapely

from poly import dynamics, geology, sphere
from poly.acceptance import GeometryBudgetError
from poly.build import cap_world, voronoi_world
from poly.checkpoint import CheckpointError, atomic_json, load_checkpoint, save_checkpoint
from poly.regions import from_network


def provenance():
    root = Path(__file__).resolve().parent
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "spherely", "shapely")},
            "geos": shapely.geos_version_string,
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}}


def diagnostics(world):
    """Measured state and accumulated ledger terms; no claim of geological calibration."""
    boundary_km = {}
    for b in world.boundaries:
        for kind, length in zip(b.kinds, sphere.arc_km(b.points[:-1], b.points[1:])):
            boundary_km[kind] = boundary_km.get(kind, 0.) + float(length)
    ages = {"0_to_20": 0., "20_to_100": 0., "100_to_200": 0., "200_plus": 0., "unknown": 0.}
    pending_arc = np.zeros(2)
    for b in world.boundaries:
        if b.arc_km3 is not None:
            bank = np.asarray(b.arc_km3)
            pending_arc += [float(bank.sum()), 0.] if bank.ndim == 1 else bank.sum(axis=0)
    for c in world.crust:
        if c.kind == "ocean":
            age = None if c.born_myr is None else max(world.time_myr - c.born_myr, 0.)
            key = "unknown" if age is None else ("0_to_20" if age < 20. else "20_to_100" if age < 100.
                                                   else "100_to_200" if age < 200. else "200_plus")
            ages[key] += c.area_km2()
    budgets = {side: {} for side in ("gained", "lost")}
    for entry in world.ledger:
        for side, terms in budgets.items():
            for term, amount in entry.get(side, {}).items():
                terms[term] = terms.get(term, 0.) + float(amount)
    from poly.coarsen import volume_km3
    return {"time_myr": world.time_myr, "plate_count": len(world.regions),
            "plate_area_km2": {str(p): sphere.area_km2(g) for p, g in world.regions.items()},
            "crust_area_km2": world.areas(), "continent_volume_km3": world._continent_km3(),
            "resolved_continent_volume_km3": sum(c.volume_km3() for c in world.crust if c.kind == "continent"),
            "unresolved_continent_volume_km3": volume_km3(world),
            "maximum_continental_thickness_km": geology.maximum_thickness(world),
            "boundary_length_km": boundary_km, "ocean_area_by_age_myr_km2": ages,
            "pending_escape_km2": sum(getattr(c, "escape_km2", 0.) for c in world.crust),
            "pending_arc_km3": pending_arc.tolist(),
            "retired_arc_km3": np.asarray(getattr(world, "retired_arc_km3", np.zeros(2))).tolist(),
            "accepted_substeps": len(world.ledger), "area_budgets_km2": budgets,
            "cumulative_rebuild_drift_km2": sum(e.get("rebuild_drift_km2", 0.) for e in world.ledger),
            "maximum_absolute_balance_error_km2": max((abs(e.get("balance_error_km2", 0.)) for e in world.ledger), default=0.)}


def make_world(config):
    if config["fixture"] == "cap":
        network, rotations = cap_world(spacing_km=config["spacing_km"],
                                       cap_omega=(0., 0., config["speed_km_myr"] / sphere.RADIUS_KM))
    else:
        network, rotations = voronoi_world(count=config["plates"], seed=config["seed"],
                                           spacing_km=config["spacing_km"], speed_km_myr=config["speed_km_myr"],
                                           continent_fraction=config["continent_fraction"])
    world = from_network(network)
    config["rotations_per_myr"] = {int(p): np.asarray(v) for p, v in rotations.items()}
    return world


def advance(world, config, progress, intervals, commit):
    """Advance whole output intervals (a resumed partial interval counts as one).

The commit callback runs after each accepted step, with progress sufficient to
resume inside a force-driven interval without repeating its rift decision.
"""
    _validate_continuation(config, progress)
    dt = float(config["dt_myr"])
    if not np.isfinite(dt) or dt <= 0. or type(intervals) is not int or intervals < 0:
        raise ValueError("dt must be finite and positive, intervals non-negative")
    target = progress["completed_intervals"] + intervals
    balance = dynamics.Balance(dynamics.Parameters(**config["parameters"])) if config["motion"] == "forces" else None
    rifting = dynamics.Rifting(**config["rifting"]) if balance is not None and config["rifting"] is not None else None
    while progress["completed_intervals"] < target:
        remaining = progress["remaining_myr"]
        events = []
        if remaining == 0.:
            remaining = dt
            if rifting is not None:
                rotations = balance.solve(world)
                candidates = [c for p in world.plates() for c in dynamics.cuts(balance, world, rotations, p, rifting)]
                best = max(candidates, key=lambda c: c.ratio, default=None)
                if best is not None and best.ratio > 1.:
                    events.append(world.rift(best))
        rotations = balance.solve(world) if balance is not None else world._complete(config["rotations_per_myr"])
        speeds = {p: float(np.linalg.norm(w)) * sphere.RADIUS_KM for p, w in rotations.items()}
        fastest = max(speeds.values(), default=0.)
        sub = min(remaining, config["parameters"]["step_limit_km"] / max(fastest, 1e-9))
        rejected = []
        while True:
            try:
                entry = world.step(rotations, sub)
                break
            except GeometryBudgetError as exc:
                # RegionWorld.step is atomic. Refine finite spatial defects without changing
                # any budget, repeating the rift decision, or retrying volume/cumulative errors.
                if getattr(world, 'geology', None) is None or len(rejected) >= 6:
                    raise
                rejected.append(dict(substep_myr=sub, metric=exc.metric,
                                     defect_km2=exc.value, budget_km2=exc.limit))
                sub *= .5
        if rejected:
            entry['geometry_rejections'] = rejected
        entry["events"] = events + entry["events"]
        entry["speed_km_myr"] = speeds
        entry["substep_myr"] = sub
        remaining -= sub
        if remaining < 1e-9:
            progress["completed_intervals"] += 1
            remaining = 0.
        progress["remaining_myr"] = remaining
        commit(world, progress)
    return world


def _validate_continuation(config, progress):
    expected = {"fixture", "plates", "seed", "spacing_km", "speed_km_myr", "continent_fraction", "dt_myr",
                "motion", "parameters", "rifting", "rotations_per_myr"}
    # The public driver also accepts a minimal supplied-world configuration.
    minimal = {"dt_myr", "motion", "parameters", "rifting", "rotations_per_myr"}
    if not isinstance(config, dict) or set(config) not in (expected, minimal):
        raise CheckpointError("invalid runner configuration fields")
    if config["motion"] not in ("forces", "prescribed"):
        raise CheckpointError("unsupported motion mode")
    if not isinstance(config["parameters"], dict) or set(config["parameters"]) != set(asdict(dynamics.Parameters())):
        raise CheckpointError("force parameter schema changed; explicit migration required")
    if config["rifting"] is not None and (not isinstance(config["rifting"], dict)
                                         or set(config["rifting"]) != set(asdict(dynamics.Rifting()))):
        raise CheckpointError("rifting parameter schema changed; explicit migration required")
    if not isinstance(progress, dict) or set(progress) != {"completed_intervals", "remaining_myr"}:
        raise CheckpointError("invalid continuation phase fields")
    completed, remaining, dt = progress["completed_intervals"], progress["remaining_myr"], config["dt_myr"]
    if type(completed) is not int or completed < 0 or not isinstance(dt, (float, int)) or isinstance(dt, bool):
        raise CheckpointError("invalid continuation interval count or timestep")
    if not isinstance(remaining, (int, float)) or isinstance(remaining, bool) or not np.isfinite(dt) or dt <= 0.:
        raise CheckpointError("invalid continuation timestep/remaining time")
    if not np.isfinite(remaining) or not 0. <= remaining < dt:
        raise CheckpointError("invalid continuation remaining time")
    limit = config["parameters"]["step_limit_km"]
    if not isinstance(limit, (int, float)) or isinstance(limit, bool) or not np.isfinite(limit) or limit <= 0.:
        raise CheckpointError("step limit must be finite and positive")
    if not isinstance(config["rotations_per_myr"], dict):
        raise CheckpointError("rotations must map integer plate IDs to vectors")
    for plate, vector in config["rotations_per_myr"].items():
        array = np.asarray(vector)
        if type(plate) is not int or array.shape != (3,) or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
            raise CheckpointError("invalid prescribed rotation")


def _configuration(args, dt=None):
    parameters = asdict(dynamics.Parameters(sample_points=args.samples))
    return {"fixture": args.fixture, "plates": args.plates, "seed": args.seed,
            "spacing_km": args.spacing, "speed_km_myr": args.speed,
            "continent_fraction": args.continent_fraction, "dt_myr": args.dt if dt is None else dt,
            "motion": args.motion, "parameters": parameters,
            "rifting": None if args.no_rifting else asdict(dynamics.Rifting())}


def _new_directory(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}; use resume or a new directory")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _run(output, world, config, progress, source, intervals, manifest):
    def commit(current, phase):
        save_checkpoint(current, output / "checkpoint.json",
                        {"config": config, "progress": dict(phase), "provenance": source})
        atomic_json(output / "diagnostics.json", diagnostics(current))
    manifest["status"] = "running"
    atomic_json(output / "manifest.json", manifest)
    try:
        commit(world, progress)
        advance(world, config, progress, intervals, commit)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(output / "manifest.json", manifest)
        raise
    manifest.update(status="complete", completed_intervals=progress["completed_intervals"], time_myr=world.time_myr)
    atomic_json(output / "manifest.json", manifest)
    return diagnostics(world)


def start(args):
    config = _configuration(args)
    world = make_world(config)
    output = _new_directory(args.output)
    source = provenance()
    manifest = {"format": "poly-time-run", "version": 1, "provenance": source,
                "configuration": {k: v for k, v in config.items() if k != "rotations_per_myr"}}
    result = _run(output, world, config, {"completed_intervals": 0, "remaining_myr": 0.}, source, args.steps, manifest)
    print(json.dumps({"time_myr": result["time_myr"], "plates": result["plate_count"], "checkpoint": str(output / "checkpoint.json")}))


def resume(args):
    output = Path(args.output)
    world, metadata = load_checkpoint(output / "checkpoint.json")
    if not isinstance(metadata, dict) or set(metadata) != {"config", "progress", "provenance"}:
        raise CheckpointError("checkpoint does not contain a Poly run continuation")
    if metadata["provenance"] != provenance():
        raise CheckpointError("source or numerical environment changed; use the recorded code/environment to resume")
    config, progress = metadata["config"], metadata["progress"]
    _validate_continuation(config, progress)
    manifest = json.loads((output / "manifest.json").read_text())
    result = _run(output, world, config, progress, metadata["provenance"], args.steps, manifest)
    print(json.dumps({"time_myr": result["time_myr"], "plates": result["plate_count"], "checkpoint": str(output / "checkpoint.json")}))


def qualify(args):
    output = _new_directory(args.output)
    results = []
    failures = []
    for label, dt, spacing in (("baseline", args.dt, args.spacing), ("half-step", args.dt / 2., args.spacing),
                               ("half-spacing", args.dt, args.spacing / 2.)):
        count = args.duration / dt
        if not math.isclose(count, round(count), abs_tol=1e-9):
            raise ValueError("qualification duration must be a whole number of baseline timesteps")
        config = _configuration(args, dt)
        config["spacing_km"] = spacing
        folder = _new_directory(output / label)
        source = provenance()
        try:
            world = make_world(config)
            summary = _run(folder, world, config, {"completed_intervals": 0, "remaining_myr": 0.}, source,
                           round(count), {"format": "poly-time-qualification-run", "version": 1, "provenance": source,
                                          "configuration": {k: v for k, v in config.items() if k != "rotations_per_myr"}})
            results.append({"label": label, "dt_myr": dt, "spacing_km": spacing, "summary": summary})
        except Exception as exc:
            failures.append({"label": label, "error": f"{type(exc).__name__}: {exc}"})
    report = {"format": "poly-time-qualification", "version": 1, "duration_myr": args.duration,
              "runs": results, "failures": failures,
              "interpretation": "Resolution-sensitivity measurements, not calibrated geological acceptance thresholds."}
    if len(results) == 3:
        baseline = results[0]["summary"]
        report["differences_from_baseline"] = {r["label"]: {
            key: r["summary"][key] - baseline[key]
            for key in ("plate_count", "continent_volume_km3", "cumulative_rebuild_drift_km2", "maximum_absolute_balance_error_km2")}
            for r in results[1:]}
    atomic_json(output / "qualification.json", report)
    print(json.dumps({"report": str(output / "qualification.json"), "completed": len(results), "failed": len(failures)}))
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "resume", "qualify"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.set_defaults(action={"start": start, "resume": resume, "qualify": qualify}[name])
        if name != "qualify":
            command.add_argument("--steps", type=int, default=2, help="output intervals, including any partially completed resumed interval")
        if name == "resume":
            continue
        command.add_argument("--fixture", choices=("cap", "voronoi"), default="voronoi")
        command.add_argument("--motion", choices=("prescribed", "forces"), default="forces")
        command.add_argument("--dt", type=float, default=.5, help="output interval in Myr")
        command.add_argument("--plates", type=int, default=8)
        command.add_argument("--seed", type=int, default=1)
        command.add_argument("--continent-fraction", type=float, default=.3)
        command.add_argument("--spacing", type=float, default=250., help="initial boundary spacing in km")
        command.add_argument("--speed", type=float, default=30., help="initial/prescribed angular speed times planetary radius, km/Myr")
        command.add_argument("--samples", type=int, default=5000, help="force quadrature samples; recorded explicitly")
        command.add_argument("--no-rifting", action="store_true")
        if name == "qualify":
            command.add_argument("--duration", type=float, default=2., help="Myr per trial")
    args = parser.parse_args(argv)
    for key in ("dt", "spacing", "speed", "duration"):
        if hasattr(args, key) and (not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0.):
            parser.error(f"--{key} must be finite and positive")
    if hasattr(args, "steps") and args.steps < 0:
        parser.error("--steps must be non-negative")
    if hasattr(args, "samples") and (args.samples < 4 or args.plates < 4 or not 0. <= args.continent_fraction <= 1.):
        parser.error("at least four plates/samples and a continent fraction in [0, 1] are required")
    try:
        return args.action(args) or 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Poly run failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
