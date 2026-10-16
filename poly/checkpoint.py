"""Versioned, data-only RegionWorld checkpoints; no pickle or executable payloads.

Spherical polygons use spherely's exact native S2 data with a pinned codec version,
including polygons larger than a hemisphere. WKB's degree conversion can collapse
near-coincident vertices of valid S2 geometry, so it is unsuitable for restart.
The fixed native decoder never unpickles or imports caller-specified objects.
Derived caches are omitted; other state is represented or rejected as unsupported.
"""
from __future__ import annotations

import json
import hashlib
import importlib.metadata
import os
import tempfile
from dataclasses import fields
from pathlib import Path

import numpy as np
import spherely

from poly import sphere
from poly.mesh import ContinentMesh
from poly.regions import Boundary, RegionWorld
from poly.world import Crust

FORMAT = "poly-time-region-world"
VERSION = 1
GEOMETRY_CODEC = {"name": "spherely-native-s2-v1", "spherely_version": importlib.metadata.version("spherely")}
WORLD_FIELDS = {"crust", "frames", "regions", "boundaries", "names", "parents", "time_myr",
                "ledger", "next_ident", "sutures"}
WORLD_EXTRAS = {"simplified_myr", "simplified_count", "plate_policy", "retired_arc_km3"}
CRUST_FIELDS = {"plate", "kind", "shape", "born_myr", "thickness_km", "mesh"}
BOUNDARY_FIELDS = {"left", "right", "points", "kinds", "overriding", "closed_km", "idents", "arc_km3"}


class CheckpointError(ValueError):
    """A checkpoint is unsupported, corrupt, or inconsistent with this runtime."""


def _keys(value, expected, where):
    if not isinstance(value, dict) or set(value) != set(expected):
        actual = set(value) if isinstance(value, dict) else set()
        raise CheckpointError(f"{where}: missing {sorted(set(expected) - actual)}, "
                              f"unexpected {sorted(actual - set(expected))}")


def _known(obj, allowed, where):
    extra = set(vars(obj)) - set(allowed)
    if extra:
        raise CheckpointError(f"{where}: unsupported state {sorted(extra)}; extend the checkpoint schema")


def _number(value, where, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
        raise CheckpointError(f"{where}: expected a finite number")
    if minimum is not None and value < minimum:
        raise CheckpointError(f"{where}: must be at least {minimum}")
    return float(value)


def _integer(value, where, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise CheckpointError(f"{where}: expected an integer >= {minimum}")
    return int(value)


def _array(value, shape, where, integer=False):
    try:
        arr = np.asarray(value)
        if arr.dtype.kind not in ("i", "u", "f") or not np.isfinite(arr).all():
            raise ValueError("non-numeric or non-finite")
        if len(arr.shape) != len(shape) or any(n is not None and n != a for n, a in zip(shape, arr.shape)):
            raise ValueError(f"expected shape {shape}, got {arr.shape}")
        if integer and (not np.equal(arr, np.floor(arr)).all() or np.any(arr < 0)):
            raise ValueError("expected non-negative integer indices")
        return arr.astype(int if integer else float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CheckpointError(f"{where}: {exc}") from exc


def _unit_vectors(value, where):
    arr = _array(value, (None, 3), where)
    if len(arr) and not np.allclose(np.linalg.norm(arr, axis=1), 1., atol=1e-10, rtol=0.):
        raise CheckpointError(f"{where}: points must be unit vectors")
    return arr


def _data(value):
    """Encode ledger/metadata without losing integer dict keys, tuples or ndarray types."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _number(value, "data")
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "iufb" or not np.isfinite(value).all():
            raise CheckpointError("data: unsupported ndarray")
        return {"type": "ndarray", "dtype": value.dtype.str, "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, (list, tuple)):
        return {"type": "tuple" if isinstance(value, tuple) else "list", "items": [_data(v) for v in value]}
    if isinstance(value, dict):
        if any(not isinstance(k, (str, int, np.integer)) or isinstance(k, bool) for k in value):
            raise CheckpointError("data: only string/integer dictionary keys are supported")
        return {"type": "dict", "items": [[_data(k), _data(v)] for k, v in value.items()]}
    raise CheckpointError(f"data: unsupported value {type(value).__name__}")


def _undata(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _number(value, "data")
    if not isinstance(value, dict) or "type" not in value:
        raise CheckpointError("data: malformed tagged value")
    kind = value["type"]
    if kind == "ndarray":
        _keys(value, {"type", "dtype", "shape", "values"}, "ndarray")
        try:
            dtype = np.dtype(value["dtype"])
            if dtype.kind not in "iufb":
                raise ValueError("unsupported dtype")
            result = np.asarray(value["values"], dtype=dtype)
            if list(result.shape) != value["shape"] or not np.isfinite(result).all():
                raise ValueError("invalid shape or values")
            return result
        except (TypeError, ValueError) as exc:
            raise CheckpointError(f"ndarray: {exc}") from exc
    _keys(value, {"type", "items"}, "data")
    if not isinstance(value["items"], list):
        raise CheckpointError("data.items must be a list")
    if kind in ("list", "tuple"):
        result = [_undata(v) for v in value["items"]]
        return tuple(result) if kind == "tuple" else result
    if kind == "dict":
        result = {}
        for item in value["items"]:
            if not isinstance(item, list) or len(item) != 2:
                raise CheckpointError("data: malformed dictionary item")
            key, entry = item
            if isinstance(key, bool) or not isinstance(key, (str, int)) or key in result:
                raise CheckpointError("data: invalid or duplicate dictionary key")
            result[key] = _undata(entry)
        return result
    raise CheckpointError(f"data: unsupported tag {kind!r}")


def _shape(shape):
    kind, empty, payload = shape.__getstate__()
    # Native types: polygon=2, multipolygon=5, collection=6. Boolean operations
    # may retain zero-area line/point debris in a collection; preserve it exactly.
    if kind not in (2, 5, 6) or not isinstance(empty, bool) or not isinstance(payload, bytes):
        raise CheckpointError("geography: unsupported native polygon/collection state")
    return {"type": kind, "empty": empty, "s2_hex": payload.hex(),
            "sha256": hashlib.sha256(payload).hexdigest(), "area_km2": sphere.area_km2(shape)}


def _unshape(value):
    _keys(value, {"type", "empty", "s2_hex", "sha256", "area_km2"}, "geography")
    area = _number(value["area_km2"], "geography.area_km2", 0.)
    try:
        if type(value["type"]) is not int or value["type"] not in (2, 5, 6) or type(value["empty"]) is not bool:
            raise ValueError("expected native polygon/collection type and empty flag")
        payload = bytes.fromhex(value["s2_hex"])
        if hashlib.sha256(payload).hexdigest() != value["sha256"]:
            raise ValueError("payload checksum mismatch")
        shape = spherely.Geography.__new__(spherely.Geography)
        shape.__setstate__((value["type"], value["empty"], payload))
        actual = sphere.area_km2(shape)
        if not np.isfinite(actual) or not np.isclose(actual, area, rtol=1e-11, atol=1e-6):
            raise ValueError(f"area mismatch: {area} -> {actual}")
        if spherely.get_dimension(shape) != 2:
            raise ValueError("expected polygonal geography")
        return shape
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CheckpointError(f"geography: {exc}") from exc


def _mesh(mesh):
    if mesh is None:
        return None
    _known(mesh, {"nodes", "triangles", "thickness", "lost_nodes"}, "mesh")
    return {"nodes": mesh.nodes.tolist(), "triangles": mesh.triangles.tolist(),
            "thickness": mesh.thickness.tolist(),
            "lost_nodes": mesh.lost_nodes.tolist() if hasattr(mesh, "lost_nodes") else None}


def _unmesh(value):
    if value is None:
        return None
    _keys(value, {"nodes", "triangles", "thickness", "lost_nodes"}, "mesh")
    nodes = _unit_vectors(value["nodes"], "mesh.nodes")
    triangles = _array(value["triangles"], (None, 3), "mesh.triangles", integer=True)
    thickness = _array(value["thickness"], (len(nodes),), "mesh.thickness")
    if not len(nodes) or not len(triangles) or triangles.max() >= len(nodes) or np.any(thickness < 0.):
        raise CheckpointError("mesh: invalid triangle indices or thickness")
    result = ContinentMesh(nodes, triangles, thickness)
    if value["lost_nodes"] is not None:
        result.lost_nodes = (_unit_vectors(value["lost_nodes"], "mesh.lost_nodes")
                             if value["lost_nodes"] else np.empty((0, 3)))
    return result


def world_to_data(world):
    """Return an explicit JSON-ready state; fail if a new persistent field is unhandled."""
    _known(world, WORLD_FIELDS | WORLD_EXTRAS | {"_continent_cache"}, "world")
    crust = []
    for c in world.crust:
        _known(c, CRUST_FIELDS | {"escape_km2"}, "crust")
        crust.append({"plate": int(c.plate), "kind": c.kind, "shape": _shape(c.shape),
                      "born_myr": c.born_myr, "thickness_km": c.thickness_km, "mesh": _mesh(c.mesh),
                      "escape_km2": getattr(c, "escape_km2", None)})
    boundaries = []
    for b in world.boundaries:
        _known(b, BOUNDARY_FIELDS, "boundary")
        boundaries.append({"left": int(b.left), "right": int(b.right), "points": b.points.tolist(),
                           "kinds": list(b.kinds), "overriding": [None if o is None else int(o) for o in b.overriding],
                           "closed_km": b.closed_km.tolist(),
                           "idents": None if b.idents is None else np.asarray(b.idents).tolist(),
                           "arc_km3": None if b.arc_km3 is None else np.asarray(b.arc_km3).tolist()})
    extras = {k: getattr(world, k) for k in WORLD_EXTRAS if hasattr(world, k)}
    if "plate_policy" in extras:
        policy = extras["plate_policy"]
        _known(policy, {"minimum_plate_km2", "numerical_crumb_km2"}, "plate_policy")
        extras["plate_policy"] = {k: getattr(policy, k) for k in ("minimum_plate_km2", "numerical_crumb_km2")}
    return {"crust": crust, "boundaries": boundaries,
            "frames": [[int(p), m.tolist()] for p, m in world.frames.items()],
            "regions": [[int(p), _shape(g)] for p, g in world.regions.items()],
            "names": _data(world.names), "parents": _data(world.parents), "time_myr": world.time_myr,
            "ledger": _data(world.ledger), "next_ident": int(world.next_ident),
            "sutures": _data(world.sutures), "extras": _data(extras)}


def _mapping(items, decode, where):
    if not isinstance(items, list):
        raise CheckpointError(f"{where}: expected list of plate/value pairs")
    result = {}
    for item in items:
        if not isinstance(item, list) or len(item) != 2:
            raise CheckpointError(f"{where}: malformed pair")
        plate = _integer(item[0], where + ".plate")
        if plate in result:
            raise CheckpointError(f"{where}: duplicate plate {plate}")
        result[plate] = decode(item[1])
    return result


def world_from_data(value):
    _keys(value, WORLD_FIELDS | {"extras"}, "world")
    regions = _mapping(value["regions"], _unshape, "regions")
    frames = _mapping(value["frames"], lambda v: _array(v, (3, 3), "frame"), "frames")
    if not regions or not set(regions) <= set(frames):
        raise CheckpointError("world: every active plate needs a frame")
    for frame in frames.values():
        if not np.allclose(frame.T @ frame, np.eye(3), atol=1e-9, rtol=0.) or not np.isclose(np.linalg.det(frame), 1.):
            raise CheckpointError("frame: expected a proper rotation")
    crust = []
    for item in value["crust"]:
        _keys(item, CRUST_FIELDS | {"escape_km2"}, "crust")
        plate = _integer(item["plate"], "crust.plate")
        if plate not in regions or item["kind"] not in ("continent", "ocean"):
            raise CheckpointError("crust: invalid plate or kind")
        born = None if item["born_myr"] is None else _number(item["born_myr"], "born_myr")
        c = Crust(plate, item["kind"], _unshape(item["shape"]), born,
                  _number(item["thickness_km"], "thickness_km", 0.), _unmesh(item["mesh"]))
        if item["escape_km2"] is not None:
            c.escape_km2 = _number(item["escape_km2"], "escape_km2", 0.)
        crust.append(c)
    next_ident = _integer(value["next_ident"], "next_ident")
    boundaries = []
    for item in value["boundaries"]:
        _keys(item, BOUNDARY_FIELDS, "boundary")
        left, right = (_integer(item[k], "boundary." + k) for k in ("left", "right"))
        if left not in regions or right not in regions or left >= right:
            raise CheckpointError("boundary: invalid plate pair")
        points = _unit_vectors(item["points"], "boundary.points")
        n = len(points) - 1
        if n < 1 or len(item["kinds"]) != n or len(item["overriding"]) != n:
            raise CheckpointError("boundary: segment length mismatch")
        if not set(item["kinds"]) <= {"ridge", "transform", "trench", "converging", "collision"}:
            raise CheckpointError("boundary: unsupported boundary kind")
        if any(o is not None and (isinstance(o, bool) or o not in (left, right)) for o in item["overriding"]):
            raise CheckpointError("boundary: invalid overriding plate")
        closed = _array(item["closed_km"], (n,), "boundary.closed_km")
        ids = None if item["idents"] is None else _array(item["idents"], (n,), "boundary.idents", integer=True)
        if np.any(closed < 0.) or (ids is not None and np.any(ids >= next_ident)):
            raise CheckpointError("boundary: invalid closure or identity counter")
        arcs = None
        if item["arc_km3"] is not None:
            # Preserve the one-column legacy representation until Boundary.arcs upgrades it.
            shape = (n,) if np.ndim(item["arc_km3"]) == 1 else (n, 2)
            arcs = _array(item["arc_km3"], shape, "boundary.arc_km3")
            if np.any(arcs < 0.):
                raise CheckpointError("boundary: negative pending magma")
        boundaries.append(Boundary(left, right, points, item["kinds"], item["overriding"], closed, ids, arcs))
    names, parents, ledger, sutures, extras = (_undata(value[k]) for k in ("names", "parents", "ledger", "sutures", "extras"))
    if not isinstance(names, dict) or not isinstance(parents, dict) or not isinstance(ledger, list) or not isinstance(sutures, list):
        raise CheckpointError("world: invalid history containers")
    time = _number(value["time_myr"], "time_myr", 0.)
    last_time = -np.inf
    for entry in ledger:
        if not isinstance(entry, dict) or "time_myr" not in entry:
            raise CheckpointError("ledger: missing entry time")
        entry_time = _number(entry["time_myr"], "ledger.time_myr", 0.)
        if entry_time < last_time or entry_time > time + 1e-9:
            raise CheckpointError("ledger: nonchronological or future entry")
        last_time = entry_time
    if ledger and not np.isclose(last_time, time, rtol=0., atol=1e-9):
        raise CheckpointError("ledger: last entry does not match world time")
    if {c.plate for c in crust} != set(regions):
        raise CheckpointError("world: every active plate needs crust")
    for child, parent in parents.items():
        _integer(child, "parents.child")
        _integer(parent, "parents.parent")
        seen = {child}
        while parent in parents:
            if parent in seen:
                raise CheckpointError("parents: cyclic ancestry")
            seen.add(parent)
            parent = parents[parent]
    for s in sutures:
        _keys(s, {"plate", "points", "time_myr"}, "suture")
        if _integer(s["plate"], "suture.plate") not in regions:
            raise CheckpointError("suture: plate no longer exists")
        s["points"] = _unit_vectors(s["points"], "suture.points")
        if _number(s["time_myr"], "suture.time_myr", 0.) > time + 1e-9:
            raise CheckpointError("suture: future formation time")
    if not isinstance(extras, dict) or not set(extras) <= WORLD_EXTRAS:
        raise CheckpointError("world: unsupported extra state")
    required = {f.name for f in fields(RegionWorld)} & WORLD_EXTRAS
    if not required <= set(extras):
        raise CheckpointError(f"world: runtime requires explicit {sorted(required - set(extras))}")
    if "plate_policy" in extras:
        from poly.policy import PlatePolicy
        _keys(extras["plate_policy"], {"minimum_plate_km2", "numerical_crumb_km2"}, "plate_policy")
        extras["plate_policy"] = PlatePolicy(**extras["plate_policy"])
    if "retired_arc_km3" in extras:
        extras["retired_arc_km3"] = _array(extras["retired_arc_km3"], (2,), "retired_arc_km3")
        if np.any(extras["retired_arc_km3"] < 0.):
            raise CheckpointError("retired_arc_km3: negative reservoir")
    if "simplified_myr" in extras:
        _number(extras["simplified_myr"], "simplified_myr")
    if "simplified_count" in extras:
        _integer(extras["simplified_count"], "simplified_count")
    world = RegionWorld(crust, frames, regions, boundaries, names, parents,
                        time, ledger, next_ident, sutures)
    for key, entry in extras.items():
        setattr(world, key, entry)
    return world


def atomic_json(path, data):
    """Replace one file only after its complete JSON has reached disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, allow_nan=False, ensure_ascii=False, separators=(",", ":")) + "\n"
    name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".checkpoint-", delete=False) as stream:
            name = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)


def save_checkpoint(world, path, metadata=None):
    """Save complete state plus caller-owned continuation metadata, atomically."""
    state = world_to_data(world)
    # Validate before replacing the last accepted checkpoint.
    world_from_data(state)
    atomic_json(path, {"format": FORMAT, "version": VERSION, "geometry_codec": GEOMETRY_CODEC, "radius_km": sphere.RADIUS_KM,
                       "world": state, "metadata": _data({} if metadata is None else metadata)})


def load_checkpoint(path):
    """Return (RegionWorld, metadata); unknown versions and missing fields fail closed."""
    def unique_pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise CheckpointError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result
    try:
        with Path(path).open(encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=unique_pairs,
                              parse_constant=lambda x: (_ for _ in ()).throw(CheckpointError(f"non-finite JSON: {x}")))
        _keys(value, {"format", "version", "geometry_codec", "radius_km", "world", "metadata"}, "checkpoint")
        if value["format"] != FORMAT or type(value["version"]) is not int or value["version"] != VERSION:
            raise CheckpointError("unsupported checkpoint format/version; no implicit migration")
        if value["radius_km"] != sphere.RADIUS_KM:
            raise CheckpointError("checkpoint sphere radius differs from runtime")
        if value["geometry_codec"] != GEOMETRY_CODEC:
            raise CheckpointError("unsupported geometry codec/spherely version; explicit migration required")
        return world_from_data(value["world"]), _undata(value["metadata"])
    except (KeyError, TypeError, json.JSONDecodeError, OverflowError) as exc:
        raise CheckpointError(f"malformed checkpoint: {exc}") from exc
