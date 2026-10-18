"""Planetary material resolution for the optional artist model.

A sub-resolution continental footprint is covered by its plate's ocean mask;
its rock is not deleted or squeezed into the remaining pixels. Nearby resolved
crust receives the volume over the normal material reach. Isolated material is
a passive inventory carried in plate coordinates until a receiver arrives.
This is explicit numerical coarsening, not a model of transport across the sea.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
import spherely

from poly import sphere
from poly.mesh import MINIMUM_KM2, REACH_KM


def inventory(world):
    state = getattr(world, 'geology', None)
    return [] if state is None else state.get('unresolved_material', [])


def volume_km3(world):
    return sum(p['volume_km3'] for p in inventory(world))


def _carry(parcel, plate, source_frame, target_frame):
    from poly.geology import carry
    rotation = target_frame.T @ source_frame
    parcel['point'] = sphere.rotate(parcel['point'], rotation)
    parcel['geology'] = carry(parcel['geology'], rotation)
    parcel['plate'] = plate


def transfer(world, plate, to, domain, source_frame, target_frame, whole=False):
    """Keep passive material with the surface transferred by a split or weld."""
    for parcel in inventory(world):
        if parcel['plate'] != plate:
            continue
        point = sphere.rotate(parcel['point'], source_frame)
        lon, lat = sphere.to_lonlat(point)
        if whole or spherely.distance(domain, spherely.points(float(lon), float(lat))) <= 1.:
            _carry(parcel, to, source_frame, target_frame)


def _owner(world, parcel, owned):
    """An unresolved tracer crossed by a moving boundary follows its new owner."""
    point = sphere.rotate(parcel['point'], world.frame(parcel['plate']))
    old = parcel['plate']
    def distance(plate):
        lon, lat = sphere.to_lonlat(sphere.rotate(point, world.frame(plate).T))
        return float(spherely.distance(owned[plate], spherely.points(float(lon), float(lat))))
    if old in owned and distance(old) <= 1.:
        return
    owner = min(owned, key=lambda p: (distance(p), p))
    _carry(parcel, owner, world.frame(old), world.frame(owner))


def _receiver(world, parcel, bodies):
    point = sphere.rotate(parcel['point'], world.frame(parcel['plate']))
    local = {}
    best = None
    for body in bodies:
        if body.plate not in local:
            xyz = sphere.rotate(point, world.frame(body.plate).T)
            lon, lat = sphere.to_lonlat(xyz)
            local[body.plate] = xyz, spherely.points(float(lon), float(lat))
        xyz, probe = local[body.plate]
        distance = float(spherely.distance(body.shape, probe)) / 1000.
        if distance <= REACH_KM and (best is None or distance < best[0]):
            best = distance, body, xyz
    return best


def consolidate(world, entry=None):
    """Remove independent continental bodies below the existing mesh resolution.

    The full surface footprint stays on the same plate, so coarsening creates
    neither a plate gap nor a ridge. Its seafloor age is unknown. Volume books
    include the passive inventory; its source province and material reference
    survive restart and ownership changes. Deposited material is homogenized
    into the receiving field, with the source reference retained in the ledger.
    """
    from poly.geology import enabled, separate_bodies
    from poly.regions import _point_in
    from poly.world import Crust
    if not enabled(world):
        return
    separate_bodies(world, entry)
    small = [c for c in world.crust if c.kind == 'continent' and 0. < c.area_km2() < MINIMUM_KM2]
    parcels = world.geology.setdefault('unresolved_material', [])
    count, area, material = len(small), 0., 0.
    for piece in small:
        a, v = piece.area_km2(), piece.volume_km3()
        point = _point_in(piece.shape)
        lon, lat = sphere.to_lonlat(point)
        if spherely.distance(piece.shape, spherely.points(float(lon), float(lat))) > 1.:
            point = sphere.polygons(piece.shape)[0][0][0].copy()
        parcels.append(dict(plate=piece.plate, point=point, volume_km3=v,
                            geology=deepcopy(piece.geology)))
        # Keep exactly the same native shape and owner. This is an unresolved
        # marine cover, not newly formed mafic crust with a fabricated age.
        world.crust.append(Crust(piece.plate, 'ocean', piece.shape, born_myr=None))
        area += a
        material += v
    removed = {id(c) for c in small}
    world.crust = [c for c in world.crust if id(c) not in removed]
    # All unresolved marine cover on one plate is a single age-unknown mask,
    # rather than an ever-growing list of tiny mechanical surface objects.
    for plate in sorted({c.plate for c in world.crust}):
        cover = [c for c in world.crust if c.plate == plate and c.kind == 'ocean' and c.born_myr is None]
        if len(cover) > 1:
            shape = sphere.union(*(c.shape for c in cover))
            change = sphere.area_km2(shape) - sum(c.area_km2() for c in cover)
            remove = {id(c) for c in cover}
            world.crust = [c for c in world.crust if id(c) not in remove]
            world.crust.append(Crust(plate, 'ocean', shape, born_myr=None))
            if entry is not None:
                entry['numerical_area_added_km2'] += max(change, 0.)
                entry['numerical_area_removed_km2'] += max(-change, 0.)
                entry['gained']['junction'] += max(change, 0.)
                entry['lost']['sliver'] += max(-change, 0.)
    bodies = [c for c in world.crust if c.kind == 'continent' and c.area_km2() >= MINIMUM_KM2]
    owned = {plate: sphere.union(*(c.shape for c in world.crust if c.plate == plate))
             for plate in {c.plate for c in world.crust}}
    kept, transferred = [], 0.
    for parcel in parcels:
        _owner(world, parcel, owned)
        found = _receiver(world, parcel, bodies)
        if found is None:
            kept.append(parcel)
            continue
        distance, body, point = found
        volume = parcel['volume_km3']
        if body.mesh is not None:
            body.mesh.add_volume(volume, body.area_km2(), near=point[None])
        body.thickness_km += volume / body.area_km2()
        transferred += volume
        if entry is not None:
            entry.setdefault('coarsening_transfers', []).append(dict(
                source_plate=parcel['plate'], receiver_plate=body.plate,
                source_geology=deepcopy(parcel['geology']),
                source_reference_point=(parcel['point'].copy() if parcel['geology'] is None else
                                        parcel['geology']['reference_from_plate'] @ parcel['point']),
                receiver_province=None if body.geology is None else body.geology['province_id'],
                distance_km=distance, volume_km3=volume))
    world.geology['unresolved_material'] = kept
    if entry is not None:
        entry['coarsened_bodies'] = entry.get('coarsened_bodies', 0) + count
        entry['coarsened_continent_km2'] = entry.get('coarsened_continent_km2', 0.) + area
        entry['coarsened_continent_km3'] = entry.get('coarsened_continent_km3', 0.) + material
        entry['coarsening_deposited_km3'] = entry.get('coarsening_deposited_km3', 0.) + transferred
        entry['unresolved_continent_km3'] = volume_km3(world)


def validate(world):
    from poly.geology import validate_piece
    parcels = inventory(world)
    if not isinstance(parcels, list):
        raise ValueError('Invalid unresolved continental inventory.')
    for parcel in parcels:
        if not isinstance(parcel, dict) or set(parcel) != {'plate', 'point', 'volume_km3', 'geology'}:
            raise ValueError('Invalid unresolved continental material.')
        point = np.asarray(parcel['point'])
        if (type(parcel['plate']) is not int or parcel['plate'] not in world.regions
                or point.shape != (3,) or not np.isfinite(point).all()
                or not np.isclose(np.linalg.norm(point), 1., atol=1e-10, rtol=0.)
                or not np.isfinite(parcel['volume_km3']) or parcel['volume_km3'] < 0.):
            raise ValueError('Invalid unresolved material position, owner or volume.')
        validate_piece(parcel['geology'])
        if parcel['geology'] is not None and parcel['geology']['province_id'] >= world.geology['next_province']:
            raise ValueError('Province counter precedes unresolved material identity.')
