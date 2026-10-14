"""Read-only source-zone partition for one ordered continental face pair.

This uses the same affine entry and spherical stack clipping kernels as native
entry/stack mechanics. It assigns geometry only: vertical order, upper basal
depth, temperature, ownership and native source commitment remain external.
"""

import numpy as np

import burial_depth
import finite_entry_arc
import mesh_coverage
from exact_polygon import ExactPolygon
from channel_region_geometry import (
    _checked_polygon, spherical_partition_intersections,
    remove_roundoff_vertices as _remove_roundoff_vertices)


def _entry_pieces(lower, normal, radius_km, midpoint, half_length):
    left = right = np.ones(3)
    if midpoint is not None:
        l, r, _, _ = finite_entry_arc.endpoint_planes(
            normal, midpoint, half_length, radius_km)
        left, right = lower @ l, lower @ r
    q = radius_km * 1000. * np.arcsin(np.clip(lower @ normal, -1., 1.))
    _, _, _, _, fraction, pieces = finite_entry_arc.integrate(q, left, right)
    polygons = []
    for _, barycentric in pieces:
        entered = barycentric @ lower
        entered /= np.linalg.norm(entered, axis=1)[:, None]
        polygons.append(entered)
    return float(fraction), polygons


def _split_entry_region(polygon, pieces, radius_km, tolerance):
    regions = [(polygon, False)]
    for entered in pieces:
        planes = mesh_coverage._triangle_planes(entered)
        next_regions = []
        for part, active in regions:
            if active:
                duplicate = mesh_coverage._clip_planes(part, planes)
                if mesh_coverage._polygon_area(duplicate, radius_km) > tolerance:
                    raise ValueError('Entry source pieces overlap each other.')
                next_regions.append((part, active))
                continue
            inside, outside = burial_depth._split(part, planes, radius_km)
            next_regions.extend((outside_part, False) for outside_part in outside)
            if mesh_coverage._polygon_area(inside, radius_km) > 0.:
                next_regions.append((inside, True))
        regions = next_regions
    return regions


def partition_entry_pair(lower_triangle, upper_triangle, hinge_normal, *,
                         radius_km=6371., finite_midpoint=None,
                         finite_half_length_km=None):
    """Disjoint lower-face polygons: unentered, entered-free, entered-contact.

    Only one upper material face is accepted. A caller must establish that it
    is a distinct overriding sheet and must supply the physical basal depth.
    """
    lower, _ = _checked_polygon(lower_triangle, 'Lower material face', radius_km)
    upper, upper_planes = _checked_polygon(
        upper_triangle, 'Upper material face', radius_km)
    normal = np.asarray(hinge_normal, float)
    if (len(lower) != 3 or len(upper) != 3 or normal.shape != (3,)
            or not np.isfinite(normal).all()
            or not np.isclose(np.linalg.norm(normal), 1., rtol=0., atol=1e-10)
            or (finite_midpoint is None) != (finite_half_length_km is None)):
        raise ValueError('Pair zones need two faces, a unit hinge and matched endpoints.')
    if finite_midpoint is not None:
        midpoint = np.asarray(finite_midpoint, float)
        if (midpoint.shape != (3,) or not np.isfinite(midpoint).all()
                or not np.isclose(np.linalg.norm(midpoint), 1.,
                                  rtol=0., atol=1e-10)
                or not np.isfinite(finite_half_length_km)
                or finite_half_length_km <= 0.):
            raise ValueError('Finite pair zones need a unit midpoint and positive half-length.')
    else:
        midpoint = None
    reference_fraction, pieces = _entry_pieces(
        lower, normal, radius_km, midpoint, finite_half_length_km)
    face_area = mesh_coverage._polygon_area(lower, radius_km)
    tolerance = 2e-10 * face_area
    regions = [(polygon, 'entered' if active else 'unentered')
               for polygon, active in _split_entry_region(
                   lower, pieces, radius_km, tolerance)]
    final = []
    for polygon, kind in regions:
        if kind != 'entered':
            final.append((polygon, kind))
            continue
        contact, free = burial_depth._split(
            polygon, upper_planes, radius_km)
        final.extend((part, 'entered-free') for part in free)
        if mesh_coverage._polygon_area(contact, radius_km) > 0.:
            final.append((contact, 'entered-contact'))
    cleaned = [(_remove_roundoff_vertices(polygon), kind)
               for polygon, kind in final]
    final = [(polygon, kind) for polygon, kind in cleaned
             if len(polygon) >= 3
             and mesh_coverage._polygon_area(polygon, radius_km) > 0.]
    polygons = [polygon for polygon, _ in final]
    geometry = spherical_partition_intersections(
        lower, polygons, polygons, radius_km=radius_km)
    zones = [dict(zone_id=f'{kind}-{index}', kind=kind, polygon=polygon,
                  area_km2=float(geometry['old_fractions'][index] * face_area),
                  fraction=float(geometry['old_fractions'][index]))
             for index, (polygon, kind) in enumerate(final)]
    return dict(zones=zones, face_area_km2=face_area,
                entry_reference_fraction=float(reference_fraction),
                entered_physical_fraction=sum(zone['fraction'] for zone in zones
                                              if zone['kind'] != 'unentered'),
                contact_area_km2=sum(zone['area_km2'] for zone in zones
                                     if zone['kind'] == 'entered-contact'),
                scope='single ordered face pair geometry only; no pressure, thermal or native state')


def partition_entry_stack(lower_triangle, upper_triangles, upper_sheet_ids,
                          lower_sheet_id, descendants, hinge_normal, *,
                          radius_km=6371., finite_midpoint=None,
                          finite_half_length_km=None, upper_face_ids=None):
    """Partition entry against several explicitly ordered upper sheets.

    Persistent sheet ancestry must order every locally co-covering pair and
    place each upper sheet above the lower one. Multiple disjoint faces of the
    same sheet are allowed; positive-area self-overlap is not. No geometric
    depth or pressure is invented.
    """
    lower, _ = _checked_polygon(lower_triangle, 'Lower material face', radius_km)
    triangles = np.asarray(upper_triangles, float)
    if triangles.size == 0:
        triangles = np.empty((0, 3, 3), float)
    sheets = np.asarray(upper_sheet_ids)
    if not len(triangles) and sheets.size == 0:
        sheets = sheets.astype(np.int64)
    face_ids = (np.asarray(upper_face_ids) if upper_face_ids is not None
                else sheets.copy())
    if not len(triangles) and face_ids.size == 0:
        face_ids = face_ids.astype(np.int64)
    normal = np.asarray(hinge_normal, float)
    if (len(lower) != 3 or triangles.ndim != 3
            or triangles.shape[1:] != (3, 3)
            or sheets.shape != (len(triangles),)
            or not np.issubdtype(sheets.dtype, np.integer)
            or face_ids.shape != sheets.shape
            or not np.issubdtype(face_ids.dtype, np.integer)
            or len(np.unique(face_ids)) != len(face_ids)
            or (upper_face_ids is None
                and len(np.unique(sheets)) != len(sheets))
            or not isinstance(lower_sheet_id, (int, np.integer))
            or not isinstance(descendants, dict)
            or normal.shape != (3,) or not np.isfinite(normal).all()
            or not np.isclose(np.linalg.norm(normal), 1., rtol=0., atol=1e-10)
            or (finite_midpoint is None) != (finite_half_length_km is None)):
        raise ValueError('Ordered entry stack needs stable upper face IDs and a unit hinge.')
    order = np.lexsort((face_ids, sheets))
    sheets = sheets[order]
    face_ids = face_ids[order]
    triangles = triangles[order]
    for triangle in triangles:
        _checked_polygon(triangle, 'Upper material face', radius_km)
    for sheet in sheets:
        if int(lower_sheet_id) not in descendants.get(int(sheet), ()):
            raise ValueError('Every covering upper sheet needs an inherited order above the lower sheet.')
    midpoint = None
    if finite_midpoint is not None:
        midpoint = np.asarray(finite_midpoint, float)
        if (midpoint.shape != (3,) or not np.isfinite(midpoint).all()
                or not np.isclose(np.linalg.norm(midpoint), 1., rtol=0., atol=1e-10)
                or not np.isfinite(finite_half_length_km)
                or finite_half_length_km <= 0.):
            raise ValueError('Finite ordered stack needs a unit midpoint and positive half-length.')
    reference_fraction, pieces = _entry_pieces(
        lower, normal, radius_km, midpoint, finite_half_length_km)
    face_area = mesh_coverage._polygon_area(lower, radius_km)
    tolerance = 2e-10 * face_area
    all_triangles = np.concatenate((lower[None], triangles))
    selected = np.arange(len(triangles))
    stack_regions, stack_areas, _ = burial_depth.partition_face(
        all_triangles, 0, selected, selected + 1, face_area, radius_km)
    final = []
    for (polygon, covering), stack_area in zip(stack_regions, stack_areas):
        if isinstance(polygon, ExactPolygon):
            raise ValueError('Ordered entry-stack clipping does not support exact-ray '
                             'polygon precision; the stack region cannot be rounded to binary64.')
        covered_sheets = [int(sheets[pair]) for pair in covering]
        if len(set(covered_sheets)) != len(covered_sheets):
            if stack_area <= tolerance:
                continue
            raise ValueError('One upper sheet self-overlaps the same material region.')
        for a in covered_sheets:
            for b in covered_sheets:
                if a != b and (b in descendants.get(a, ())) == (
                        a in descendants.get(b, ())):
                    raise ValueError('Co-covering upper sheets lack a unique vertical order.')
        ordered = sorted(covering, key=lambda pair: -sum(
            int(sheets[other]) in descendants.get(int(sheets[pair]), ())
            for other in covering))
        ordered_sheets = tuple(int(sheets[pair]) for pair in ordered)
        ordered_faces = tuple(int(face_ids[pair]) for pair in ordered)
        adjacent = ordered[-1] if ordered else None
        for part, entered in _split_entry_region(
                polygon, pieces, radius_km, tolerance):
            cleaned = _remove_roundoff_vertices(part)
            if len(cleaned) < 3 or mesh_coverage._polygon_area(cleaned, radius_km) <= 0.:
                continue
            kind = ('unentered' if not entered else
                    'entered-contact' if adjacent is not None else 'entered-free')
            final.append(dict(kind=kind, polygon=cleaned,
                              covering_sheets_top_to_bottom=ordered_sheets,
                              covering_upper_face_ids_top_to_bottom=ordered_faces,
                              adjacent_upper_sheet_id=(None if adjacent is None
                                                       else int(sheets[adjacent])),
                              adjacent_upper_face_id=(None if adjacent is None
                                                      else int(face_ids[adjacent]))))
    geometry = spherical_partition_intersections(
        lower, [row['polygon'] for row in final],
        [row['polygon'] for row in final], radius_km=radius_km)
    zones = [dict(row, zone_id=f"{row['kind']}-{index}",
                  area_km2=float(fraction * face_area), fraction=float(fraction))
             for index, (row, fraction) in enumerate(
                 zip(final, geometry['old_fractions']))]
    return dict(zones=zones, face_area_km2=face_area,
                entry_reference_fraction=reference_fraction,
                entered_physical_fraction=sum(zone['fraction'] for zone in zones
                                              if zone['kind'] != 'unentered'),
                contact_area_km2=sum(zone['area_km2'] for zone in zones
                                     if zone['kind'] == 'entered-contact'),
                scope='ordered multi-sheet geometry only; no basal depth, pressure or native update')
