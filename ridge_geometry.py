"""Pure spherical geometry for midpoint spreading, without raster mutations.

Rotation vectors are radians, angular velocities radians/Myr, and time Myr.
Cell searches use cell-centre distances: callers add any footprint allowance
to their requested radius. No latitude-dependent longitude padding is assumed.
"""
from __future__ import annotations

import math
import numpy as np

RADIUS_KM = 6371.
_EPS = 32*np.finfo(np.float64).eps


def normal_motion_threshold(shear, *, obliquity=.15, spreading_version=0):
    """Opening/closing threshold in km/Myr for native boundary kinematics.

    Classification, finite ocean production and front caching must share this
    policy. The retained raster edition explicitly supplies its historical
    obliquity of .35. Equality stays in the transform/stationary category.
    This is a kinematic gate, not a model of fault nucleation or strain partition.
    Normal spreading version 1 instead admits any positive normal opening;
    tangential motion continues to act independently at an oblique front.
    """
    if (isinstance(spreading_version, (bool, np.bool_))
            or not isinstance(spreading_version, (int, np.integer))
            or spreading_version not in (0, 1)):
        raise ValueError('Unsupported normal spreading version.')
    # Version 1 partitions positive opening independently of tangential slip,
    # matching the reviewed model's existing any-closing-speed consumption.
    # Geometric clipping and local paired-capacity checks still govern birth.
    if spreading_version == 1:
        return np.zeros_like(np.asarray(shear), dtype=float)
    return np.maximum(2., obliquity*np.asarray(shear))


def _vectors(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] != 3 or not np.isfinite(value).all():
        raise ValueError(f'{name} must contain finite three-component vectors.')
    return value


def rotate(points, rotation):
    """Apply a Rodrigues rotation; leading point/vector dimensions broadcast.

    The sinc formulation is stable at zero and for very small rotations.
    Input vectors need not be normalized and are never modified.
    """
    points, rotation = np.broadcast_arrays(_vectors(points, 'Points'),
                                           _vectors(rotation, 'Rotation'))
    angle = np.linalg.norm(rotation, axis=-1, keepdims=True)
    sine_scale = np.sinc(angle/np.pi)
    cosine_scale = .5*np.sinc(angle/(2*np.pi))**2
    return (points*np.cos(angle) + np.cross(rotation, points)*sine_scale
            + rotation*np.sum(rotation*points, axis=-1, keepdims=True)*cosine_scale)


def _quaternion(rotation):
    angle = np.linalg.norm(rotation, axis=-1, keepdims=True)
    return np.concatenate((np.cos(angle*.5),
                           rotation*(.5*np.sinc(angle/(2*np.pi)))), axis=-1)


def finite_half_stage(omega_a, omega_b, dt):
    """Return the shortest rotational midpoint of the two finite plate stages.

    This is the normalized quaternion sum after hemisphere alignment, exactly
    halfway along the shorter rotational path. It is not the arithmetic mean
    of two rotation vectors. At a relative rotation of exactly pi there are two
    equivalent shortest paths; a nonnegative quaternion dot selects one.
    """
    if np.ndim(dt) != 0 or not np.isfinite(float(dt)):
        raise ValueError('Time must be one finite scalar.')
    a, b = np.broadcast_arrays(_vectors(omega_a, 'First angular velocity'),
                               _vectors(omega_b, 'Second angular velocity'))
    qa, qb = _quaternion(a*float(dt)), _quaternion(b*float(dt))
    qb = qb*np.where(np.sum(qa*qb, axis=-1, keepdims=True) < 0, -1., 1.)
    mid = qa+qb
    mid /= np.linalg.norm(mid, axis=-1, keepdims=True)
    # A principal rotation vector avoids a spurious nearly-2*pi representation
    # of an identity stage when the input quaternions lie in the other hemisphere.
    mid *= np.where(mid[..., :1] < 0, -1., 1.)
    length = np.linalg.norm(mid[..., 1:], axis=-1, keepdims=True)
    angle = 2*np.arctan2(length, mid[..., :1])
    scale = np.divide(angle, length, out=np.full_like(length, 2.), where=length > 1e-15)
    return mid[..., 1:]*scale


def candidate_cells(centres, radius_km, width, height):
    """Return matching segment/cell indices within each exact spherical cap.

    Centres may be one vector or an (N,3) array. Radius may be a scalar or N
    values. Returned int64 arrays have equal length, are grouped by input
    segment, and contain no duplicate cell within a segment. Longitude wraps;
    caps crossing a pole include every eligible longitude. The distance test
    includes a small floating-point tolerance at the circular boundary.

    Only latitude rows and longitude spans that can intersect each cap are
    constructed. There is no full-world temporary array for each segment.
    """
    if (isinstance(width, (bool, np.bool_)) or isinstance(height, (bool, np.bool_))
            or int(width) != width or int(height) != height or width < 2 or height < 2):
        raise ValueError('The spherical grid needs integer width/height of at least two.')
    width, height = int(width), int(height)
    centres = _vectors(centres, 'Cap centres')
    if centres.ndim == 1:
        centres = centres[None, :]
    if centres.ndim != 2:
        raise ValueError('Cap centres must have shape (N,3).')
    norm = np.linalg.norm(centres, axis=1)
    if np.any(norm <= 0):
        raise ValueError('A cap centre must be a nonzero vector.')
    centres = centres/np.maximum(norm[:, None], 1e-30)
    try:
        radii = np.broadcast_to(np.asarray(radius_km, dtype=np.float64), (len(centres),))
    except ValueError as exc:
        raise ValueError('Supply one radius or one radius per cap centre.') from exc
    if not np.isfinite(radii).all() or np.any(radii < 0):
        raise ValueError('Cap radii must be finite and nonnegative.')
    angular = np.minimum(radii/RADIUS_KM, np.pi)
    dlat, dlon = np.pi/height, 2*np.pi/width
    latitudes = np.pi/2-(np.arange(height)+.5)*dlat
    sin_lat, cos_lat = np.sin(latitudes), np.cos(latitudes)
    all_columns = np.arange(width, dtype=np.int64)
    segments, cells = [], []
    for index, (center, radius) in enumerate(zip(centres, angular)):
        cos_center = float(np.hypot(center[0], center[1]))
        latitude = math.atan2(float(center[2]), cos_center)
        longitude = math.atan2(float(center[1]), float(center[0]))
        # The epsilon is applied before rounding so exact cap/cell tangencies
        # cannot be lost to inverse trigonometric or grid-coordinate roundoff.
        low = max(0, math.ceil((np.pi/2-min(np.pi/2, latitude+radius))/dlat-.5-_EPS*height))
        high = min(height-1, math.floor((np.pi/2-max(-np.pi/2, latitude-radius))/dlat-.5+_EPS*height))
        cos_radius = math.cos(float(radius))
        for row in range(low, high+1):
            constant = sin_lat[row]*center[2]
            denominator = cos_lat[row]*cos_center
            if denominator <= 1e-15:
                if constant < cos_radius-_EPS:
                    continue
                columns = all_columns
            else:
                threshold = (cos_radius-constant-_EPS)/denominator
                if threshold > 1:
                    continue
                if threshold <= -1:
                    columns = all_columns
                else:
                    half_span = math.acos(float(np.clip(threshold, -1., 1.)))
                    start = math.ceil((longitude-half_span+np.pi)/dlon-.5-_EPS*width)
                    stop = math.floor((longitude+half_span+np.pi)/dlon-.5+_EPS*width)
                    columns = (all_columns if stop-start+1 >= width
                               else np.arange(start, stop+1, dtype=np.int64) % width)
            if not len(columns):
                continue
            delta = (columns+.5)*dlon-np.pi-longitude
            inside = constant+denominator*np.cos(delta) >= cos_radius-_EPS
            selected = columns[inside]+row*width
            if len(selected):
                cells.append(selected)
                segments.append(np.full(len(selected), index, dtype=np.int64))
    if not cells:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(segments), np.concatenate(cells)
