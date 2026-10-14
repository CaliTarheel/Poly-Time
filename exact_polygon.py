"""Exact source-ray polygons for rare ill-conditioned spherical partitions.

The integer homogeneous vertices are the geometry. Binary64 coordinates are
available only as an explicit diagnostic projection; they can change the
area or winding of a thin region and must not become its physical geometry.
"""
from dataclasses import dataclass
from decimal import Decimal, localcontext
from math import atan2, fsum, isfinite

import numpy as np


def determinant(a, b, c):
    """Exact oriented determinant of three homogeneous spatial rays."""
    return (a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])
            +a[2]*(b[0]*c[1]-b[1]*c[0]))


@dataclass(frozen=True, slots=True)
class ExactPolygon:
    """Immutable, pickle-safe polygon with rational vertices (X/W,Y/W,Z/W)."""
    homogeneous: tuple

    def __post_init__(self):
        points = tuple(tuple(point) for point in self.homogeneous)
        if len(points) < 3 or any(len(point) != 4 for point in points):
            raise ValueError('Exact polygon requires at least three homogeneous vertices.')
        if any(type(value) is not int for point in points for value in point):
            raise TypeError('Exact polygon homogeneous coordinates must be integers.')
        if any(point[3] <= 0 or not any(point[:3]) for point in points):
            raise ValueError('Exact polygon requires nonzero rays with positive denominators.')
        object.__setattr__(self, 'homogeneous', points)

    def __len__(self):
        return len(self.homogeneous)

    def __array__(self, dtype=None, copy=None):
        raise TypeError('Exact polygon cannot be cast implicitly; represented() is diagnostic only.')

    def represented(self):
        """Return a detached binary64 projection, never a replacement geometry."""
        with localcontext() as context:
            context.prec = 80
            result = []
            for point in self.homogeneous:
                values = [Decimal(value) for value in point[:3]]
                length = sum(value*value for value in values).sqrt()
                result.append([float(value/length) for value in values])
        return np.asarray(result)

    def solid_angle(self):
        """Independently integrate the exact ray polygon without a float cast."""
        source = self.homogeneous
        signs = [determinant(source[0], b, c) for b, c in zip(source[1:-1], source[2:])]
        if any(value < 0 for value in signs) or not any(value > 0 for value in signs):
            raise ValueError('Exact polygon requires positive convex winding.')
        with localcontext() as context:
            context.prec = 80
            points, lengths = [], []
            for point in source:
                values = [Decimal(value) for value in point[:3]]
                length = sum(value*value for value in values).sqrt()
                points.append([value/length for value in values])
                lengths.append(length)
            a, angles = points[0], []
            for index, (b, c, sign) in enumerate(zip(points[1:-1], points[2:], signs), 1):
                if sign == 0:
                    continue
                numerator = Decimal(sign)/(lengths[0]*lengths[index]*lengths[index+1])
                denominator = 1+sum(a[i]*b[i]+b[i]*c[i]+c[i]*a[i] for i in range(3))
                if numerator <= 0 or denominator <= 0:
                    raise ValueError('Exact polygon integral requires resolved minor convex winding.')
                angles.append(2*atan2(float(numerator), float(denominator)))
        result = fsum(angles)
        if not isfinite(result) or result <= 0:
            raise ValueError('Exact polygon integral requires positive finite area.')
        return result
