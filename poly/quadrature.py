"""Exact basal-drag area moments for spherical polygons with great-circle edges."""
import numpy as np

from poly import sphere


def basal_moment_m2(region):
    """Return integral (I - r r.T) dA in m² for unit surface position r.

    On the unit sphere, the surface Laplacian of r_i*r_j is
    2*delta_ij - 6*r_i*r_j. The surface divergence theorem therefore gives

        integral (I - r r.T) dOmega = (2*area/3)*I
            + (1/6) integral_boundary (n*r.T + r*n.T) ds.

    Here n is the outward conormal. It is constant on a great-circle edge:
    minus its oriented normal. The edge integral of r is
    tan(angle/2)*(a+b). All rings contribute with their S2 orientation,
    including clockwise holes; disconnected polygons add. The spherical
    area term handles regions larger than a hemisphere and the full sphere.
    This integrates the existing linear drag law without choosing new
    rheology or requiring global samples inside a microplate.
    """
    area = sphere.area_km2(region) / sphere.RADIUS_KM ** 2
    moment = np.eye(3) * (2. * area / 3.)
    for polygon in sphere.polygons(region):
        for ring in polygon:
            other = np.roll(ring, -1, axis=0)
            normal = np.cross(ring, other)
            sine = np.linalg.norm(normal, axis=1)
            cosine = np.einsum('ij,ij->i', ring, other)
            valid = sine > 1e-15
            if np.any(valid & (1. + cosine < 1e-12)):
                raise ValueError('basal quadrature requires edges shorter than a semicircle')
            outward = -normal[valid] / sine[valid, None]
            integral = (ring[valid] + other[valid]) * (sine[valid] / (1. + cosine[valid]))[:, None]
            moment += (outward.T @ integral + integral.T @ outward) / 6.
    return moment * (sphere.RADIUS_KM * 1000.) ** 2
