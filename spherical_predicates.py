"""Filtered physical halfspace predicates on represented spherical geometry.

The edge predicate decides the sign of det(a,b,p) for the original stored edge,
not the dot product with a separately rounded normal.  Ambiguous signs are
evaluated exactly on binary64 inputs.  There is no geometric inside band.
Distances are unnormalized; this common edge scale cancels in crossing ratios.
"""
from fractions import Fraction

import numpy as np


def _inputs(points, a, b=None, valid=None):
    inputs=[np.asarray(value,float) for value in (points,a) if value is not None]
    if b is not None:inputs.append(np.asarray(b,float))
    if any(value.ndim<1 or value.shape[-1]!=3 for value in inputs):
        raise ValueError('Spherical predicates need three-dimensional coordinates.')
    arrays=np.broadcast_arrays(*inputs)
    shape=arrays[0].shape[:-1]
    active=np.ones(shape,bool) if valid is None else np.broadcast_to(valid,shape).astype(bool,copy=False)
    if any(not np.isfinite(value[active]).all() for value in arrays):
        raise ValueError('Spherical predicates need finite active coordinates.')
    return arrays,active


def _dyadic(vector):
    """Binary64 coordinates as exact integers over one power-of-two denominator."""
    ratios=[float(value).as_integer_ratio() for value in vector]
    scale=max(denominator for _,denominator in ratios)
    return [numerator*(scale//denominator) for numerator,denominator in ratios],scale


def _record(stats,active,uncertain,incident):
    if stats is not None:
        for key,value in [('calls',1),('evaluations',np.count_nonzero(active)),
                          ('exact_fallbacks',np.count_nonzero(uncertain)),
                          ('incident_shortcuts',np.count_nonzero(incident&active))]:
            stats[key]=stats.get(key,0)+int(value)


def edge_distances(points,a,b,*,valid=None,stats=None):
    """Broadcast det(a,b,points), with exact signs on uncertain binary64 data.

    For a paired polygon buffer use points (N,V,3), a/b (N,1,3). ``valid``
    excludes padding. Exact endpoints and zero points are known incidences and
    do not need rational work. This follows convex_partition.Edge's filter.
    """
    (p,a,b),active=_inputs(points,a,b,valid)
    value=np.einsum('...i,...i->...',a,np.cross(b-a,p-a))
    scale=np.sum(np.abs(a)*(np.abs(b[..., [1,2,0]]*p[..., [2,0,1]])
                           +np.abs(b[..., [2,0,1]]*p[..., [1,2,0]])),axis=-1)
    incident=np.all(p==a,axis=-1)|np.all(p==b,axis=-1)|np.all(p==0.,axis=-1)
    uncertain=active&~incident&(np.abs(value)<=64*np.finfo(float).eps*scale)
    value=np.where(active&~incident,value,0.)
    if np.any(uncertain):
        # A clipping plane is shared by many vertices; compute its exact
        # coefficients once per original edge for this bounded call.
        # Binary64 values are dyadic rationals, so the determinant is an
        # integer over a power of two: the same rational a Fraction would
        # hold, and integer division rounds it to binary64 the same way.
        coefficients={}
        flat=value.reshape(-1)
        aa=a.reshape(-1,3);bb=b.reshape(-1,3);pp=p.reshape(-1,3)
        for index in np.flatnonzero(uncertain):
            key=tuple(aa[index])+tuple(bb[index])
            coefficient=coefficients.get(key)
            if coefficient is None:
                x,x_scale=_dyadic(aa[index])
                y,y_scale=_dyadic(bb[index])
                coefficient=(x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0],x_scale*y_scale)
                coefficients[key]=coefficient
            point,scale=_dyadic(pp[index])
            exact=coefficient[0]*point[0]+coefficient[1]*point[1]+coefficient[2]*point[2]
            flat[index]=exact/(coefficient[3]*scale)
            if exact and flat[index]==0.:
                raise ValueError('Spherical determinant is below the representable coordinate domain.')
    _record(stats,active,uncertain,incident)
    return value


def plane_distances(points,planes,*,valid=None,stats=None):
    """Broadcast a filtered dot product for a supplied *stored* plane.

    Exactness refers to these supplied normal coordinates. It cannot recover
    an original edge from a rounded normal; prefer edge_distances with edges.
    """
    (p,n),active=_inputs(points,planes,valid=valid)
    value=np.sum(p*n,axis=-1)
    incident=np.all(p==0.,axis=-1)
    uncertain=active&~incident&(np.abs(value)<=16*np.finfo(float).eps*np.sum(np.abs(p*n),axis=-1))
    value=np.where(active&~incident,value,0.)
    if np.any(uncertain):
        flat=value.reshape(-1);pp=p.reshape(-1,3);nn=n.reshape(-1,3)
        for index in np.flatnonzero(uncertain):
            exact=sum((Fraction(float(a))*Fraction(float(b)) for a,b in zip(pp[index],nn[index])),Fraction(0))
            flat[index]=float(exact)
            if exact and flat[index]==0.:
                raise ValueError('Spherical dot product is below the representable coordinate domain.')
    _record(stats,active,uncertain,incident)
    return value
