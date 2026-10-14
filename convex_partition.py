"""Disjoint convex spherical partitions with filtered exact side predicates.

The two halves share one classification and each computed crossing. There is
no overlapping inside/outside tolerance band. A near-zero determinant is
evaluated with exact rational arithmetic on the stored binary coordinates;
crossing coordinates themselves remain floating-point unit vectors.
"""
from fractions import Fraction
import numpy as np
import vector_norm


def _clean(points):
    points=np.asarray(points,float).reshape(-1,3)
    if len(points)>1:
        # Compare the same cyclic neighbours without copying all coordinates
        # into a rolled array. Keep the first-vs-last comparison separate so
        # closing duplicates, signed zeros and nonfinite values retain exactly
        # the original comparison semantics and surviving vertex order.
        different=np.empty(points.shape,bool)
        np.not_equal(points[0],points[-1],out=different[0])
        np.not_equal(points[1:],points[:-1],out=different[1:])
        points=points[np.any(different,axis=1)]
    return points


class Edge:
    def __init__(self,a,b,sign=1):
        self.a=np.asarray(a,float);self.b=np.asarray(b,float);self.sign=sign
        if self.a.shape!=(3,) or self.b.shape!=(3,) or not np.isfinite([self.a,self.b]).all() or sign not in (-1,1):
            raise ValueError('Spherical partition edge needs finite three-dimensional endpoints and a winding sign.')
        self.exact=None

    def distances(self,points):
        p=np.asarray(points,float)
        if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():
            raise ValueError('Spherical partition needs finite three-dimensional points.')
        # Translated determinant retains precision on short edges. The filter
        # deliberately uses the larger unshifted product scale, so ambiguous
        # nearly coplanar cases go to exact arithmetic, including edge corners.
        result=np.einsum('i,ji->j',self.a,np.cross(self.b-self.a,p-self.a))
        a,b=self.a,self.b
        scale=(abs(a[0])*(abs(b[1])*np.abs(p[:,2])+abs(b[2])*np.abs(p[:,1]))+
               abs(a[1])*(abs(b[2])*np.abs(p[:,0])+abs(b[0])*np.abs(p[:,2]))+
               abs(a[2])*(abs(b[0])*np.abs(p[:,1])+abs(b[1])*np.abs(p[:,0])))
        uncertain=np.abs(result)<=64.*np.finfo(float).eps*scale
        if np.any(uncertain):
            if self.exact is None:
                aa=[Fraction(float(x)) for x in a];bb=[Fraction(float(x)) for x in b]
                self.exact=(aa[1]*bb[2]-aa[2]*bb[1],aa[2]*bb[0]-aa[0]*bb[2],aa[0]*bb[1]-aa[1]*bb[0])
            for i in np.flatnonzero(uncertain):
                value=sum((coefficient*Fraction(float(x)) for coefficient,x in zip(self.exact,p[i])),Fraction(0))
                result[i]=float(value)
                if value and result[i]==0.:
                    raise ValueError('Spherical partition determinant is below the representable coordinate domain.')
        return result*self.sign


def prepare(polygon):
    polygon=_clean(polygon)
    if not np.isfinite(polygon).all():raise ValueError('Spherical partition needs finite coordinates.')
    if len(polygon)<3:return ()
    edge=Edge(polygon[0],polygon[1]);values=edge.distances(polygon[2:])
    nonzero=values[values!=0.]
    if not len(nonzero):return ()
    # A nominally collinear leading fan triangle can have a tiny stored-data
    # determinant of either sign. Use the resolved interior, not that edge
    # roundoff, to choose the winding of the convex footprint.
    sign=1 if values[np.argmax(np.abs(values))]>0. else -1
    return tuple(Edge(a,b,sign) for a,b in zip(polygon,np.roll(polygon,-1,axis=0)))


def split(polygon,edge):
    """Return the closed positive and negative halves; shared edges have no area."""
    polygon=np.asarray(polygon,float)
    if not len(polygon):return polygon,polygon
    d=edge.distances(polygon)
    if np.all(d>=0.):return polygon,np.empty((0,3))
    if np.all(d<=0.):return np.empty((0,3)),polygon
    positive=[];negative=[]
    for i,point in enumerate(polygon):
        previous=(i-1)%len(polygon)
        if (d[i]>0. and d[previous]<0.) or (d[i]<0. and d[previous]>0.):
            fraction=abs(d[previous])/(abs(d[previous])+abs(d[i]))
            crossing=(1.-fraction)*polygon[previous]+fraction*point
            crossing/=vector_norm.length(crossing)
            positive.append(crossing);negative.append(crossing)
        if d[i]>=0.:positive.append(point)
        if d[i]<=0.:negative.append(point)
    return _clean(positive),_clean(negative)


def partition(polygon,edges):
    """One convex intersection and disjoint first-failed-halfspace complements."""
    if not edges:return np.empty((0,3)),[polygon]
    inside=np.asarray(polygon,float);outside=[]
    for edge in edges:
        inside,piece=split(inside,edge)
        if len(piece)>=3:outside.append(piece)
        if len(inside)<3:break
    return inside,outside
