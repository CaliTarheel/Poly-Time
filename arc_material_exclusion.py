"""Exact new-footprint/material exclusion for juvenile emplacement.

This is a Boolean set guard, not an area budget or a contact polarity rule.
Original represented rays define a Fraction barycentric chart. Positive new
intersection cannot be spent from an area tolerance. Old footprint overlap is
retained by exact set subtraction. Conservative cap certificates and exact
bin floors include small faces and boundary incidences without a locator floor.
"""
from fractions import Fraction as F
import hashlib
import math
import numpy as np

_RESOLUTION=64

def _dot(a,b):return sum((x*y for x,y in zip(a,b)),F(0))
def _cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def _area(poly):return sum((a[0]*b[1]-a[1]*b[0] for a,b in zip(poly,poly[1:]+poly[:1])),F(0))/2

def _rays(triangle):
    result=tuple(tuple(F(float(x)) for x in p) for p in triangle)
    if _dot(result[0],_cross(result[1],result[2]))<=0:
        raise ValueError('Exact material exclusion requires positive original winding.')
    return result

def _split(poly,plane):
    values=[plane[0]+x*(plane[1]-plane[0])+y*(plane[2]-plane[0]) for x,y in poly]
    if all(v>=0 for v in values):return poly,[]
    if all(v<=0 for v in values):return [],poly
    positive=[];negative=[]
    for i,p in enumerate(poly):
        j=(i-1)%len(poly)
        if values[i]*values[j]<0:
            t=values[j]/(values[j]-values[i])
            q=tuple((1-t)*a+t*b for a,b in zip(poly[j],p));positive.append(q);negative.append(q)
        if values[i]>=0:positive.append(p)
        if values[i]<=0:negative.append(p)
    return positive,negative

def _partition(poly,planes):
    inside=poly;outside=[]
    for plane in planes:
        inside,part=_split(inside,plane)
        if len(part)>=3 and _area(part)>0:outside.append(part)
        if len(inside)<3 or _area(inside)<=0:return [],outside
    return inside,outside

def _planes(blocker,basis):
    return [tuple(_dot(_cross(a,b),p) for p in basis) for a,b in zip(blocker,blocker[1:]+blocker[:1])]


def _excludes_reference(planes):
    """Exact empty-area certificate on the uncut barycentric triangle.

    A plane's three stored values are its values at (0,0), (1,0), and
    (0,1). Affinity makes the interior values convex combinations of them.
    If all are nonpositive and at least one is negative, the positive
    half-plane reaches at most an edge or vertex: it cannot enclose area.
    Check every plane before constructing potentially large rational cut
    vertices. An identically zero plane imposes no constraint.
    """
    return any(all(value <= 0 for value in plane) and any(value < 0 for value in plane)
               for plane in planes)

def _box(triangle, *, rays=None):
    # For any unit normalized positive ray combination q, c.q>=d/L>0.
    # Thus |q-c|²<=1+|c|²-2d/L. The squared certificate below is exact.
    center=np.sum(triangle,axis=0);length=np.linalg.norm(center)
    if not np.isfinite(length) or length==0: return np.zeros(3,int),np.full(3,_RESOLUTION-1,int)
    center=center/length
    radius=float(np.max(np.linalg.norm(triangle/np.linalg.norm(triangle,axis=1)[:,None]-center,axis=1)))+1e-12
    rays=_rays(triangle) if rays is None else rays;c=tuple(F(float(x)) for x in center)
    d=min(_dot(c,v) for v in rays);l2=max(_dot(v,v) for v in rays);c2=_dot(c,c)
    if d<=0:return np.zeros(3,int),np.full(3,_RESOLUTION-1,int)
    for _ in range(64):
        r=F(radius);k=(1+c2-r*r)/2
        if k<=0 or d*d>=k*k*l2:break
        radius=float(np.nextafter(2*radius+1e-12,np.inf))
    else:raise ValueError('Could not certify a material exclusion candidate cap.')
    bounds=[]
    for sign in (-1,1):
        row=[]
        for x in c:
            value=(x+sign*r+1)*_RESOLUTION/2
            row.append(max(0,min(_RESOLUTION-1,value.numerator//value.denominator)))
        bounds.append(np.asarray(row,dtype=np.int32))
    return bounds

def _keys(lo,hi):
    x,y,z=(np.arange(a,b+1,dtype=np.int32) for a,b in zip(lo,hi))
    return (x[:,None,None]+_RESOLUTION*(y[None,:,None]+_RESOLUTION*z[None,None,:])).ravel()

class _Index:
    def __init__(self,triangles):
        triangles=np.asarray(triangles,float)
        if (triangles.ndim!=3 or triangles.shape[1:]!=(3,3) or not np.isfinite(triangles).all()
                or np.any(abs(np.linalg.norm(triangles,axis=2)-1)>2e-12)):
            raise ValueError('Material exclusion needs finite unit original triangles.')
        self.triangles=triangles.copy();self.triangles.setflags(write=False)
        self.signature=hashlib.sha256(triangles.tobytes()).hexdigest();self.rational={}
        self.normals={};self.candidate_boxes={}
        keys=[];faces=[];glob=[]
        for i,triangle in enumerate(triangles):
            lo,hi=_box(triangle)
            if math.prod(map(int,hi-lo+1))>4096:glob.append(i);continue
            part=_keys(lo,hi);keys.append(part);faces.append(np.full(len(part),i,np.int32))
        if keys:
            keys=np.concatenate(keys);faces=np.concatenate(faces);order=np.argsort(keys,kind='stable')
            keys=keys[order];self.faces=faces[order];self.keys,offsets=np.unique(keys,return_index=True)
            self.offsets=np.r_[offsets,len(keys)]
        else:self.keys=np.empty(0,np.int32);self.faces=np.empty(0,np.int32);self.offsets=np.array([0])
        self.global_faces=np.asarray(glob,np.int32)
    def candidates(self,triangle, *, bounds=None):
        lo,hi=_box(triangle) if bounds is None else bounds
        if math.prod(map(int,hi-lo+1))>4096:return np.arange(len(self.triangles))
        key=tuple(map(int,np.r_[lo,hi]))
        if key in self.candidate_boxes:return self.candidate_boxes[key]
        keys=_keys(lo,hi);at=np.searchsorted(self.keys,keys);valid=at<len(self.keys)
        at=at[valid];at=at[self.keys[at]==keys[valid]]
        parts=[self.faces[self.offsets[k]:self.offsets[k+1]] for k in at]
        if len(self.global_faces):parts.append(self.global_faces)
        result=np.unique(np.concatenate(parts)) if parts else np.empty(0,int)
        result.setflags(write=False)
        if len(self.candidate_boxes)>=4096:self.candidate_boxes.pop(next(iter(self.candidate_boxes)))
        self.candidate_boxes[key]=result
        return result
    def exact(self,i):
        if i not in self.rational:self.rational[i]=_rays(self.triangles[i])
        return self.rational[i]
    def _plane_normals(self,i):
        if i not in self.normals:
            blocker=self.exact(i)
            self.normals[i]=tuple(_cross(a,b) for a,b in zip(blocker,blocker[1:]+blocker[:1]))
        return self.normals[i]
    def planes(self,i,basis):
        return [tuple(_dot(normal,p) for p in basis) for normal in self._plane_normals(i)]
    def reference_planes(self,i,basis):
        """Preserve plane order, stopping only on an exact empty-area proof."""
        result=[]
        for normal in self._plane_normals(i):
            plane=tuple(_dot(normal,p) for p in basis)
            if _excludes_reference((plane,)):return None
            result.append(plane)
        return result

def _index(context):
    cache=context.setdefault('_exact_material_cache',{})
    if 'index' not in cache:cache['index']=_Index(context['triangles'])
    index=cache['index']
    if hashlib.sha256(np.asarray(context['triangles'],float).tobytes()).hexdigest()!=index.signature:
        raise ValueError('Material exclusion context geometry changed without a new index.')
    return index

def obstruction(triangles,contexts,old=None):
    """Return first exact-positive (candidate minus old) intersection witness.

    A false result establishes zero positive-area obstruction against every
    supplied active context. It does not certify foreign-owner water, source
    volume, or contact dynamics; those remain separate admission requirements.
    """
    indices=[_index(c) for c in contexts];old_index=None if old is None else _index(old)
    selections=[]
    for context,index in zip(contexts,indices):
        count=len(index.triangles)
        active=None if 'active' not in context else np.asarray(context['active'])
        if active is not None and (active.dtype.kind!='b' or active.shape!=(count,)):
            raise ValueError('Material exclusion active mask must be Boolean with one entry per face.')
        excluded=None if 'exclude' not in context else np.asarray(context['exclude'])
        if excluded is not None and (excluded.dtype.kind not in 'iu' or excluded.ndim!=1
                or np.any(excluded<0) or np.any(excluded>=count)):
            raise ValueError('Material exclusion excluded indices must be a one-dimensional integer array in range.')
        selections.append((active,excluded))
    reference=[(F(0),F(0)),(F(1),F(0)),(F(0),F(1))]
    for candidate,triangle in enumerate(np.asarray(triangles,float)):
        basis=_rays(triangle)
        bounds=_box(triangle,rays=basis) if indices or old_index is not None else None
        old_planes=[] if old_index is None else [old_index.planes(int(i),basis)
                                                 for i in old_index.candidates(triangle,bounds=bounds)]
        for context_number,(context,index) in enumerate(zip(contexts,indices)):
            selected=index.candidates(triangle,bounds=bounds)
            active,excluded=selections[context_number]
            if active is not None:selected=selected[active[selected]]
            if excluded is not None:selected=selected[~np.isin(selected,excluded)]
            for face in selected:
                planes=index.reference_planes(int(face),basis)
                if planes is None:continue
                inside,_=_partition(reference,planes)
                if not inside:continue
                parts=[inside]
                for planes in old_planes:
                    parts=[q for p in parts for q in _partition(p,planes)[1]]
                    if not parts:break
                if parts:
                    polygon=parts[0]
                    return dict(candidate_face=int(candidate),material_context=context_number,material_face=int(face),
                        exact_chart_area=str(_area(polygon)),chart_polygon=[[str(x),str(y)] for x,y in polygon])
    return None
