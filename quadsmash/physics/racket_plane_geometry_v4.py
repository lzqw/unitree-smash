"""Convex racket/volume intersections using current engine poses, no forces.

CPU scene queries cannot supply live articulation poses under DirectGPU API.
These tests transform the declared USD primitive geometry with the measured
GPU body pose instead. Boxes/capsules are analytic; cylinders use a tiny convex
quadratic program only when their finite end caps matter.
"""
from itertools import combinations,product
import numpy as np
from .full_racket_v4 import parts,string_strips
from .court_v4 import segment_top


def segment_box_distance_squared(start,end,lower,upper):
    """Exact convex piecewise-quadratic minimum along a segment."""
    delta=end-start;cuts=[0.,1.]
    for j in range(3):
        if abs(delta[j])>1e-15:
            for bound in [lower[j],upper[j]]:
                t=(bound-start[j])/delta[j]
                if 0<t<1:cuts.append(float(t))
    cuts=np.unique(cuts);candidates=list(cuts)
    for left,right in zip(cuts[:-1],cuts[1:]):
        middle=(left+right)/2;p=start+middle*delta
        active=(p<lower)|(p>upper)
        target=np.where(p<lower,lower,upper)
        denom=float(np.dot(delta[active],delta[active]))
        if denom>1e-30:
            t=-float(np.dot(delta[active],(start-target)[active]))/denom
            candidates.append(float(np.clip(t,left,right)))
    t=np.asarray(candidates);p=start[None]+t[:,None]*delta
    nearest=np.minimum(np.maximum(p,lower),upper)
    dist=np.sum((p-nearest)**2,axis=-1);i=int(dist.argmin())
    return float(dist[i]),float(t[i]),nearest[i]


def oriented_box_overlap(center,rotation,half,lower,upper):
    """Separating-axis theorem for the actual strip OBB and query AABB."""
    query_center=(lower+upper)/2;query_half=(upper-lower)/2;offset=center-query_center
    axes=[*np.eye(3),*rotation.T]
    axes += [np.cross(a,b) for a in np.eye(3) for b in rotation.T]
    axes=np.asarray(axes);axes=axes[np.linalg.norm(axes,axis=1)>1e-12]
    radius=np.abs(axes@rotation)@half+np.abs(axes)@query_half
    return bool(np.all(np.abs(axes@offset)<=radius+1e-10))


def cylinder_box_overlap(center,axis,half_length,radius,lower,upper):
    start=center-half_length*axis;end=center+half_length*axis
    distance,t,point=segment_box_distance_squared(start,end,lower,upper)
    if distance>(radius+1e-10)**2:return False
    axial=float(np.dot(point-center,axis))
    if abs(axial)<=half_length+1e-10:return True
    # Remove points that belong only to the hemispherical extension of a
    # capsule. Keep the real finite circular cylinder's two flat end planes.
    low=lower-center;high=upper-center
    middle=(low+high)/2;extent=(high-low)/2
    projected=float(np.dot(axis,middle));spread=float(np.dot(np.abs(axis),extent))
    if projected-spread>half_length or projected+spread<-half_length:return False
    matrix=np.r_[np.eye(3),-np.eye(3),axis[None],-axis[None]]
    bounds=np.r_[high,-low,half_length,half_length]
    # This regularization has a rigorous maximum objective perturbation of
    # epsilon*half_length^2; it never changes the physical cylinder dimensions.
    epsilon=1e-8
    radial=np.eye(3)-np.outer(axis,axis)
    hessian=2*(radial+epsilon*np.outer(axis,axis))
    for count in range(4):
        for active in combinations(range(8),count):
            a=matrix[list(active)] if count else np.empty((0,3))
            b=bounds[list(active)] if count else np.empty(0)
            if count and np.linalg.matrix_rank(a,tol=1e-10)!=count:continue
            kkt=np.block([[hessian,a.T],[a,np.zeros((count,count))]])
            try:solution=np.linalg.solve(kkt,np.r_[np.zeros(3),b])
            except np.linalg.LinAlgError:continue
            u=solution[:3];multipliers=solution[3:]
            if np.max(matrix@u-bounds)>1e-8 or (count and multipliers.min()<-1e-8):continue
            value=float(u@radial@u)
            objective=value+epsilon*float(np.dot(u,axis))**2
            if value<=(radius+1e-8)**2:return True
            if objective-epsilon*half_length**2>radius**2:return False
            return value<=(radius+1e-8)**2
    raise RuntimeError('Finite-cylinder/volume intersection unresolved; refuse a silent legality verdict')


def quaternion_matrix(wxyz):
    w,x,y,z=np.asarray(wxyz,dtype=float)/np.linalg.norm(wxyz)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
        [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
        [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


class RacketPlaneGeometry:
    def __init__(self,config,court):
        self.court=court;self.primitives=[]
        for part in parts(config):
            if part['name']=='Clamp' or not part['collision']:continue
            if part['kind']=='string_prism':
                for i,strip in enumerate(string_strips(config)):
                    self.primitives.append(dict(name=f'Collider/strip_{i:02d}',kind='box',
                        center=np.array(strip['center']),half=np.array(strip['size'])/2))
            else:self.primitives.append(dict(part))

    def first_above_net_overlap(self,position,quaternion):
        """Return one actual primitive/volume witness, or None."""
        rotation=quaternion_matrix(quaternion);position=np.asarray(position,dtype=float)
        court=self.court;n=court['net_profile_segments'];half=court['net_width_m']/2
        plane=court['net_plane_x_m']+1e-6
        for primitive in self.primitives:
            center=position+rotation@primitive['center'];kind=primitive['kind']
            if kind=='box':extent=np.abs(rotation)@primitive['half']
            else:
                axis=rotation@primitive['axis'];length=primitive['length']/2;radius=primitive['radius']
                extent=np.abs(axis)*length+radius*(1 if kind=='capsule' else np.sqrt(np.maximum(0,1-axis*axis)))
            low,high=center-extent,center+extent
            if high[0]<=plane or high[2]<court['net_center_top_m']:continue
            first=max(0,min(n-1,int(np.floor((low[1]+half)*n/(2*half)))))
            last=max(0,min(n-1,int(np.floor((high[1]+half)*n/(2*half)))))
            for strip in range(first,last+1):
                lower=np.array([plane,max(low[1],-half+2*half*strip/n),segment_top(strip,court)])
                upper=np.array([high[0]+1e-9,min(high[1],-half+2*half*(strip+1)/n),high[2]+1e-9])
                if np.any(upper<=lower):continue
                if kind=='box':hit=oriented_box_overlap(center,rotation,primitive['half'],lower,upper)
                elif kind=='capsule':
                    distance,_,_=segment_box_distance_squared(center-axis*length,center+axis*length,lower,upper)
                    hit=distance<=(radius+1e-10)**2
                elif kind=='cylinder':hit=cylinder_box_overlap(center,axis,length,radius,lower,upper)
                else:raise ValueError('Unsupported racket primitive: '+kind)
                if hit:return dict(primitive=primitive['name'],net_strip=strip,
                    query_box_local_min_m=lower.tolist(),query_box_local_max_m=upper.tolist(),
                    plane_numerical_tolerance_m=1e-6,
                    provider='Declared collision primitives transformed by current GPU engine body pose; no forces')
        return None
