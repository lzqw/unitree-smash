"""Convex authored robot geometry against the actual suspended net and posts.

Uses installed hpp-fcl, not an infinite keep-out plane. Authored convex hulls
are not claimed identical to PhysX collision cooking/contact offsets. Far
pairs use conservative sphere/AABB lower bounds; possible intersections are
checked by convex distance. Original native collision remains authoritative.
"""
import numpy as np
import pinocchio as pin
import hppfcl as fcl
from scipy.spatial import ConvexHull
from quadsmash.physics.court_v4 import net_band_segments


def convex(points):
    points=np.unique(np.asarray(points,float),axis=0)
    hull=ConvexHull(points);vertices=fcl.StdVec_Vec3f();triangles=fcl.StdVec_Triangle()
    for p in points:vertices.append(p)
    for face, equation in zip(hull.simplices,hull.equations):
        face=face.copy();a,b,c=points[face]
        if np.dot(np.cross(b-a,c-a),equation[:3])<0:face[[1,2]]=face[[2,1]]
        triangles.append(fcl.Triangle(*(int(x) for x in face)))
    return fcl.Convex(vertices,triangles)


def authored_shape(shape):
    T=np.asarray(shape['transform']);linear=T[:3,:3].T;translation=T[3,:3]
    scales=np.linalg.norm(linear,axis=0);rotation=linear/scales
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6):raise ValueError('Unsupported authored shear')
    kind=shape['kind']
    if kind=='Mesh':
        points=np.asarray(shape['points'])@linear.T+translation
        low,high=points.min(0),points.max(0);center=(low+high)/2
        return convex(points),np.eye(3),np.zeros(3),center,float(np.linalg.norm(points-center,axis=1).max())
    if kind=='Cube':
        dimensions=scales*shape['size'];geometry=fcl.Box(*dimensions);radius=np.linalg.norm(dimensions)/2
    elif kind=='Sphere':
        if not np.allclose(scales,scales[0],atol=1e-7):raise ValueError('Nonuniform sphere')
        radius=shape['radius']*scales[0];geometry=fcl.Sphere(radius)
    elif kind in ['Cylinder','Capsule']:
        axis=shape['axis'];radial=[i for i in range(3) if i!=axis]
        if not np.isclose(scales[radial[0]],scales[radial[1]],atol=1e-7):raise ValueError('Noncircular authored primitive')
        r=shape['radius']*scales[radial[0]];length=shape['height']*scales[axis]
        if kind=='Capsule' and not np.allclose(scales,scales[0],atol=1e-7):raise ValueError('Nonuniform capsule')
        geometry=(fcl.Cylinder(r,length) if kind=='Cylinder' else fcl.Capsule(r,length))
        radius=np.hypot(r,length/2) if kind=='Cylinder' else r+length/2
        z=np.eye(3)[:,axis]
        align=pin.Quaternion.FromTwoVectors(np.array([0.,0.,1.]),z).matrix()
        rotation=rotation@align
    else:raise ValueError('Unknown authored primitive '+kind)
    return geometry,rotation,translation,translation.copy(),float(radius)


class AuthoredNetClearanceV1:
    def __init__(self,ground_geometry,court):
        self.ground=ground_geometry;self.model=ground_geometry.model;self.shapes=ground_geometry.shapes
        self.bodies=list(dict.fromkeys(s['body'] for s in self.shapes))
        self.body_index={b:i for i,b in enumerate(self.bodies)}
        self.robot=[authored_shape(s) for s in self.shapes]
        self.obstacles=[];self.low=[];self.high=[]
        x=court['net_plane_x_m']
        for i,(points,_) in enumerate(net_band_segments(court)):
            points=np.asarray(points)+np.array([x,0.,0.]);lo,hi=points.min(0),points.max(0)
            self.obstacles.append((fcl.Box(*(hi-lo)),fcl.Transform3f(np.eye(3),(lo+hi)/2),f'net_segment_{i}'))
            self.low.append(lo);self.high.append(hi)
        for sign in [-1.,1.]:
            r=court['post_radius_m'];h=court['post_height_m'];center=np.array([x,sign*court['post_y_center_abs_m'],h/2])
            self.obstacles.append((fcl.Cylinder(r,h),fcl.Transform3f(np.eye(3),center),f'post_{sign:g}'))
            self.low.append(center-[r,r,h/2]);self.high.append(center+[r,r,h/2])
        self.low=np.asarray(self.low);self.high=np.asarray(self.high)
        self.request=fcl.DistanceRequest();self.request.enable_nearest_points=True;self.request.gjk_tolerance=1e-8
        self.last_violation=None

    def evaluate(self,q,*,jacobian=False,placements_current=False,near_distance_m=.08):
        m=self.model
        if not placements_current:
            if jacobian:pin.computeJointJacobians(m.model,m.data,q)
            else:pin.forwardKinematics(m.model,m.data,q)
            pin.updateFramePlacements(m.model,m.data)
        elif jacobian:raise ValueError('Explicit Jacobian refresh required')
        result={b:dict(distance_m=float('inf'),exact=False,shape_index=None,obstacle=None) for b in self.bodies}
        for i,(shape,robot) in enumerate(zip(self.shapes,self.robot)):
            geometry,local_rotation,translation,center,radius=robot
            frame=m.body_frames[shape['body']];pose=m.data.oMf[frame]
            world_center=pose.act(center)
            lower=np.linalg.norm(world_center-np.clip(world_center,self.low,self.high),axis=1)-radius
            record=result[shape['body']]
            candidates=np.flatnonzero(lower<=near_distance_m)
            if len(candidates)==0:
                d=float(lower.min())
                if d<record['distance_m']:record.update(distance_m=d,exact=False,shape_index=i,obstacle='broad_phase_lower_bound')
                continue
            transform=fcl.Transform3f(pose.rotation@local_rotation,pose.act(translation))
            for j in candidates:
                obstacle,obstacle_transform,name=self.obstacles[j];answer=fcl.DistanceResult()
                d=float(fcl.distance(geometry,transform,obstacle,obstacle_transform,self.request,answer))
                if not np.isfinite(d):raise ValueError('Nonfinite authored convex distance')
                if d<record['distance_m']:
                    record.update(distance_m=d,exact=True,shape_index=i,obstacle=name)
                    if jacobian and d>1e-7:
                        p1=answer.getNearestPoint1();p2=answer.getNearestPoint2();normal=(p1-p2)/np.linalg.norm(p1-p2)
                        J=pin.getFrameJacobian(m.model,m.data,frame,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                        record['jacobian']=normal@(J[:3]-pin.skew(p1-pose.translation)@J[3:])
            # For collision detection no culled pair can intersect. Its bound
            # need not equal the nearest exact distance and is not a gradient.
        return result

    def check(self,q,*,placements_current=False):
        records=self.evaluate(q,placements_current=placements_current,near_distance_m=.0)
        for body,row in records.items():
            if row['exact'] and row['distance_m']< -1e-6:
                self.last_violation=dict(body=body,**row)
                raise ValueError('authored_net_collision:'+str(self.last_violation))
        return records


class CourtCollisionPathConstraintV1:
    def __init__(self,base,net):
        self.base=base;self.net=net;self.model=base.model;self.shapes=base.shapes;self.last_violation=None
    def evaluate(self,q,*,jacobian=False):
        try:
            result=self.base.evaluate(q,jacobian=jacobian)
            self.net.check(q,placements_current=True)
            return result
        except ValueError:
            self.last_violation=self.net.last_violation or getattr(self.base,'last_violation',None)
            raise
