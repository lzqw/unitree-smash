"""Authored nonfoot geometry against a horizontal plane, not PhysX collision.

Analytic support witnesses provide configuration gradients. Mesh witnesses
are exact for the authored vertex hull; ties are nonsmooth. No self-collision
or collision-cooking equivalence is claimed.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pinocchio as pin


def load_usd_shapes(asset):
    from pxr import Usd, UsdGeom, UsdPhysics
    stage=Usd.Stage.Open(str(asset));cache=UsdGeom.XformCache();shapes=[]
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        owner=prim
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            if prim.GetTypeName()!='Mesh':continue
            owner=prim.GetParent()
            while owner and not owner.HasAPI(UsdPhysics.CollisionAPI):
                if owner.HasAPI(UsdPhysics.RigidBodyAPI):break
                owner=owner.GetParent()
        if not owner or not owner.HasAPI(UsdPhysics.CollisionAPI) or owner.GetAttribute('physics:collisionEnabled').Get() is False:continue
        kind=prim.GetTypeName()
        if kind=='Xform':continue
        if kind not in ['Sphere','Cylinder','Capsule','Cube','Mesh']:raise ValueError('Unknown collider '+kind)
        body=prim
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
        if not body:raise ValueError('Missing collider body')
        if body.GetName().endswith('_foot'):continue
        transform=np.array(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse())
        shape=dict(body=body.GetName(),path=str(prim.GetPath()),kind=kind,transform=transform.tolist())
        if kind=='Mesh':shape['points']=np.asarray(prim.GetAttribute('points').Get(),dtype=float).tolist()
        if kind=='Cube':shape['size']=float(prim.GetAttribute('size').Get())
        if kind in ['Sphere','Cylinder','Capsule']:shape['radius']=float(prim.GetAttribute('radius').Get())
        if kind in ['Cylinder','Capsule']:
            shape.update(height=float(prim.GetAttribute('height').Get()),axis='XYZ'.index(prim.GetAttribute('axis').Get()))
        shapes.append(shape)
    return shapes


def support_witness(shape, rotation):
    """Lowest surface point in body coordinates for this body rotation."""
    T=shape['transform'];direction=(rotation@T[:3,:3].T)[2];kind=shape['kind']
    if kind=='Mesh':local=shape['points'][np.argmin(shape['points']@direction)]
    elif kind=='Cube':local=-.5*shape['size']*np.sign(direction)
    else:
        radial=direction.copy()
        if kind=='Cylinder':radial[shape['axis']]=0.
        norm=np.linalg.norm(radial)
        local=-shape['radius']*radial/norm if norm>1e-14 else np.zeros(3)
        if kind in ['Cylinder','Capsule']:
            local[shape['axis']]-=.5*shape['height']*np.sign(direction[shape['axis']])
    return T[:3,:3].T@local+T[3,:3]


class AuthoredGroundClearance:
    def __init__(self, path, model, *, margin_m=.005, ground_z_m=0.):
        self.path=Path(path);self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        record=json.loads(self.path.read_text())
        for name,digest in record['asset_sha256'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Geometry asset changed')
        if not np.isfinite([margin_m,ground_z_m]).all() or margin_m<0:raise ValueError('Invalid plane/margin')
        self.margin_m=margin_m;self.ground_z_m=ground_z_m;self.model=model;self.shapes=record['shapes']
        if not self.shapes:raise ValueError('Missing collision shapes')
        for s in self.shapes:
            if s['body'] not in model.body_frames:raise ValueError('Geometry/model body mismatch')
            s['transform']=np.asarray(s['transform'])
            if s['kind']=='Mesh':s['points']=np.asarray(s['points'])

    def evaluate(self,q,*,jacobian=False):
        m=self.model
        if jacobian:pin.computeJointJacobians(m.model,m.data,q)
        else:pin.forwardKinematics(m.model,m.data,q)
        pin.updateFramePlacements(m.model,m.data)
        distances=[];rows=[];cached={}
        for s in self.shapes:
            fid=m.body_frames[s['body']];pose=m.data.oMf[fid]
            local=support_witness(s,pose.rotation);offset=pose.rotation@local
            distances.append(float(pose.translation[2]+offset[2]-self.ground_z_m))
            if jacobian:
                if fid not in cached:
                    cached[fid]=pin.getFrameJacobian(m.model,m.data,fid,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).copy()
                J=cached[fid];rows.append(J[2]+offset[1]*J[3]-offset[0]*J[4])
        return np.asarray(distances),np.asarray(rows) if jacobian else None


    def minimum_witness(self, q):
        """Exact authored lowest support point; does not assert native contact."""
        gaps,_=self.evaluate(q)
        index=int(np.argmin(gaps));shape=self.shapes[index]
        pose=self.model.data.oMf[self.model.body_frames[shape['body']]]
        local=support_witness(shape,pose.rotation)
        return dict(body=shape['body'],shape_path=shape['path'],shape_index=index,
            gap_m=float(gaps[index]),witness_world_m=(pose.translation+pose.rotation@local).tolist(),
            root_position_world_m=np.asarray(q[:3]).tolist(),
            root_quaternion_xyzw=np.asarray(q[3:7]).tolist(),
            geometry_sha256=self.sha256,ground_z_m=self.ground_z_m,
            native_contact_verified=False)
