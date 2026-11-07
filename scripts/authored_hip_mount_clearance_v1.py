"""Authored hip-cylinder / mounting-box intersection checks.

This is a convex geometric calculation, not a PhysX contact-force observation.
The radial gap is measured within the cylinder axial slab; it is not the
Euclidean distance between the complete solids. An axial separation certifies
nonintersection without solving the radial problem.
"""
import numpy as np
import pinocchio as pin
from scipy.optimize import minimize

from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance


def cylinder_box_gap(center, rotation, radius, half_height, axis, box_center, box_linear, box_half):
    center,rotation,box_center,box_linear=map(np.asarray,(center,rotation,box_center,box_linear))
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-10):
        raise ValueError('Cylinder transform must be a rotation')
    if radius<=0 or half_height<=0 or box_half<=0 or axis not in [0,1,2]:
        raise ValueError('Invalid primitive dimensions')
    A=rotation.T@box_linear;b=rotation.T@(box_center-center)
    extent=box_half*np.sum(abs(A[axis]))
    low,high=b[axis]-extent,b[axis]+extent
    axial_gap=max(low-half_height,-half_height-high)
    if axial_gap>1e-12:
        return dict(separated=True,axis_separation_m=float(axial_gap),radial_gap_m=None,
                    solver_required=False,solver_verified=True)
    radial=[i for i in range(3) if i!=axis]
    def objective(t):return float(np.sum((A@t+b)[radial]**2))
    def gradient(t):return 2*A[radial].T@(A@t+b)[radial]
    def constraints(t):
        z=(A@t+b)[axis];return np.array([half_height-z,half_height+z])
    result=minimize(objective,np.zeros(3),jac=gradient,bounds=[(-box_half,box_half)]*3,
        constraints={'type':'ineq','fun':constraints,'jac':lambda t:np.stack([-A[axis],A[axis]])},
        method='SLSQP',options={'ftol':1e-14,'maxiter':100})
    residual=max(float(np.max(abs(result.x))-box_half),float(-constraints(result.x).min()),0.)
    if not result.success or residual>1e-8 or not np.isfinite(result.fun):
        raise ValueError('Authored convex geometry solve failed: '+str(result.message))
    gap=float(np.sqrt(max(result.fun,0.))-radius)
    return dict(separated=gap>0.,axis_separation_m=float(max(axial_gap,0.)),radial_gap_m=gap,
        solver_required=True,solver_verified=True,constraint_residual_m=residual,
        box_witness_local=result.x.tolist(),cylinder_witness_local=(A@result.x+b).tolist())


class AuthoredHipMountClearanceV1:
    def __init__(self, path, model):
        self.model=model
        self.geometry=AuthoredGroundClearance(path,model,margin_m=0.)
        self.hips=['FL_hip','FR_hip','RL_hip','RR_hip']
        self.shapes={}
        for name in self.hips+['MountPlate']:
            matches=[s for s in self.geometry.shapes if s['body']==name]
            if len(matches)!=1:raise ValueError('Unique collider required for '+name)
            self.shapes[name]=matches[0]
        if self.shapes['MountPlate']['kind']!='Cube' or any(self.shapes[h]['kind']!='Cylinder' for h in self.hips):
            raise ValueError('Declared primitive geometry changed')

    def evaluate(self,q,hips=None):
        model=self.model
        pin.forwardKinematics(model.model,model.data,q);pin.updateFramePlacements(model.model,model.data)
        def pose(s):
            body=model.data.oMf[model.body_frames[s['body']]];T=s['transform']
            return body.translation+body.rotation@T[3,:3],body.rotation@T[:3,:3].T
        box=self.shapes['MountPlate'];bc,B=pose(box);rows=[]
        for name in self.hips if hips is None else hips:
            s=self.shapes[name];center,rotation=pose(s)
            rows.append(dict(hip=name,mount='MountPlate',**cylinder_box_gap(center,rotation,s['radius'],
                s['height']/2,s['axis'],bc,B,box['size']/2)))
        return rows
