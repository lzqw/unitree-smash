"""Authored sphere separation and configuration derivative, no fake pair force."""
import itertools
import numpy as np
import pinocchio as pin


def foot_pair_clearance(model,q,spheres):
    if [s['body'] for s in spheres]!=model.feet:raise ValueError('Audited physical foot order required')
    pin.computeJointJacobians(model.model,model.data,q);pin.updateFramePlacements(model.model,model.data)
    centers=[];jac=[]
    for sphere in spheres:
        frame=model.body_frames[sphere['body']];pose=model.data.oMf[frame]
        offset=pose.rotation@np.asarray(sphere['center_body_m'])
        centers.append(pose.translation+offset)
        j=pin.getFrameJacobian(model.model,model.data,frame,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        jac.append(j[:3]-pin.skew(offset)@j[3:])
    gaps=[];rows=[]
    for a,b in itertools.combinations(range(4),2):
        delta=centers[a]-centers[b];distance=np.linalg.norm(delta)
        if distance<1e-9:raise ValueError('Coincident foot centers')
        gaps.append(distance-spheres[a]['radius_m']-spheres[b]['radius_m'])
        rows.append(delta/distance@(jac[a]-jac[b]))
    return np.asarray(gaps),np.asarray(rows)
