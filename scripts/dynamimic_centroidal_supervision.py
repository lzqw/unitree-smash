"""Estimate interval-average external support from native whole-robot momentum changes."""
import json
import torch
from torch.nn import functional as F

from quadsmash.rl.dynamimic_lagrangian_v1 import mv, rotation
from quadsmash.rl.dynamimic_recovery_wm_v1 import local_robot_state
from quadsmash.rl.supermimic_full_body_geometry_v1 import MODEL


def link_inertias(geometry):
    record = json.loads(MODEL.read_text())
    if record['asset_sha256'] != geometry.asset_sha256 or tuple(b['name'] for b in record['bodies']) != geometry.names:
        raise ValueError('Centroidal inertia/bodies differ from native geometry')
    principal = rotation(geometry.mass.new_tensor([b['principal_axes_wxyz'] for b in record['bodies']]))
    diagonal = geometry.mass.new_tensor([b['diagonal_inertia_kg_m2'] for b in record['bodies']])
    return principal @ torch.diag_embed(diagonal) @ principal.mT


def endpoints(physical, targets, geom, inertias):
    q,v = local_robot_state(physical)
    rot = rotation(q[:,3:7])
    fk = geom(q[:,7:],v[:,6:],q[:,:3],q[:,3:7],mv(rot,v[:,:3]),mv(rot,v[:,3:6]))
    link_rot, omega = fk['rotation'], fk['angular_velocity']
    offset = mv(link_rot,geom.local_com)
    centers = fk['position'] + offset
    velocities = fk['velocity'] + torch.linalg.cross(omega,offset)
    orbital = torch.linalg.cross(centers-fk['center'][:,None],geom.mass[None,:,None]*velocities)
    spin = mv(link_rot,mv(inertias,mv(link_rot.mT,omega)))
    h = fk['feet'][...,2]
    features = torch.cat((v[:,:6],-rot[:,2],q[:,7:],v[:,6:],h,
                          fk['velocity'][:,geom.feet].flatten(1),targets),-1)
    return features, fk['feet']-fk['center'][:,None], fk['center_velocity'], (orbital+spin).sum(1)


@torch.no_grad()
def native_support_samples(data, geom, excluded, dt):
    if dt != .01:
        raise ValueError('Unexpected native control interval')
    # Use only still-live post-release steps, avoiding body/rigid-net collision impulses.
    keep = data['gate'] & data['live'] & ~data['body'] & ~data['net'] & ~excluded
    rows = keep.nonzero()
    if not len(rows):
        raise ValueError('No native training-side support samples')
    outputs = [[] for _ in range(7)]
    inertia = link_inertias(geom)
    for group in rows.split(512):
        t,l = group.T
        target = data['final_joint_targets'][t,l].to(geom.mass)
        before,after = (endpoints(data[key][t,l].to(geom.mass),target,geom,inertia) for key in ('before','after'))
        force = geom.mass.sum()*((after[2]-before[2])/dt + target.new_tensor([0.,0.,9.81]))
        moment = (after[3]-before[3])/dt
        labels = data['feet'][t,l].to(target)
        for out,value in zip(outputs,(before[0],after[0],before[1],after[1],force,moment,labels),strict=True):
            out.append(value.cpu())
    return [torch.cat(b) for b in outputs], rows


def support_loss(member, tensors):
    start,end,lever0,lever1,measured_force,measured_moment,feet = tensors
    force0, _, residual0, _ = member(start,start[:,45:49])
    force1, p1, residual1, _ = member(end,end[:,45:49])
    force = (force0+force1).sum(1)*.5
    moment = (torch.linalg.cross(lever0,force0)+torch.linalg.cross(lever1,force1)).sum(1)*.5
    # Momentum supplies the aggregate wrench. Individual foot forces remain latent.
    error = (F.smooth_l1_loss(force/50.,measured_force/50.) +
             F.smooth_l1_loss(moment/10.,measured_moment/10.) +
             F.binary_cross_entropy(p1.clamp(1e-6,1-1e-6),feet) +
             .001*(residual0.square().mean()+residual1.square().mean()))
    return error, force, moment


def fit_support(member, optimizer, tensors, gen, device, steps=240):
    sizes = tensors[-1].sum(1)
    pools = [(sizes==k).nonzero().flatten() for k in range(5)]
    supported = [p for p in pools if len(p)]
    for _ in range(steps):
        indices = torch.cat([p[torch.randint(len(p),(64,),generator=gen)] for p in supported])
        loss, _, _ = support_loss(member,[a[indices].to(device) for a in tensors])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(member.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
    return float(loss.detach())
