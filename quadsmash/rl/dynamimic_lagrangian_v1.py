"""Batched CRBA/RNEA in link-local linear/angular tangent coordinates."""
import hashlib
from pathlib import Path

import torch
from torch import nn


def skew(v):
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).unflatten(-1, (3, 3))


def mv(mat, vec):
    return (mat @ vec.unsqueeze(-1)).squeeze(-1)


def rotation(wxyz):
    wxyz = torch.nn.functional.normalize(wxyz, dim=-1)
    c = skew(wxyz[..., 1:])
    return torch.eye(3, device=c.device, dtype=c.dtype) + 2 * (wxyz[..., :1,None] * c + c @ c)


class DynaMimicLagrangianV1(nn.Module):
    """Configuration is xyz, wxyz, 18 policy-order angles; velocity is local root twist, 18 rates."""
    def __init__(self, asset_path, model_path=None):
        super().__init__()
        from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1, MODEL
        parsed = LoadedPinocchioModelV1(model_path or MODEL)
        self.asset_sha256 = hashlib.sha256(Path(asset_path).read_bytes()).hexdigest()
        if self.asset_sha256 != parsed.record['asset_sha256']:
            raise ValueError('Inertia record does not match the supplied USD')
        model = parsed.model
        names = parsed.joint_order
        ids = [1] + [parsed.actuated_joint_ids[name] for name in names]
        self.parents = [0] + [ids.index(model.parents[i]) for i in ids[1:]]
        if any(parent >= i for i,parent in enumerate(self.parents[1:],1)):
            raise ValueError('Policy joint order must be topological')
        axes = {e['name']: e['axis'] for e in parsed.record['joints']}
        import numpy as np
        def buffer(name, vals):
            self.register_buffer(name, torch.from_numpy(np.array(vals, dtype=np.float64)))
        buffer('axis', [np.eye(3)['XYZ'.index(axes[n])] for n in names])
        buffer('rest_R', [model.jointPlacements[i].rotation for i in ids[1:]])
        buffer('rest_t', [model.jointPlacements[i].translation for i in ids[1:]])
        buffer('inertia', [model.inertias[i].matrix() for i in ids])
        feet = [model.frames[parsed.body_frames[n]] for n in parsed.feet]
        self.foot_parents = [ids.index(f.parent) for f in feet]
        buffer('foot_offset', [f.placement.translation for f in feet])

    def forward(self, q, v):
        if q.shape != (v.shape[0], 25) or v.shape[-1] != 24:
            raise ValueError('Expected batch of 25 configuration and 24 tangent coordinates')
        b = len(q)
        eye = torch.eye(3, device=q.device, dtype=q.dtype)
        axis_cross = skew(self.axis)
        angle = q[:, 7:, None, None]
        rot = self.rest_R @ (eye + angle.sin()*axis_cross + (1-angle.cos())*(axis_cross @ axis_cross))
        rt = rot.transpose(-1,-2)
        top = torch.cat((rt, -rt @ skew(self.rest_t)), -1)
        X = torch.cat((top, torch.cat((torch.zeros_like(rt), rt), -1)), -2)
        S = torch.cat((torch.zeros_like(self.axis), self.axis), -1)
        R = [rotation(q[:,3:7])]
        pos = [q[:,:3]]
        vel = [v[:,:6]]
        grav = q.new_tensor([0.,0.,9.81]).expand(b,-1)
        acc = [torch.cat((mv(R[0].mT,grav),torch.zeros_like(grav)), -1)]
        J = [torch.eye(6,24,device=q.device,dtype=q.dtype).expand(b,-1,-1)]
        basis = torch.eye(24,device=q.device,dtype=q.dtype)
        for j, parent in enumerate(self.parents[1:]):
            x = X[:,j]
            pos.append(pos[parent] + mv(R[parent], self.rest_t[j]))
            R.append(R[parent] @ rot[:,j])
            joint_v = v[:,6+j,None] * S[j]
            vi = mv(x, vel[parent]) + joint_v
            vel.append(vi)
            joint_w = joint_v[:,3:]
            cross = torch.cat((torch.linalg.cross(vi[:,:3], joint_w),
                               torch.linalg.cross(vi[:,3:], joint_w)), -1)
            acc.append(mv(x, acc[parent]) + cross)
            J.append(x @ J[parent] + S[j,:,None]*basis[6+j])
        vc, ac = torch.stack(vel,1), torch.stack(acc,1)
        mom = mv(self.inertia,vc)
        wrench = mv(self.inertia,ac) + torch.cat((
            torch.linalg.cross(vc[...,3:],mom[...,:3]),
            torch.linalg.cross(vc[...,:3],mom[...,:3])+torch.linalg.cross(vc[...,3:],mom[...,3:])), -1)
        forces = list(wrench.unbind(1))
        composite = [m.expand(b,-1,-1) for m in self.inertia.unbind(0)]
        rows, tau = [], []
        for i in range(18,0,-1):
            j, parent = i-1, self.parents[i]
            x = X[:,j]
            tau.append((forces[i]*S[j]).sum(-1))
            forces[parent] = forces[parent] + mv(x.mT,forces[i])
            rows.append(mv(J[i].mT,mv(composite[i],S[j])))
            composite[parent] = composite[parent] + x.mT @ composite[i] @ x
        base = torch.nn.functional.pad(composite[0], (0,18,0,18))
        lower = torch.nn.functional.pad(torch.stack(rows[::-1],1),(0,0,6,0))
        mass = base + lower + lower.tril(-1).mT
        feet, jac = [], []
        for parent, off in zip(self.foot_parents,self.foot_offset):
            feet.append(pos[parent] + mv(R[parent],off))
            jac.append(R[parent] @ (J[parent][:,:3] - skew(off) @ J[parent][:,3:]))
        jac = torch.stack(jac,1)
        return mass, torch.cat((forces[0],torch.stack(tau[::-1],1)),-1), torch.stack(feet,1), mv(jac,v[:,None]), jac
