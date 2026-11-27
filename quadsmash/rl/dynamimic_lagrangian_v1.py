"""Batched CRBA/RNEA in link-local linear/angular tangent coordinates."""
import hashlib
import json
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

        spheres = json.loads((Path(__file__).resolve().parents[2] / 'configs' /
                              'go2_piper_authored_foot_spheres_v1.json').read_text())
        if self.asset_sha256 not in spheres['asset_sha256'].values():
            raise ValueError('Foot collision geometry differs from robot asset')
        by_name = {s['body']: s for s in spheres['spheres']}
        shapes = json.loads((Path(__file__).resolve().parents[2] / 'configs' /
                             'go2_piper_authored_ground_geometry_v1.json').read_text())
        if self.asset_sha256 not in shapes['asset_sha256'].values():
            raise ValueError('Head collision geometry differs from robot asset')
        head, = [s for s in shapes['shapes'] if s['body'] == 'Head_lower']
        transform = np.asarray(head['transform'])
        if head['kind'] != 'Sphere' or not np.allclose(transform[:3, :3], np.eye(3)):
            raise ValueError('Expected the authored lower-head sphere')
        head_frame = model.frames[parsed.body_frames[head['body']]]
        self.surface_parents = self.foot_parents + [ids.index(head_frame.parent)]
        buffer('sphere_offset', [f.placement.act(np.asarray(by_name[n]['center_body_m']))
                                 for f, n in zip(feet, parsed.feet)] +
                                [head_frame.placement.act(transform[3, :3])])
        buffer('sphere_radius', [by_name[n]['radius_m'] for n in parsed.feet] + [head['radius']])

        # Fixed tree topology. Batch independent limbs at each depth, preserving
        # the spatial inertia recursion and the public policy joint ordering.
        levels = []
        previous = [0]
        while previous:
            children = [i for i in range(1, len(ids)) if self.parents[i] in previous]
            if not children:
                break
            parents = [previous.index(self.parents[i]) for i in children]
            level = nn.Module()
            level.register_buffer('joints', torch.tensor(children) - 1, persistent=False)
            level.register_buffer('sources', torch.tensor(parents), persistent=False)
            level.register_buffer('sum_children', torch.from_numpy(np.eye(len(previous))[parents].T.copy()),
                                  persistent=False)
            levels.append(level)
            previous = children
        self.levels = nn.ModuleList(levels)
        traversal = torch.cat([level.joints for level in levels])
        self.register_buffer('joint_order', traversal.argsort(), persistent=False)
        self.level_sizes = [1] + [len(level.joints) for level in levels]
        self.register_buffer('link_order', torch.cat((torch.zeros(1, dtype=torch.long),
                                                      1 + traversal.argsort())), persistent=False)
        self.register_buffer('surface_ids', torch.tensor(self.surface_parents), persistent=False)

    def forward(self, q, v, *, collision_surface=False):
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
        R = [rotation(q[:,3:7])[:, None]]
        pos = [q[:, None, :3]]
        vel = [v[:, None, :6]]
        grav = q.new_tensor([0., 0., 9.81]).expand(b, 1, -1)
        acc = [torch.cat((mv(R[0].mT, grav), torch.zeros_like(grav)), -1)]
        basis = torch.eye(24, device=q.device, dtype=q.dtype)
        J = [basis[:6].expand(b, 1, -1, -1)]
        joint_motion = S[..., None] * basis[6:, None]
        joint_velocity = v[:, 6:, None] * S
        transforms = []
        for level in self.levels:
            joints, parents = level.joints, level.sources
            x = X.index_select(1, joints)
            transforms.append(x)
            parent_rot = R[-1].index_select(1, parents)
            pos.append(pos[-1].index_select(1, parents) + mv(parent_rot, self.rest_t[joints]))
            R.append(parent_rot @ rot.index_select(1, joints))
            jv = joint_velocity.index_select(1, joints)
            vi = mv(x, vel[-1].index_select(1, parents)) + jv
            vel.append(vi)
            cross = torch.cat((torch.linalg.cross(vi[..., :3], jv[..., 3:]),
                               torch.linalg.cross(vi[..., 3:], jv[..., 3:])), -1)
            acc.append(mv(x, acc[-1].index_select(1, parents)) + cross)
            J.append(x @ J[-1].index_select(1, parents) + joint_motion[joints])
        traversal = torch.cat((self.inertia[:1], *(self.inertia[level.joints+1] for level in self.levels)))
        vc, ac = torch.cat(vel, 1), torch.cat(acc, 1)
        mom = mv(traversal, vc)
        wrench = mv(traversal, ac) + torch.cat((
            torch.linalg.cross(vc[..., 3:], mom[..., :3]),
            torch.linalg.cross(vc[..., :3], mom[..., :3]) + torch.linalg.cross(vc[..., 3:], mom[..., 3:])), -1)
        forces = list(wrench.split(self.level_sizes, dim=1))
        composite = list(traversal.expand(b, -1, -1, -1).split(self.level_sizes, dim=1))
        rows, tau = [], []
        for i in range(len(self.levels), 0, -1):
            level, x = self.levels[i-1], transforms[i-1]
            motion = S[level.joints]
            tau.append((forces[i] * motion).sum(-1))
            rows.append(mv(J[i].mT, mv(composite[i], motion)))
            forces[i-1] = forces[i-1] + level.sum_children @ mv(x.mT, forces[i])
            projected = (x.mT @ composite[i] @ x).flatten(-2)
            composite[i-1] = composite[i-1] + (level.sum_children @ projected).unflatten(-1, (6, 6))
        base = torch.nn.functional.pad(composite[0][:, 0], (0,18,0,18))
        lower = torch.nn.functional.pad(torch.cat(rows[::-1],1).index_select(1,self.joint_order),(0,0,6,0))
        mass = base + lower + lower.tril(-1).mT
        bias = torch.cat((forces[0][:, 0], torch.cat(tau[::-1],1).index_select(1,self.joint_order)), -1)
        offsets = self.sphere_offset if collision_surface else self.foot_offset
        parents = self.link_order[self.surface_ids[:len(offsets)]]
        rot_parent = torch.cat(R, 1).index_select(1, parents)
        jac_parent = torch.cat(J, 1).index_select(1, parents)
        feet = torch.cat(pos, 1).index_select(1, parents) + mv(rot_parent, offsets)
        jac = rot_parent @ (jac_parent[..., :3, :] - skew(offsets) @ jac_parent[..., 3:, :])
        if collision_surface:
            bottom = q.new_tensor([0., 0., -1.]) * self.sphere_radius[:, None]
            feet = feet + bottom
            jac = jac - skew(bottom) @ rot_parent @ jac_parent[..., 3:, :]
        return mass, bias, feet, mv(jac, v[:,None]), jac
