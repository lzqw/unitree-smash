"""Differentiable all-body geometry from the pinned Go2/PiPER/racket asset.

No controller, contact inference, or simulator calls. Link origins are not
collision-surface clearances. Used to build physically consistent model inputs
and analytical reward features, never to replace native collision evidence.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .supermimic_physical_observation_v1 import JOINT_NAMES

MODEL = Path(__file__).resolve().parents[2] / 'configs/go2_piper_racket_landing_model_v1.json'


def rotation(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def rotate(r, v):
    return (r @ v.unsqueeze(-1)).squeeze(-1)


class FullBodyGeometryV1(nn.Module):
    def __init__(self, asset_sha256, *, model_path=MODEL, dtype=torch.float32):
        super().__init__()
        path = Path(model_path)
        record = json.loads(path.read_text())
        if record['asset_sha256'] != asset_sha256:
            raise ValueError('Full-body geometry does not match native asset')
        if tuple(record['joint_order']) != JOINT_NAMES:
            raise ValueError('Full-body geometry joint order differs from actor contract')
        self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.asset_sha256 = asset_sha256
        self.names = tuple(record['body_names'])
        self.ids = {name: i for i, name in enumerate(self.names)}
        self.root_id = self.ids[record['root_body']]
        self.feet = tuple(self.ids[name+'_foot'] for name in ('FL','FR','RL','RR'))
        self.heads = tuple(self.ids[name] for name in ('Head_upper','Head_lower'))
        self.register_buffer('mass', torch.tensor([b['mass_kg'] for b in record['bodies']], dtype=dtype))
        self.register_buffer('local_com', torch.tensor([b['center_of_mass_m'] for b in record['bodies']], dtype=dtype))
        self.register_buffer('eye', torch.eye(3, dtype=dtype))
        pending, seen, edges, constants = list(record['joints']), {record['root_body']}, [], []
        while pending:
            ready = [j for j in pending if j['parent'] in seen]
            if not ready:
                raise ValueError('Disconnected geometry tree')
            for j in ready:
                axis = torch.zeros(3, dtype=dtype)
                if j['axis'] is not None:
                    axis['XYZ'.index(j['axis'])] = 1.
                x,y,z = axis.unbind(); zero = axis.new_zeros(())
                skew = torch.stack((zero,-z,y,z,zero,-x,-y,x,zero)).reshape(3,3)
                constants.append((j['parent_position_m'],j['child_position_m'],
                                  rotation(torch.tensor(j['parent_rotation_wxyz'],dtype=dtype)),
                                  rotation(torch.tensor(j['child_rotation_wxyz'],dtype=dtype)),axis,skew))
                edges.append((self.ids[j['parent']],self.ids[j['child']],
                              JOINT_NAMES.index(j['name']) if j['name'] in JOINT_NAMES else None))
                seen.add(j['child']); pending.remove(j)
        self.edges = tuple(edges)
        # Preserve the asset's joint order and the original buffer indices.
        # Off-chain links cannot affect the world pose or twist of the racket.
        ancestor = self.ids.get('Racket')
        chain = []
        for i, edge in reversed(tuple(enumerate(self.edges))):
            if edge[1] == ancestor:
                chain.append((i, edge))
                ancestor = edge[0]
        self.racket_chain = tuple(reversed(chain))
        for i,name in enumerate(('parent_offset','child_offset','parent_rotation','child_rotation','axis','skew')):
            self.register_buffer(name, torch.stack([torch.as_tensor(c[i],dtype=dtype) for c in constants]))

    @staticmethod
    def _validate_inputs(joints, velocities, root_position, root_quaternion, root_velocity, root_omega):
        n = joints.shape[0]
        if joints.shape != (n,18) or velocities.shape != joints.shape:
            raise ValueError('Expected named18 joint positions and velocities')
        if root_position.shape != (n,3) or root_quaternion.shape != (n,4):
            raise ValueError('Expected world root link position and wxyz quaternion')
        if root_velocity.shape != (n,3) or root_omega.shape != (n,3):
            raise ValueError('Expected root LINK velocity and world angular velocity')

    def enable_depth_batches(self, expected_backend=None):
        """Batch sibling links for full-body FK while preserving recorded body and joint order."""
        backend = dict(schema='full-body-depth-batches-v1', model_sha256=self.model_sha256,
                       source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if expected_backend is not None and backend != expected_backend:
            raise ValueError('Resumed full-body FK backend differs from checkpoint')
        if hasattr(self, '_depth_ranges'):
            return backend

        def indices(name, values):
            self.register_buffer(name, torch.tensor(values, dtype=torch.long, device=self.mass.device),
                                 persistent=False)

        # The asset loader emits all joints at the current depth before their children.
        parents = [self.root_id]
        order = parents.copy()
        ranges = []
        start = 0
        while start < len(self.edges):
            end = start
            parent_rows = []
            while end < len(self.edges) and self.edges[end][0] in parents:
                parent_rows.append(parents.index(self.edges[end][0]))
                end += 1
            if end == start:
                raise ValueError('Joint order does not form consecutive tree depths')
            name = f'_parents_at_depth_{len(ranges)}'
            indices(name, parent_rows)
            ranges.append((slice(start, end), name))
            parents = [child for _, child, _ in self.edges[start:end]]
            order.extend(parents)
            start = end
        indices('_named_body_order', [order.index(link) for link in range(len(self.names))])
        # Fixed joints use the appended zero angle and zero relative speed.
        indices('_edge_joint', [len(JOINT_NAMES) if j is None else j for _, _, j in self.edges])
        self._depth_ranges = tuple(ranges)
        return backend

    def _depth_kinematics(self, joints, velocities, root_position, root_quaternion, root_velocity, root_omega):
        self._validate_inputs(joints, velocities, root_position, root_quaternion, root_velocity, root_omega)
        zero = joints.new_zeros(len(joints), 1)
        angle = torch.cat((joints, zero), 1).index_select(1, self._edge_joint)[:, :, None, None]
        rate = torch.cat((velocities, zero), 1).index_select(1, self._edge_joint)[:, :, None]
        motion = self.eye + angle.sin()*self.skew + (1-angle.cos())*(self.skew @ self.skew)
        layers = [(root_position[:, None], rotation(root_quaternion)[:, None],
                   root_velocity[:, None], root_omega[:, None])]
        for edges, parent_buffer in self._depth_ranges:
            parents = getattr(self, parent_buffer)
            p, r, v, w = (field.index_select(1, parents) for field in layers[-1])
            joint_rotation = r @ self.parent_rotation[edges]
            offset0 = rotate(r, self.parent_offset[edges])
            child_rotation = joint_rotation @ motion[:, edges] @ self.child_rotation[edges].transpose(1, 2)
            offset1 = rotate(child_rotation, self.child_offset[edges])
            child_omega = w + rotate(joint_rotation, self.axis[edges])*rate[:, edges]
            layers.append((p + offset0 - offset1, child_rotation,
                           v + torch.linalg.cross(w, offset0) - torch.linalg.cross(child_omega, offset1),
                           child_omega))
        return tuple(torch.cat(parts, 1).index_select(1, self._named_body_order) for parts in zip(*layers))

    def _link_kinematics(self, edges, joints, velocities, root_position, root_quaternion, root_velocity, root_omega):
        self._validate_inputs(joints, velocities, root_position, root_quaternion, root_velocity, root_omega)
        # Each body's tensors have their own storage/graph. Avoid writing views
        # of a shared FK output while autograd still needs an earlier body.
        p, r, v, w = {}, {}, {}, {}
        p[self.root_id], r[self.root_id] = root_position, rotation(root_quaternion)
        v[self.root_id], w[self.root_id] = root_velocity, root_omega
        for i,(parent,child,joint) in edges:
            joint_rotation = r[parent] @ self.parent_rotation[i]
            offset0 = rotate(r[parent],self.parent_offset[i])
            if joint is None:
                motion, added_omega = self.eye, 0.
            else:
                angle = joints[:,joint,None,None]
                skew = self.skew[i]
                motion = self.eye + angle.sin()*skew + (1-angle.cos())*(skew@skew)
                added_omega = rotate(joint_rotation,self.axis[i])*velocities[:,joint,None]
            r[child] = joint_rotation @ motion @ self.child_rotation[i].T
            offset1 = rotate(r[child],self.child_offset[i])
            p[child] = p[parent] + offset0 - offset1
            w[child] = w[parent] + added_omega
            v[child] = v[parent] + torch.linalg.cross(w[parent],offset0) - torch.linalg.cross(w[child],offset1)
        return p, r, v, w

    def forward(self, joints, velocities, root_position, root_quaternion, root_velocity, root_omega):
        inputs = (joints, velocities, root_position, root_quaternion, root_velocity, root_omega)
        if hasattr(self, '_depth_ranges'):
            positions, rotations, linear, angular = self._depth_kinematics(*inputs)
        else:
            links = self._link_kinematics(enumerate(self.edges), *inputs)
            positions, rotations, linear, angular = (
                torch.stack([field[i] for i in range(len(self.names))], 1) for field in links)
        offset = rotate(rotations,self.local_com)
        weights = self.mass[None,:,None] / self.mass.sum()
        return dict(position=positions, rotation=rotations, velocity=linear, angular_velocity=angular,
                    feet=positions[:,self.feet], head_origins=positions[:,self.heads],
                    center=((positions+offset)*weights).sum(1),
                    center_velocity=((linear+torch.linalg.cross(angular,offset))*weights).sum(1))


def racket_kinematics(fields, geometry):
    """Return world racket position, rotation, linear and angular velocity."""
    inputs = (fields['joint_position'], fields['joint_velocity'], fields['root_position_w'],
              fields['root_quaternion_w'], fields['root_link_velocity_w'], fields['root_angular_velocity_w'])
    i = geometry.ids['Racket']
    if type(geometry) is FullBodyGeometryV1 and hasattr(geometry, 'racket_chain'):
        positions, rotations, velocities, omegas = geometry._link_kinematics(geometry.racket_chain, *inputs)
        return positions[i], rotations[i], velocities[i], omegas[i]
    bodies = geometry(*inputs)
    return tuple(bodies[name][:, i] for name in ('position', 'rotation', 'velocity', 'angular_velocity'))
