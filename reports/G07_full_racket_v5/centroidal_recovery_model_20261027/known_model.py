"""Rigid-body kinematics and centroidal angular momentum from public q/qd.

All model constants come from the composed USD. Public root velocities are
expressed in the root frame. Velocities below remain inertial velocities
expressed in that same instantaneous frame, including root angular motion.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def rotation(wxyz):
    value = np.asarray(wxyz, dtype=np.float64)
    flat = value[..., [1, 2, 3, 0]].reshape(-1, 4)
    return Rotation.from_quat(flat).as_matrix().reshape(value.shape[:-1]+(3, 3))


def apply(matrix, vector):
    return np.einsum('...ij,...j->...i', matrix, vector)


class KnownModel:
    def __init__(self, path=Path(__file__).with_name('MODEL.json')):
        self.record = json.loads(Path(path).read_text())
        self.names = self.record['body_names']
        self.indices = {name: i for i, name in enumerate(self.names)}
        self.joint_indices = {name: i for i, name in enumerate(self.record['joint_order'])}
        self.root = self.indices[self.record['root_body']]
        self.mass = np.array([b['mass_kg'] for b in self.record['bodies']], dtype=np.float64)
        self.com = np.array([b['center_of_mass_m'] for b in self.record['bodies']], dtype=np.float64)
        axes = rotation([b['principal_axes_wxyz'] for b in self.record['bodies']])
        diagonal = np.array([b['diagonal_inertia_kg_m2'] for b in self.record['bodies']])
        self.inertia = (axes*diagonal[:, None, :]) @ axes.transpose(0, 2, 1)
        assert (self.mass > 0).all() and (diagonal > 0).all()
        pending = list(self.record['joints'])
        self.edges = []
        seen = {self.record['root_body']}
        while pending:
            ready = [j for j in pending if j['parent'] in seen]
            if not ready:
                raise ValueError('Physical body graph must be a connected tree')
            for joint in ready:
                assert joint['child'] not in seen
                seen.add(joint['child'])
                axis = np.zeros(3)
                if joint['axis'] is not None:
                    axis['XYZ'.index(joint['axis'])] = 1.
                self.edges.append(dict(parent=self.indices[joint['parent']], child=self.indices[joint['child']],
                    p0=np.asarray(joint['parent_position_m']), p1=np.asarray(joint['child_position_m']),
                    r0=rotation(joint['parent_rotation_wxyz']), r1=rotation(joint['child_rotation_wxyz']),
                    axis=axis, coordinate=self.joint_indices.get(joint['name'])))
                pending.remove(joint)
        assert len(seen) == len(self.names)

    def forward(self, q, qd=None, root_linear=None, root_angular=None):
        q = np.asarray(q, dtype=np.float64)
        assert q.ndim == 2 and q.shape[1] == 18
        count = len(q)
        if qd is None:
            qd = np.zeros_like(q)
        positions = np.zeros((count, len(self.names), 3))
        rotations = np.broadcast_to(np.eye(3), (count, len(self.names), 3, 3)).copy()
        linear, angular = np.zeros_like(positions), np.zeros_like(positions)
        if root_linear is not None:
            linear[:, self.root] = root_linear
        if root_angular is not None:
            angular[:, self.root] = root_angular
        for edge in self.edges:
            parent, child, coordinate = edge['parent'], edge['child'], edge['coordinate']
            joint_rotation = rotations[:, parent] @ edge['r0']
            if coordinate is None:
                moving_rotation = np.broadcast_to(np.eye(3), (count, 3, 3))
                added_angular = np.zeros((count, 3))
            else:
                moving_rotation = Rotation.from_rotvec(q[:, coordinate, None]*edge['axis']).as_matrix()
                added_angular = apply(joint_rotation, edge['axis'])*qd[:, coordinate, None]
            rotations[:, child] = joint_rotation @ moving_rotation @ edge['r1'].T
            offset_parent = apply(rotations[:, parent], edge['p0'])
            offset_child = apply(rotations[:, child], edge['p1'])
            positions[:, child] = positions[:, parent]+offset_parent-offset_child
            angular[:, child] = angular[:, parent]+added_angular
            linear[:, child] = linear[:, parent]+np.cross(angular[:, parent], offset_parent)-np.cross(angular[:, child], offset_child)
        offset_com = apply(rotations, self.com)
        com_positions = positions+offset_com
        com_velocities = linear+np.cross(angular, offset_com)
        center = np.sum(com_positions*self.mass[None, :, None], axis=1)/self.mass.sum()
        center_velocity = np.sum(com_velocities*self.mass[None, :, None], axis=1)/self.mass.sum()
        tensors = rotations @ self.inertia @ rotations.transpose(0, 1, 3, 2)
        spin = apply(tensors, angular)
        orbital = self.mass[None, :, None]*np.cross(com_positions-center[:, None], com_velocities-center_velocity[:, None])
        momentum = spin+orbital
        return dict(position=positions, rotation=rotations, linear=linear, angular=angular,
            com_position=com_positions, com_velocity=com_velocities, center=center, center_velocity=center_velocity,
            angular_momentum_by_body=momentum, angular_momentum=momentum.sum(axis=1))

    def public_state(self, public, defaults):
        public = np.asarray(public)
        return self.forward(defaults[None]+public[:, 9:27], public[:, 27:45], public[:, :3], public[:, 3:6])

    def public_world_state(self, public, defaults, known_command=(1.10, 0., 1.80)):
        public = np.asarray(public, dtype=np.float64)
        value = self.public_state(public, defaults)
        # This admitted task commands a fixed known world+X normal and a
        # fixed known point. Gravity supplies world+Z; no true pose is read.
        forward, up = public[:, 763:766], -public[:, 6:9]
        forward = forward/np.linalg.norm(forward, axis=1, keepdims=True)
        up = up/np.linalg.norm(up, axis=1, keepdims=True)
        left = np.cross(up, forward)
        left = left/np.linalg.norm(left, axis=1, keepdims=True)
        root_rotation = np.stack((forward, left, up), axis=1)
        root_position = np.asarray(known_command)[None]-apply(root_rotation, public[:, 760:763])
        value.update(root_rotation_world=root_rotation, root_position_world=root_position,
            center_world=root_position+apply(root_rotation, value['center']),
            center_velocity_world=apply(root_rotation, value['center_velocity']),
            angular_momentum_world=apply(root_rotation, value['angular_momentum']),
            body_position_world=root_position[:, None]+apply(root_rotation[:, None], value['position']),
            body_velocity_world=apply(root_rotation[:, None], value['linear']))
        return value

    def angular_momentum_matrix(self, q):
        q = np.asarray(q, dtype=np.float64)
        count = len(q)
        matrix = []
        for column in range(21):
            root_angular = np.zeros((count, 3))
            joint_velocity = np.zeros((count, 18))
            if column < 3:
                root_angular[:, column] = 1.
            else:
                joint_velocity[:, column-3] = 1.
            matrix.append(self.forward(q, joint_velocity, root_angular=root_angular)['angular_momentum'])
        return np.stack(matrix, axis=-1)
