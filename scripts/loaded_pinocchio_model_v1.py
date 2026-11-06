"""Full floating-base rigid-body model of the unchanged loaded USD asset.

CPU planning foundation, not an executed controller. Fixed bodies retain their
mass and inertia. Joint order is mapped explicitly; Pinocchio uses a depth-first
tree. Public root velocity is the LINK-origin velocity, expressed in the body,
which is the free-flyer tangent convention. No simulator state is queried.

The source record contains authored rigid-body inertia, not freshly measured
PhysX tensors. Motor armature/friction and implicit-PD/impact dynamics are NOT
silently included here. They require separate native-interface verification.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'configs/go2_piper_racket_landing_model_v1.json'


def rotation_wxyz(value):
    w, x, y, z = np.asarray(value, dtype=np.float64)
    quat = pin.Quaternion(w, x, y, z)
    if quat.norm() < 1e-12:
        raise ValueError('Zero quaternion')
    quat.normalize()
    return quat.matrix()


@dataclass
class PublicState:
    q: np.ndarray
    v: np.ndarray
    root_rotation_world: np.ndarray
    root_position_world: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    contacts: np.ndarray
    rotation_projection_error: float


class LoadedPinocchioModelV1:
    def __init__(self, path=MODEL):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.record = json.loads(raw)
        if (self.record['stage_meters_per_unit'] != 1. or
                self.record['stage_kilograms_per_unit'] != 1.):
            raise ValueError('Require SI asset record')
        self.names = self.record['body_names']
        self.joint_order = self.record['joint_order']
        bodies = {b['name']: b for b in self.record['bodies']}
        if len(bodies) != len(self.names) or set(bodies) != set(self.names):
            raise ValueError('Duplicate or missing bodies')
        edges = {}
        children = set()
        for joint in self.record['joints']:
            if joint['child'] in children:
                raise ValueError('Body has multiple parents')
            children.add(joint['child'])
            edges.setdefault(joint['parent'], []).append(joint)
        self.model = pin.Model()
        self.model.name = 'Go2_PiPER_full_racket_rigid_v1'
        self.model.gravity.linear = np.array([0., 0., -9.81])
        root_joint = self.model.addJoint(0, pin.JointModelFreeFlyer(),
                                        pin.SE3.Identity(), 'floating_base')
        self.body_frames = {}
        self.actuated_joint_ids = {}

        def add_body(name, joint_id, placement, parent_frame):
            if name in self.body_frames:
                raise ValueError('Cyclic body tree')
            body = bodies[name]
            principal = rotation_wxyz(body['principal_axes_wxyz'])
            inertia = principal @ np.diag(body['diagonal_inertia_kg_m2']) @ principal.T
            if body['mass_kg'] <= 0 or np.linalg.eigvalsh(inertia).min() <= 0:
                raise ValueError('Nonpositive authored body mass/inertia')
            self.model.appendBodyToJoint(joint_id, pin.Inertia(
                body['mass_kg'], np.array(body['center_of_mass_m']), inertia), placement)
            frame_id = self.model.addFrame(pin.Frame(
                name, joint_id, parent_frame, placement, pin.FrameType.BODY))
            self.body_frames[name] = frame_id
            for edge in edges.get(name, []):
                parent_to_joint = pin.SE3(rotation_wxyz(edge['parent_rotation_wxyz']),
                                         np.array(edge['parent_position_m']))
                child_to_joint = pin.SE3(rotation_wxyz(edge['child_rotation_wxyz']),
                                        np.array(edge['child_position_m']))
                parent_joint_to_new_joint = placement * parent_to_joint
                if edge['kind'] == 'revolute':
                    if edge['name'] not in self.joint_order or edge['axis'] not in 'XYZ':
                        raise ValueError('Unknown actuated joint')
                    axis = np.eye(3)['XYZ'.index(edge['axis'])]
                    child_id = self.model.addJoint(joint_id,
                        pin.JointModelRevoluteUnaligned(axis), parent_joint_to_new_joint,
                        edge['name'])
                    self.actuated_joint_ids[edge['name']] = child_id
                    child_placement = child_to_joint.inverse()
                elif edge['kind'] == 'fixed':
                    child_id = joint_id
                    child_placement = parent_joint_to_new_joint * child_to_joint.inverse()
                else:
                    raise ValueError('Unsupported asset joint kind')
                add_body(edge['child'], child_id, child_placement, frame_id)

        add_body(self.record['root_body'], root_joint, pin.SE3.Identity(), 0)
        if set(self.body_frames) != set(bodies):
            raise ValueError('Disconnected asset tree')
        if set(self.actuated_joint_ids) != set(self.joint_order):
            raise ValueError('Actuated joint mismatch')
        self.q_indices = np.array([self.model.joints[self.actuated_joint_ids[n]].idx_q
                                   for n in self.joint_order])
        self.v_indices = np.array([self.model.joints[self.actuated_joint_ids[n]].idx_v
                                   for n in self.joint_order])
        if (self.model.nq, self.model.nv) != (25, 24):
            raise ValueError('Expected floating base and all 18 active joints')
        self.feet = [name + '_foot' for name in ('FL', 'FR', 'RL', 'RR')]
        self.mass = float(sum(i.mass for i in self.model.inertias[1:]))
        if not np.isclose(self.mass, self.record['total_authored_mass_kg'], atol=1e-10, rtol=0):
            raise ValueError('Lost fixed-body mass')
        self.data = self.model.createData()

    def state_from_public(self, public, default,
                          known_goal_world=(1.10, 0., 1.80)):
        public = np.asarray(public, dtype=np.float64)
        default = np.asarray(default, dtype=np.float64)
        if public.shape != (794,) or default.shape != (18,) or not np.isfinite(public).all():
            raise ValueError('Require one finite public794 row and 18 defaults')
        # Existing command fixes its world normal to +X. Gravity supplies +Z.
        forward, up = public[763:766], -public[6:9]
        measured = np.stack((forward, np.cross(up, forward), up))
        u, _, vh = np.linalg.svd(measured)
        rotation = u @ np.diag([1., 1., np.linalg.det(u @ vh)]) @ vh
        projection_error = float(np.linalg.norm(rotation - measured))
        if projection_error > 1e-3:
            raise ValueError('Public command/gravity does not define a valid rotation')
        root = np.asarray(known_goal_world) - rotation @ public[760:763]
        q = pin.neutral(self.model)
        q[:3], q[3:7] = root, pin.Quaternion(rotation).coeffs()
        q[self.q_indices] = default + public[9:27]
        v = np.zeros(self.model.nv)
        v[:6], v[self.v_indices] = public[:6], public[27:45]
        return PublicState(q, v, rotation, root, q[self.q_indices].copy(),
                           v[self.v_indices].copy(), public[57:61] > .5, projection_error)

    @staticmethod
    def _rotate_world_to_body(rotation, value):
        """Rotate a world vector into the floating-base body frame."""
        return rotation.T @ np.asarray(value, dtype=np.float64)

    def state_from_native(self, env, lane, default=None, contact_force_threshold_n=0.0):
        """Build a planning state from Isaac Lab's live articulation tensors.

        The previous evaluator reconstructed q/v from the 794-d public policy
        observation.  That observation contains relative quantities and a
        synthetic goal position, so it is not an adequate state source for a
        contact-sensitive landing QP.  This adapter reads the actual root pose,
        COM twist and joint state from ``env.robot.data`` and the measured foot
        contact hysteresis from ``env.observed_foot_contact``.  It deliberately
        does not query PhysX solver internals; these are the same tensors used
        by the native observation and event code.

        ``default`` is retained for API symmetry and validation only.  Joint
        positions are read directly from the simulator, rather than inferred
        as an offset from a policy observation.
        """
        if not hasattr(env, 'robot') or not hasattr(env.robot, 'data'):
            raise ValueError('native state requires env.robot.data')
        data = env.robot.data
        lane = int(lane)
        root_pos = (data.root_link_pos_w[lane] - env.scene.env_origins[lane]).detach().cpu().numpy()
        quat_wxyz = data.root_link_quat_w[lane].detach().cpu().numpy()
        if quat_wxyz.shape != (4,) or not np.isfinite(quat_wxyz).all():
            raise ValueError('invalid native root quaternion')
        rotation = rotation_wxyz(quat_wxyz)

        # Isaac stores the COM twist in world coordinates.  Pinocchio's free
        # flyer tangent uses the link-origin linear velocity and body-frame
        # angular/linear components, matching the public observation contract.
        omega_w = data.root_com_ang_vel_w[lane].detach().cpu().numpy()
        com_v_w = data.root_com_lin_vel_w[lane].detach().cpu().numpy()
        com_offset_b = data.body_com_pos_b[lane, 0].detach().cpu().numpy()
        com_offset_w = rotation @ com_offset_b
        link_v_w = com_v_w - np.cross(omega_w, com_offset_w)
        v = np.zeros(self.model.nv, dtype=np.float64)
        v[:3] = self._rotate_world_to_body(rotation, link_v_w)
        v[3:6] = self._rotate_world_to_body(rotation, omega_w)

        joint_ids = np.asarray(env.joint_ids, dtype=np.int64)
        joint_position = data.joint_pos[lane, joint_ids].detach().cpu().numpy().astype(np.float64)
        joint_velocity = data.joint_vel[lane, joint_ids].detach().cpu().numpy().astype(np.float64)
        if joint_position.shape != (18,) or joint_velocity.shape != (18,):
            raise ValueError('native state requires the admitted 18-joint order')
        q = pin.neutral(self.model)
        q[:3] = root_pos
        q[3:7] = pin.Quaternion(rotation).coeffs()
        q[self.q_indices] = joint_position
        v[self.v_indices] = joint_velocity
        geometric_contacts = np.asarray(env.observed_foot_contact[lane].detach().cpu().numpy(), dtype=bool)
        if (isinstance(contact_force_threshold_n, bool) or
                not isinstance(contact_force_threshold_n, (int, float)) or
                not np.isfinite(contact_force_threshold_n) or contact_force_threshold_n < 0.):
            raise ValueError('contact_force_threshold_n must be finite and nonnegative')
        contacts = geometric_contacts.copy()
        # Always retain the native per-foot force telemetry when available.
        # The threshold argument controls whether force is used to gate the
        # support mask; it should not disable the measurement itself.  The
        # landing MPC needs the raw load transfer signal even in the default
        # zero-threshold diagnostic mode.
        measured_force_world = None
        if hasattr(env, 'feet_contact'):
            measured_force_world = env.feet_contact.data.net_forces_w[lane].detach().cpu().numpy()
            if measured_force_world.shape != (4, 3) or not np.isfinite(measured_force_world).all():
                raise ValueError('native foot force measurement must be finite 4x3')
            if contact_force_threshold_n > 0.:
                # A geometric contact is usable as a support constraint only
                # when the contact sensor reports a positive upward load. This
                # avoids giving the QP reaction-force variables to a foot that
                # merely grazes the ground or has already unloaded.
                contacts &= measured_force_world[:, 2] >= float(contact_force_threshold_n)
        if contacts.shape != (4,):
            raise ValueError('native state requires four measured foot contacts')
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError('nonfinite native q/v')
        result = PublicState(q, v, rotation, root_pos.copy(), joint_position.copy(),
                             joint_velocity.copy(), contacts, 0.0)
        result.geometric_contacts = geometric_contacts
        result.measured_force_world = measured_force_world
        return result

    def poses(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return ([self.data.oMf[self.body_frames[n]].translation.copy() for n in self.names],
                [self.data.oMf[self.body_frames[n]].rotation.copy() for n in self.names])

    def snapshot(self, q, v):
        """All matrices use Pinocchio's local free-flyer tangent coordinates.

        Foot Jacobians map v to world linear velocity at each foot-body origin.
        Their time variation follows the same moving configuration and columns.
        """
        model, data = self.model, self.data
        mass = pin.crba(model, data, q).copy()
        mass = np.triu(mass) + np.triu(mass, 1).T
        nonlinear = pin.nonLinearEffects(model, data, q, v).copy()
        gravity = pin.computeGeneralizedGravity(model, data, q).copy()
        pin.computeJointJacobiansTimeVariation(model, data, q, v)
        pin.updateFramePlacements(model, data)
        reference = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        foot_jacobians, foot_jacobian_dots = [], []
        for name in self.feet:
            fid = self.body_frames[name]
            jacobian = pin.getFrameJacobian(model, data, fid, reference).copy()
            # Pinocchio 2.7's LWA time-variation helper only translates dJ.
            # Differentiate the moving-point shift explicitly: Jp=Jw-p×Jw_ang;
            # dJp=dJw-p×dJw_ang-pdot×Jw_ang. WORLD has a fixed origin.
            world_dot = pin.getFrameJacobianTimeVariation(
                model, data, fid, pin.ReferenceFrame.WORLD).copy()
            point = data.oMf[fid].translation
            velocity = jacobian[:3] @ v
            point_dot = (world_dot[:3] - np.cross(point, world_dot[3:].T).T
                         - np.cross(velocity, jacobian[3:].T).T)
            foot_jacobians.append(jacobian[:3])
            foot_jacobian_dots.append(point_dot)
        poses = [data.oMf[self.body_frames[n]].translation.copy() for n in self.names]
        rotations = [data.oMf[self.body_frames[n]].rotation.copy() for n in self.names]
        pin.ccrba(model, data, q, v)
        centroidal_matrix, momentum = data.Ag.copy(), data.hg.vector.copy()
        centroidal_inertia = data.Ig.inertia.copy()
        pin.centerOfMass(model, data, q, v)
        return dict(mass_matrix=mass, nonlinear=nonlinear, gravity=gravity,
                    foot_jacobian=np.array(foot_jacobians),
                    foot_jacobian_dot=np.array(foot_jacobian_dots),
                    body_positions_world=np.array(poses), body_rotations_world=np.array(rotations),
                    com_world=data.com[0].copy(), com_velocity_world=data.vcom[0].copy(),
                    centroidal_matrix=centroidal_matrix, momentum_world=momentum,
                    centroidal_inertia_world=centroidal_inertia)
