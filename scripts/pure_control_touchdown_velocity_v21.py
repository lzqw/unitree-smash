"""State-driven preloading foot motion for uncovered recovery states.

Approaching swing feet receive soft stopping-position and tangential-velocity
references. Vertical buffering, all-body dynamics and original actuator/path
constraints are unchanged. This is a WBC reference candidate, not certified
MPC or proof of recoverability. Selected optimized trajectories are preserved.
"""
import numpy as np
import pinocchio as pin
from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15


class PureControlTouchdownVelocityV21(PureControlStanceFootReferenceV15):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.touchdown_reference = dict(active=False)

    def reference(self, state, snapshot, joint_reference):
        result = super().reference(state, snapshot, joint_reference)
        self.touchdown_reference = dict(active=False, reason='selected_full_reference_preserved')
        if self.chosen is not None:
            return result
        pin.computeJointJacobians(self.model.model, self.model.data, state.q)
        pin.updateFramePlacements(self.model.model, self.model.data)
        weights = np.zeros(4)
        velocities = np.zeros((4, 3))
        gaps = np.zeros(4)
        times = np.zeros(4)
        for j, sphere in enumerate(self.geometry.spheres):
            frame = self.model.body_frames[sphere['body']]
            pose = self.model.data.oMf[frame]
            offset = pose.rotation @ np.asarray(sphere['center_body_m']) - np.array([0., 0., sphere['radius_m']])
            jac = pin.getFrameJacobian(self.model.model, self.model.data, frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            twist = jac @ state.v
            velocity = twist[:3] + np.cross(twist[3:], offset)
            gap = float(pose.translation[2] + offset[2])
            velocities[j], gaps[j] = velocity, gap
            # Actual supporting feet are handled by the original contact
            # constraints; no force is assigned to an airborne/light foot.
            if state.contacts[j] or velocity[2] >= -1e-6:
                continue
            time_to_plane = max(0., gap) / -velocity[2]
            times[j] = time_to_plane
            if time_to_plane >= .12:
                continue
            x = 1. - time_to_plane/.12
            weight = x*x*(3.-2.*x)
            weights[j] = weight
            stopping_time = max(time_to_plane, .02)
            target_position = pose.translation[:2] + .5*time_to_plane*velocity[:2]
            # The WBC swing task uses the foot body origin. Convert a zero
            # surface tangential-speed goal to that origin without assuming
            # its angular velocity is zero. The acceleration is a local
            # braking reference, not an exact moving-point inverse dynamics.
            target_velocity = -np.cross(twist[3:], offset)[:2]
            target_acceleration = -velocity[:2]/stopping_time
            for name, target in [('swing_foot_target_world', target_position),
                ('swing_foot_velocity_world', target_velocity),
                ('swing_foot_acceleration_world', target_acceleration)]:
                result[name][j, :2] = (1.-weight)*result[name][j, :2] + weight*target
        self.touchdown_reference = dict(active=bool(np.any(weights>0)),
            reason='current_swing_foot_descent', weights=weights,
            surface_velocity_world=velocities, sphere_gap_m=gaps,
            constant_velocity_time_to_plane_s=times,
            time_estimate_is_not_contact_certificate=True,
            vertical_reference_unchanged=True, support_contacts_unchanged=True)
        self.reference_diagnostics['touchdown_reference'] = self.touchdown_reference
        return result

    def command(self, state, phase, time_s):
        target, diagnostic = super().command(state, phase, time_s)
        diagnostic.update(controller='pure_control_touchdown_velocity_v21',
            touchdown_reference=self.touchdown_reference,
            original_physical_constraints=True, selected_plan_preserved=True)
        return target, diagnostic
