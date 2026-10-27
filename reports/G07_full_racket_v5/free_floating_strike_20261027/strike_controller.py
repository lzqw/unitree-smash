"""Public pre-hit air-only velocity correction using conserved momentum.

Every result is an ordinary bounded joint position request through the native
actuators. No root velocity, momentum, body pose, or shuttle state is written.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
import torch

from quadsmash.control.full_racket_residual_v4 import ResidualActionMixerV4

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidates():
    return [dict(candidate_id=i, name=name, gain=gain, leg_velocity_delta_cap_rad_s=cap)
        for i,(name,gain,cap) in enumerate([
            ('unchanged_source', 0., 0.), ('momentum_arm_only', 1., 0.),
            ('momentum_leg_delta2', 1., 2.), ('momentum_leg_delta5', 1., 5.)])]


class PublicStrikeRequest:
    def __init__(self, asset_sha256, gain, leg_velocity_delta_cap_rad_s):
        if gain not in [0., 1.] or leg_velocity_delta_cap_rad_s not in [0., 2., 5.]:
            raise ValueError('Require a declared strike gain/leg velocity change cap')
        check = json.loads((HERE/'ACTUAL_CHECK.json').read_text())
        for path in [HERE/'free_floating.py', HERE.parent/'centroidal_recovery_model_20261027/known_model.py',
                     HERE.parent/'centroidal_recovery_model_20261027/MODEL.json']:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=check['source_sha256'][str(path)]:
                raise ValueError('Contact-validated model/linearization changed')
        self.core = load('public_strike_free_floating_core', HERE/'free_floating.py')
        known = load('public_strike_known_model', HERE.parent/'centroidal_recovery_model_20261027/known_model.py')
        self.model = known.KnownModel()
        if self.model.record['asset_sha256']!=asset_sha256:
            raise ValueError('Require the validated original full-racket physical asset')
        self.gain, self.leg_cap = float(gain), float(leg_velocity_delta_cap_rad_s)
        self.preview_s, self.scale_rad = .03, .15
        self.desired_velocity_world = np.array([3., 0., -.5])

    def request(self, public, defaults, limits, event, half_window, planned_valid, local_point, face_sign):
        public = np.asarray(public, dtype=float)
        defaults, limits = np.asarray(defaults), np.asarray(limits)
        if public.ndim!=2 or public.shape[1]!=794 or defaults.shape!=(18,) or limits.shape!=(18,2):
            raise ValueError('Require original public794D input and18D joint limits/defaults')
        if not np.isfinite(public).all() or not np.isfinite(limits).all() or not np.isfinite(defaults).all():
            raise ValueError('Finite public inputs required')
        event, half_window = np.asarray(event), np.asarray(half_window)
        if event.shape!=(len(public),) or half_window.shape!=event.shape or not (half_window>0).all():
            raise ValueError('Require original positive stroke windows per lane')
        local_point = np.asarray(local_point, dtype=float)
        if local_point.shape!=(3,) or face_sign not in [-1.,1.]:
            raise ValueError('Require source material contact point/face')
        relative = (public[:,775]-event)/half_window
        envelope = np.where(abs(relative)<1., (1.-relative**2)**2, 0.)
        up = -public[:,6:9]
        height = 1.80-np.sum(up*public[:,760:763], axis=1)
        active = ((self.gain>0.) & np.asarray(planned_valid,dtype=bool) & (public[:,772]>.5)
                  & (public[:,67]>.5) & (~(public[:,57:61]>.5).any(1))
                  & (height>=.30) & (up[:,2]>.25) & (envelope>0.))
        requests = np.zeros((len(public),18),dtype=np.float32)
        selected = np.flatnonzero(active)
        diagnostics = []
        if not len(selected):
            return requests, active, diagnostics
        obs = public[selected]
        q, qd = defaults[None]+obs[:,9:27], obs[:,27:45]
        points = np.broadcast_to(local_point,(len(obs),3))
        normals = np.broadcast_to([face_sign,0.,0.],(len(obs),3))
        item = self.core.linearization(self.model,q,points,normals,obs[:,:3],obs[:,3:6],qd)
        world = self.model.public_world_state(obs,defaults)['root_rotation_world']
        for i,lane in enumerate(selected):
            jv = world[i]@item['hit_jacobian'][i]
            jn = world[i]@item['normal_rate_jacobian'][i]
            velocity = world[i]@item['velocity'][i]
            normal_rate = world[i]@item['normal_rate'][i]
            # Total joint rates obey original simulation limits and an80ms
            # original position-limit preview. The correction is limited
            # further to5rad/s in arms, and the declared cap in legs.
            velocity_cap = np.r_[np.full(12,30.), np.full(6,5.)]
            delta_cap = np.r_[np.full(12,self.leg_cap), np.full(6,5.)]
            lower = np.maximum(-delta_cap, np.maximum(-velocity_cap-qd[i],(limits[:,0]-q[i])/.08-qd[i]))
            upper = np.minimum(delta_cap, np.minimum(velocity_cap-qd[i],(limits[:,1]-q[i])/.08-qd[i]))
            if np.any(lower>upper):
                active[lane]=False
                diagnostics.append(dict(lane=int(lane), solved=False, reason='Original velocity/position box infeasible'))
                continue
            fixed = (upper-lower)<=1e-10
            delta = np.where(fixed,.5*(lower+upper),0.)
            free = ~fixed
            a = np.vstack((jv,.3*jn,.05*np.eye(18)))
            b = np.r_[self.desired_velocity_world-velocity,-.3*normal_rate,np.zeros(18)]
            if free.any():
                fit = lsq_linear(a[:,free],b-a@delta,bounds=(lower[free],upper[free]),method='bvls',tol=1e-10)
                if not fit.success or not np.isfinite(fit.x).all():
                    raise RuntimeError('Conserved-momentum strike bounded solve failed')
                delta[free]=fit.x
            # Do not emit a request whose combined velocity/orientation
            # tracking objective is worse than leaving the source unchanged.
            before = float(np.linalg.norm(b))
            after = float(np.linalg.norm(a@delta-b))
            if after>before+1e-8:
                active[lane]=False
                diagnostics.append(dict(lane=int(lane),solved=False,reason='No local objective improvement'))
                continue
            displacement = self.gain*envelope[lane]*self.preview_s*delta
            assert (abs(displacement)<=self.scale_rad+1e-8).all()
            requests[lane] = (displacement/self.scale_rad).astype(np.float32)
            predicted_w = item['root_angular_offset'][i]+item['root_angular_jacobian'][i]@(qd[i]+delta)
            predicted_v = item['root_linear_offset'][i]+item['root_linear_jacobian'][i]@(qd[i]+delta)
            diagnostics.append(dict(lane=int(lane),solved=True,envelope=float(envelope[lane]),
                before_weighted_objective=before,after_weighted_objective=after,
                actual_point_velocity_world_m_s=velocity.tolist(),
                predicted_full_delta_point_velocity_world_m_s=(velocity+jv@delta).tolist(),
                requested_joint_velocity_delta_rad_s=delta.tolist(),
                requested_joint_position_delta_rad=displacement.tolist(),
                measured_joint_velocity_rad_s=qd[i].tolist(),
                predicted_root_angular_body_rad_s=predicted_w.tolist(),
                predicted_root_linear_body_m_s=predicted_v.tolist()))
        return requests, active, diagnostics


class MomentumStrikeController:
    def __init__(self, source_controller, asset_sha256, count, device, gain, leg_velocity_delta_cap_rad_s):
        self.source = source_controller
        self.reference = self.source.reference_controller
        self.teacher = self.reference.teacher
        if self.teacher.joint_preparation is None or self.teacher.arm_target_rate.rate!=5.:
            raise ValueError('Retain complete-tool support guard and original5rad/s arm limiter')
        self.requester = PublicStrikeRequest(asset_sha256,gain,leg_velocity_delta_cap_rad_s)
        self.mixer = ResidualActionMixerV4(count,device,[.15]*18)
        self.valid_plan = torch.zeros(count,dtype=torch.bool,device=device)
        self.plan_cursor = 0
        self.last_diagnostic = None

    @property
    def reference_controller(self):
        return self.reference

    @property
    def logs(self):
        return self.source.logs

    @logs.setter
    def logs(self,value):
        self.source.logs=value

    def reset(self,ids=None):
        self.source.reset(ids)
        self.mixer.reset(ids)
        if ids is None:
            self.valid_plan.zero_()
            self.plan_cursor=0
            self.last_diagnostic=None
        else:
            self.valid_plan[ids]=False

    @torch.inference_mode()
    def action(self,observation,default,limits):
        public = observation['policy']
        base = self.source.action(observation,default,limits)
        if self.requester.gain==0.:
            return base
        for entry in self.reference.logs[self.plan_cursor:]:
            ids=torch.as_tensor(entry['env_ids'],device=public.device,dtype=torch.long)
            self.valid_plan[ids]=torch.as_tensor(entry['valid'],device=public.device,dtype=torch.bool)
        self.plan_cursor=len(self.reference.logs)
        center = self.reference.event_time-self.reference.parameters[:,14]
        half_window = self.reference.parameters[:,19]
        requests,active,predictions=self.requester.request(public.detach().cpu().numpy(),
            default.detach().cpu().numpy(),limits.detach().cpu().numpy(),center.detach().cpu().numpy(),
            half_window.detach().cpu().numpy(),self.valid_plan.detach().cpu().numpy(),
            self.teacher.contact_offset.detach().cpu().numpy(),float(self.teacher.face_sign))
        requests=torch.as_tensor(requests,device=public.device,dtype=public.dtype)
        base_target=limits[:,0]+.5*(base+1.)*(limits[:,1]-limits[:,0])
        desired=(base_target+.15*requests).clamp(limits[:,0],limits[:,1])
        arm_actual=(default[12:][None]+public[:,21:27]).clamp(limits[12:,0],limits[12:,1])
        rows=torch.as_tensor(np.flatnonzero(active),device=public.device,dtype=torch.long)
        guard_rejected=torch.zeros(len(public),dtype=torch.bool,device=public.device)
        if len(rows):
            # Run the original complete-tool guard on the candidate target.
            # It accepts a full public batch; only active rows are emitted.
            guarded=self.teacher.joint_preparation(desired[:,12:],public,limits[12:],
                self.teacher.kinematics,arm_actual,post_stroke_mask=public[:,775]>=center+half_window)
            delta_arm=(guarded-base_target[:,12:])/.15
            accepted=(abs(delta_arm[rows])<=1.).all(1)
            rejected=rows[~accepted]
            guard_rejected[rejected]=True
            requests[rejected]=0.
            requests[rows[accepted],12:]=delta_arm[rows[accepted]]
        action=self.mixer.action(base,requests,public,default,limits)
        self.last_diagnostic=dict(active=active.copy(),predictions=predictions,
            guard_rejected=guard_rejected.clone(),requests=requests.clone(),
            base_target=base_target.clone(),original_complete_tool_guard_applied=True)
        return action
