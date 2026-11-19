"""Single-impact moving-surface prior, not a replacement for native PhysX.

Uses original geometry and restitution with a kinematic/infinite-mass surface.
Friction, ball spin, articulated effective mass, multiple impacts, and contact
persistence are NOT solved. Discrete contact selection is not a smooth actor
objective. This module is a diagnostic prior for later supervised residuals.
"""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .supermimic_ball_geometry_preview_v1 import BallGeometryPreviewV1,box_gap,round_gap
from .supermimic_physical_dataset_v1 import decode_physical


class BallImpulseCalibrationV1(nn.Module):
    """Two-parameter calibration for the frozen analytic contact prior.

    The parameters are deliberately low dimensional. ``gap_bias_m`` adjusts
    the predicted contact boundary and ``normal_impulse_scale`` calibrates the
    normal response magnitude. The bounded parameterization prevents a small
    native probe set from producing an unphysical correction. This module is
    used by the offline calibration utility and is not an Actor or WM head.
    """
    schema = 'ball-impulse-calibration-v1'

    def __init__(self, *, max_gap_bias_m=0.01, scale_bounds=(0.5, 1.5)):
        super().__init__()
        if max_gap_bias_m <= 0 or not torch.isfinite(torch.tensor(max_gap_bias_m)):
            raise ValueError('max_gap_bias_m must be finite and positive')
        lo, hi = map(float, scale_bounds)
        if not 0 < lo < hi:
            raise ValueError('scale_bounds must be positive and ordered')
        self.max_gap_bias_m = float(max_gap_bias_m)
        self.scale_bounds = (lo, hi)
        self.raw_gap_bias = nn.Parameter(torch.zeros(()))
        midpoint = (lo + hi) / 2.0
        fraction = (midpoint - lo) / (hi - lo)
        self.raw_scale = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(fraction)))))

    def values(self):
        lo, hi = self.scale_bounds
        gap_bias = self.max_gap_bias_m * torch.tanh(self.raw_gap_bias)
        normal_scale = lo + (hi - lo) * torch.sigmoid(self.raw_scale)
        return gap_bias, normal_scale

    def manifest(self):
        gap_bias, normal_scale = self.values()
        return dict(schema=self.schema, gap_bias_m=float(gap_bias.detach()),
                    normal_impulse_scale=float(normal_scale.detach()),
                    max_gap_bias_m=self.max_gap_bias_m,
                    scale_bounds=list(self.scale_bounds))


def box_normal(points,centers,halves):
    d=points[...,None,:]-centers
    outside=d-d.clamp(-halves,halves)
    q=d.abs()-halves
    axis=q.argmax(-1)
    sign=torch.where(d>=0.,torch.ones_like(d),-torch.ones_like(d))
    inside=F.one_hot(axis,3).to(d.dtype)*sign
    return torch.where((q.amax(-1)>0)[...,None],F.normalize(outside,dim=-1),inside)


def round_normal(points,centers,axes,halves,radii,capsule):
    d=points[...,None,:]-centers;a=(d*axes).sum(-1)
    radial=d-a[...,None]*axes;radial_length=radial.norm(dim=-1)
    radial_unit=F.normalize(radial,dim=-1)
    axial_unit=torch.where(a>=0.,1.,-1.)[...,None]*axes
    capsule_unit=F.normalize(d-a.clamp(-halves,halves)[...,None]*axes,dim=-1)
    q=torch.stack((radial_length-radii,a.abs()-halves),-1)
    outside=F.normalize(q[...,0:1].clamp_min(0)*radial_unit+q[...,1:2].clamp_min(0)*axial_unit,dim=-1)
    inside=torch.where((q[...,0]>=q[...,1])[...,None],radial_unit,axial_unit)
    cylinder=torch.where((q.amax(-1)>0)[...,None],outside,inside)
    return torch.where(capsule[...,None],capsule_unit,cylinder)


def rotate_for_time(rotation,omega,time):
    angle=omega*time[:,None]
    x,y,z=angle.unbind(-1);zero=torch.zeros_like(x)
    skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(-1,3,3)
    theta=angle.norm(dim=-1)
    # sinc expressions avoid division singularities at zero angular speed.
    a=torch.sinc(theta/torch.pi)[:,None,None]
    b=(.5*torch.sinc(theta/(2*torch.pi)).square())[:,None,None]
    return (torch.eye(3,device=omega.device,dtype=omega.dtype)+a*skew+b*(skew@skew))@rotation


class BallImpulsePriorV1(nn.Module):
    schema='single-impact-moving-surface-ball-prior-v1'
    def __init__(self, *, gap_bias_m=0.0, normal_impulse_scale=1.0):
        super().__init__()
        # Geometry is a deterministic execution-contract object.  Keep it
        # available on the frozen prior while avoiding a second copy of its
        # constant buffers in every action-ball residual checkpoint.
        object.__setattr__(self, 'preview', BallGeometryPreviewV1())
        path=Path(__file__).resolve().parents[2]/'configs/shuttle_physics_v2.json'
        c=json.loads(path.read_text());self.restitution=float(c['restitution'])
        self.drag=float(c['quadratic_drag_k_per_m']);self.dt=float(c['policy_dt_s'])
        self.sub_dt=float(c['native_physics_dt_s']);self.steps=round(self.dt/self.sub_dt)
        if not torch.isfinite(torch.tensor(gap_bias_m)):
            raise ValueError('gap_bias_m must be finite')
        if not torch.isfinite(torch.tensor(normal_impulse_scale)) or normal_impulse_scale <= 0:
            raise ValueError('normal_impulse_scale must be finite and positive')
        self.gap_bias_m=float(gap_bias_m)
        self.normal_impulse_scale=float(normal_impulse_scale)
        self.register_buffer('gravity',torch.tensor([0.,0.,-9.81]))
        self.register_buffer('wind',torch.tensor(c['wind_m_s']))

    def _apply(self, fn):
        # ``preview`` is intentionally unregistered, so move/cast it along
        # with the registered prior buffers when the Hybrid module changes
        # device or dtype.
        super()._apply(fn)
        self.preview._apply(fn)
        return self

    def calibration_manifest(self):
        return dict(gap_bias_m=self.gap_bias_m,
                    normal_impulse_scale=self.normal_impulse_scale)

    # The analytic prior is a frozen execution component.  Its physical
    # configuration and source hash are recorded in the hybrid manifest, so
    # it is intentionally reconstructed from the contract instead of being
    # duplicated inside action-ball residual checkpoints.
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        return

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        return

    def advance(self,p,v,duration):
        remaining=duration[:,None]
        for _ in range(self.steps):
            dt=remaining.clamp(0.,self.sub_dt)
            relative=v-self.wind
            v=v+dt*(self.gravity-self.drag*relative.norm(dim=-1,keepdim=True)*relative)
            p=p+dt*v;remaining=remaining-dt
        return p,v

    def advance_trajectory(self, p, v):
        """Integrate one control tick and retain every native substep.

        Collision search and post-impact integration must use the same
        drag/gravity trajectory.  This avoids using a ballistic, no-drag path
        for event timing and a drag path for the resulting state.
        """
        positions=[p]
        velocities=[v]
        for _ in range(self.steps):
            relative=v-self.wind
            v=v+self.sub_dt*(self.gravity-
                self.drag*relative.norm(dim=-1,keepdim=True)*relative)
            p=p+self.sub_dt*v
            positions.append(p)
            velocities.append(v)
        return torch.stack(positions,1),torch.stack(velocities,1)

    def racket_surface(self,local):
        g=self.preview
        bg=box_gap(local,g.racket_box_centers,g.racket_box_halves)
        rg=round_gap(local,g.racket_round_centers,g.racket_round_axes,g.racket_round_halves,g.racket_round_radii,g.racket_round_capsule)
        normals=torch.cat((box_normal(local,g.racket_box_centers,g.racket_box_halves),
            round_normal(local,g.racket_round_centers,g.racket_round_axes,g.racket_round_halves,g.racket_round_radii,g.racket_round_capsule)),-2)
        index=torch.cat((bg,rg),-1).argmin(-1)
        normal=normals.gather(-2,index[...,None,None].expand(*index.shape,1,3)).squeeze(-2)
        return normal,index

    def contact_state_at_time(self, ball_position, ball_velocity, initial_rotation,
                              initial_center, initial_velocity, initial_angular_velocity,
                              impact_time, *, endpoint_rotation=None,
                              endpoint_center=None, endpoint_velocity=None,
                              endpoint_angular_velocity=None):
        """Reconstruct ball and racket state at one common continuous TOI.

        Collision search still uses the sampled trajectory. All quantities used
        by the impulse calculation are then recomputed at the interpolated
        impact time, so position, normal, orientation and surface velocity are
        evaluated on one time slice.
        """
        pre_p, pre_v=self.advance(ball_position,ball_velocity,impact_time)
        alpha=(impact_time/self.dt).clamp(0.,1.)[:,None]
        if endpoint_rotation is None:
            racket_rotation=rotate_for_time(initial_rotation,initial_angular_velocity,impact_time)
            racket_center=initial_center+initial_velocity*impact_time[:,None]
            racket_velocity=initial_velocity
            racket_angular_velocity=initial_angular_velocity
        else:
            if any(value is None for value in (endpoint_center,endpoint_velocity,endpoint_angular_velocity)):
                raise ValueError('Complete endpoint racket state is required')
            racket_rotation=self._interpolate_rotation(initial_rotation,endpoint_rotation,alpha)
            racket_center=torch.lerp(initial_center,endpoint_center,alpha)
            racket_velocity=torch.lerp(initial_velocity,endpoint_velocity,alpha)
            racket_angular_velocity=torch.lerp(initial_angular_velocity,endpoint_angular_velocity,alpha)
        delta=pre_p-racket_center
        local=(racket_rotation.transpose(-1,-2)@delta[...,None]).squeeze(-1)
        normal_local,part=self.racket_surface(local)
        normal_world=(racket_rotation@normal_local[...,None]).squeeze(-1)
        contact_arm=(racket_rotation@(local-self.preview.radius*normal_local)[...,None]).squeeze(-1)
        surface_velocity=racket_velocity+torch.linalg.cross(racket_angular_velocity,contact_arm)
        return dict(position=pre_p,velocity=pre_v,rotation=racket_rotation,
                    center=racket_center,local=local,normal_local=normal_local,
                    normal=normal_world,part=part,surface_velocity=surface_velocity)

    @staticmethod
    def _interpolate_rotation(start,end,alpha):
        """Differentiable orthonormal interpolation of two racket frames."""
        x=torch.nn.functional.normalize(torch.lerp(start[..., :, 0],end[..., :, 0],alpha),dim=-1)
        y_raw=torch.lerp(start[..., :, 1],end[..., :, 1],alpha)
        y=torch.nn.functional.normalize(y_raw-x*(x*y_raw).sum(-1,keepdim=True),dim=-1)
        z=torch.linalg.cross(x,y)
        return torch.stack((x,y,z),-1)

    def net_surface(self,world):
        g=self.preview
        bg=box_gap(world,g.net_centers,g.net_halves)
        rg=round_gap(world,g.post_centers,g.post_axes,g.post_halves,g.post_radii,g.post_capsule)
        normals=torch.cat((box_normal(world,g.net_centers,g.net_halves),
            round_normal(world,g.post_centers,g.post_axes,g.post_halves,g.post_radii,g.post_capsule)),-2)
        index=torch.cat((bg,rg),-1).argmin(-1)
        return normals.gather(-2,index[...,None,None].expand(*index.shape,1,3)).squeeze(-2),index

    def forward(self,state,geometry,present,predicted_robot=None):
        if len(state)>256:
            robot_chunks = (None if predicted_robot is None else predicted_robot.split(256))
            chunks=[self(s,geometry,p, None if robot_chunks is None else robot_chunks[i])
                    for i,(s,p) in enumerate(zip(state.split(256),present.split(256)))]
            details={}
            for key, value in chunks[0][1].items():
                if torch.is_tensor(value):
                    details[key]=torch.cat([item[1][key] for item in chunks])
                else:
                    details[key]=value
            return torch.cat([x[0] for x in chunks]), details
        p=decode_physical(state)
        if predicted_robot is not None and predicted_robot.shape != state.shape:
            raise ValueError('Predicted robot state must align with current state')
        bodies=geometry(p['joint_position'],p['joint_velocity'],p['root_position_w'],p['root_quaternion_w'],
            p['root_link_velocity_w'],p['root_angular_velocity_w'])
        i=geometry.ids['Racket'];r=bodies['rotation'][:,i];center=bodies['position'][:,i]
        vc=bodies['velocity'][:,i];omega=bodies['angular_velocity'][:,i]
        delta=p['ball_position_w']-center
        local=(r.transpose(-1,-2)@delta[...,None]).squeeze(-1)
        relative=(r.transpose(-1,-2)@(p['ball_velocity_w']-vc-torch.linalg.cross(omega,delta))[...,None]).squeeze(-1)
        times=self.preview.times[None,:,None]
        if predicted_robot is None:
            # Use the same normalized-time Hermite preview as the learned
            # sweep gate.  The helper converts m/s endpoint tangents using
            # the control duration, so the analytic and learned paths share
            # both geometry and time units.
            from .supermimic_action_ball_v1 import hermite_sweep_samples
            local_path=hermite_sweep_samples(
                local, local + self.dt * relative, relative, relative,
                dt=self.dt, count=self.steps + 1)
        else:
            # The endpoint uses the predicted current-command robot pose.  A
            # cubic sweep is unnecessary for the analytic prior itself, but
            # using both endpoint poses prevents the prior from becoming a
            # state-only collision term in the main transition.
            endpoint_fields=decode_physical(torch.cat((predicted_robot[:, :49],
                state[:, 49:]), -1))
            endpoint_bodies=geometry(endpoint_fields['joint_position'],
                endpoint_fields['joint_velocity'],endpoint_fields['root_position_w'],
                endpoint_fields['root_quaternion_w'],endpoint_fields['root_link_velocity_w'],
                endpoint_fields['root_angular_velocity_w'])
            endpoint_id=geometry.ids['Racket']
            endpoint_rotation=endpoint_bodies['rotation'][:, endpoint_id]
            endpoint_center=endpoint_bodies['position'][:, endpoint_id]
            endpoint_ball,endpoint_ball_velocity=self.advance(
                p['ball_position_w'],p['ball_velocity_w'],
                torch.full_like(p['ball_position_w'][:, 0],self.dt))
            endpoint_local=(endpoint_rotation.transpose(-1,-2) @
                            (endpoint_ball-endpoint_center)[...,None]).squeeze(-1)
            endpoint_delta=endpoint_ball-endpoint_center
            endpoint_surface=endpoint_bodies['velocity'][:, endpoint_id] + torch.linalg.cross(
                endpoint_bodies['angular_velocity'][:, endpoint_id], endpoint_delta)
            endpoint_relative=(endpoint_rotation.transpose(-1,-2) @
                               (endpoint_ball_velocity - endpoint_surface)[...,None]).squeeze(-1)
            from .supermimic_action_ball_v1 import hermite_sweep_samples
            local_path=hermite_sweep_samples(
                local, endpoint_local, relative, endpoint_relative,
                dt=self.dt, count=self.steps + 1)
        world_path,world_velocity_path=self.advance_trajectory(
            p['ball_position_w'],p['ball_velocity_w'])
        gap=torch.stack((self.preview.racket_gap(local_path),world_path[:,:,2]-self.preview.radius,
                         self.preview.net_gap(world_path)),-1)
        gap=torch.cat((gap[..., :1]-self.gap_bias_m, gap[..., 1:]), -1)
        hit=gap<=0.
        indices=torch.where(hit,torch.arange(self.steps+1,device=state.device)[None,:,None],self.steps+1).amin(1)
        index,object_id=indices.min(-1);active=(index<=self.steps)&present.bool()
        safe=index.clamp_max(self.steps)
        t=self.preview.times[safe]
        row=torch.arange(len(state),device=state.device)
        before=(safe-1).clamp_min(0)
        g0=gap[row,before,object_id];g1=gap[row,safe,object_id]
        fraction=(g0/(g0-g1).clamp_min(1e-12)).clamp(0.,1.)
        time=torch.where(safe>0,self.preview.times[before]+fraction*self.sub_dt,t)
        time=torch.where(active,time,torch.full_like(time,self.dt))
        if predicted_robot is None:
            contact=self.contact_state_at_time(
                p['ball_position_w'],p['ball_velocity_w'],r,center,vc,omega,time)
        else:
            contact=self.contact_state_at_time(
                p['ball_position_w'],p['ball_velocity_w'],r,center,vc,omega,time,
                endpoint_rotation=endpoint_rotation,endpoint_center=endpoint_center,
                endpoint_velocity=endpoint_bodies['velocity'][:, endpoint_id],
                endpoint_angular_velocity=endpoint_bodies['angular_velocity'][:, endpoint_id])
        pre_p,pre_v=contact['position'],contact['velocity']
        racket_normal=contact['normal']
        racket_surface_velocity=contact['surface_velocity']
        part=contact['part']
        net_normal,_=self.net_surface(pre_p)
        ground_normal=torch.zeros_like(pre_p);ground_normal[:,2]=1.
        normal=torch.where((object_id==0)[:,None],racket_normal,
                           torch.where((object_id==1)[:,None],ground_normal,net_normal))
        surface_velocity=torch.where((object_id==0)[:,None],racket_surface_velocity,torch.zeros_like(vc))
        normal_speed=((pre_v-surface_velocity)*normal).sum(-1)
        impulse_dv=self.normal_impulse_scale * (-(1.+self.restitution)*
            normal_speed.clamp_max(0.)[:,None]*normal)
        impulse_dv=torch.where(active[:,None],impulse_dv,torch.zeros_like(impulse_dv))
        post_p,post_v=self.advance(pre_p,pre_v+impulse_dv,self.dt-time)
        result=torch.cat((post_p,post_v),-1)
        result=torch.where(present[:,None],result,state[:,49:])
        info=dict(active=active,object_id=torch.where(active,object_id,torch.full_like(object_id,3)),
            time=time,normal=normal,normal_speed=normal_speed,impulse_dv=impulse_dv,racket_part=part,
            min_gap=gap.amin(1),gap_bias_m=self.gap_bias_m,
            normal_impulse_scale=self.normal_impulse_scale,
            # Keep the continuous TOI state available to the optional
            # low-dimensional residual head.  These are diagnostics and
            # causal model inputs, never native future labels.
            pre_position=pre_p,
            pre_velocity=pre_v,
            remaining_time=(self.dt-time).clamp_min(0.),
            post_position=post_p,
            post_velocity=post_v)
        return result,info


def hybrid_ball_manifest(*, gap_bias_m=0.0, normal_impulse_scale=1.0):
    """Identity for the frozen analytic part of the Hybrid WM transition."""
    source = Path(__file__).resolve()
    config = source.parents[2] / 'configs/shuttle_physics_v2.json'
    def sha(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                h.update(block)
        return h.hexdigest()
    return dict(
        schema='supermimic-hybrid-ball-transition-v3',
        impulse_prior_schema=BallImpulsePriorV1.schema,
        impulse_prior_code_sha256=sha(source),
        shuttle_physics_config_sha256=sha(config),
        contact_objects=('racket', 'ground', 'net'),
        control_dt_s=BallImpulsePriorV1().dt,
        native_substep_dt_s=BallImpulsePriorV1().sub_dt,
        legacy_residual_policy='other_residual_only',
        calibration=dict(gap_bias_m=float(gap_bias_m),
                         normal_impulse_scale=float(normal_impulse_scale)),
    )


def attach_hybrid_impulse_prior(action_model, device=None, *, calibration=None):
    """Attach one registered, frozen prior to every Hybrid WM caller.

    The prior has no trainable or persistent checkpoint state.  Registering it
    as a submodule still makes ``to()``, ``eval()`` and parameter-freezing
    semantics uniform across loaders, fitting scripts and imagination.
    """
    if action_model is None:
        raise ValueError('Cannot attach a Hybrid prior to None')
    existing = getattr(action_model, 'impulse_prior', None)
    if existing is not None:
        if not isinstance(existing, BallImpulsePriorV1):
            raise TypeError('Existing impulse_prior has an incompatible type')
        expected=hybrid_ball_manifest(**(calibration or {}))
        if getattr(action_model, 'hybrid_ball_manifest', None) != expected:
            raise ValueError('Existing Hybrid prior manifest differs from current code')
        existing.to(device=device) if device is not None else None
        existing.requires_grad_(False).eval()
        return action_model
    calibration=calibration or {}
    unknown=set(calibration)-{'gap_bias_m','normal_impulse_scale'}
    if unknown:raise ValueError('Unknown Hybrid calibration fields: '+','.join(sorted(unknown)))
    prior = BallImpulsePriorV1(**calibration)
    if device is not None:
        prior = prior.to(device)
    prior.requires_grad_(False).eval()
    action_model.add_module('impulse_prior', prior)
    action_model.hybrid_ball_manifest = hybrid_ball_manifest(**calibration)
    return action_model
