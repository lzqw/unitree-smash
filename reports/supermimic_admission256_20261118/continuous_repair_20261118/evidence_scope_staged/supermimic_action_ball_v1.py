"""Opt-in, supervised action-conditioned racket-sweep correction.

Uses the predicted post-command robot pose (not future/native labels) and
native racket geometry to estimate a soft collision support. Legacy ball
predictions remain byte-for-byte unchanged unless a verified candidate is
explicitly loaded. This is a differentiable surrogate, not a PhysX solver.
"""
import torch
from torch import nn
from .supermimic_ball_residual_v1 import ball_geometry_features
from .supermimic_ball_geometry_preview_v1 import BallGeometryPreviewV1
from .supermimic_v1 import MLP


def preimpact_hard_negatives(contact_edge, episode, terminated, timeout):
    """Offline replay label for the transition immediately before a hit edge.

    Never consumed by Actor/WM forward; does not cross a reset boundary.
    """
    if not (contact_edge.shape==episode.shape==terminated.shape==timeout.shape):
        raise ValueError('Preimpact label shape mismatch')
    out=torch.zeros_like(contact_edge,dtype=torch.bool)
    out[:-1]=contact_edge[1:].bool()&(episode[:-1]==episode[1:])&~(
        terminated[:-1].bool()|timeout[:-1].bool())
    return out&~contact_edge.bool()


def postimpact_hard_negatives(contact_edge, episode, terminated, timeout):
    """Observed transition after an edge, excluding another edge or reset.

    This is an offline sampling index, not a claim that the ball has separated
    and not an extra input to the model. Native next state remains the label.
    """
    if not (contact_edge.shape == episode.shape == terminated.shape == timeout.shape):
        raise ValueError('Postimpact label shape mismatch')
    out = torch.zeros_like(contact_edge, dtype=torch.bool)
    out[1:] = contact_edge[:-1].bool() & (episode[1:] == episode[:-1]) & ~(
        terminated[:-1].bool() | timeout[:-1].bool())
    return out & ~contact_edge.bool()


def contact_neighborhood_pools(contact_edge, episode, terminated, timeout, eligible):
    """Disjoint contact/pre/post/background pools; future labels stay offline."""
    if eligible.shape != contact_edge.shape:
        raise ValueError('Contact eligibility shape mismatch')
    pre = preimpact_hard_negatives(contact_edge, episode, terminated, timeout)
    post = postimpact_hard_negatives(contact_edge, episode, terminated, timeout)
    # Consecutive impacts may place a row in both neighborhoods; retain its
    # previous-impact association so post-impact coverage remains explicit.
    pre = pre & ~post
    masks = dict(contact=contact_edge.bool(), pre_contact=pre, post_contact=post,
                 background=~(contact_edge.bool() | pre | post))
    return {key: (mask & eligible.bool()).nonzero() for key, mask in masks.items()}


class ActionConditionedBallV1(nn.Module):
    schema = 'action-conditioned-racket-sweep-v1'
    feature_dim = 48

    def __init__(self, *, preview=None, hidden=(128, 128), softness_m=0.008):
        super().__init__()
        if not 0 < softness_m <= 0.05:
            raise ValueError('Racket sweep support width must be in (0, 0.05] m')
        self.preview = preview if preview is not None else BallGeometryPreviewV1()
        self.softness_m = float(softness_m)
        self.head = MLP(self.feature_dim, hidden, 6)
        nn.init.zeros_(self.head.net[-1].weight)
        nn.init.zeros_(self.head.net[-1].bias)
        one = [3.]*3 + [12.]*3 + [.2]*3 + [12.]*3 + [1.]*3 + [8.]*3 + [15.]*3
        self.register_buffer('input_scale', torch.tensor(one+one+[.03]*3+[.1]*3))
        self.register_buffer('output_scale', torch.tensor([.05]*3+[10.]*3))

    def forward(self, state, predicted_robot, free_ball, geometry, present, *, details=False):
        """Predict [B,6] additional ball correction from current to next pose.

        state is the current 55D physical state, predicted_robot is 55D after
        the learned robot dynamics for the current final joint target, and
        free_ball is [B,6] from the causal free-flight model. No future native
        observations, events, or contact labels are consumed.
        """
        if state.ndim != 2 or state.shape[1] != 55 or predicted_robot.shape != state.shape:
            raise ValueError('Sweep requires [batch,55] current and predicted robot states')
        if free_ball.shape != (len(state),6) or present.numel() != len(state):
            raise ValueError('Sweep free-flight/ball-presence batch mismatch')
        current = ball_geometry_features(state, geometry)
        future = ball_geometry_features(torch.cat((predicted_robot[:, :49], free_ball),-1), geometry)
        first = current[:, 6:9]
        last = future[:, 6:9]
        middle = (first + last) * .5
        gaps = torch.stack((self.preview.racket_gap(first),
                            self.preview.racket_gap(middle),
                            self.preview.racket_gap(last)),-1)
        # Soft minimum keeps the impulse pathway differentiable near grazing
        # trajectories; the original collision geometry is not enlarged.
        smallest = -self.softness_m * torch.logsumexp(-gaps/self.softness_m,-1)
        support = torch.sigmoid(-smallest/self.softness_m)
        features = torch.cat((current[:, :21],future[:, :21],gaps,last-first),-1)
        normalized = (features/self.input_scale).clamp(-30.,30.)
        residual = (self.head(normalized)*self.output_scale)*support[:,None]*present.reshape(-1,1).to(state.dtype)
        if not torch.isfinite(residual).all():
            raise FloatingPointError('Nonfinite action-conditioned ball response')
        if details:
            return residual, dict(support=support, gaps=gaps,
                                  local_displacement=last-first)
        return residual


class RacketModeActionBallV2(ActionConditionedBallV1):
    """Action correction belongs to the predicted racket mixture component.

    V1 adds its correction even when the shared ball model predicts free
    flight. This explicit schema revision uses that model's CURRENT soft
    racket probability, never a native/future event label. Geometry, command
    authority and learned parameter shapes are unchanged. This remains a
    candidate model, not certified contact or a native execution change.
    """
    schema = 'racket-mode-conditioned-action-sweep-v2'
    requires_racket_probability = True

    def forward(self, state, predicted_robot, free_ball, geometry, present, *,
                racket_probability, details=False):
        if racket_probability.shape != (len(state),):
            raise ValueError('Racket mode probability must be [batch]')
        if (not torch.isfinite(racket_probability).all() or
                not ((racket_probability >= 0) & (racket_probability <= 1)).all()):
            raise ValueError('Invalid predicted racket mode probability')
        residual, info = super().forward(state, predicted_robot, free_ball,
                                         geometry, present, details=True)
        residual = residual * racket_probability[:, None]
        if details:
            return residual, dict(info, racket_probability=racket_probability,
                                  effective_support=info['support']*racket_probability)
        return residual


class ImpactTimeActionBallV3(RacketModeActionBallV2):
    """Explicit candidate using causal impact-time motion, not tick-end motion."""
    schema='impact-time-racket-mode-action-sweep-v3'

    def forward(self,state,predicted_robot,free_ball,geometry,present,*,racket_probability,details=False):
        from .supermimic_impact_timing_v1 import causal_sweep_time,interpolate_physical_state
        current=ball_geometry_features(state,geometry)
        endpoint=torch.cat((predicted_robot[:,:49],free_ball),-1)
        future=ball_geometry_features(endpoint,geometry)
        time,info=causal_sweep_time(current[:,6:9],current[:,9:12],future[:,6:9],self.preview)
        at_impact=interpolate_physical_state(state,endpoint,time,self.preview.times[-1])
        result=super().forward(state,at_impact,at_impact[:,49:],geometry,present,
                               racket_probability=racket_probability,details=details)
        if details:
            residual,metadata=result
            return residual,dict(metadata,impact_time_s=time,**info)
        return result


class MaterialInertiaActionBallV4(ImpactTimeActionBallV3):
    """Learned impact correction conditioned on measured component materials.

    Articulated inertia and normal-impact velocity scale are causal features,
    not an executed collision solver. Tangential/spin/multi-contact response
    remains supervised. In particular no missing ball-spin state is filled
    with a fictitious zero. Within-part gradients are continuous; nearest
    primitive and crossing-bin selection remain explicit discrete choices.
    """
    schema='native-material-articulated-impact-action-v4'

    def __init__(self, *, material_contract, **kwargs):
        from .supermimic_racket_material_contract_v1 import validate_contract
        from .supermimic_articulated_inertia_v1 import ArticulatedInertiaV1
        super().__init__(**kwargs)
        self.material_contract=validate_contract(material_contract,material_contract['asset_sha256'])
        self.inertia=ArticulatedInertiaV1(material_contract['asset_sha256'],dtype=torch.float32)
        self.register_buffer('component_material',torch.tensor(material_contract['effective'],dtype=torch.float32))
        self.register_buffer('mechanical_scale',torch.tensor([1.]*3+[12.]*6+[50.]*9+[1.]*3+[15.]*3))
        self.mechanical_head=MLP(24,(128,128),6)
        nn.init.zeros_(self.mechanical_head.net[-1].weight)
        nn.init.zeros_(self.mechanical_head.net[-1].bias)
        from .supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1
        self.impulse_geometry=BallImpulsePriorV1()
        import json
        from pathlib import Path
        self.ball_mass=float(json.loads((Path(__file__).resolve().parents[2]/'configs/shuttle_physics_v2.json').read_text())['mass_kg'])

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        # Parent adapter.load_state_dict recursively calls this hook; overriding
        # only public load_state_dict would leave resumed mechanical buffers unchecked.
        expected=self.state_dict()
        for key,value in expected.items():
            if key.startswith(('inertia.','impulse_geometry.')) or key in ('component_material','mechanical_scale'):
                saved=state_dict.get(prefix+key)
                if saved is None or not torch.equal(value.detach().cpu(),saved.detach().cpu()):
                    raise ValueError('Immutable mechanical contract differs: '+prefix+key)
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def mechanical_features(self, impact, geometry):
        from .supermimic_physical_dataset_v1 import decode_physical
        if geometry.asset_sha256!=self.material_contract['asset_sha256']:
            raise ValueError('Impact inertia/physical geometry asset mismatch')
        fields=decode_physical(impact)
        f=ball_geometry_features(impact,geometry)
        local_normal,part=self.impulse_geometry.racket_surface(f[:,6:9])
        local_point=f[:,6:9]-self.preview.radius*local_normal
        inverse,info=self.inertia(fields['joint_position'],fields['root_quaternion_w'],local_point)
        bodies=geometry(fields['joint_position'],fields['joint_velocity'],fields['root_position_w'],fields['root_quaternion_w'],fields['root_link_velocity_w'],fields['root_angular_velocity_w'])
        rid=geometry.ids['Racket'];normal=(bodies['rotation'][:,rid]@local_normal[...,None]).squeeze(-1)
        generalized=torch.cat((fields['root_link_velocity_w'],fields['root_angular_velocity_w'],fields['joint_velocity']),-1)
        surface=(info['point_jacobian']@generalized[...,None]).squeeze(-1)
        relative=fields['ball_velocity_w']-surface
        material=self.component_material[part]
        vn=(relative*normal).sum(-1)
        normal_inverse=(normal*(inverse@normal[...,None]).squeeze(-1)).sum(-1)
        normal_dv=-(1+material[:,2])*vn.clamp_max(0)/(1+self.ball_mass*normal_inverse)
        features=torch.cat((normal,relative,surface,inverse.flatten(1),material,normal_dv[:,None]*normal),-1)
        if features.shape!=(len(impact),24) or not torch.isfinite(features).all():
            raise FloatingPointError('Invalid causal articulated material features')
        return features,part

    def forward(self,state,predicted_robot,free_ball,geometry,present,*,racket_probability,details=False):
        from .supermimic_impact_timing_v1 import causal_sweep_time,interpolate_physical_state
        current=ball_geometry_features(state,geometry)
        endpoint=torch.cat((predicted_robot[:,:49],free_ball),-1)
        future=ball_geometry_features(endpoint,geometry)
        time,timing=causal_sweep_time(current[:,6:9],current[:,9:12],future[:,6:9],self.preview)
        impact=interpolate_physical_state(state,endpoint,time,self.preview.times[-1])
        # Evaluate the same V2 correction on the same causal impact state as V3.
        legacy,metadata=RacketModeActionBallV2.forward(self,state,impact,impact[:,49:],geometry,present,
            racket_probability=racket_probability,details=True)
        features,part=self.mechanical_features(impact,geometry)
        extra=self.mechanical_head((features/self.mechanical_scale).clamp(-30.,30.))*self.output_scale
        extra=extra*metadata['effective_support'][:,None]*present.reshape(-1,1).to(state.dtype)
        result=legacy+extra
        if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite material-conditioned ball correction')
        if details:return result,dict(metadata,impact_time_s=time,material_part=part,mechanical_features=features,**timing)
        return result


def migrate_material_inertia_v4(source,material_contract):
    """Explicit V3 migration. Check every legacy tensor; never strict=False."""
    if type(source) is not ImpactTimeActionBallV3:
        raise ValueError('Native-material migration requires explicit V3 source')
    candidate=MaterialInertiaActionBallV4(material_contract=material_contract,
        softness_m=source.softness_m).to(next(source.parameters()).device)
    old=source.state_dict();complete=candidate.state_dict()
    for key,value in old.items():
        if key not in complete or complete[key].shape!=value.shape:
            raise ValueError('Legacy migration field mismatch: '+key)
        complete[key]=value
    candidate.load_state_dict(complete,strict=True)
    return candidate


def action_ball_from_checkpoint(checkpoint):
    kwargs={}
    if checkpoint.get('schema')==MaterialInertiaActionBallV4.schema:
        if 'material_contract' not in checkpoint:raise ValueError('Missing native material receipt in V4 checkpoint')
        kwargs['material_contract']=checkpoint['material_contract']
    return action_ball_for_schema(checkpoint.get('schema'),**kwargs)


def action_ball_for_schema(schema, **kwargs):
    """Explicit dispatch: old checkpoints retain their original computation."""
    classes = {cls.schema: cls for cls in (ActionConditionedBallV1, RacketModeActionBallV2, ImpactTimeActionBallV3, MaterialInertiaActionBallV4)}
    if schema not in classes:
        raise ValueError('Unknown action-ball checkpoint schema: '+str(schema))
    return classes[schema](**kwargs)
