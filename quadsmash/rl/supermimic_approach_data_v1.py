"""Native-only Approach sampling labels. Never inputs to actor/model rollout."""
import torch

APPROACH_QUOTA = dict(uniform=24, approach_active=32, hit_prepare=24,
                     contact_release=24, approach_failure=16, post_hit=8)
FIELDS = ('approach_gate', 'racket_distance', 'had_contact', 'released',
          'qualified', 'frame_hit', 'attack', 'crossed_net', 'landed_target')

def active_gate(gate, threshold=.25, minimum_batch_lanes=4):
    gate = gate.detach().reshape(-1)
    if not len(gate):return False
    return bool((gate > threshold).sum() >= min(minimum_batch_lanes, len(gate)))

@torch.no_grad()
def gate_on_actor_device(actor, observation):
    """Archive tensors stay on CPU; only the gate input moves to actor device."""
    if isinstance(observation, dict): observation = observation['policy']
    if observation.ndim != 2 or observation.shape[-1] != 794:
        raise ValueError('Expected [batch,794] public observation')
    parameter = next(iter(actor.parameters()), None) if hasattr(actor, 'parameters') else None
    # Stateless gates have no device-bearing weights; preserve their input.
    input_value = observation if parameter is None else observation.to(device=parameter.device, dtype=parameter.dtype)
    value = actor.approach_gate(input_value)
    if value.numel() != len(observation) or not torch.isfinite(value).all():
        raise ValueError('Invalid Approach gate shape or values')
    return value.reshape(-1)

def _task_labels(task, unknown):
    def field(name):
        return task[name].float() if name in task else unknown
    qualified = field('legal_initial_contact') * field('string_hit') * field('airborne_string_hit')
    velocity = task.get('outgoing_velocity')
    attack = unknown
    if velocity is not None:
        attack = qualified * field('released') * (velocity.norm(dim=-1) >= 5.) * (velocity[..., 0] > 0.)
        attack = torch.where(torch.isfinite(velocity).all(-1) | (qualified == 0) | (field('released') == 0),
                             attack, unknown)
    return dict(had_contact=field('had_contact'), released=field('released'), qualified=qualified,
                frame_hit=field('frame_hit'), attack=attack, crossed_net=field('crossed_net'),
                landed_target=field('landed_target'))


def _racket_distance(physical, geometry):
    kin = geometry(physical['joint_position'], physical['joint_velocity'], physical['root_position_w'],
                   physical['root_quaternion_w'], physical['root_link_velocity_w'], physical['root_angular_velocity_w'])
    return (kin['position'][:, geometry.ids['Racket']] - physical['ball_position_w']).norm(dim=-1)


@torch.no_grad()
def native_approach_metadata(row, actor, geometry):
    before = row['before']
    result = dict(approach_gate=gate_on_actor_device(actor, row['observation']).cpu(),
                  racket_distance=_racket_distance(before['physical'], geometry).cpu())
    for prefix, record in (('', before), ('next_', row['after'])):
        q = record['physical']['joint_position']
        labels = _task_labels(record['task'], q.new_full((len(q),), float('nan')))
        result.update({prefix+k:v.detach().cpu() for k,v in labels.items()})
    return result


@torch.no_grad()
def recorded_approach_metadata(tensors, schema, actor, geometry):
    """Reconstruct replay labels from recorded observations and native value/valid pairs."""
    from .supermimic_physical_dataset_v1 import decode_physical
    state = tensors['state']
    shape = state.shape[:2]
    unknown = state.new_full(shape, float('nan'))
    result = {}
    for prefix in ('', 'next_'):
        values, valid = tensors[prefix+'values'], tensors[prefix+'valid']
        if values.shape != (*shape, schema.width) or valid.shape != values.shape:
            raise ValueError('Recorded Approach task fields have incompatible dimensions')
        task = {}
        offset = 0
        for field in schema.fields:
            if field.path.startswith('rally/'):
                section = slice(offset, offset+field.width)
                value = values[..., section].reshape(*shape, *field.shape)
                mask = valid[..., section].reshape_as(value) > .5
                task[field.path.removeprefix('rally/')] = value.masked_fill(~mask, float('nan'))
            offset += field.width
        result.update({prefix+k:v.cpu() for k,v in _task_labels(task, unknown).items()})
    observation = tensors['observation']
    if observation.shape != (*shape,794):
        raise ValueError('Recorded gate requires the original per-lane public observation')
    geom_device = next(geometry.buffers(), state).device
    gates, distances = [], []
    for obs, physical in zip(observation.flatten(0,1).split(1024), state.flatten(0,1).split(1024)):
        gates.append(gate_on_actor_device(actor, obs).cpu())
        distances.append(_racket_distance(decode_physical(physical.to(geom_device)), geometry).cpu())
    result['approach_gate'] = torch.cat(gates).reshape(shape)
    result['racket_distance'] = torch.cat(distances).reshape(shape)
    return result

def approach_window_masks(metadata, valid, steps, *, gate_threshold=.25, min_gate_steps=4, near_distance=.30):
    """Unknown historical fields stay unknown; no terminal miss is invented.

    Near miss requires a *completed* same-episode suffix in replay. Before
    completion the close/no-hit sample remains hit_prepare, not a failure.
    """
    gate=metadata['approach_gate']; near=metadata['racket_distance']<=near_distance
    pre=(metadata['had_contact']<.5)
    def any_window(x): return x.unfold(0,steps,1).any(-1)&valid
    contact=(metadata['next_had_contact']>.5)&(metadata['had_contact']<.5)
    release=(metadata['next_released']>.5)&(metadata['released']<.5)
    illegal=((metadata['next_frame_hit']>.5)&(metadata['frame_hit']<.5)) | (contact&(metadata['next_qualified']<.5))
    bad_out=release & (metadata['next_qualified']>.5)&(metadata['next_attack']<.5)
    miss=metadata.get('completed_miss',torch.zeros_like(pre)) & near & pre
    failed_net=metadata.get('failed_net_crossing',torch.zeros_like(pre))
    failed_target=metadata.get('failed_target_landing',torch.zeros_like(pre))
    return dict(approach_active=((gate>gate_threshold).unfold(0,steps,1).sum(-1)>=min(min_gate_steps,steps))&valid,
                hit_prepare=any_window(near & pre), contact_release=any_window(contact|release),
                approach_failure=any_window(illegal|bad_out|miss|((failed_net|failed_target)&(pre|contact|release))),post_hit=any_window(metadata['released']>.5),
                near_miss=any_window(miss),illegal_contact=any_window(illegal),invalid_outgoing=any_window(bad_out),failed_net_crossing=any_window(failed_net&(pre|contact|release)),failed_target_landing=any_window(failed_target&(pre|contact|release)))

def masked_lane_mean(value, weights):
    weights=weights.detach().reshape(-1).to(value)
    value=value.reshape(len(weights),-1).mean(-1)
    return (value*weights).sum()/weights.sum().clamp_min(1.)

def gradient_parts(terms, parameters):
    """Diagnostics only; does not accumulate .grad or step an optimizer."""
    result={}; vectors={}
    for name,term in terms.items():
        grads=torch.autograd.grad(term,parameters,retain_graph=True,allow_unused=True) if term.requires_grad else [None]*len(parameters)
        vector=torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(parameters,grads)])
        if not torch.isfinite(vector).all(): raise FloatingPointError('Nonfinite '+name+' gradient')
        vectors[name]=vector; result[name+'_gradient_norm']=float(vector.norm())
    a,b=vectors['task'],vectors['tracking']
    result['task_tracking_cosine']=float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-20))
    return result
