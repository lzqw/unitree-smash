"""Native-only Approach sampling labels. Never inputs to actor/model rollout."""
import hashlib
import json

import torch

APPROACH_QUOTA = dict(uniform=24, approach_active=32, hit_prepare=24,
                     contact_release=24, approach_failure=16, post_hit=8)
FIELDS = ('approach_gate', 'racket_distance', 'had_contact', 'released',
          'qualified', 'frame_hit', 'attack', 'crossed_net', 'landed_target')


def approach_quick_panel(training_cases):
    """Predeclared half of the ordered TRAIN128 suite for repeated native checks."""
    if len(training_cases) != 128 or len({row['case_id'] for row in training_cases}) != 128:
        raise ValueError('Quick-check requires the original ordered TRAIN128')
    indices = list(range(0, 128, 2))
    cases = [training_cases[index] for index in indices]
    digest = hashlib.sha256(json.dumps(cases, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return cases, dict(schema='approach-train64-quick-v1', sha256=digest, source='TRAIN128',
                       source_case_indices=indices, stage_end_holdout=False)

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


def early_approach_start(observation, lane_activity):
    """Include causal early starts that can reach the swing within the rollout."""
    if isinstance(observation, dict):
        observation = observation['policy']
    displacement = observation[:, 61:64] - observation[:, 45:48]
    relative = observation[:, 64:67] - observation[:, 51:54]
    lead = -(displacement * relative).sum(-1) / relative.square().sum(-1).clamp_min(1e-6)
    closest = displacement + lead.clamp(0., .4)[:, None] * relative
    return ((lane_activity.to(observation.device).reshape(-1) > .25)
            | ((observation[:, 67] > .5) & (lead > 0.) & (lead <= .4) & (closest.norm(dim=-1) < .6)))


class NativeArmIdentification:
    """Small signed command pulses on a rotating subset of training lanes."""
    def __init__(self, lanes, device, excluded, *, ticks=4, amplitude=.01, stride=8):
        self.ticks, self.amplitude, self.stride = ticks, amplitude, stride
        self.lanes = torch.arange(lanes, device=device)
        self.allowed = torch.ones(lanes, dtype=torch.bool, device=device)
        self.allowed[list(excluded)] = False
        self.episode = torch.full_like(self.lanes, -1)
        self.remaining = torch.zeros_like(self.lanes)
        self.pulse = torch.zeros((lanes, 6), device=device)
        self.excited = torch.zeros_like(self.allowed)
        self.active = torch.zeros_like(self.allowed)

    def apply(self, action, observation, episode, had_contact):
        reset = episode != self.episode
        self.episode.copy_(episode)
        self.excited &= ~reset
        self.remaining.masked_fill_(reset | had_contact, 0)
        candidates = self.allowed & ((self.lanes + episode) % self.stride == 0) & ~had_contact
        approaching = early_approach_start(observation, torch.zeros_like(self.lanes))
        start = candidates & ~self.excited & approaching
        index = start.nonzero().flatten()
        if len(index):
            number = (self.lanes[index] // self.stride) + episode[index]
            self.pulse[index] = 0
            self.pulse[index, number % 6] = self.amplitude * (1 - 2 * ((number // 6) % 2))
        self.remaining.masked_fill_(start, self.ticks)
        self.active.copy_(self.remaining > 0)
        self.excited |= self.active
        self.remaining.sub_(self.active.long())
        arm = action[:, 12:]
        arm = torch.where(self.active[:, None], (arm + self.pulse.to(action)).clamp(-1., 1.), arm)
        return torch.cat((action[:, :12], arm), dim=-1)


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

def prioritize_flight_gradient(parameters, flight_gradients):
    """Remove uphill flight interference while retaining transverse gradients."""
    combined = torch.cat([torch.zeros_like(p).flatten() if p.grad is None else p.grad.flatten()
                          for p in parameters])
    flight = torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten()
                        for p, g in zip(parameters, flight_gradients, strict=True)])
    norm_squared = flight.square().sum()
    alignment = torch.dot(combined, flight)
    # The bisector makes both costs descend to first order when their
    # gradients conflict and are not exactly opposed.
    coefficient = float(torch.where(
        (alignment < 0) & (norm_squared > 1e-20),
        combined.norm() / norm_squared.clamp_min(1e-20).sqrt(), 0.))
    if coefficient:
        for parameter, gradient in zip(parameters, flight_gradients, strict=True):
            if gradient is None:
                continue
            if parameter.grad is None:
                parameter.grad = gradient.detach() * coefficient
            else:
                parameter.grad.add_(gradient.detach(), alpha=coefficient)
    return coefficient


def comparable_flight_merit(before, after, baseline_loss, candidate_loss):
    """Compare flight losses on the exact baseline releases during line search."""
    if before.shape != after.shape:
        raise ValueError('Flight line search requires the same packed start lanes')
    mask = before.isfinite()
    count = int(mask.sum())
    if not count:
        return candidate_loss < baseline_loss, dict(matched_lanes=0)
    initial = float(before[mask].mean())
    comparable = after[mask]
    if not comparable.isfinite().all():
        return False, dict(matched_lanes=count, before=initial, after=None, lost_release=True)
    trial = float(comparable.mean())
    accepted = trial <= initial and candidate_loss < baseline_loss
    return accepted, dict(matched_lanes=count, before=initial, after=trial, lost_release=False)


def approach_guard_merit(before, after, *, flight_tolerance=.002):
    """Preserve aggregate flight and contact quality on held-out proposal starts."""
    old, new = (rollout['flight_cost_by_lane'] for rollout in (before, after))
    margins, candidate_margins = (rollout['string_margin_by_lane'] for rollout in (before, after))
    if old.shape != new.shape or margins.shape != candidate_margins.shape or old.shape != margins.shape:
        raise ValueError('The guard must score the same lanes for each candidate')
    released = torch.isfinite(old)
    candidate_released = torch.isfinite(new)
    n_release = int(released.sum())
    lost = released & ~candidate_released
    # Lost predicted releases still contribute a unit cost, even if a different
    # lane gained an outgoing trajectory.
    candidate = torch.where(candidate_released, new, old + 1.)
    old_flight = float(old[released].mean()) if n_release else None
    new_flight = float(candidate[released].mean()) if n_release else None
    margin = torch.isfinite(margins) & (margins >= .010)
    n_margin = int(margin.sum())
    old_q25 = float(margins[margin].quantile(.25)) if n_margin else None
    compared = candidate_margins.nan_to_num(nan=-.015)
    new_q25 = float(compared[margin].quantile(.25)) if n_margin else None
    count = int(torch.isfinite(margins).sum())
    candidate_count = int(torch.isfinite(candidate_margins).sum())
    allowed_losses = max(1, count // 20)
    flight_ok = not n_release or new_flight <= old_flight + flight_tolerance
    margin_ok = not n_margin or new_q25 >= old_q25 - .003
    count_ok = candidate_count + allowed_losses >= count
    objective_ok = float(after['loss']) < float(before['loss'])
    return bool(flight_ok and margin_ok and count_ok and objective_ok), dict(
        matched_lanes=n_release, before=old_flight, after=new_flight,
        lost_release=bool(lost.any()), lost_release_lanes=int(lost.sum()),
        legal_string_before=count, legal_string_after=candidate_count,
        stable_margin_lanes=n_margin, margin_q25_before_m=old_q25, margin_q25_after_m=new_q25,
        flight_ok=bool(flight_ok), margin_ok=bool(margin_ok), count_ok=count_ok, objective_ok=objective_ok)


def approach_step_sizes(previous):
    """Try the last successful scale and its neighbors before distant scales."""
    scales = (1., .5, .25, .125, .0625)
    if previous not in scales:
        return scales
    index = scales.index(previous)
    return tuple(scales[i] for i in sorted(range(len(scales)), key=lambda i: (abs(i - index), i)))


def gradient_parts(terms, parameters):
    """Diagnostics only; does not accumulate .grad or step an optimizer."""
    result={}; vectors={}
    for name,term in terms.items():
        grads=torch.autograd.grad(term,parameters,retain_graph=True,allow_unused=True) if term.requires_grad else [None]*len(parameters)
        vector=torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(parameters,grads)])
        if not torch.isfinite(vector).all(): raise FloatingPointError('Nonfinite '+name+' gradient')
        vectors[name]=vector; result[name+'_gradient_norm']=float(vector.norm())
    for left, right, key in (('task', 'tracking', 'task_tracking_cosine'),
                             ('ball_speed', 'ball_geometry', 'ball_speed_geometry_cosine')):
        if left in vectors and right in vectors:
            a, b = vectors[left], vectors[right]
            denominator = a.norm() * b.norm()
            cosine = float(torch.dot(a, b) / denominator.clamp_min(1e-20))
            result[key] = None if key == 'ball_speed_geometry_cosine' and float(denominator) <= 1e-20 else cosine
    return result
