"""Training-side native branches for Approach distillation and WM supervision."""
from copy import deepcopy
import math

import torch
from torch.nn import functional as F

from .supermimic_native_approach_search import validate_offsets
from .supermimic_physical_dataset_v1 import STATE_FIELDS, encode_physical
from .supermimic_physical_observation_v1 import capture_native_inputs


SCHEMA = 'supermimic-hybrid-native-v1'
BRANCHES = 4
QUICK_LANES = 64
FIRST_CAPTURE_TICK = 200
PERTURBATION_START = 226


def native_layout(cases, spec, holdout):
    offsets = validate_offsets(spec['offsets'], minimum=BRANCHES, maximum=BRANCHES)
    indices = spec['case_indices']
    if (len(cases) != 128 or len(indices) != 16 or len(set(indices)) != 16
            or any(type(i) is not int or not 0 <= i < 128 or i % 2 != 1 or i in holdout for i in indices)
            or any(offsets[0])):
        raise ValueError('Hybrid exploration needs 16 distinct training cases and an unmodified control')
    return [*cases[::2], *(cases[i] for _ in offsets for i in indices)]


def perturbation_envelope(tick):
    """Smooth, pre-impact-only pulse with zero offset on the common prefix."""
    x = max(0., min(1., (tick - PERTURBATION_START) / 82))
    return math.sin(math.pi * x) ** 2 if 0 < x < 1 else 0.


def native_state(env):
    physical = capture_native_inputs(env)
    keys = ('had_contact', 'released', 'outgoing_velocity')
    return dict(
        state=encode_physical({key: getattr(physical, key) for key, _ in STATE_FIELDS})[QUICK_LANES:],
        previous_action=physical.final_action[QUICK_LANES:],
        elapsed=physical.elapsed_s[QUICK_LANES:], present=physical.ball_present[QUICK_LANES:],
        feet=physical.foot_contact[QUICK_LANES:],
        episode=env.episode_index[QUICK_LANES:].clone(),
        **{key: getattr(env.rally, key)[QUICK_LANES:].clone() for key in keys},
    )


def pack_tapes(frames, outcomes, indices, launch, actor_sha256):
    """Match settled outcomes to the same lane's first-episode tape."""
    tape = {key: torch.stack([row[key] for row in frames]).cpu() for key in frames[0]}
    examples = {'teacher': [], 'preserve': []}
    effect_by_branch = []
    for branch in range(1, BRANCHES):
        nominal = slice(0, len(indices))
        alternate = slice(branch * len(indices), (branch + 1) * len(indices))
        valid = ((tape['episode'][:, nominal] == 0) & (tape['episode'][:, alternate] == 0)
                 & ~tape['had_contact'][:, nominal] & ~tape['had_contact'][:, alternate])
        target_delta = (tape['targets'][:, alternate] - tape['targets'][:, nominal]).abs().amax(-1)
        effect_by_branch.append(target_delta.masked_fill(~valid, 0.).amax(0))
    for i in range(len(indices)):
        nominal = outcomes[0]['rows'][i]
        if nominal['success']:
            examples['preserve'].append(i)
        useful = []
        for branch in range(1, BRANCHES):
            result = outcomes[branch]['rows'][i]
            if result['case_id'] != nominal['case_id']:
                raise ValueError('Teacher is not matched to the nominal case')
            executed = effect_by_branch[branch - 1][i] > 1e-5
            safe = (executed and result['string'] and not result['robot_net'] and not result['ball_net']
                    and not result['low_speed'])
            if safe and (result['success'] and not nominal['success']
                         or not nominal['string'] and (result['success'] or result['landed_target'])
                         or nominal['success'] and result['success']
                         and result['margin_score'] >= nominal['margin_score'] + .08):
                useful.append((result['success'], result['margin_score'], branch))
        if useful:
            examples['teacher'].append(max(useful)[2] * len(indices) + i)
    return dict(schema=SCHEMA, case_indices=indices, actor_sha256=actor_sha256,
                launch=launch, outcomes=outcomes, examples=examples, tape=tape,
                native_command_delta_rad=torch.stack(effect_by_branch))


def load_training_data(path, holdout):
    data = torch.load(path, map_location='cpu', weights_only=True)
    indices = data['case_indices']
    if (data['schema'] != SCHEMA or len(indices) != 16 or len(set(indices)) != 16
            or any(i % 2 != 1 or not 0 <= i < 128 or i in holdout for i in indices)):
        raise ValueError('Hybrid tape includes an ineligible original training lane')
    tape = data['tape']
    data['supervision'] = {}
    for name in ('teacher', 'preserve'):
        members = data['examples'][name]
        if not members:
            continue
        valid = ((tape['episode'][:, members] == 0) & ~tape['had_contact'][:, members]
                 & (tape['gate'][:, members] > .1) & tape['present'][:, members])
        data['supervision'][name] = {
            key: tape[key][:, members][valid] for key in ('observation', 'request', 'targets')}
    return data


def native_actor_loss(actor, samples, attempt):
    parameter = next(actor.parameters())
    loss, stats = None, {}
    for name, weight in (('teacher', .20), ('preserve', .25)):
        group = samples.get(name)
        if not group or not len(group['request']):
            continue
        n = len(group['request'])
        ids = (torch.arange(min(n, 512)) + 512 * max(attempt - 1, 0)) % n
        observation = group['observation'][ids].to(parameter)
        request = group['request'][ids].to(parameter)
        # The teacher supplies its own observation, including action history.
        # Use the request at the unchanged G07 interface; WM uses actual PD targets.
        no_recovery = torch.zeros(len(observation), dtype=torch.bool, device=observation.device)
        prediction = actor.forward_with_recovery_gate(observation, no_recovery, sample=False)[0].clamp(-1., 1.)
        term = F.smooth_l1_loss(prediction[:, 12:], request[:, 12:], beta=.02)
        loss = weight * term if loss is None else loss + weight * term
        stats[f'native_{name}_frames'] = len(ids)
        stats[f'native_{name}_loss'] = float(term.detach())
    return loss, stats


@torch.no_grad()
def flight_validation(data, prior, court, device):
    """Check free flight from a measured release, with no learned Body transition."""
    from .supermimic_outgoing_flight import predict_flight_events

    tape = data['tape']
    released = tape['released'][1:] & ~tape['released'][:-1] & (tape['episode'][1:] == 0)
    ticks, lanes = released.nonzero(as_tuple=True)
    truth = data['flight_truth']
    if not len(lanes):
        return dict(releases=0)
    positions = tape['state'][ticks + 1, lanes, 49:52].to(device)
    velocities = tape['outgoing_velocity'][ticks + 1, lanes].to(device)
    if not torch.isfinite(positions).all() or not torch.isfinite(velocities).all():
        raise ValueError('Nonfinite first-release state')
    events = predict_flight_events(positions, velocities, prior, court)
    predicted_net = events['net_contact'].cpu()
    native_net = truth['ball_net'][lanes]
    predicted = events['landing_position'].cpu()
    native_landing = truth['landing'][lanes]
    valid = predicted.isfinite().all(-1) & native_landing.isfinite().all(-1) & ~native_net & ~predicted_net
    landing_rmse = (predicted[valid, :2] - native_landing[valid, :2]).square().sum(-1).mean().sqrt()
    return dict(releases=len(lanes), native_net_contacts=int(native_net.sum()),
                missed_net_contacts=int((native_net & ~predicted_net).sum()),
                extra_net_contacts=int((~native_net & predicted_net).sum()),
                clean_landings=int(valid.sum()), landing_rmse_m=float(landing_rmse) if valid.any() else None)


def refresh_world(trainer, data, device, verify):
    """Fit the measured branch tape, then commit only after fixed native validation."""
    tape = data['tape']
    transition = ((tape['episode'][:-1] == 0) & (tape['episode'][1:] == 0))
    before_contact = ~tape['had_contact'][:-1] & ~tape['had_contact'][1:]
    window = {}
    for key in ('state', 'targets', 'previous_action', 'feet', 'present', 'elapsed'):
        window[key] = tape[key][:-1].to(device)
    window['next_state'] = tape['state'][1:].to(device)
    launch = data['launch'].to(device)
    flight = flight_validation(data, trainer.adapter.action_ball_model.impulse_prior,
                               trainer.research_court, device)
    model, optimizer = trainer.adapter, trainer.world_opt
    saved = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()), trainer.world_updates
    body_report, contact_report = [], []
    paired_report = None
    try:
        # Use an uninterrupted pre-contact window from every available branch,
        # including near-misses, frame hits and successful impacts.
        for horizon in (20, 32):
            in_window = tape['elapsed'][:-1] < 3.15
            eligibility = ((transition & before_contact & tape['present'][:-1] & in_window)
                           .unfold(0, horizon, 1).all(-1))
            lanes = eligibility.any(0).nonzero().flatten()
            if len(lanes):
                ends = len(eligibility) - 1 - eligibility[:, lanes].flip(0).int().argmax(0)
                steps = ends[None] + torch.arange(horizon)[:, None]
                batch = {key: value[steps, lanes] for key, value in window.items()}
                fit = trainer.fit_event_aligned_approach_body(batch, launch[lanes])
                body_report.append(dict(horizon=horizon, branches=len(lanes), **fit))
        # The existing contact fit only supervises complete first contact to
        # release segments. Never cross an autoreset in either array.
        fields = trainer.adapter.task_model.memory_schema.fields
        columns, width = {}, 0
        for field in fields:
            columns[field.path] = slice(width, width + field.width)
            width += field.width
        def contact_window(offset):
            lanes = slice(offset, offset + 16)
            contact = {key: value[:, lanes] for key, value in window.items()}
            values = window['state'].new_zeros((len(tape['episode']), launch[lanes].shape[0], width))
            valid = torch.zeros_like(values, dtype=torch.bool)
            for key in ('had_contact', 'released', 'outgoing_velocity'):
                column = columns['rally/' + key]
                field = tape[key][:, lanes].to(device)
                if field.ndim == 2:
                    field = field.unsqueeze(-1)
                original = (tape['episode'][:, lanes] == 0).to(device).unsqueeze(-1)
                values[..., column] = torch.where(original, field.nan_to_num(), 0.)
                valid[..., column] = field.isfinite() & original
            return dict(contact, values=values[:-1], next_values=values[1:], next_valid=valid[1:])

        contacts = [contact_window(offset) for offset in range(0, launch.shape[0], 16)]
        start = PERTURBATION_START - FIRST_CAPTURE_TICK
        paired = []
        for index in (3, 1):
            batch = {key: value[start:start + 32] for key, value in contacts[index].items()}
            batch['present'] = batch['present'] & transition[start:start + 32, index*16:(index+1)*16].to(device)
            codes = [0 if row['low_speed'] else
                     1 if row['success'] and row['speed_margin_mps'] is not None and row['speed_margin_mps'] < .25
                     else 2 if row['success'] else 3 for row in data['outcomes'][index]['rows']]
            batch['terminal_response_group'] = torch.tensor([codes], device=device)
            paired.append(batch)
        paired_report = trainer.fit_approach_body_pair(*paired, launch[:16])
        for index, contact in enumerate(contacts):
            contact_report.append(trainer.fit_ball_only_contact(contact, launch[index*16:(index+1)*16]))
        validation = verify()
    except BaseException:
        model.load_state_dict(saved[0])
        optimizer.load_state_dict(saved[1])
        trainer.world_updates = saved[2]
        raise
    if not validation['passed']:
        model.load_state_dict(saved[0])
        optimizer.load_state_dict(saved[1])
        trainer.world_updates = saved[2]
        if not verify()['passed']:
            raise RuntimeError('Restored WM failed fixed native forward validation')
    return dict(fixed_native_forward=validation, committed=validation['passed'],
                body=body_report, body_paired=paired_report, contact=contact_report, flight_native_branch_check=flight,
                branches=launch.shape[0], actor_supervision={k: len(v['request']) for k, v in data['supervision'].items()})
