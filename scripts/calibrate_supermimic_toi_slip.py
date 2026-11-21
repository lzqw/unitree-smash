#!/usr/bin/env python3
"""Fit a bounded tangential response at the deployable ball prior's first racket impact."""
import argparse
import json
from pathlib import Path

import torch

from evaluate_supermimic_contact_rollout import racket, sha
from fit_supermimic_native_racket_toi_head import fixture, free_rollout, g07, metrics, prediction, predicted_state, prepare
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1


class TOISlip:
    """Velocity-unit sticking impulse capped by a fraction of the predicted normal impulse."""

    def __init__(self, friction, sticking, residual=None):
        self.residual = residual
        self.friction = friction
        self.sticking = sticking

    def toi_response(self, row, present):
        normal = row['normal']
        relative = row['relative_velocity']
        slip = relative - (relative * normal).sum(-1, keepdim=True) * normal
        bound = self.friction * row['impulse_dv'].norm(dim=-1, keepdim=True)
        scale = torch.minimum(torch.full_like(bound, self.sticking), bound / slip.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        active = present.reshape(-1).bool() & row['active'].reshape(-1).bool() & (row['object_id'] == 0)
        dv = -slip * scale
        if self.residual is not None:
            dv = dv + self.residual.toi_response(row, present)['impulse_residual_dv']
        return dict(impulse_residual_active=active, impulse_residual_dv=dv * active[:, None])


def integrated_friction(root, split, physics, prior):
    audit = json.loads((root / (split + '.json')).read_text())
    design = json.loads((root / 'design.json').read_text())
    seeds = sorted(int(s) for s, label in design['assignments'].items() if label == split)
    result = {}
    for branch in ('minus', 'plus'):
        labels = []
        for seed in seeds:
            path = root / branch / f'spec_{seed}' / 'trace.pt'
            if sha(path) != audit['archives'][str(path)]:
                raise ValueError('Native friction trace differs from the signed dataset')
            record = torch.load(path, weights_only=True, mmap=True, map_location='cpu')
            trace = record['raw']['substep_trace']
            forces = trace['tensors']['racket_friction_force_w'][1:]
            if forces.shape != (prior.steps, design['lanes'], 3) or abs(trace['physics_dt_s']-prior.sub_dt) > 1e-12:
                raise ValueError('Unexpected sensor frame count or time step')
            labels.append(forces.sum(0) * (prior.sub_dt / physics['mass_kg']))
        result[branch] = torch.cat(labels)
        if not torch.isfinite(result[branch]).all():
            raise ValueError('Nonfinite friction measurement')
    return result


def friction_errors(model, dataset, observations):
    rows, groups, _ = dataset
    residual = {}
    for branch, row in rows.items():
        estimate = model.toi_response(row, torch.ones_like(row['active']))['impulse_residual_dv']
        residual[branch] = estimate - observations[branch]
    paired = residual['plus'] - residual['minus']
    stable = torch.tensor([g == 'racket_stable' for g in groups])
    return dict(absolute=float(torch.cat(list(residual.values())).square().mean().sqrt()),
                signed=float(paired.square().mean().sqrt()), stable_signed=float(paired[stable].square().mean().sqrt()) if stable.any() else None)


def evaluate(prior, model, dataset):
    outputs = {b: prediction(model, prior, r) for b, r in dataset[0].items()}
    return metrics(dataset, outputs)


def signed_training_capture(root, development_root, prior, mass):
    from audit_supermimic_native_window_response import complete_window
    from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch

    reg = json.loads((root.parent / 'preregistration.json').read_text())
    start = json.loads((root / 'startup.json').read_text())
    dev = json.loads((development_root / 'startup.json').read_text())
    if (reg['role'] != 'training_action_identification' or len(start['specs']) != 1 or
            start['specs'][0]['sha256'] != reg['spec_sha256'] or start['source_sha256'] != reg['actor_checkpoint_sha256']):
        raise ValueError('Training capture differs from registration')
    training = set(start['specs'][0]['spec']['lanes'])
    reserved = set(reg['held_out_lanes'])
    if training & reserved or not set(dev['specs'][0]['spec']['lanes']) <= reserved:
        raise ValueError('G07 training and development cases overlap')
    rows, groups, repeats, by_lane, _, commands = g07(root, prior)
    signed = rows['plus']['target'] - rows['minus']['target']
    above_noise = signed.norm(dim=-1) > (3 * repeats.norm(dim=-1)).clamp_min(1e-8)
    comparable = torch.tensor([kind in ('racket_stable','toi_switch','manifold_switch') for kind in groups])
    if 'nominal_mismatch' in by_lane:
        raise ValueError('Nominal repeat changed native contact timing or topology')
    lane_ids = commands['lanes']
    target_changed = torch.tensor(commands['max_joint_target_difference_rad']) > 0
    usable = (above_noise & comparable).reshape(-1, len(lane_ids)) & target_changed
    if not usable.any():
        raise ValueError('No loaded control interval resolves a signed native response above repeat noise with a changed target')
    supported_lanes = [lane for lane, ok in zip(lane_ids, usable.any(0).tolist()) if ok]
    stable = torch.tensor([kind == 'racket_stable' for kind in groups]).reshape_as(usable)
    stable_lanes = [lane for lane, ok in zip(lane_ids, (usable & stable).any(0).tolist()) if ok]
    labels = {}
    status = json.loads((root / 'status.json').read_text())
    for b in rows:
        run = _load_branch(root / status['completed'][0]['spec'], b)
        ts = complete_window(run['raw'], run['spec'], prior.steps)
        labels[b] = torch.cat([tr['tensors']['racket_friction_force_w'][1:].sum(0) * (prior.sub_dt/mass) for tr in ts])
        if labels[b].shape != rows[b]['target'].shape or not torch.isfinite(labels[b]).all():
            raise ValueError('G07 friction labels do not match native transitions')
    return (rows, groups, repeats), labels, dict(status_sha256=sha(root/'status.json'), commands=commands,
        lane_counts={k:by_lane.count(k) for k in sorted(set(by_lane))},
        control_interval_counts={k:groups.count(k) for k in sorted(set(groups))},
        supported_racket_lane_ticks=int(usable.sum()), supported_racket_lane_ids=supported_lanes,
        supported_stable_contact_lane_ids=stable_lanes)


def body_transfer(root, checkpoint, prior, model, groups, repeats, sequences, commands, replay=None, source_stage=None, measured=None, racket_outputs=None, body_anchor_interval=None):
    from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch
    from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical
    from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2

    status = json.loads((root / 'status.json').read_text())
    folder = root / status['completed'][0]['spec']
    ck = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    task = ck['source_manifest']['task_contract']
    lanes = commands['lanes']
    if set(lanes) & set(ck['calibration']['train_lanes']):
        raise ValueError('Body training and G07 development lanes overlap')
    body = PhysicalWorldV2(task['asset']['sha256']).eval()
    weights = ck['model']
    replay_identity = None
    if replay:
        from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
        fit = json.loads((replay / 'fit.json').read_text())
        stage = torch.load(source_stage, map_location='cpu', mmap=True, weights_only=False)
        metadata = stage['metadata']['physical_imagination']
        if (sha(source_stage) != fit['stage_checkpoint_sha256'] or
                metadata['checkpoint_sha256']['physical'] != sha(checkpoint) or
                not set(lanes) <= set(fit['excluded_lanes'])):
            raise ValueError('Online Body source or excluded development lanes differ')
        world = torch.load(replay / 'online_world.pt', map_location='cpu', mmap=True, weights_only=True)
        replay_identity = tensor_digest(world.items())
        if replay_identity != fit['online_world_sha256']:
            raise ValueError('Online Body candidate differs from its fit')
        weights = {k.removeprefix('physical_model.'): v for k,v in world.items() if k.startswith('physical_model.')}
    body.load_state_dict(weights, strict=True)
    suite = Path(task['suite']['path'])
    if sha(suite) != task['suite']['sha256']:
        raise ValueError('Different native launch suite')
    launch = torch.tensor([r['launch']['position_m'] + r['launch']['velocity_mps']
                           for r in json.loads(suite.read_text())['cases']])[lanes]
    predictions, targets, first_by_branch = {}, {}, {}
    for branch, seq in sequences.items():
        # g07() verified the branch hashes, native frames and signed executed targets.
        run = _load_branch(folder, branch)
        if run['task_contract'] != task or run['spec']['lanes'] != lanes:
            raise ValueError('Body and measured-racket input contracts differ')
        raw = run['raw']
        initial = raw['start']['before']['physical']
        state = encode_physical(initial)[lanes]
        torch.testing.assert_close(state[:, 49:], seq[0][0])
        previous, feet, elapsed, present = [initial[k][lanes] for k in
                                            ('final_action', 'foot_contact', 'elapsed_s', 'ball_present')]
        first = torch.full((len(lanes),), float('inf'))
        if racket_outputs is not None:
            racket_outputs[branch] = []
        for i, (control, (_, label, native_racket)) in enumerate(zip(raw['step_sequence'], seq, strict=True)):
            if i and body_anchor_interval and i % body_anchor_interval == 0:
                observed = raw['step_sequence'][i-1]['next_state'][lanes]
                state = torch.cat((observed[:, :49], state[:, 49:]), 1)
            robot, contact, present_next = body(state, control['final_joint_targets'][lanes],
                                                previous, feet.float(), present, elapsed, launch)
            if not present.all() or not present_next.all():
                raise ValueError('Unexpected launch boundary in the Body oracle')
            stream = [torch.stack((v, w)) for v, w in zip(racket(state, body.geometry), racket(robot, body.geometry), strict=True)]
            if racket_outputs is not None:
                racket_outputs[branch].append(tuple(v[-1] for v in stream))
            if measured:
                # Diagnostic exchange of pose and twist independently. Only interval endpoints enter the prior.
                replace = {'center': (0,), 'rotation': (1,), 'pose': (0,1), 'twist': (2,3), 'both': (0,1,2,3)}[measured]
                for k in replace:
                    stream[k] = native_racket[k]
            row = prepare(prior, state[:, 49:], label, stream)
            loaded = row['active'] & (row['object_id'] == 0)
            first = torch.where(loaded & first.isinf(), i * prior.dt + row['time'], first)
            state = torch.cat((robot[:, :49], predicted_state(model, prior, row)), 1)
            previous, feet = control['final_action'][lanes], contact.sigmoid()
            elapsed, present = elapsed + prior.dt, present_next
        first_by_branch[branch] = [float(t) if torch.isfinite(t) else None for t in first]
        targets[branch] = {'target': seq[-1][1]}
        predictions[branch] = state[:, 52:55]
    native = targets['plus']['target'] - targets['minus']['target']
    delta = predictions['plus'] - predictions['minus']
    return dict(base_body_sha256=sha(checkpoint), replay_world_sha256=replay_identity, metrics=metrics((targets, groups, repeats[-len(lanes):]), predictions),
                lane_ids=lanes, native_delta_mps=native.tolist(), predicted_delta_mps=delta.tolist(),
                first_racket_impact_time_s=first_by_branch)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('fixture', 'calibration', 'g07-development', 'output'):
        p.add_argument('--' + arg, type=Path, required=True)
    p.add_argument('--body', type=Path, help='Compare the same ball model with a held-lane Body WM under recorded G07 targets')
    p.add_argument('--g07-training', type=Path)
    p.add_argument('--racket-oracles', action='store_true', help='Exchange native pose and twist independently in the Body rollout')
    p.add_argument('--online-body', type=Path)
    p.add_argument('--source-stage', type=Path)
    a = p.parse_args()
    if bool(a.source_stage) != bool(a.online_body) or ((a.online_body or a.racket_oracles) and not a.body):
        p.error('--online-body needs its --source-stage and --body')
    torch.set_num_threads(2)
    cfg = json.loads(a.calibration.read_text())
    prior = BallImpulsePriorV1(normal_impulse_scale=cfg['fitted']['normal_impulse_scale']).eval()
    physics_file = Path(__file__).resolve().parents[1] / 'configs/shuttle_physics_v2.json'
    if sha(physics_file) != prior.preview.geometry_hashes['configs/shuttle_physics_v2.json']:
        raise ValueError('Inconsistent physical units')
    physics = json.loads(physics_file.read_text())
    train = fixture(a.fixture, 'train', prior)
    labels = integrated_friction(a.fixture, 'train', physics, prior)
    initial = TOISlip(0., 0.)
    reference = friction_errors(initial, train, labels)
    panels = [(train,labels,reference)]
    training_g07 = None
    if a.g07_training:
        gtrain, glabels, training_g07 = signed_training_capture(a.g07_training,a.g07_development,prior,physics['mass_kg'])
        panels.append((gtrain,glabels,friction_errors(initial,gtrain,glabels)))
    if any(ref['absolute'] <= 0 or ref['signed'] <= 0 for _,_,ref in panels):
        raise ValueError('No measurable signed friction in a requested calibration source')
    grid = []
    for mu in (0., .04, .08, .12, .16, .24, .32, .48):
        for sticking in (.125, .25, 2/7, .375, .5, .75, 1.):
            source_errors = []
            source_scores = []
            for dataset, observation, ref in panels:
                errors = friction_errors(TOISlip(mu, sticking), dataset, observation)
                source_errors.append(errors)
                # A source with no stable signed signal still trains absolute and signed load.
                keys = [k for k, value in ref.items() if value is not None and value > 0]
                source_scores.append(sum(errors[k]/ref[k] for k in keys)/len(keys))
            grid.append(dict(friction=mu, sticking=sticking, score=sum(source_scores)/len(panels), source_errors=source_errors))
    selected = min(grid, key=lambda candidate: candidate['score'])
    fitted = TOISlip(selected['friction'], selected['sticking'])
    dev = fixture(a.fixture, 'development', prior)
    observed = integrated_friction(a.fixture, 'development', physics, prior)
    development = {name: dict(friction=friction_errors(model, dev, observed), shuttle=evaluate(prior, model, dev))
                   for name, model in [('analytic', initial), ('fitted', fitted)]}
    _, _, repeats, groups, trajectories, commands = g07(a.g07_development, prior)
    transfer = free_rollout(fitted, prior, trajectories, groups, repeats, commands)
    checks = {}
    for label, base, new in [('fixture', development['analytic']['shuttle'], development['fitted']['shuttle']),
                             ('g07', transfer['baseline'], transfer['candidate'])]:
        signed = new['stable_signed']
        checks[label] = dict(endpoint=new['absolute_rmse'] <= base['absolute_rmse'],
                             signed=new['all_signed']['absolute_rmse'] <= base['all_signed']['absolute_rmse'],
                             stable_error=signed is not None and signed['absolute_rmse'] < base['stable_signed']['absolute_rmse'],
                             direction=bool(signed and signed.get('cosine_median', -1) >= .8 and signed.get('cosine_p10', -1) >= .5),
                             gain=bool(signed and .5 <= signed.get('gain_median', 0) <= 2),
                             no_contact=(new['no_contact_signed_rmse'] is not None and
                                         new['no_contact_signed_rmse'] <= base['no_contact_signed_rmse']))
    report = dict(g07_training=training_g07,train_pairs=len(train[1]), development_pairs=len(dev[1]), selected=selected, train_analytic=reference,
                  train_scores=grid, development=development, g07=transfer, checks=checks,
                  development_passed=all(all(v.values()) for v in checks.values()), actor_updates=0, final_holdout_opened=False,
                  physics_sha256=sha(physics_file), calibration_sha256=sha(a.calibration),
                  fixture_sha256={s:sha(a.fixture/(s+'.json')) for s in ('train','development')},
                  g07_sha256=sha(a.g07_development/'status.json'),
                  scope='Single first-racket-TOI analytic model. Measured racket drives inference. Native friction is a training label only. No shuttle spin state or contact-gradient receipt.')
    if a.body:
        report['body_racket'] = body_transfer(a.g07_development, a.body, prior, fitted, groups, repeats, trajectories, commands)
    if a.online_body:
        report['online_body_racket'] = body_transfer(a.g07_development, a.body, prior, fitted, groups, repeats,
                                                    trajectories, commands, a.online_body, a.source_stage)
    if a.racket_oracles:
        report['racket_oracles'] = {k: body_transfer(a.g07_development, a.body, prior, fitted, groups, repeats, trajectories,
                                                     commands, a.online_body, a.source_stage, measured=k)
                                     for k in ('center','rotation','pose','twist','both')}
        native_delta = torch.tensor([c['candidate']['delta_mps'] for c in transfer['cases']])
        measured = torch.tensor(report['racket_oracles']['both']['predicted_delta_mps'])
        torch.testing.assert_close(measured,native_delta,atol=1e-6,rtol=1e-6)
    if a.body:
        key = 'online_body_racket' if a.online_body else 'body_racket'
        stable = report[key]['metrics']['stable_signed']
        report['body_racket_response_passed'] = bool(stable and stable.get('cosine_median', -1) >= .8
            and stable.get('cosine_p10', -1) >= .5 and .5 <= stable.get('gain_median', 0) <= 2)
    with a.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
