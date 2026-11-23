"""Fit a bounded Body WM candidate on E1 native paths and measured command pairs."""

import argparse
import json
import math
from pathlib import Path

import torch

from fit_supermimic_mixed_body_rollout import digest
from scripts.replay_supermimic_recovery_wm_updates import fit_readout_rows
from quadsmash.rl.supermimic_archive_storage_v1 import read_archive_rows
from quadsmash.rl.supermimic_ball_geometry_preview_v1 import BallGeometryPreviewV1
from quadsmash.rl.supermimic_native_response_data_v1 import normalized_response_loss
from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical, valid_windows
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_policy_execution_v1 import contract
from quadsmash.rl.supermimic_racket_dynamics_v1 import racket_state, robot_loss
from quadsmash.rl.supermimic_toi_racket_objective import string_boundary
from quadsmash.rl.supermimic_training_probe_replay import training_windows


def native_data(directory, task, actor_hash):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['task_contract'] != task or manifest['policy_execution'] != contract(arm_preparation_seconds=.25):
        raise ValueError('Expected the compatible E1 native collection')
    columns = ('state', 'next_state', 'feet', 'present', 'elapsed', 'previous_action',
               'targets', 'episode', 'terminated', 'timeout', 'new_racket', 'tick', 'stream',
               'new_contact', 'qualified_release', 'outgoing_velocity', 'before_contact',
               'net_contact_at_end', 'ball_clock', 'next_ball_clock')
    merged = {key: [] for key in columns}
    used = []
    for receipt in manifest['chunks']:
        with read_archive_rows(directory, receipt, scratch=directory.parent) as records:
            original = [row for row in records if row['actor_state_sha256'] == actor_hash]
            if not original:
                continue
            used.append(receipt['sha256'])
            for row in original:
                before, after = row['before'], row['after']
                source = before['physical']
                events = after['event_state'] > before['event_state']
                rally = after['task']
                release = rally['released'] & ~before['task']['released']
                qualify = (release & rally['legal_initial_contact'] & rally['string_hit']
                           & rally['airborne_string_hit'])
                sample = (encode_physical(source), encode_physical(after['physical']),
                          source['foot_contact'].float(), source['ball_present'], source['elapsed_s'],
                          source['final_action'], row['final_joint_targets'], before['episode_index'],
                          row['terminated'], row['timeout'], events[:, 1] | events[:, 6],
                          torch.tensor(row['native_tick']), torch.tensor(row['stream_id']),
                          rally['had_contact'] & ~before['task']['had_contact'], qualify,
                          rally['outgoing_velocity'], ~before['task']['had_contact'],
                          rally['shuttle_net_contact'],
                          before['auxiliary']['environment']['ball_clock'],
                          after['auxiliary']['environment']['ball_clock'])
                for key, value in zip(columns, sample, strict=True):
                    merged[key].append(value.clone())
    if not used:
        raise ValueError('Missing native records with the initial E1 actor digest')
    return {key: torch.stack(values) for key, values in merged.items()}, used


def sample_windows(data, eligible, horizon, count, rng, *, clearance=False):
    valid = valid_windows(data['episode'], data['terminated'], data['timeout'], horizon)
    valid &= ~(data['terminated'] | data['timeout']).unfold(0, horizon, 1).any(-1)
    continuous = ((data['tick'].unfold(0, horizon, 1).diff() == 1).all(-1)
                  & (data['stream'].unfold(0, horizon, 1).diff() == 0).all(-1))
    valid &= continuous[:, None] & eligible
    near = data['new_racket'].unfold(0, horizon, 1).any(-1)
    approaching = data['present'][:len(valid)] & (data['state'][:len(valid), :, 52] < 0)
    pools = (valid & near, valid & ~near & approaching, valid & ~near & ~approaching)
    close = (data['near_miss'].unfold(0, horizon, 1).any(-1) if clearance else None)
    selected = []
    for index, (pool, quota) in enumerate(zip(pools, (count // 2, count // 4, count // 4), strict=True)):
        # Select at most one start per native lane inside each stratum.
        lanes = pool.any(0).nonzero().flatten()
        if len(lanes) < quota:
            raise ValueError('Insufficient distinct natural or racket trajectories')
        lanes = lanes[torch.randperm(len(lanes), generator=rng)]
        if index == 1 and close is not None:
            preferred = (pool & close).any(0)[lanes]
            lanes = torch.cat((lanes[preferred], lanes[~preferred]))
        for lane in lanes[:quota]:
            options = pool[:, lane]
            if index == 1 and close is not None and (options & close[:, lane]).any():
                options = options & close[:, lane]
            ticks = options.nonzero().flatten()
            selected.append(torch.stack((ticks[torch.randint(len(ticks), (), generator=rng)], lane)))
    return torch.stack(selected)


def racket_at_ball(racket, observed_ball_position, *, full_orientation=False):
    orientation = racket[:, 3:12].reshape(-1, 3, 3)
    axes = orientation.flatten(1) if full_orientation else orientation[:, :, 0]
    offset = observed_ball_position - racket[:, :3]
    # Far from the racket, supervise the twist at its center.
    lever = torch.where(offset.norm(dim=-1, keepdim=True) <= .4, offset, torch.zeros_like(offset))
    surface = racket[:, 12:15] + torch.linalg.cross(racket[:, 15:18], lever)
    return torch.cat((racket[:, :3], surface, axes, racket[:, 15:18]), dim=-1)


def ball_in_racket_frame(racket, observed_ball_position):
    attitude = racket[..., 3:12].unflatten(-1, (3, 3))
    delta = observed_ball_position - racket[..., :3]
    return (attitude.transpose(-1, -2) @ delta.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def paired_precontact_mask(signed, schema, geometry, preview, device):
    offset = 0
    for field in schema['fields']:
        if field['path'] == 'rally/had_contact':
            if field['kind'] != 'bool' or field['width'] != 1:
                raise ValueError('Expected a scalar native contact latch')
            break
        offset += field['width']
    else:
        raise ValueError('Native contact latch is required for pre-impact pairing')

    preimpact, near_racket = [], []
    for branch in ('minus', 'plus'):
        data = signed[branch]
        truth = data['next_state'].to(device)
        flat = truth.flatten(0, 1)
        local = ball_in_racket_frame(racket_state(flat, geometry), flat[:, 49:52])
        gap = preview.racket_gap(local).reshape(truth.shape[:2]).cpu()
        known = data['next_valid'][..., offset] > .5
        no_contact = data['next_values'][..., offset] < .5
        arriving = data['present'].bool() & (data['next_state'][..., 52] < 0.)
        preimpact.append(known & no_contact & arriving)
        near_racket.append((gap > .002) & (gap <= .08))
    command_delta = signed['plus']['targets'] - signed['minus']['targets']
    different_commands = command_delta.square().sum(-1).cumsum(0) >= 1e-12
    return preimpact[0] & preimpact[1] & (near_racket[0] | near_racket[1]) & different_commands


def paired_precontact_pose_loss(minus, plus, signed, active, boundary):
    inward, offset = boundary

    def features(racket, point):
        pose = torch.cat((racket[..., :3], racket[..., 6:15]), dim=-1)
        local = ball_in_racket_frame(pose, point)
        margin = (local[..., 1:] @ inward.T - offset).amin(-1, keepdim=True)
        return torch.cat((local, margin), dim=-1)

    predicted_minus, native_minus = (features(racket, signed['minus']['next_state'][..., 49:52])
                                     for racket in minus[:2])
    predicted_plus, native_plus = (features(racket, signed['plus']['next_state'][..., 49:52])
                                   for racket in plus[:2])
    residual = ((predicted_plus - predicted_minus) - (native_plus - native_minus)) / .01
    error = torch.nn.functional.smooth_l1_loss(residual, torch.zeros_like(residual), beta=.2, reduction='none')
    return torch.where(active, error.mean(-1), 0.).sum() / active.sum().clamp_min(1)


@torch.no_grad()
def label_precontact_clearance(data, geometry, eligible, preview, device):
    # Use transitions without racket contact before the first racket event. The
    # clearance label is computed from native endpoints and authored geometry.
    valid = (eligible & data['present'] & data['before_contact']
             & ~data['new_racket'] & ~data['new_contact']
             & ~data['terminated'] & ~data['timeout'] & (data['next_state'][..., 52] < 0))
    gaps = torch.zeros_like(data['elapsed'])
    rows = valid.nonzero()
    for offset in range(0, len(rows), 1024):
        chunk = rows[offset:offset + 1024]
        ticks, lanes = chunk.T
        truth = data['next_state'][ticks, lanes].to(device)
        local = ball_in_racket_frame(racket_state(truth, geometry), truth[:, 49:52])
        gaps[ticks, lanes] = preview.racket_gap(local).cpu()
    return gaps, valid & (gaps > .002) & (gaps <= .08)


def body_predictions(body, data, launch):
    state, feet = data['state'][0], data['feet'][0]
    for tick in range(len(data['state'])):
        observed = data['state'][tick]
        # A Body-only fit sees the measured shuttle and never alters contact
        # weights to absorb trajectory error from the wrong robot pose.
        state = torch.cat((state[:, :49], observed[:, 49:]), dim=-1)
        state, logits, _ = body(
            state, data['targets'][tick], data['previous_action'][tick], feet,
            data['present'][tick], data['elapsed'][tick], launch, detach_contacts=True)
        feet = logits.sigmoid()
        yield state, data['next_state'][tick]


def body_state_loss(state, truth, root_angular_weight):
    loss = robot_loss(state, truth)
    if root_angular_weight:
        root_angular = torch.nn.functional.smooth_l1_loss(
            state[:, 10:13], truth[:, 10:13], beta=.1)
        loss = loss + root_angular_weight * root_angular
    return loss


def body_rollout(body, data, launch, *, root_angular_weight=0., clearance_preview=None,
                 supervise_precontact_pose=False, full_orientation=False, boundary=None):
    estimates, labels, robot_errors = [], [], []
    clearance_errors, pose_errors = [], []
    for tick, (state, truth) in enumerate(body_predictions(body, data, launch)):
        point = truth[:, 49:52]
        predicted_racket = racket_state(state, body.geometry)
        measured_racket = racket_state(truth, body.geometry)
        estimates.append(racket_at_ball(predicted_racket, point, full_orientation=full_orientation))
        labels.append(racket_at_ball(measured_racket, point, full_orientation=full_orientation))
        robot_errors.append(body_state_loss(state, truth, root_angular_weight))
        if clearance_preview is not None or supervise_precontact_pose:
            local = ball_in_racket_frame(predicted_racket, point)
            close = data['near_miss'][tick]
        if clearance_preview is not None:
            estimated_gap = clearance_preview.racket_gap(local)
            measured_gap = data['clearance'][tick]
            error = torch.nn.functional.smooth_l1_loss(
                estimated_gap / .01, measured_gap / .01, beta=.2, reduction='none')
            clearance_errors.append(torch.where(close, error, 0.).sum())
        if supervise_precontact_pose:
            measured_local = ball_in_racket_frame(measured_racket, point)
            error = torch.nn.functional.smooth_l1_loss(
                local / .01, measured_local / .01, beta=.2, reduction='none').mean(-1)
            if boundary is not None:
                inward, offset = boundary
                margin = (local[:, 1:] @ inward.T - offset).amin(-1)
                measured_margin = (measured_local[:, 1:] @ inward.T - offset).amin(-1)
                error = .5 * (error + torch.nn.functional.smooth_l1_loss(
                    margin / .01, measured_margin / .01, beta=.2, reduction='none'))
            pose_errors.append(torch.where(close, error, 0.).sum())
    losses = [
        torch.stack(errors).sum() / data['near_miss'].sum().clamp_min(1)
        if errors else launch.new_zeros(()) for errors in (clearance_errors, pose_errors)]
    return torch.stack(estimates), torch.stack(labels), torch.stack(robot_errors).mean(), *losses


def fitting_terms(body, natural, natural_launch, signed, signed_launch, *, root_angular_weight=0.,
                  clearance_preview=None, supervise_precontact_pose=False, full_orientation=False, boundary=None,
                  paired_pose_mask=None):
    def pointwise(predicted, measured):
        return torch.nn.functional.smooth_l1_loss(predicted / scale, measured / scale, beta=.2)

    scale = natural_launch.new_tensor([.025] * 3 + [1.] * 3 + [.15] * (9 if full_orientation else 3) + [3.] * 3)
    estimate, measured, robot, clearance, pose = body_rollout(
        body, natural, natural_launch, root_angular_weight=root_angular_weight, clearance_preview=clearance_preview,
        supervise_precontact_pose=supervise_precontact_pose, full_orientation=full_orientation, boundary=boundary)
    natural_loss = robot + pointwise(estimate, measured)
    minus, plus = (body_rollout(body, signed[b], signed_launch, root_angular_weight=root_angular_weight,
                               full_orientation=full_orientation)
                   for b in ('minus', 'plus'))
    pair_trajectory = (minus[2] + plus[2] + pointwise(*minus[:2]) + pointwise(*plus[:2])) / 2
    response = []
    for tick in range(len(minus[0])):
        targets = [signed[key]['targets'][:tick + 1].transpose(0, 1).flatten(1) for key in ('minus', 'plus')]
        loss, _ = normalized_response_loss(minus[0][tick], plus[0][tick], minus[1][tick], plus[1][tick],
                                           *targets, scale)
        response.append(loss)
    pair_pose = (paired_precontact_pose_loss(minus, plus, signed, paired_pose_mask, boundary)
                 if paired_pose_mask is not None else natural_launch.new_zeros(()))
    return natural_loss, pair_trajectory, torch.stack(response).mean(), clearance, pose, pair_pose


def draw_windows(data, launch, eligible, fields, horizon, count, rng, *, clearance=False):
    starts, lanes = sample_windows(data, eligible, horizon, count, rng, clearance=clearance).T
    index = starts[None, :] + torch.arange(horizon)[:, None]
    batch = {key: data[key][index, lanes].to(launch.device) for key in fields}
    return batch, launch[lanes.to(launch.device)]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'native-archive', 'output'):
        cli.add_argument('--' + name, type=Path, required=True)
    cli.add_argument('--updates-per-horizon', type=int, default=12)
    cli.add_argument('--lr', type=float, default=1e-5)
    cli.add_argument('--root-angular-weight', type=float, default=0.)
    cli.add_argument('--long-body-weight', type=float, default=0.,
                     help='Mix free robot rollouts over alternating 32- and 64-tick native windows')
    cli.add_argument('--joint-readout-only', action='store_true',
                     help='Freeze the encoder and update only the joint position and velocity outputs')
    cli.add_argument('--precontact-clearance-weight', type=float, default=0.,
                     help='Prefer near misses and match racket clearance on measured no-contact steps')
    cli.add_argument('--precontact-pose-weight', type=float, default=0.,
                     help='Match all three racket-frame ball coordinates before measured contact')
    cli.add_argument('--full-racket-pose-supervision', action='store_true',
                     help='Supervise signed motion of the full racket frame and measured string margin')
    cli.add_argument('--paired-precontact-pose-weight', type=float, default=0.,
                     help='Match the executed-action response of racket-frame position and string margin before impact')
    cli.add_argument('--device', default='cuda:0', choices=('cpu', 'cuda:0'))
    args = cli.parse_args()
    if args.updates_per_horizon < 1 or not 0 < args.lr < .001:
        cli.error('Invalid bounded Body fitting budget or learning rate')
    if args.full_racket_pose_supervision and not args.precontact_pose_weight:
        cli.error('Full racket pose supervision requires a positive precontact pose weight')
    if args.paired_precontact_pose_weight and not args.full_racket_pose_supervision:
        cli.error('Paired precontact pose supervision requires the full racket frame')
    if any(not math.isfinite(weight) or weight < 0 for weight in
           (args.root_angular_weight, args.long_body_weight, args.precontact_clearance_weight,
            args.precontact_pose_weight, args.paired_precontact_pose_weight)):
        cli.error('Body supervision weights must be finite and nonnegative')
    if args.device == 'cuda:0' and not torch.cuda.is_available():
        raise RuntimeError('Body response fitting requires CUDA')
    torch.set_num_threads(4)
    device = torch.device(args.device)
    rng = torch.Generator().manual_seed(20261123)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    parent_hash = tensor_digest(checkpoint['world'].items())
    if parent_hash != checkpoint['wm_fit_provenance']['world_state_sha256']:
        raise ValueError('Source WM digest mismatch')
    task = checkpoint['metadata']['task_contract']
    suite = Path(task['suite']['path'])
    if digest(suite) != task['suite']['sha256']:
        raise ValueError('Launch suite changed')
    cases = json.loads(suite.read_text())['cases']
    launch = torch.tensor([r['launch']['position_m'] + r['launch']['velocity_mps'] for r in cases], device=device)
    holdout = set(checkpoint['config']['wm_holdout_lanes'])
    eligible = torch.tensor([i not in holdout for i in range(len(cases))])
    identity_file = args.native_archive.parent / 'evaluation_boundary.json'
    initial_actor = json.loads(identity_file.read_text())['invocation']['start']['actor_sha256']
    data, chunks = native_data(args.native_archive, task, initial_actor)
    collections = checkpoint['metadata']['wm_action_collections']
    if not collections:
        raise ValueError('No registered native command pairs')
    sequences = []
    # The saved phase adds ball-only evaluation and return semantics. The
    # historical interventions are imported under the shared physics source
    # contract, with actual executed targets and no task reward supervision.
    physics_source = {key: value for key, value in task.items() if key != 'approach_ball_scoring'}
    for source in collections:
        windows, receipt = training_windows(source['path'], physics_source, contract(), holdout)
        if receipt['status_sha256'] != source['status_sha256'] or receipt['spec_branches'] != source['spec_branches']:
            raise ValueError('Native pair data changed since source run')
        for name, (lanes, signed) in zip(receipt['included_lanes'], windows, strict=True):
            schema = None
            if args.paired_precontact_pose_weight:
                path = Path(source['path']) / name / 'minus' / 'native_probe.pt'
                schema = torch.load(path, map_location='cpu', mmap=True, weights_only=False)['raw']['wm_window']['schema']
            sequences.append((lanes, signed, schema))
    body = PhysicalWorldV2(task['asset']['sha256']).to(device).eval()
    original_body = {k.removeprefix('physical_model.'): v for k, v in checkpoint['world'].items()
                     if k.startswith('physical_model.')}
    body.load_state_dict(original_body, strict=True)
    if args.joint_readout_only:
        fit_readout_rows(body, body.body.net[-1], slice(13, 49))
    else:
        body.requires_grad_(False)
        body.body.requires_grad_(True)
    trainable = {name: p for name, p in body.body.named_parameters() if p.requires_grad}
    initial = {name: p.detach().clone() for name, p in trainable.items()}
    opt = torch.optim.Adam(trainable.values(), lr=args.lr)
    fields = ('state', 'next_state', 'targets', 'previous_action', 'feet', 'present', 'elapsed')
    clearance_preview = None
    clearance_fields = ()
    boundary = (string_boundary(Path('configs/full_racket_v4_r3.json'), launch)
                if args.full_racket_pose_supervision else None)
    if args.precontact_clearance_weight or args.precontact_pose_weight:
        clearance_preview = BallGeometryPreviewV1().to(device)
        data['clearance'], data['near_miss'] = label_precontact_clearance(
            data, body.geometry, eligible, clearance_preview, device)
        if not data['near_miss'].any():
            raise ValueError('No native training transitions near the racket without contact')
        clearance_fields = ('clearance', 'near_miss')
    pair_masks = [paired_precontact_mask(signed, schema, body.geometry, clearance_preview, device)
                  if schema is not None else None for _, signed, schema in sequences]
    pair_support = sum(int(mask.sum()) for mask in pair_masks if mask is not None)
    if args.paired_precontact_pose_weight and not pair_support:
        raise ValueError('No signed training pairs with measured pre-impact racket geometry')
    args.output.mkdir(parents=True, exist_ok=False)
    # Keep the short-window/command-pair draws comparable with the prior fit.
    long_rng = torch.Generator().manual_seed(rng.initial_seed() + 1)
    step = 0
    with (args.output / 'fit.jsonl').open('x', buffering=1) as stream:
        for horizon in (4, 8, 16):
            for _ in range(args.updates_per_horizon):
                natural, natural_launch = draw_windows(
                    data, launch, eligible, fields + clearance_fields, horizon, 32, rng,
                    clearance=clearance_preview is not None)
                sequence = step % len(sequences)
                pair_lanes, paired, _ = sequences[sequence]
                if len(pair_lanes) < 16:
                    raise ValueError('Too few independent command pairs for the batch')
                selected = torch.randperm(len(pair_lanes), generator=rng)[:16]
                signed = {b: {k: values[k][:horizon, selected].to(device) for k in fields}
                          for b, values in paired.items()}
                pair_launch = launch[torch.tensor(pair_lanes)[selected].to(device)]
                mask = pair_masks[sequence]
                mask = mask[:horizon, selected].to(device) if mask is not None else None
                nat, absolute, delta, clearance, pose, pair_pose = fitting_terms(
                    body, natural, natural_launch, signed, pair_launch,
                    root_angular_weight=args.root_angular_weight,
                    clearance_preview=clearance_preview if args.precontact_clearance_weight else None,
                    supervise_precontact_pose=bool(args.precontact_pose_weight),
                    full_orientation=args.full_racket_pose_supervision, boundary=boundary, paired_pose_mask=mask)
                anchor = sum((p - initial[name]).square().mean() for name, p in trainable.items())
                loss = .55 * nat + .30 * absolute + .15 * delta + .01 * anchor
                loss = loss + args.precontact_clearance_weight * clearance + args.precontact_pose_weight * pose
                loss = loss + args.paired_precontact_pose_weight * pair_pose
                long_horizon, long_loss = 0, loss.new_zeros(())
                if args.long_body_weight:
                    long_horizon = (32, 64)[step % 2]
                    long_batch, long_launch = draw_windows(
                        data, launch, eligible, fields, long_horizon, 16, long_rng)
                    long_loss = torch.stack([body_state_loss(state, truth, args.root_angular_weight)
                                             for state, truth in body_predictions(body, long_batch, long_launch)]).mean()
                    loss = loss + args.long_body_weight * long_loss
                opt.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite Body loss')
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(trainable.values(), 5., error_if_nonfinite=True)
                opt.step()
                step += 1
                row = dict(update=step, horizon=horizon, natural=float(nat.detach()),
                           paired_absolute=float(absolute.detach()), paired_response=float(delta.detach()),
                           long_horizon=long_horizon, long_body=float(long_loss.detach()), gradient_norm=float(grad),
                           precontact_clearance=float(clearance.detach()),
                           precontact_pose=float(pose.detach()),
                           paired_precontact_pose=float(pair_pose.detach()),
                           paired_precontact_pose_steps=int(mask.sum()) if mask is not None else 0,
                           near_miss_steps=int(natural['near_miss'].sum()) if clearance_fields else 0)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
    candidate = args.output / 'world.pt'
    state = {name: value.cpu() for name, value in body.state_dict().items()}
    if args.joint_readout_only:
        readout = f'body.net.{len(body.body.net) - 1}'
        for name, value in state.items():
            expected = original_body[name]
            protected = torch.cat((value[:13], value[49:])) if name.startswith(readout + '.') else value
            if name.startswith(readout + '.'):
                expected = torch.cat((expected[:13], expected[49:]))
            if not torch.equal(protected, expected):
                raise RuntimeError(f'Joint readout fitting modified protected Body weights: {name}')
    report = dict(parent_world_sha256=parent_hash, schema=body.schema, train_lanes=eligible.nonzero().flatten().tolist(),
                  e1_actor_sha256=initial_actor, e1_native_chunk_sha256=chunks, command_pair_sources=collections,
                  lr=args.lr, updates=step, root_angular_weight=args.root_angular_weight,
                  long_body_weight=args.long_body_weight, long_body_horizons=[32, 64] if args.long_body_weight else [],
                  precontact_clearance_weight=args.precontact_clearance_weight,
                  precontact_pose_weight=args.precontact_pose_weight,
                  paired_precontact_pose_weight=args.paired_precontact_pose_weight,
                  available_paired_precontact_steps=pair_support,
                  full_racket_pose_supervision=args.full_racket_pose_supervision,
                  available_near_miss_steps=int(data['near_miss'].sum()) if clearance_fields else 0,
                  available_near_miss_lanes=int(data['near_miss'].any(0).sum()) if clearance_fields else 0,
                  joint_readout_only=args.joint_readout_only, device=str(device),
                  actor_updates=0, critic_updates=0,
                  gpu_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                  body_sha256=tensor_digest(state.items()))
    pending = candidate.with_suffix('.pending')
    torch.save(dict(model=state, **report), pending)
    pending.replace(candidate)
    (args.output / 'completed.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
