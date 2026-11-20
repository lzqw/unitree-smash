#!/usr/bin/env python3
"""Fit a frozen-policy world model on paired action-identification rollouts.

This fit is deliberately separate from policy-distribution WM fitting.  It
uses the measured plus/minus PhysX trajectories to improve action-to-racket
sensitivity, while keeping the contact branch frozen and never updating an
actor or critic.  The output is diagnostic evidence and is not a policy
training admission receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_racket_dynamics_v1 import racket_loss, racket_state


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_suite(path: Path) -> torch.Tensor:
    suite = json.loads(path.read_text())
    cases = suite.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('Launch suite has no cases')
    return torch.tensor(
        [case['launch']['position_m'] + case['launch']['velocity_mps']
         for case in cases], dtype=torch.float32)


def rows_to_tensors(rows, indices, device):
    selected = [rows[int(i)] for i in indices]
    keys = (
        'state', 'previous_action', 'foot_contact', 'present', 'elapsed',
        'step_targets_minus', 'step_targets_plus',
        'step_final_actions_minus', 'step_final_actions_plus',
        'step_next_states_minus', 'step_next_states_plus',
        'step_foot_contact_minus', 'step_foot_contact_plus',
        'step_ball_present_minus', 'step_ball_present_plus',
        'step_elapsed_minus', 'step_elapsed_plus', 'lane_index',
    )
    result = {}
    for key in keys:
        values = [row[key] for row in selected]
        if key == 'lane_index':
            result[key] = torch.tensor(values, dtype=torch.long, device=device)
        else:
            result[key] = torch.stack(values).to(device)
    return result


def branch_rollout(model, batch, launch, branch):
    state = batch['state']
    previous = batch['previous_action']
    feet = batch['foot_contact'].float()
    present = batch['present']
    elapsed = batch['elapsed']
    targets = batch[f'step_targets_{branch}']
    actions = batch[f'step_final_actions_{branch}']
    target_states = batch[f'step_next_states_{branch}']
    target_feet = batch[f'step_foot_contact_{branch}']
    target_present = batch[f'step_ball_present_{branch}']
    target_elapsed = batch[f'step_elapsed_{branch}']
    outputs = []
    for step in range(targets.shape[1]):
        state, logits, predicted_present = model(
            state, targets[:, step], previous, feet, present, elapsed,
            launch, detach_contacts=True)
        outputs.append(dict(
            state=state,
            logits=logits,
            present=predicted_present,
            target_state=target_states[:, step],
            target_feet=target_feet[:, step].float(),
            target_present=target_present[:, step],
            target_elapsed=target_elapsed[:, step],
        ))
        previous = actions[:, step]
        feet = logits.sigmoid()[:, :feet.shape[1]]
        present = predicted_present
        elapsed = elapsed.new_tensor(.01) + elapsed
    return outputs


def rollout_loss(model, outputs, geometry):
    state_loss = outputs[0]['state'].new_zeros(())
    racket_value = state_loss.clone()
    contact_value = state_loss.clone()
    for output in outputs:
        state_loss = state_loss + model.state_loss(
            output['state'], output['target_state']) / len(outputs)
        racket_value = racket_value + racket_loss(
            racket_state(output['state'], geometry),
            racket_state(output['target_state'], geometry)) / len(outputs)
        contact_value = contact_value + F.binary_cross_entropy_with_logits(
            output['logits'], output['target_feet']) / len(outputs)
    return state_loss, racket_value, contact_value


@torch.no_grad()
def evaluate(model, rows, indices, launch_cpu, device):
    if len(indices) == 0:
        return dict(count=0)
    batch = rows_to_tensors(rows, indices, device)
    launch = launch_cpu[batch['lane_index'].cpu()].to(device)
    result = {'count': len(indices), 'branches': {}}
    for branch in ('minus', 'plus'):
        outputs = branch_rollout(model, batch, launch, branch)
        state_errors = []
        racket_errors = []
        for output in outputs:
            state_errors.append((output['state'] - output['target_state']).abs())
            racket_errors.append(
                (racket_state(output['state'], model.geometry) -
                 racket_state(output['target_state'], model.geometry)).abs())
        state_error = torch.stack(state_errors)
        racket_error = torch.stack(racket_errors)
        result['branches'][branch] = dict(
            horizon=len(outputs),
            state_mae=float(state_error.mean()),
            robot_state_mae=float(state_error[..., :49].mean()),
            ball_state_mae=float(state_error[..., 49:].mean()),
            racket_position_mae_m=float(racket_error[..., :3].mean()),
            racket_linear_velocity_mae_m_s=float(racket_error[..., 12:15].mean()),
            racket_angular_velocity_mae_rad_s=float(racket_error[..., 15:18].mean()),
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--source-world', type=Path, required=True)
    parser.add_argument('--launch-suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=256)
    parser.add_argument('--batch-rows', type=int, default=4)
    parser.add_argument('--racket-loss-weight', type=float, default=1.0)
    parser.add_argument('--anchor-weight', type=float, default=1e-7)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.updates < 1 or args.batch_rows < 1:
        raise ValueError('updates and batch-rows must be positive')
    if args.racket_loss_weight < 0 or args.anchor_weight < 0:
        raise ValueError('loss weights must be nonnegative')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('GPU required for action-identification WM fit')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'startup.json', dict(
        arguments={k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()},
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        dirty_diff=subprocess.check_output(['git', 'diff'], text=True),
        actor_updates=0, critic_updates=0,
        policy_training_admitted=False,
    ))

    dataset_hash = sha256(args.dataset)
    source_hash = sha256(args.source_world)
    dataset = torch.load(args.dataset, map_location='cpu', weights_only=True)
    if dataset.get('schema') != 'supermimic-action-identification-v1':
        raise ValueError('Unexpected action-identification dataset schema')
    rows = dataset.get('rows', [])
    if not rows:
        raise ValueError('Action-identification dataset has no rows')
    launch_cpu = load_suite(args.launch_suite)
    source = torch.load(args.source_world, map_location='cpu', weights_only=True)
    if source.get('schema') != PhysicalWorldV2.schema:
        raise ValueError('source-world must be a PhysicalWorldV2 checkpoint')
    manifest = source.get('source_manifest', {})
    asset_sha = manifest.get('task_contract', {}).get('asset', {}).get('sha256')
    if not asset_sha:
        raise ValueError('source-world is missing asset identity')
    if len(launch_cpu) <= max(int(row['lane_index']) for row in rows):
        raise ValueError('launch suite does not cover all probe lanes')
    device = torch.device(args.device)
    model = PhysicalWorldV2(asset_sha).to(device)
    model.load_state_dict(source['model'], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.body.parameters():
        parameter.requires_grad_(True)
    frozen = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith('body.')
    }
    source_body = {
        name: value.detach().cpu().clone()
        for name, value in model.body.state_dict().items()
    }
    spec_names = [str(row['spec_name']) for row in rows]
    unique_specs = sorted(set(spec_names))
    validation_spec = unique_specs[-1]
    train_indices = torch.tensor(
        [i for i, name in enumerate(spec_names) if name != validation_spec],
        dtype=torch.long)
    validation_indices = torch.tensor(
        [i for i, name in enumerate(spec_names) if name == validation_spec],
        dtype=torch.long)
    if not len(train_indices) or not len(validation_indices):
        raise ValueError('Need at least one train and one validation spec')
    initial = evaluate(model, rows, validation_indices, launch_cpu, device)
    write_json(args.output / 'validation_initial.json', initial)
    optimizer = torch.optim.Adam(model.body.parameters(), lr=2e-5)
    generator = torch.Generator().manual_seed(20261120)
    train_list = train_indices.tolist()
    start = time.perf_counter()
    with (args.output / 'training.jsonl').open('w', buffering=1) as log:
        for update in range(1, args.updates + 1):
            if len(train_list) <= args.batch_rows:
                selected = train_list
            else:
                choice = torch.randint(len(train_list), (args.batch_rows,), generator=generator)
                selected = [train_list[int(i)] for i in choice]
            batch = rows_to_tensors(rows, selected, device)
            launch = launch_cpu[batch['lane_index'].cpu()].to(device)
            loss = torch.zeros((), device=device)
            state_total = torch.zeros_like(loss)
            racket_total = torch.zeros_like(loss)
            contact_total = torch.zeros_like(loss)
            for branch in ('minus', 'plus'):
                outputs = branch_rollout(model, batch, launch, branch)
                state_value, racket_value, contact_value = rollout_loss(
                    model, outputs, model.geometry)
                loss = loss + state_value + args.racket_loss_weight * racket_value
                state_total = state_total + state_value
                racket_total = racket_total + racket_value
                contact_total = contact_total + contact_value
            anchor = torch.zeros_like(loss)
            for name, parameter in model.body.named_parameters():
                anchor = anchor + (parameter - source_body[name].to(device)).square().mean()
            loss = loss / 2 + args.anchor_weight * anchor
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite action-identification loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                model.body.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            row = dict(
                update=update, loss=float(loss.detach()),
                state_loss=float((state_total / 2).detach()),
                racket_loss=float((racket_total / 2).detach()),
                contact_loss=float((contact_total / 2).detach()),
                anchor_loss=float(anchor.detach()),
                gradient_before_clip=float(gradient),
                elapsed_s=time.perf_counter() - start,
                actor_updates=0, critic_updates=0,
                policy_training_admitted=False,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
            log.write(json.dumps(row) + '\n')
            if update == 1 or update % 32 == 0:
                print(json.dumps(row), flush=True)
    final = evaluate(model, rows, validation_indices, launch_cpu, device)
    write_json(args.output / 'validation_final.json', final)
    for name, value in frozen.items():
        current = model.state_dict()[name].detach().cpu()
        if not torch.equal(current, value):
            raise RuntimeError('Frozen world-model parameter changed: ' + name)
    body_change = sum(
        (value.detach().cpu() - source_body[name]).square().sum().item()
        for name, value in model.body.state_dict().items()) ** .5
    state = dict(
        schema=model.schema, model=model.state_dict(),
        source_manifest=manifest,
        source_world=str(args.source_world.resolve()), source_world_sha256=source_hash,
        dataset=str(args.dataset.resolve()), dataset_sha256=dataset_hash,
        launch_suite=str(args.launch_suite.resolve()), launch_suite_sha256=sha256(args.launch_suite),
        world_updates=int(source.get('world_updates', 0)) + args.updates,
        action_identification_updates=args.updates,
        actor_updates=0, critic_updates=0,
        train_spec_names=[name for name in unique_specs if name != validation_spec],
        validation_spec_name=validation_spec,
        policy_training_admitted=False,
        scope='Offline paired action-identification WM fit; no policy training admission',
    )
    torch.save(state, args.output / 'world.pt')
    write_json(args.output / 'completed.json', dict(
        world_updates=args.updates, actor_updates=0, critic_updates=0,
        body_parameter_l2_change=body_change,
        train_rows=len(train_indices), validation_rows=len(validation_indices),
        training_s=time.perf_counter() - start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        policy_training_admitted=False,
        scope='Offline paired action-identification WM fit; diagnostic only',
    ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
