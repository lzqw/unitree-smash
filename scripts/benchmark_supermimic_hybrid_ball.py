#!/usr/bin/env python3
"""Benchmark legacy and Hybrid ball transitions without updating parameters."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
import torch

from quadsmash.rl.supermimic_action_ball_v1 import action_ball_from_checkpoint
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import attach_hybrid_impulse_prior
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2


def load_launch(manifest, execution_root, device):
    suite = Path(execution_root) / manifest['source_manifest']['task_contract']['suite']['path']
    cases = json.loads(suite.read_text())['cases']
    return torch.tensor(
        [case['launch']['position_m'] + case['launch']['velocity_mps'] for case in cases],
        device=device, dtype=torch.float32)


def timed_rollout(base, ball, action_model, tensors, launch, start_tick,
                  horizon, backward):
    end_tick = start_tick + horizon
    if end_tick >= tensors['state'].shape[0]:
        raise ValueError(f'Rollout {start_tick}:{end_tick} exceeds packed data')
    state = tensors['state'][start_tick].clone()
    targets = tensors['targets'][start_tick:end_tick]
    previous = tensors['previous_action'][start_tick].clone()
    feet = tensors['feet'][start_tick].clone()
    present = tensors['present'][start_tick].clone()
    elapsed = tensors['elapsed'][start_tick].clone()
    state = state.detach()
    if backward:
        for module in (base, ball, action_model):
            if module is not None:
                module.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    if backward:
        for step in range(horizon):
            state, feet, present = ball_transition(
                base, ball, state, targets[step], previous, feet, present,
                elapsed, launch, action_model=action_model)
            previous = tensors['final_action'][start_tick + step]
            elapsed = elapsed + .01
    else:
        with torch.no_grad():
            for step in range(horizon):
                state, feet, present = ball_transition(
                    base, ball, state, targets[step], previous, feet, present,
                    elapsed, launch, action_model=action_model)
                previous = tensors['final_action'][start_tick + step]
                elapsed = elapsed + .01
    loss = state[:, 49:].square().mean()
    if backward:
        loss.backward()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - started
    return elapsed_s, float(loss.detach())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--physical-checkpoint', type=Path, required=True)
    parser.add_argument('--ball-checkpoint', type=Path, required=True)
    parser.add_argument('--packed-data', type=Path, required=True)
    parser.add_argument('--action-ball-checkpoint', type=Path, default=None)
    parser.add_argument('--execution-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--start-tick', type=int, default=267)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    packed = torch.load(args.packed_data, map_location='cpu', weights_only=True)
    # Keep only the measured rollout window before transferring data to the
    # accelerator.  The packed archive can contain many thousands of ticks,
    # while this benchmark needs one reset-safe 64-step segment.
    window_start = args.start_tick
    window_end = window_start + 65
    tensors = {}
    for key, value in packed['tensors'].items():
        selected = value
        if value.ndim >= 2 and value.shape[0] >= window_end:
            selected = value[window_start:window_end]
        if selected.ndim >= 2:
            selected = selected[:, :args.batch]
        tensors[key] = selected.to(device)
    manifest = packed
    physical_state = torch.load(args.physical_checkpoint, map_location='cpu', weights_only=True)
    asset_sha = physical_state.get('asset_sha256') or packed['source_manifest']['task_contract']['asset']['sha256']
    base = PhysicalWorldV2(asset_sha).to(device)
    base.load_state_dict(physical_state['model'], strict=True)
    base.eval()
    ball_state = torch.load(args.ball_checkpoint, map_location='cpu', weights_only=True)
    ball = BallResidualV2().to(device)
    ball.load_state_dict(ball_state['model'], strict=True)
    ball.eval()
    action_model = None
    action_sha = None
    if args.action_ball_checkpoint is not None:
        candidate = torch.load(args.action_ball_checkpoint, map_location='cpu', weights_only=True)
        action_model = action_ball_from_checkpoint(candidate).to(device)
        action_model.load_state_dict(candidate['model'], strict=True)
        attach_hybrid_impulse_prior(action_model, device)
        action_model.eval()
        # Candidate metadata does not necessarily embed its own file digest.
        # Bind the benchmark to the actual checkpoint bytes instead of a
        # nullable optional field.
        action_sha = hashlib.sha256(args.action_ball_checkpoint.read_bytes()).hexdigest()
    launch = load_launch(manifest, args.execution_root, device)[:args.batch]
    results = []
    # Data was sliced above, so all timing below starts at local tick zero.
    local_start_tick = 0
    if tensors['state'].shape[0] < 65:
        raise ValueError('start-tick must leave at least 64 packed transitions')
    for label, model in (('legacy', None), ('hybrid', action_model)):
        if label == 'hybrid' and model is None:
            continue
        for horizon in (1, 16, 32, 64):
            for _ in range(args.warmups):
                timed_rollout(base, ball, model, tensors, launch, local_start_tick,
                              horizon, False)
            forwards = [timed_rollout(base, ball, model, tensors, launch,
                                      local_start_tick, horizon, False)[0]
                        for _ in range(args.repeats)]
            peak_memory = 0
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            backward = [timed_rollout(base, ball, model, tensors, launch,
                                      local_start_tick, horizon, True)[0]
                        for _ in range(args.repeats)]
            if device.type == 'cuda':
                peak_memory = torch.cuda.max_memory_allocated(device)
            results.append(dict(model=label, horizon=horizon,
                               forward_mean_s=sum(forwards) / len(forwards),
                               backward_mean_s=sum(backward) / len(backward),
                               forward_batch=args.batch,
                               backward_peak_memory_bytes=peak_memory))
    def sha256(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                digest.update(block)
        return digest.hexdigest()
    output = dict(schema='supermimic-hybrid-ball-benchmark-v2',
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                  text=True).strip(),
                  device=str(device), start_tick=args.start_tick,
                  batch=args.batch, repeats=args.repeats, warmups=args.warmups,
                  physical_checkpoint_sha256=sha256(args.physical_checkpoint),
                  ball_checkpoint_sha256=sha256(args.ball_checkpoint),
                  packed_data_sha256=sha256(args.packed_data),
                  action_ball_sha256=action_sha, results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
