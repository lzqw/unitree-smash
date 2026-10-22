#!/usr/bin/env python3
"""Synthetic state-machine/reward audit and CPU-only checkpoint noise check.

No Isaac imports, simulator, optimizer step or GPU use. Synthetic signals are
counterexamples to reward alignment, not evidence of feasible dynamics.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from quadsmash.tasks.stand import jump_metrics
from quadsmash.tasks.stand.jump_reward_audit import v2c_components, recovery_draft
from quadsmash.rl.action_diagnostics import reset_initial_std

PROTOCOL = json.loads((ROOT / 'configs/evaluation_protocol_v2.json').read_text())


def signal_probe(kind):
    state = SimpleNamespace(); jump_metrics.initialize(state, 1, 'cpu')
    weak_paid = torch.zeros(1); time = 0.; trace = []
    def tick(support=True, height=.4, force=0., horizontal_speed=0.):
        nonlocal weak_paid, time
        time += .01
        _, stable = jump_metrics.advance(state, dt=.01, time=torch.tensor([time]),
            com_height=torch.tensor([height]), foot_contact=torch.full((1, 4), support),
            foot_clearance=torch.tensor([.02 if support else .1]), roll_pitch=torch.zeros(1, 2),
            horizontal_speed=torch.tensor([horizontal_speed]), angular_speed=torch.zeros(1),
            joint_speed=torch.zeros(1), body_ground_force=torch.tensor([force]), protocol=PROTOCOL)
        z = torch.zeros(1); b = torch.zeros(1, dtype=torch.bool)
        quality = (state.landed_once & support & ~state.body_ground_seen).float() * torch.exp(torch.tensor([-2 * horizontal_speed**2]))
        f = dict(height_tracking=z, flight=z, recovery=stable.float(), prepare_quality=z,
            recovery_quality=quality, arm_error=z, torque_square_sum=z, action_delta=z,
            settling=z, settle_cost=z, first_jump=b, first_success=b, first_body=b, terminated=b)
        old = v2c_components(f, .01)
        draft, weak_paid = recovery_draft(old, valid_jump_landed=state.valid_jump_landed,
            body_ground_seen=state.body_ground_seen, recovery_quality=quality, weak_paid=weak_paid, dt=.01)
        trace.append(dict(time_s=time, supported=support, height_m=height,
            valid_jump_landed=bool(state.valid_jump_landed), recovery_seen=bool(state.recovery_seen),
            formal_success=bool(jump_metrics.success(state)),
            old_primary_recovery=float(old['stable_recovery'] + old['recovery_quality']),
            draft_primary_recovery=float(draft['stable_recovery'] + draft['recovery_quality']),
            draft_weak_recovery=float(draft['weak_safe_settling'])))
    tick()
    if kind != 'standing':
        tick(False, .4)
        short = kind in ('tiny_hop', 'short_high_hop')
        high = kind in ('qualified', 'body_contact', 'broken_recovery', 'short_high_hop')
        for _ in range(3 if short else 12): tick(False, .49 if high else .402)
        for i in range(100):
            tick(force=2. if kind == 'body_contact' and i == 0 else 0.,
                 horizontal_speed=.31 if kind == 'broken_recovery' and i % 30 == 29 else 0.)
    else:
        for _ in range(100): tick()
    return dict(kind=kind, domain='synthetic CPU signals, NOT physics or learned performance',
        valid_jump_landed=bool(state.valid_jump_landed), recovery_seen=bool(state.recovery_seen),
        formal_success=bool(jump_metrics.success(state)), max_continuous_recovery_s=float(state.recovery_time),
        old_primary_recovery=sum(r['old_primary_recovery'] for r in trace),
        draft_primary_recovery=sum(r['draft_primary_recovery'] for r in trace),
        draft_weak_recovery=sum(r['draft_weak_recovery'] for r in trace), trace=trace)


def checkpoint_noise_probe(checkpoint):
    from tensordict import TensorDict
    from rsl_rl.modules import ActorCritic
    cfg = json.loads((ROOT / 'configs/jump_ppo_v2c.json').read_text())
    options = dict(cfg['runner']['policy']); options.pop('class_name')
    with contextlib.redirect_stdout(io.StringIO()):
        policy = ActorCritic(TensorDict({'policy': torch.zeros(1, 794)}, batch_size=[1]),
                            cfg['runner']['obs_groups'], 18, **options)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    policy.load_state_dict(saved['model_state_dict'])
    original = {k: v.clone() for k, v in policy.state_dict().items()}
    noise = reset_initial_std(policy, .15)
    unchanged = all(torch.equal(v, policy.state_dict()[k]) for k, v in original.items() if k != 'log_std')
    assert unchanged and not torch.cuda.is_initialized()
    return dict(checkpoint=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        **noise, all_non_noise_state_unchanged=unchanged, optimizer_steps=0, cuda_initialized=False)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    cases = [signal_probe(k) for k in ('tiny_hop', 'short_high_hop', 'long_low_hop', 'qualified', 'body_contact', 'broken_recovery', 'standing')]
    result = dict(baseline_commit='65396a21bd8065400df91437432bd12a0a46bf81', cases=cases,
        noise=checkpoint_noise_probe(ROOT / 'runs/G04/corrected_v2c_main01/native/checkpoint.pt'),
        gpu_jobs_started=0, physics_simulations=0, training_runs=0)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({**result, 'cases': [{k: v for k, v in c.items() if k != 'trace'} for c in cases]}, indent=2))


if __name__ == '__main__': main()
