#!/usr/bin/env python3
"""CPU supervised initialization from verified native recovery commands.

This is an in-sample fit, not PPO, closed-loop validation or generalization.
The original frozen actor, its normalizer and hardware limits are unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.control.learned_recovery_command_v1 import (
    SCHEMA, RecoveryCommandHead, command_features, compose_native_delta)
from quadsmash.rl.episode_trace_v4 import unpack_public
from quadsmash.rl.recovery_command_dataset_v1 import recovery_eligibility


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_episodes(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['schema'] != 'verified-recovery-native-command-dataset-v1':
        raise ValueError('Verified native command dataset required')
    limits = np.asarray(manifest['joint_limits_rad'])
    width = limits[:, 1] - limits[:, 0]
    episodes = []
    for entry in manifest['episodes']:
        path = directory / entry['data_file']
        if sha(path) != entry['data_sha256']:
            raise ValueError('Dataset changed: ' + str(path))
        with np.load(path, allow_pickle=False) as data:
            public = unpack_public(data['packed_public'])
            if hashlib.sha256(public.tobytes()).hexdigest() != entry['full_public794_sha256']:
                raise ValueError('Observation history changed')
            active, elapsed = recovery_eligibility(public)
            mask = data['supervision_mask']
            np.testing.assert_array_equal(mask, active & data['previous_command_known'])
            target = data['native_action'][mask].astype(float)
            previous = 2. * (data['previous_joint_target_rad'][mask] - limits[:, 0]) / width - 1.
            roles = data['label_role'][mask]
            known = data['student_request_known'][mask]
            if np.any((roles != 1) & ~known):
                raise ValueError('Missing actual student request')
            base = target.copy()
            base[known] = 2. * (data['student_requested_joint_target_rad'][mask][known] - limits[:, 0]) / width - 1.
            delta = (target - base) / 2.
            if not np.isfinite(delta).all() or abs(delta).max() > 1. + 1e-6:
                raise ValueError('Invalid native delta')
            tensors = [torch.tensor(x, dtype=torch.float32) for x in
                       (public[mask], base, previous, elapsed[mask], target, delta)]
            pub, b, prev, elapsed_t, target_t, delta_t = tensors
            episodes.append(dict(case=entry['case'], outcome=entry['outcome'],
                corrected=bool(np.any(roles == 2)), roles=roles,
                features=command_features(pub, b, prev, elapsed_t), base=b,
                previous=prev, target=target_t, delta=delta_t))
    return manifest, episodes, torch.tensor(limits, dtype=torch.float32)


@torch.inference_mode()
def evaluate_fit(head, episodes, limits):
    rows = []
    width = limits[:, 1] - limits[:, 0]
    for episode in episodes:
        delta = head(episode['features'])
        command = compose_native_delta(episode['base'], delta, episode['previous'],
            torch.ones(len(delta), dtype=torch.bool), limits)
        error = (command - episode['target']) * width / 2.
        rows.append(dict(case=episode['case'], corrected=episode['corrected'],
            outcome=episode['outcome'], samples=len(delta),
            command_rmse_rad=float(error.square().mean().sqrt()),
            command_error_p95_rad=float(torch.quantile(abs(error).flatten(), .95)),
            command_error_max_rad=float(abs(error).max())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=882076)
    parser.add_argument('--max-seconds', type=float, default=600.)
    args = parser.parse_args()
    if args.output.exists() or args.steps <= 0 or args.batch_size <= 0:
        raise ValueError('Fresh output and positive training limits required')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    manifest, episodes, limits = load_episodes(args.dataset)
    if manifest['control_dt_s'] != .01 or manifest['arm_target_rate_rad_s'] != 10.:
        raise ValueError('Runtime command assumptions differ')
    features = torch.cat([row['features'] for row in episodes])
    head = RecoveryCommandHead(features.mean(0), features.std(0).clamp_min(.05))
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-4)
    groups = [[i for i, e in enumerate(episodes) if e['corrected'] == flag] for flag in (False, True)]
    if not all(groups):
        raise ValueError('Need both success preservation and executed correction episodes')
    before = evaluate_fit(head, episodes, limits)
    args.output.mkdir(parents=True)
    start = time.monotonic()
    history = []
    width = limits[:, 1] - limits[:, 0]
    # Equal corrected/unmodified weight, uniform episodes within each group.
    # Half each episode's probability targets its first recovery second.
    for step in range(args.steps):
        selected = []
        for _ in range(args.batch_size):
            episode = episodes[int(rng.choice(groups[int(rng.integers(2))]))]
            n = len(episode['features'])
            upper = min(n, 100) if rng.random() < .5 else n
            selected.append((episode, int(rng.integers(upper))))
        x = torch.stack([e['features'][k] for e, k in selected])
        target_delta = torch.stack([e['delta'][k] for e, k in selected])
        predicted_delta = head(x)
        loss = ((predicted_delta - target_delta) * width).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(grad):
            raise ValueError('Nonfinite supervised update')
        optimizer.step()
        if step % 100 == 0 or step + 1 == args.steps:
            row = dict(step=step + 1, loss_rad2=float(loss.detach()), elapsed_s=time.monotonic() - start)
            history.append(row)
            print(json.dumps(row), flush=True)
        if time.monotonic() - start >= args.max_seconds:
            break
    after = evaluate_fit(head, episodes, limits)
    checkpoint = dict(schema=SCHEMA, head_state_dict=head.state_dict(), hidden=head.hidden,
        base_policy_sha256=manifest['base_policy_sha256'], protocol_sha256=manifest['protocol_sha256'],
        joint_limits_rad=manifest['joint_limits_rad'], control_dt_s=.01, arm_target_rate_rad_s=10.,
        optimizer_state_dict=optimizer.state_dict(), torch_rng_state=torch.get_rng_state(),
        numpy_rng_state=rng.bit_generator.state, steps=step + 1,
        dataset_manifest_sha256=sha(args.dataset / 'manifest.json'),
        interface='native base plus two times learned delta; original limits and actual-command arm slew',
        gate='observed airborne hit then support, latched to episode reset')
    torch.save(checkpoint, args.output / 'head.pt')
    report = dict(schema='native-recovery-command-fit-v1', dataset=str(args.dataset.resolve()),
        seed=args.seed, steps=step + 1, elapsed_s=time.monotonic() - start, device='cpu', threads=1,
        complete5_episodes=manifest['complete5_episodes'], recovery_only_episodes=manifest['recovery_only_episodes'],
        corrected_cases=manifest['corrected_complete5_cases'], samples=len(features),
        training_objective='native delta imitation in radians; outcome-filtered; balanced episode groups and early recovery',
        validation_scope='in-sample command fit only; no held-out or native closed-loop evaluation',
        teacher_off_learning_verified=False, base_actor_changed=False, hardware_limits_changed=False,
        case_id_or_future_outcome_as_input=False, before=before, after=after, history=history,
        checkpoint_sha256=sha(args.output / 'head.pt'),
        source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),
            ROOT / 'quadsmash/control/learned_recovery_command_v1.py', args.dataset / 'manifest.json']})
    (args.output / 'fit_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('steps','elapsed_s','samples','checkpoint_sha256')}), flush=True)


if __name__ == '__main__':
    main()
