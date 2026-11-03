#!/usr/bin/env python3
"""Preserve successful native trajectories from compatible historical policies.

The old best policy owns its successes. Greedy coverage adds one independently
successful full episode for each remaining condition. Source actors and all
dispatched commands must pass the existing production-controller CPU replay.
No teacher is queried on a failed student state, and no online selector or
student success rate is claimed. This never edits a live training config.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_frozen_complete5_v2 import read_run, sha


def fingerprint(value):
    """Content identity for the complete nested frozen controller payload."""
    digest = hashlib.sha256()
    def add(v):
        if isinstance(v, dict):
            digest.update(b'dict')
            for key in sorted(v):
                add(key); add(v[key])
        elif isinstance(v, (list, tuple)):
            digest.update(b'list' + str(len(v)).encode())
            for item in v:
                add(item)
        elif isinstance(v, torch.Tensor):
            a = v.detach().cpu().contiguous().numpy()
            digest.update(b'tensor' + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
        elif isinstance(v, np.ndarray):
            digest.update(b'array' + str(v.dtype).encode() + str(v.shape).encode() + v.tobytes())
        elif isinstance(v, bytes):
            digest.update(b'bytes' + hashlib.sha256(v).digest())
        elif v is None or isinstance(v, (str, int, float, bool)):
            digest.update(json.dumps(v, sort_keys=True).encode() + b'\0')
        else:
            raise TypeError('Unsupported controller payload field: ' + str(type(v)))
    add(value)
    return digest.hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-index', type=Path, required=True)
    parser.add_argument('--baseline-dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh pool directory')
    torch.set_num_threads(1)
    index = json.loads(args.source_index.read_text())
    baseline_path = Path(index['baseline_policy'])
    baseline = torch.load(baseline_path, map_location='cpu', weights_only=False)
    if fingerprint(baseline['base_controller']) != index['base_controller_fingerprint']:
        raise ValueError('Baseline frozen controller changed')
    protocol = json.loads((ROOT / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json').read_text())
    protocol['_root'] = str(ROOT)
    candidates = []
    seen = set()
    for source in sorted(index['sources'], key=lambda s: s['evaluation']):
        if source['policy_sha256'] in seen:
            continue
        policy_path = Path(source['policy'])
        if sha(policy_path) != source['policy_sha256']:
            raise ValueError('Historical source policy changed')
        policy = torch.load(policy_path, map_location='cpu', weights_only=False)
        if (fingerprint(policy['base_controller']) != index['base_controller_fingerprint']
                or policy['residual_joint_scale_rad'] != baseline['residual_joint_scale_rad']
                or policy['residual_arm_target_rate_rad_s'] != baseline['residual_arm_target_rate_rad_s']
                or policy.get('residual_scale_limit_rad') != baseline.get('residual_scale_limit_rad')
                or policy['residual_training_metadata']['recovery_assistance'] !=
                    baseline['residual_training_metadata']['recovery_assistance']):
            raise ValueError('Historical source does not share the exact base/action contract')
        summary, _ = read_run(Path(source['evaluation']), protocol)
        successes = sorted(s['training_index'] for s in summary['stages'] if s['complete5'])
        if summary['policy_sha256'] != source['policy_sha256'] or successes != source['complete_ids']:
            raise ValueError('Historical success labels changed')
        candidates.append(source)
        seen.add(source['policy_sha256'])
    primary = next(s for s in candidates if s['policy_sha256'] == sha(baseline_path))
    chosen = [dict(source=primary, assigned=primary['complete_ids'])]
    covered = set(primary['complete_ids'])
    while True:
        available = [s for s in candidates if set(s['complete_ids']) - covered]
        if not available:
            break
        best = max(available, key=lambda s: len(set(s['complete_ids']) - covered))
        new = sorted(set(best['complete_ids']) - covered)
        chosen.append(dict(source=best, assigned=new)); covered.update(new)
    args.output.mkdir(parents=True)
    plan = dict(schema='common-base-native-success-pool-plan-v1', status='BUILDING',
                source_index_sha256=sha(args.source_index), baseline_policy_sha256=sha(baseline_path),
                common_base_controller_fingerprint=index['base_controller_fingerprint'],
                selected_sources=chosen, unique_conditions=len(covered), goal_complete=False)
    atomic_json(args.output / 'state.json', plan)
    all_x, all_y, all_commands, offsets, episodes, sources = [], [], [], [0], [], []
    try:
        for source_id, selection in enumerate(chosen):
            source = selection['source']
            if source_id == 0:
                dataset = args.baseline_dataset
            else:
                dataset = args.output / f'source_{source_id:02d}'
                command = [sys.executable, str(ROOT / 'scripts/build_verified_fixed_retention_v2.py'),
                           '--evaluation', source['evaluation'], '--policy', source['policy'],
                           '--output', str(dataset)]
                with (args.output / f'source_{source_id:02d}.log').open('x') as log:
                    subprocess.run(command, cwd=ROOT, check=True, timeout=900,
                                   stdout=log, stderr=subprocess.STDOUT)
            manifest_path = dataset / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            source_record = manifest['sources'][0]
            if (manifest['schema'] != 'verified-fixed-success-residual-retention-v1'
                    or manifest.get('label_validation') != 'actor-and-command-replay-v1'
                    or manifest['minimum_speed_mps'] != 5. or len(manifest['sources']) != 1
                    or source_record['policy_sha256'] != source['policy_sha256']
                    or not source_record['command_replay_validated']
                    or sha(dataset / manifest['data_file']) != manifest['data_sha256']):
                raise ValueError('Source lacks complete native command replay evidence')
            selected_record = dict(source_record, original_dataset_manifest=str(manifest_path.resolve()),
                                   original_dataset_manifest_sha256=sha(manifest_path))
            sources.append(selected_record)
            with np.load(dataset / manifest['data_file'], allow_pickle=False) as z:
                for episode_id, entry in enumerate(manifest['episodes']):
                    if entry['source_training_index'] not in selection['assigned']:
                        continue
                    start, end = map(int, z['episode_offsets'][episode_id:episode_id+2])
                    if end - start != 800 or entry['max_command_error'] > 5e-4:
                        raise ValueError('Require full8s independently successful command sequences')
                    all_x.append(z['observations'][start:end].copy())
                    all_y.append(z['residual_targets'][start:end].copy())
                    all_commands.append(z['dispatched_actions'][start:end].copy())
                    episodes.append(dict(entry, source_index=source_id))
                    offsets.append(offsets[-1] + end - start)
            print(json.dumps(dict(source_id=source_id, copied_conditions=len(selection['assigned']),
                                  evaluation=source['evaluation'], total_copied=len(episodes))), flush=True)
        actual_ids = [e['source_training_index'] for e in episodes]
        if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != covered:
            raise ValueError('Pool did not preserve exactly one native success per assigned condition')
        data_path = args.output / 'residual_retention.npz'
        temp = data_path.with_suffix('.npz.tmp')
        with temp.open('xb') as stream:
            np.savez_compressed(stream, observations=np.concatenate(all_x), residual_targets=np.concatenate(all_y),
                                dispatched_actions=np.concatenate(all_commands), episode_offsets=np.asarray(offsets),
                                episode_source_policy_index=np.asarray([e['source_index'] for e in episodes]),
                                episode_training_index=np.asarray(actual_ids))
        temp.replace(data_path)
        manifest = dict(schema='verified-fixed-success-residual-retention-v1', sources=sources, episodes=episodes,
                        episode_count=len(episodes), samples=offsets[-1], minimum_speed_mps=5.,
                        protocol_sha256=sha(ROOT / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'),
                        target_interface='raw-residual18-before-mixer', data_file=data_path.name, data_sha256=sha(data_path),
                        label_validation='actor-and-command-replay-v1', cpu_gpu_command_tolerance=5e-4,
                        common_base_controller_fingerprint=index['base_controller_fingerprint'],
                        source_index_sha256=sha(args.source_index), generator_sha256=sha(__file__),
                        baseline_successes_preserved=sorted(primary['complete_ids']),
                        action_override_at_deployment=False, formal_acceptance=False,
                        online_selector_evaluated=False, student_complete5_rate=None,
                        scope='Historical successful full native episodes with source-specific actor labels and command replay. '
                              'Source option identities and episode boundaries retained; no action averaging or runtime oracle. '
                              'This is a demonstration pool, not a policy success rate or a live training modification.')
        atomic_json(args.output / 'manifest.json', manifest)
        plan.update(status='FINISHED', episodes=len(episodes), samples=offsets[-1],
                    manifest_sha256=sha(args.output / 'manifest.json'),
                    max_command_replay_error=max(e['max_command_error'] for e in episodes))
        atomic_json(args.output / 'state.json', plan)
        print(json.dumps({k: v for k, v in plan.items() if k != 'selected_sources'}), flush=True)
    except BaseException as error:
        plan.update(status='FAILED', error=repr(error)); atomic_json(args.output / 'state.json', plan)
        raise


if __name__ == '__main__':
    main()
