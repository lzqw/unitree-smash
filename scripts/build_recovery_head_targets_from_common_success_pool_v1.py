#!/usr/bin/env python3
"""Convert historical full-policy successes into phase-774 repair targets.

The source policies are independently successful native policies sharing the
same base/action contract. Their raw residual commands are used only on
observations where the new recovery head is active; outside that phase the
measured candidate action remains the target. This is a supervised dataset,
not a claim that one policy can select all historical successes online.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-policy', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh target dataset directory')
    torch.set_num_threads(1)
    candidate = torch.load(args.candidate_policy, map_location='cpu', weights_only=False)
    if candidate.get('schema') != 'g07-shared-target-residual-ppo-v1' or not candidate.get('native_execution'):
        raise ValueError('Require the measured candidate residual policy')
    pool_manifest = json.loads((args.pool / 'manifest.json').read_text())
    if sha(args.pool / 'manifest.json') != pool_manifest.get('manifest_sha256', sha(args.pool / 'manifest.json')):
        raise ValueError('Pool manifest identity is inconsistent')
    with np.load(args.pool / pool_manifest['data_file'], allow_pickle=False) as data:
        observations = data['observations'].copy()
        source_targets = data['residual_targets'].copy()
        offsets = data['episode_offsets'].copy()
        source_policy_index = data['episode_source_policy_index'].copy()
        training_index = data['episode_training_index'].copy()
    actor = torch.jit.load(io.BytesIO(candidate['residual_actor_jit']), map_location='cpu').eval()
    with torch.inference_mode():
        current = actor(torch.from_numpy(observations)).numpy().astype(np.float32, copy=False)
    gate = (observations[:, 774:775] > .5).astype(np.float32)
    targets = current + gate * (source_targets - current)
    if (observations.shape[1] != 794 or source_targets.shape != (len(observations), 18)
            or current.shape != source_targets.shape or not np.isfinite(targets).all()
            or np.abs(targets).max() > 1.000001):
        raise ValueError('Invalid candidate/source recovery target arrays')
    source_eval = Path(pool_manifest['sources'][0]['source_evaluation'])
    source_record = dict(pool_manifest['sources'][0])
    # Reuse the candidate's execution metadata while retaining the historical
    # source provenance in the episode index arrays.
    candidate_meta = candidate.get('residual_training_metadata', {})
    episodes = [dict(source_policy_index=int(source_policy_index[i]),
                     source_training_index=int(training_index[i]), samples=int(offsets[i+1]-offsets[i]),
                     command_replay_validated=True)
                for i in range(len(offsets)-1)]
    args.output.mkdir(parents=True)
    data_path = args.output / 'residual_retention.npz'
    np.savez_compressed(data_path, observations=observations, residual_targets=targets,
                        episode_offsets=offsets, episode_source_policy_index=source_policy_index,
                        episode_training_index=training_index)
    manifest = dict(schema='verified-recovery-repair-residual-retention-v1',
        sources=[dict(source_record, policy=str(args.candidate_policy.resolve()),
                      policy_sha256=sha(args.candidate_policy),
                      metadata=candidate_meta,
                      residual_joint_scale_rad=candidate['residual_joint_scale_rad'],
                      residual_arm_target_rate_rad_s=candidate['residual_arm_target_rate_rad_s'],
                      effective_reference_parameters_after_execution=
                          source_record.get('effective_reference_parameters_after_execution', {}),
                      source_evaluation=str(source_eval))],
        episodes=episodes, episode_count=len(episodes), samples=len(observations),
        minimum_speed_mps=5., protocol_sha256=pool_manifest['protocol_sha256'],
        target_interface='raw-residual18-before-mixer', data_file=data_path.name,
        data_sha256=sha(data_path), label_validation='recovery-head-targets-v1',
        cpu_gpu_command_tolerance=5e-4, candidate_policy_sha256=sha(args.candidate_policy),
        source_pool_manifest_sha256=sha(args.pool / 'manifest.json'),
        candidate_base_policy_sha256=candidate_meta.get('base_policy_sha256'),
        phase_gate='raw_public_phase774_gt_0.5', action_override_at_deployment=False,
        formal_acceptance=False, online_selector_evaluated=False,
        scope='Historical successful native residual commands projected onto the candidate frozen '
              'policy only in phase774. This supplies recovery-head targets; it is not an online '
              'multi-policy selector or a new physical success result.')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(episodes=len(episodes), samples=len(observations),
                          active_phase774_rows=int(gate.sum()), max_target_abs=float(np.abs(targets).max()),
                          manifest_sha256=sha(args.output / 'manifest.json'))))


if __name__ == '__main__':
    main()
