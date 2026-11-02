#!/usr/bin/env python3
"""Build a causal recovery-state guidance set from native failure traces.

The target is the zero residual fallback over the measured recovery controller.
These rows are recovery corrections only; they are never labeled as complete
expert demonstrations or used to change the task success definition.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--traces', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase-index', type=int, default=774)
    p.add_argument('--teacher-policy', type=Path, default=None,
                   help='Optional exported residual teacher used as the paired correction target')
    args = p.parse_args()
    if args.output.exists() or args.phase_index not in (773, 774):
        raise ValueError('Require a fresh output and public phase 773 or 774')
    root = args.traces.resolve()
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    observations, targets, episodes = [], [], []
    teacher = None
    if args.teacher_policy is not None:
        payload = torch.load(args.teacher_policy, map_location='cpu', weights_only=False)
        blob = payload.get('residual_actor_jit') if isinstance(payload, dict) else None
        if not isinstance(blob, (bytes, bytearray)):
            raise ValueError('Teacher policy must contain residual_actor_jit')
        teacher = torch.jit.load(__import__('io').BytesIO(blob), map_location='cpu').eval()
    rows = 0
    for meta_path in sorted(root.glob('*.json')):
        if meta_path.name == 'manifest.json':
            continue
        meta = json.loads(meta_path.read_text())
        reasons = meta.get('selection_reasons', [])
        if not any('ball_positive' in r and 'recovery_failure' in r for r in reasons):
            continue
        data_path = root / meta['data_file']
        if sha(data_path) != meta['data_sha256']:
            raise ValueError('Trace data hash mismatch: ' + meta['data_file'])
        with np.load(data_path, allow_pickle=False) as data:
            public = unpack_public(data['packed_public']).astype(np.float32, copy=False)
        if public.shape[1:] != (794,) or not np.isfinite(public).all():
            raise ValueError('Invalid public observation shape or values')
        mask = public[:, args.phase_index] > .5
        if not mask.any():
            continue
        selected = public[mask].copy()
        observations.append(selected)
        if teacher is None:
            target = np.zeros((len(selected), 18), dtype=np.float32)
            target_kind = 'zero_residual_base_fallback'
        else:
            with torch.inference_mode():
                target = teacher(torch.from_numpy(selected)).numpy().astype(np.float32, copy=False)
            if target.shape != (len(selected), 18) or not np.isfinite(target).all():
                raise ValueError('Teacher produced invalid residual targets')
            target_kind = 'paired_posthit_teacher_residual'
        targets.append(target)
        episodes.append(dict(data_file=meta['data_file'], source_training_index=meta['source_training_index'],
                             samples=int(len(selected)), phase_index=args.phase_index,
                             selection_reasons=reasons, target_kind=target_kind))
        rows += len(selected)
    if not observations:
        raise ValueError('No ball-positive recovery-failure states in requested phase')
    args.output.mkdir(parents=True)
    data_path = args.output / 'guidance.npz'
    np.savez_compressed(data_path, observations=np.concatenate(observations),
                        residual_targets=np.concatenate(targets))
    result = dict(schema='failure-recovery-guidance-v1', source_traces=str(root),
                  source_manifest_sha256=sha(manifest_path), trace_schema=manifest.get('schema'),
                  target_interface='raw-residual18-before-mixer',
                  target_kind=('paired_posthit_teacher_residual' if teacher is not None else 'zero_residual_base_fallback'),
                  teacher_policy=str(args.teacher_policy.resolve()) if args.teacher_policy is not None else None,
                  teacher_policy_sha256=sha(args.teacher_policy) if args.teacher_policy is not None else None,
                  phase_index=args.phase_index, episodes=episodes, episode_count=len(episodes),
                  samples=rows, data_file=data_path.name, data_sha256=sha(data_path),
                  failure_states_are_not_expert_success=True, native_execution=True,
                  formal_acceptance=False, goal_complete=False)
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(episodes=len(episodes), samples=rows, data_sha256=result['data_sha256'])))


if __name__ == '__main__':
    main()
