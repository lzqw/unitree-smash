#!/usr/bin/env python3
"""Fit actual student-state corrections while retaining verified old successes.

Requires a completed, audited assisted dataset. This CPU supervised update is
followed by native teacher-off evaluation; fitting cannot promote the policy.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_native_recovery_command_v1 import load_episodes
from scripts.train_native_recovery_command_v2 import evaluate_events
from quadsmash.rl.paired_native_recovery_dataset_v1 import load_paired_episodes
from quadsmash.control.learned_recovery_command_v1 import admit_native_recovery_head


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_batch(sources, rng, size):
    """Equal source exposure, then correction category, episode and event window.

    Case indices only identify training records; they never enter features.
    A source without one correction category retains its available category.
    """
    if len(sources) != 2 or size < 2 or size % 2:
        raise ValueError('Two nonempty sources and an even batch size required')
    selected = []
    for episodes in sources:
        groups = [[e for e in episodes if e['corrected'] == flag] for flag in (False, True)]
        groups = [g for g in groups if g]
        if not groups:
            raise ValueError('Empty supervision source')
        for _ in range(size // 2):
            group = groups[int(rng.integers(len(groups)))]
            e = group[int(rng.integers(len(group)))]
            n = len(e['features'])
            upper = (min(n, 5), min(n, 100), n)[int(rng.integers(3))]
            selected.append((e, int(rng.integers(upper))))
    return (torch.stack([e['features'][k] for e, k in selected]),
            torch.stack([e['delta'][k] for e, k in selected]))


def validate_provenance(directory, manifest):
    validation = json.loads((directory / 'validation.json').read_text())
    if (validation.get('passed') is not True or validation.get('trainable') is not True or
            validation['manifest_sha256'] != sha(directory / 'manifest.json') or
            manifest.get('failed_episodes_admitted') != 0 or
            manifest.get('teacher_increment_is_total_label') is not False or
            manifest.get('teacher_changed_rows', 0) <= 0):
        raise ValueError('Require exported successful actual teacher corrections')
    for path, digest in manifest['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Paired source evidence changed: ' + path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paired-dataset', type=Path, required=True)
    p.add_argument('--retention-dataset', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--seed', type=int, default=882086)
    p.add_argument('--max-seconds', type=float, default=600.)
    a = p.parse_args()
    if (a.output.exists() or a.steps < 1 or not math.isfinite(a.max_seconds) or
            a.max_seconds <= 0 or a.batch_size < 2 or a.batch_size % 2):
        raise ValueError('Fresh output, finite training and even batch size required')
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    paired_manifest, paired, limits = load_paired_episodes(a.paired_dataset)
    validate_provenance(a.paired_dataset, paired_manifest)
    old_manifest, old, old_limits = load_episodes(a.retention_dataset)
    for key in ('base_policy_sha256', 'protocol_sha256', 'control_dt_s', 'arm_target_rate_rad_s'):
        if paired_manifest[key] != old_manifest[key]:
            raise ValueError('Paired and retention execution differ: ' + key)
    if not torch.equal(limits, old_limits) or not old or not any(not e['corrected'] for e in old):
        raise ValueError('Original successful unchanged behavior must be retained')
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if (sha(a.checkpoint) != paired_manifest['source_student_head_sha256'] or
            checkpoint['dataset_manifest_sha256'] != sha(a.retention_dataset / 'manifest.json')):
        raise ValueError('Start from the actual corrected student and its original retention set')
    head = admit_native_recovery_head(checkpoint, paired_manifest['base_policy_sha256'],
                                     paired_manifest['protocol_sha256'], limits, .01).train()
    frozen_normalizer = {k: v.clone() for k, v in head.state_dict().items() if k.startswith('feature_')}
    # New data distribution: use a fresh optimizer rather than hidden old moments.
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-4)
    sources = [old, paired]
    before = [evaluate_events(head, episodes, limits) for episodes in sources]
    a.output.mkdir(parents=True)
    start = time.monotonic()
    history = []
    width = limits[:, 1] - limits[:, 0]
    for step in range(a.steps):
        features, target = sample_batch(sources, rng, a.batch_size)
        # Total replacement delta against original base, not teacher-only delta.
        square = ((head(features) - target) * width).square()
        loss = square.mean() + .25 * square.max(dim=1).values.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(gradient):
            raise ValueError('Nonfinite supervised update')
        optimizer.step()
        if step % 250 == 0 or step + 1 == a.steps:
            history.append(dict(step=step + 1, loss_rad2=float(loss.detach()), elapsed_s=time.monotonic() - start))
            print(json.dumps(history[-1]), flush=True)
        if time.monotonic() - start >= a.max_seconds:
            break
    if any(not torch.equal(v, head.state_dict()[k]) for k, v in frozen_normalizer.items()):
        raise ValueError('Frozen feature mapping changed')
    after = [evaluate_events(head.eval(), episodes, limits) for episodes in sources]
    parent_steps = checkpoint['steps']
    checkpoint.update(head_state_dict=head.state_dict(), optimizer_state_dict=optimizer.state_dict(),
        torch_rng_state=torch.get_rng_state(), numpy_rng_state=rng.bit_generator.state,
        steps=parent_steps + step + 1, refinement_parent_sha256=sha(a.checkpoint),
        paired_dataset_manifest_sha256=sha(a.paired_dataset / 'manifest.json'),
        refinement='actual student-state total correction plus equal original success retention')
    torch.save(checkpoint, a.output / 'head.pt')
    files = [Path(__file__), a.checkpoint, a.paired_dataset / 'manifest.json',
             a.retention_dataset / 'manifest.json', a.paired_dataset / 'validation.json',
             ROOT / 'quadsmash/rl/paired_native_recovery_dataset_v1.py',
             ROOT / 'scripts/train_native_recovery_command_v1.py',
             ROOT / 'scripts/train_native_recovery_command_v2.py',
             ROOT / 'quadsmash/control/learned_recovery_command_v1.py']
    report = dict(schema='paired-native-recovery-command-fit-v1', seed=a.seed,
        refinement_steps=step + 1, elapsed_s=time.monotonic() - start, device='cpu',
        checkpoint_sha256=sha(a.output / 'head.pt'), parent_checkpoint_sha256=sha(a.checkpoint),
        source_order=['original_verified_success_retention', 'actual_student_state_corrections'],
        source_episodes=[len(e) for e in sources], source_rows=[sum(len(e['features']) for e in s) for s in sources],
        before=before, after=after, history=history, source_sha256={str(f.resolve()): sha(f) for f in files},
        source_selection='equal sources; correction groups; uniform episodes; equal first5/first100/full window',
        optimizer_reset=True, feature_normalizer_frozen=True, base_actor_changed=False,
        ppo_training=False, native_evaluated=False, teacher_off_learning_verified=False,
        case_id_as_input=False, automatic_promotion=False, formal_acceptance=False)
    (a.output / 'fit_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('refinement_steps', 'elapsed_s', 'checkpoint_sha256')}))


if __name__ == '__main__':
    main()
