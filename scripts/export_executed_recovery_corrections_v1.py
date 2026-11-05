#!/usr/bin/env python3
"""Export result-filtered, actually executed recovery residuals from TRAIN.

The source may still be running: this exporter snapshots one atomic manifest
and reads only immutable, completed episode files named in that snapshot.
It never treats an offline teacher query as a verified correction, never
reconstructs student samples, and never claims frozen-policy performance.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_residual_v4 import ResidualActionMixerV4
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.export_complete_demonstrations_v1 import classify

SCHEMA = 'verified-executed-recovery-corrections-v1'
COMMAND_TOLERANCE = 5e-4  # Existing CPU/GPU production-command replay tolerance.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_dispatch(public, residual, base, dispatched, provenance, contract):
    """Replay the real stateful mixer from clock zero, on whole episodes.

    Array layout is [episode, control step, feature]. No inverse through
    saturation or arm slew limits is used to generate the targets.
    """
    if (public.ndim != 3 or public.shape[-1] != 794
            or residual.shape != (*public.shape[:2], 18)
            or base.shape != residual.shape or dispatched.shape != residual.shape
            or not all(np.isfinite(x).all() for x in [public, residual, base, dispatched])
            or np.abs(dispatched).max() > 1.000001):
        raise ValueError('Invalid actual observation/residual/base/command arrays')
    if not np.allclose(public[:, :, 775], np.arange(public.shape[1])[None] * .01,
                       atol=2e-6, rtol=0.):
        raise ValueError('Mixer replay requires unskipped whole episodes from clock zero')
    mixer = ResidualActionMixerV4(len(public), 'cpu', contract['joint_scale_rad'],
        contract['arm_target_rate_rad_s'], contract['residual_scale_limit_rad'])
    default = torch.tensor(provenance['default_joint_position_rad'], dtype=torch.float32)
    limits = torch.tensor(provenance['joint_limits_rad'], dtype=torch.float32)
    errors = np.zeros(len(public), dtype=np.float64)
    with torch.inference_mode():
        for tick in range(public.shape[1]):
            actual = mixer.action(torch.from_numpy(base[:, tick].copy()),
                torch.from_numpy(residual[:, tick].copy()),
                torch.from_numpy(public[:, tick].copy()), default, limits).numpy()
            errors = np.maximum(errors, np.abs(actual - dispatched[:, tick]).max(axis=1))
    if not np.isfinite(errors).all() or errors.max() > COMMAND_TOLERANCE:
        raise ValueError('Recorded residuals do not reproduce native dispatch: ' + str(errors.max()))
    return errors


def phase_rows(public):
    if public.ndim != 2 or public.shape[1] != 794 or not np.isfinite(public).all():
        raise ValueError('Require actual finite public794 observations')
    return public[:, 774] > .5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-per-case', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.max_per_case < 1:
        raise ValueError('Require a fresh output and positive per-case episode limit')
    torch.set_num_threads(1)
    source = args.source_run.resolve()
    trace_root = source / 'trajectory_traces'
    snapshot = (trace_root / 'manifest.json').read_bytes()
    manifest = json.loads(snapshot)
    inputs = json.loads((source / 'INPUTS.json').read_text())
    contract = inputs['contract']
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol = json.loads(protocol_path.read_text()); protocol['_root'] = str(ROOT)
    provenance = manifest['provenance']
    assistance = contract.get('posthit_teacher_assistance', {})
    if (not provenance.get('native_source_confirmed') or manifest.get('disabled_error')
            or manifest.get('formal_acceptance') or inputs.get('formal_cases_read')
            or provenance['protocol_sha256'] != sha(protocol_path)
            or contract['protocol_sha256'] != sha(protocol_path)
            or contract['attack_speed_threshold_mps'] != 5.
            or not assistance.get('enabled') or not assistance.get('recovery_only')
            or contract.get('recovery_assistance_execution') != 'reset-persistent-recovery-assistance-v1'):
        raise ValueError('Require original TRAIN execution with declared recovery-only teacher assistance')
    available = Counter(); selected = []; selected_count = Counter()
    for entry in manifest['saved_episodes']:
        path = trace_root / entry['data_file']
        if path.name != entry['data_file']:
            raise ValueError('Trace must be a basename in the source directory')
        metadata = path.with_suffix('.json')
        meta = json.loads(metadata.read_text())
        row = meta['raw_episode']
        label = classify(row, protocol, 5.)
        if not label['curriculum_complete']:
            continue
        case = int(meta['source_training_index']); available[case] += 1
        if selected_count[case] >= args.max_per_case:
            continue
        if (not meta.get('native_execution') or meta['protocol_sha256'] != sha(protocol_path)
                or meta.get('formal_cases_read') or row['source_training_index'] != case
                or meta['data_sha256'] != entry['data_sha256'] or sha(path) != meta['data_sha256']
                or hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest() != meta['raw_episode_sha256']):
            raise ValueError('Completed native episode identity or hash mismatch')
        with np.load(path, allow_pickle=False) as data:
            public = unpack_public(data['packed_public'])
            values = {k: data[k].copy() for k in ['residual_actions', 'base_actions', 'native_actions',
                                                 'episode_clock_s', 'learner_iteration']}
        if (len(public) != 800 or meta['samples'] != 800
                or hashlib.sha256(public.tobytes()).hexdigest() != meta['full_public794_sha256']
                or not np.array_equal(public[:, 775], values['episode_clock_s'])
                or np.unique(values['learner_iteration']).tolist() != meta['actor_iterations_encountered']):
            raise ValueError('Full episode/public observation identity differs')
        mask = phase_rows(public)
        if not mask.any():
            continue
        selected_count[case] += 1
        selected.append(dict(public=public, values=values, mask=mask, meta=meta,
            trace_path=path, metadata_path=metadata, label=label))
    if not selected:
        raise ValueError('No actual full successes with recovery-phase actions')
    errors = verify_dispatch(np.stack([e['public'] for e in selected]),
        np.stack([e['values']['residual_actions'] for e in selected]),
        np.stack([e['values']['base_actions'] for e in selected]),
        np.stack([e['values']['native_actions'] for e in selected]), provenance, contract)
    observations = []; targets = []; commands = []; offsets = [0]; episodes = []
    for e, error in zip(selected, errors):
        mask, meta = e['mask'], e['meta']
        public = e['public'][mask].copy()
        observations.append(public)
        targets.append(e['values']['residual_actions'][mask])
        commands.append(e['values']['native_actions'][mask])
        offsets.append(offsets[-1] + len(public))
        episodes.append(dict(case_id=meta['raw_episode']['case_id'],
            source_training_index=meta['source_training_index'], samples=len(public),
            source_trace=meta['data_file'], source_trace_sha256=meta['data_sha256'],
            metadata_sha256=sha(e['metadata_path']), full_public_sha256=meta['full_public794_sha256'],
            actor_iterations=meta['actor_iterations_encountered'],
            max_command_error=float(error), complete5=True,
            recovery_phase_rows_before_hit=int((public[:, 775] < meta['raw_episode']['contact_time_s']).sum()),
            recovery_phase_rows_after_release=int((public[:, 775] >= meta['raw_episode']['release_time_s']).sum()),
            actual_outgoing_speed_mps=e['label']['outgoing_speed_mps']))
    args.output.mkdir(parents=True)
    (args.output/'source_manifest_snapshot.json').write_bytes(snapshot)
    (args.output/'source_inputs_snapshot.json').write_text(json.dumps(inputs, indent=2)+'\n')
    data_path = args.output/'executed_recovery_corrections.npz'
    np.savez_compressed(data_path, observations=np.concatenate(observations),
        residual_targets=np.concatenate(targets), dispatched_actions=np.concatenate(commands),
        episode_offsets=np.array(offsets,dtype=np.int64))
    result = dict(schema=SCHEMA, source_run=str(source),
        source_manifest_sha256=hashlib.sha256(snapshot).hexdigest(), source_inputs_sha256=sha(source/'INPUTS.json'),
        execution_contract=contract, source_update_at_export=json.loads((source/'progress.json').read_text())['update'],
        source_available_complete_episode_count=sum(available.values()),
        source_available_case_indices=sorted(available), max_episodes_per_case=args.max_per_case,
        episodes=episodes, episode_count=len(episodes), case_indices=sorted(selected_count), samples=offsets[-1],
        data_file=data_path.name, data_sha256=sha(data_path),
        target_interface='raw-residual18-before-mixer', target_kind='actually_executed_teacher_mixed_residual',
        minimum_speed_mps=5., protocol_sha256=sha(protocol_path),
        phase_gate='public_phase774_gt_0.5_including_pre_hit_support_if_present',
        command_replay_validated=True, max_command_error=float(errors.max()),
        cpu_gpu_command_tolerance=COMMAND_TOLERANCE,
        raw_student_actions_available=False, teacher_only_action_validated=False,
        full_success_only=True, recovery_only_and_failed_episodes_excluded=True,
        native_execution=True, frozen_policy_performance=False, formal_acceptance=False,
        scope='Outcome-filtered executed actions from completed ground-start teacher-assisted TRAIN episodes. '
              'Only observed recovery-phase rows are labels. Not teacher-only action labels, '
              'not reconstructed student actions, and not a new frozen-policy success result.')
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(episodes=len(episodes),cases=len(selected_count),samples=offsets[-1],
        max_command_error=float(errors.max()),before_hit_rows=sum(e['recovery_phase_rows_before_hit'] for e in episodes),
        after_release_rows=sum(e['recovery_phase_rows_after_release'] for e in episodes),
        manifest_sha256=sha(args.output/'manifest.json')),indent=2),flush=True)


if __name__ == '__main__':
    main()
