"""Read-only CPU command reconstruction on real selected TRAIN traces.

No native physics replay, causal policy ablation, exact behavior-mean recovery,
or population performance is claimed. All physical commands come from saved
100Hz native traces; the existing production mixer is checked on those inputs.
"""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT/'reports/G07_full_racket_v4'
RUN = REPORTS/'residual_ppo_centered_placement075_n2048_resume32_02'
SNAPSHOT = REPORTS/'video_snapshot_placement075_additional_complete_20261026'/RUN.name
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_residual_v4 import ResidualActionMixerV4
from quadsmash.evaluation_v4 import independent_episode
from quadsmash.rl.episode_trace_v4 import unpack_public


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    absolute = np.abs(values.astype(np.float64))
    return dict(samples=len(values), mean_abs_rad=float(absolute.mean()),
        p95_abs_rad=float(np.quantile(absolute, .95)), maximum_abs_rad=float(absolute.max()))


def main():
    output = HERE/'ACTUAL_SELECTED_TRACE_ACTION_CHECK.json'
    assert not output.exists()
    manifest_path, raw_path = SNAPSHOT/'observed_trace_manifest.json', SNAPSHOT/'observed_episodes.json'
    manifest, raw = read(manifest_path), read(raw_path)
    indexed = {(r['env_id'], r['episode_index']): r for r in raw}
    provenance = manifest['provenance']
    default = torch.tensor(provenance['default_joint_position_rad'], dtype=torch.float32)
    limits = torch.tensor(provenance['joint_limits_rad'], dtype=torch.float32)
    width = (limits[:, 1]-limits[:, 0]).numpy()
    assert default.shape == (18,) and limits.shape == (18, 2) and (width > 0).all()
    protocol_path = ROOT/'configs/g07_full_racket_eval_v4_r3.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(ROOT)
    config_path = ROOT/'configs/g07_full_racket_residual_ppo_centered_placement075_v4.json'
    config = read(config_path)
    scale = np.asarray(config['joint_scale_rad'], dtype=np.float32)
    assert np.array_equal(scale, np.array([.03]*12+[.15]*6, dtype=np.float32))
    checked, table, dependencies = [], [], [Path(__file__), manifest_path, raw_path, protocol_path, config_path,
        ROOT/'quadsmash/control/full_racket_residual_v4.py', ROOT/'quadsmash/control/full_racket_arm_target_rate_v4.py',
        ROOT/'quadsmash/rl/episode_trace_v4.py']
    for entry in manifest['saved_episodes']:
        path = RUN/'trajectory_traces'/entry['data_file']
        meta = read(path.with_suffix('.json'))
        key = entry['env_id'], entry['episode_index']
        row = indexed[key]
        assert sha(path) == entry['data_sha256'] == meta['data_sha256']
        assert json.dumps(row, sort_keys=True) == json.dumps(meta['raw_episode'], sort_keys=True)
        with np.load(path) as data:
            public = unpack_public(data['packed_public'])
            residual = data['residual_actions'].copy()
            base, actual = data['base_actions'].copy(), data['native_actions'].copy()
        assert hashlib.sha256(public.tobytes()).hexdigest() == meta['full_public794_sha256']
        assert np.isfinite(residual).all() and np.max(np.abs(actual)) <= 1.000001
        mixer = ResidualActionMixerV4(1, 'cpu', scale.tolist())
        reconstructed = []
        with torch.inference_mode():
            for index in range(len(public)):
                action = mixer.action(torch.from_numpy(base[index:index+1]),
                    torch.from_numpy(residual[index:index+1]), torch.from_numpy(public[index:index+1]), default, limits)
                reconstructed.append(action[0].numpy().copy())
        reconstructed = np.asarray(reconstructed)
        command_error = float(np.max(np.abs(actual-reconstructed)))
        # Command arithmetic synchronization only, never physical acceptance.
        arithmetic_bound = float(16*np.finfo(np.float32).eps)
        assert command_error <= arithmetic_bound, (key, command_error)
        unbounded = base+2*np.clip(residual, -1., 1.)*scale/width
        requested = np.clip(unbounded, -1., 1.)
        intended_delta = np.clip(residual, -1., 1.)*scale
        actual_delta = (actual-base)*width/2
        arm_slew_delta = (actual[:, 12:]-requested[:, 12:])*width[12:]/2
        clock = public[:, 775]
        contact = row['contact_time_s']
        phases = {'all_saved': np.ones(len(clock), dtype=bool),
            'preparation_before_2p2': clock < 2.2,
            'interception_2p2_to_3p3': (clock >= 2.2) & (clock <= 3.3)}
        if np.isfinite(contact):
            phases['last_020s_before_actual_contact'] = (clock >= contact-.2) & (clock <= contact)
            phases['recovery_after_actual_contact'] = clock > contact
        outcome = independent_episode(row, protocol)
        phase_results = {}
        for phase, mask in phases.items():
            if not mask.any():
                continue
            result = dict(leg_intended=stats(intended_delta[mask, :12]),
                arm_intended=stats(intended_delta[mask, 12:]),
                leg_actual_vs_frozen_base=stats(actual_delta[mask, :12]),
                arm_actual_vs_frozen_base=stats(actual_delta[mask, 12:]),
                arm_slew_correction=stats(arm_slew_delta[mask]),
                native_requested_clipping_fraction=float((abs(unbounded[mask]) > 1.).mean()),
                residual_raw_clipping_fraction=float((abs(residual[mask]) > 1.).mean()))
            phase_results[phase] = result
            table.append(dict(TRAIN=row['source_training_index'], env_id=key[0], episode_index=key[1],
                complete=outcome['complete'], phase=phase, samples=int(mask.sum()),
                leg_mean_abs_intended_rad=result['leg_intended']['mean_abs_rad'],
                leg_mean_abs_actual_delta_rad=result['leg_actual_vs_frozen_base']['mean_abs_rad'],
                arm_mean_abs_intended_rad=result['arm_intended']['mean_abs_rad'],
                arm_mean_abs_actual_delta_rad=result['arm_actual_vs_frozen_base']['mean_abs_rad'],
                arm_mean_abs_slew_correction_rad=result['arm_slew_correction']['mean_abs_rad'],
                requested_clipped_fraction=result['native_requested_clipping_fraction'],
                residual_clipped_fraction=result['residual_raw_clipping_fraction']))
        checked.append(dict(TRAIN=row['source_training_index'], env_id=key[0], episode_index=key[1],
            source_selection_reasons=entry['selection_reasons'], actual_outcome=outcome, samples=len(public),
            CPU_mixer_vs_saved_native_action_max_error=command_error, phases=phase_results))
        dependencies.extend([path, path.with_suffix('.json')])
    assert len(checked) == manifest['saved_episode_count'] and sum(e['actual_outcome']['complete'] for e in checked) == 3
    # Use an immutable saved actual progress publication for mean/std telemetry.
    # The exact online behavior mean cannot be recovered per selected trace:
    # normalizers update using every lane during the rollout.
    lines = (RUN/'stdout.log').read_text(errors='replace').splitlines()
    telemetry = []
    for line in lines:
        if line.startswith('{'):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get('update', 0) <= 22 and 'losses' in value and 'behavior_actions' in value['losses']:
                telemetry.append(value)
    assert [p['update'] for p in telemetry] == list(range(9, 23))
    telemetry_path = HERE/'ACTUAL_PROGRESS_UPDATES09_22.json'
    assert not telemetry_path.exists()
    telemetry_path.write_text(json.dumps(telemetry, indent=2)+'\n')
    dependencies.append(telemetry_path)
    latest = telemetry[-1]
    behavior = latest['losses']['behavior_actions']
    post = latest['losses']['post_update_actions']
    mean_abs = np.asarray(behavior['mean_abs_per_joint'])*scale
    behavior_std = np.asarray(behavior['std_per_joint'])*scale
    post_mean_abs = np.asarray(post['mean_abs_per_joint'])*scale
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        scope='Read-only CPU production-mixer reconstruction and physical command statistics on15 selected actual TRAIN traces from fixed6581-row capture. Not native physics replay, a causal ablation or population performance.',
        selected_actual_traces=len(checked), all_selected_trace_samples=sum(e['samples'] for e in checked),
        selected_complete_traces=3, command_arithmetic_bound=float(16*np.finfo(np.float32).eps),
        this_is_not_an_acceptance_tolerance=True, episodes=checked,
        actual_update22_fixed_stride_behavior_4096_rows=dict(
            requested_leg_mean_abs_rad=float(mean_abs[:12].mean()), requested_arm_mean_abs_rad=float(mean_abs[12:].mean()),
            requested_leg_mean_std_rad=float(behavior_std[:12].mean()), requested_arm_mean_std_rad=float(behavior_std[12:].mean()),
            post_update_leg_mean_abs_rad=float(post_mean_abs[:12].mean()), post_update_arm_mean_abs_rad=float(post_mean_abs[12:].mean()),
            expected_raw_residual_clipping_fraction=behavior['expected_clipping_fraction'],
            actual_raw_residual_clipping_fraction=behavior['actual_clipped_fraction'],
            post_update_KL=latest['losses']['post_update_kl'],
            post_update_clip_fraction=latest['losses']['post_update_clip_fraction'],
            no_exact_per_trace_behavior_means_claimed=True),
        limitations=['Selected successes, early failures and predefined lanes are not a representative policy sample.',
            'Physical command deltas are joint-target differences, not measured joint motion or task-space racket velocity.',
            '4096 behavior rows use fixed-stride rollout telemetry; no exact per-trace stochastic-minus-mean decomposition is possible from these captures.',
            'No conclusion about counterfactual outcome without a residual can be drawn from saved action replay.'],
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in dependencies},
        formal_cases_read=False, new_native_execution=False, formal_acceptance=False, goal_complete=False)
    with (HERE/'SELECTED_TRACE_PHASE_ACTIONS.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(selected_traces=len(checked), actual_samples=result['all_selected_trace_samples'],
        maximum_command_reconstruction_error=max(e['CPU_mixer_vs_saved_native_action_max_error'] for e in checked),
        actual_update22_telemetry=result['actual_update22_fixed_stride_behavior_4096_rows']), indent=2))


if __name__ == '__main__':
    main()
