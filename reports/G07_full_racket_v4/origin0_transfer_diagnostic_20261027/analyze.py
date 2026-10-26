"""Assess completed TRAIN transfers and compare actual preparation dynamics."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.control.full_racket_contact_aim_v4 import make_rally_controller
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE/'RESULT.json').exists()
    reports = ROOT/'reports/G07_full_racket_v4'
    old, new = [reports/name for name in ['contact_aim7_sequential_n128_01', 'prepose_origin0_8_sequential_n128_01']]
    protocol = read(ROOT/'configs/g07_full_racket_eval_v4_r3_origin0.json'); protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    history = read(new/'metrics.json')['history']
    assert len(history) >= 3
    checked, paths = [], []
    for candidate_id in range(3):
        suffix = f'{candidate_id:03d}'
        row_path, initial_path = new/f'episodes_{suffix}.json', new/f'initial_ground_states_{suffix}.json'
        rows, initial = read(row_path), read(initial_path)
        assert len(rows) == 128 and initial['all_clocks_zero']
        for lane, row in enumerate(rows):
            assert row['env_id'] == row['jump']['env_id'] == lane and row['candidate_id'] == candidate_id
            assert row['episode_index'] == initial['episode_ids'][lane] and row['ground_init']
            assert row['source_training_index'] == indices[lane] and row['case_id'] == cases[lane]['case_id']
        summary = summarize_candidate(rows, protocol)
        assert summary == history[candidate_id]['summary'] and summary['numerical_failures'] == 0
        checked.append(dict(candidate_id=candidate_id, name=history[candidate_id]['candidate']['name'], summary=summary,
            qualified_flight_seen=sum(r['jump']['qualified_flight_seen'] for r in rows),
            body_landing=sum(r['jump']['body_landing'] for r in rows),
            stable_recovery=sum(r['jump']['stable_recovery'] for r in rows),
            actual_racket_contacts=sum(r['had_contact'] for r in rows),
            frame_contacts=sum(r['frame_hit'] for r in rows),
            median_closest_string_gap_m=float(np.median([r['closest_string_gap_m'] for r in rows])),
            terminal_reason_counts={key:sum(r['training_terminal_reasons'][key] for r in rows)
                                    for key in rows[0]['training_terminal_reasons']}))
        paths.extend([row_path, initial_path])
    series, samples = {}, []
    for label, run, suffix in [('origin16', old, '002'), ('origin0', new, '000')]:
        geometry = run/'geometry.json'; base_index = read(geometry)['body_names'].index('base')
        pose_path = run/f'all_first_episodes_{suffix}.npz'
        with np.load(pose_path) as arrays:
            poses, clocks, valid = [arrays[k].copy() for k in ['link_poses', 'episode_clock_s', 'valid_first_episode']]
        valid = valid.astype(bool); root = poses[:, :, base_index]
        q = root[:, :, 3:]
        tilt = np.degrees(np.arccos(np.clip(1.-2*(q[:, :, 1]**2+q[:, :, 2]**2), -1., 1.)))
        before_launch = (clocks[:, 0] <= 2.+1e-6) & valid.all(1)
        prep = root[before_launch]
        # Equal public preparation before release should remain equal across
        # independent origin0 replicas. This is an observation, not a proof
        # of collision filtering during every possible later interaction.
        spread = abs(prep-prep[:, :1]).max()
        series[label] = dict(time_s=clocks[:, 0].tolist(),
            median_root_height_m=np.nanmedian(np.where(valid, root[:, :, 2], np.nan), axis=1).tolist(),
            median_root_tilt_deg=np.nanmedian(np.where(valid, tilt, np.nan), axis=1).tolist(),
            max_prelaunch_replica_pose_component_spread=float(spread))
        for target in [.5, 1., 1.5, 2., 2.6]:
            tick = int(np.argmin(abs(clocks[:, 0]-target))); good = valid[tick]
            samples.append(dict(layout=label, requested_time_s=target, actual_time_s=float(clocks[tick, 0]),
                actual_valid_lanes=int(good.sum()), median_root_position_m=np.median(root[tick, good, :3], axis=0).tolist(),
                median_root_tilt_deg=float(np.median(tilt[tick, good])),
                root_position_range_m=np.ptp(root[tick, good, :3], axis=0).tolist()))
        paths.extend([geometry, pose_path])
    source_path = reports/'contact_aim7_sequential_n128_01/best_candidate.pt'
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    assert sha(source_path) == read(new/'INPUTS.json')['source_policy_sha256']
    controller = make_rally_controller(source, 2, 'cpu'); node, layers = controller, []
    while True:
        if hasattr(node, 'actor'):
            layers.append(node); node = node.base
        elif hasattr(node, 'source'):
            node = node.source
        else:
            break
    assert len(layers) == 2
    public_path = new/'public_and_actions_000.npz'
    with np.load(public_path) as arrays:
        public, valid = [arrays[k].copy() for k in ['observations', 'valid_first_episode']]
    keep = valid.astype(bool) & (public[:, :, 775] <= 2.+1e-6)
    selected = torch.from_numpy(public[keep])
    residuals = []
    with torch.inference_mode():
        for index, layer in enumerate(layers):
            bounded = layer.actor(selected).clamp(-1., 1.)
            requested = (bounded*layer.mixer.scale).numpy()
            residuals.append(dict(layer_outer_to_inner=index, actual_prelaunch_public_rows=int(keep.sum()),
                leg_mean_absolute_requested_correction_rad=float(abs(requested[:, :12]).mean()),
                leg_max_absolute_requested_correction_rad=float(abs(requested[:, :12]).max()),
                arm_mean_absolute_requested_correction_rad=float(abs(requested[:, 12:]).mean()),
                arm_max_absolute_requested_correction_rad=float(abs(requested[:, 12:]).max()),
                physical_request_scale_rad=layer.mixer.scale.tolist(),
                scope='Requested bounded residual before actuator command clipping and slew; not measured joint motion.'))
    paths.extend([source_path, public_path, Path(__file__)])
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), completed_native_candidates=checked,
        actual_root_samples=samples, actual_preparation_series=series, requested_source_residuals=residuals,
        source_sha256={str(path):sha(path) for path in paths},
        causal_limit='Different numerical replica layouts, not a controlled actor ablation. Residual responsibility remains a hypothesis for the registered next native comparison.',
        scope='First three completed origin0 TRAIN candidates only. No entire-course, cold or formal success inferred.',
        formal_cases_read=False, formal_acceptance=False)
    (HERE/'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(checked_candidates=len(checked), checked_native_rows=384,
        exact_origin0_prelaunch_replica_pose_spread=series['origin0']['max_prelaunch_replica_pose_component_spread'],
        requested_source_residuals=residuals)), flush=True)


if __name__ == '__main__':
    main()
