"""Check a completed gain/body candidate against original cases and actual actions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports/G07_full_racket_v4'
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from quadsmash.control.full_racket_gain_body_adaptation_v4 import GainBodyAdaptationControllerV4
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-id', type=int, required=True, choices=range(8))
    args = parser.parse_args()
    output = HERE / f'candidate_{args.candidate_id:03d}'
    assert not (output / 'RESULT.json').exists()
    run = REPORTS / 'gain_body8_sequential_n128_01'
    source_path = REPORTS / 'prepose_origin0_8_sequential_n128_01/best_candidate.pt'
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v4_r3_origin0.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    inputs = read(run / 'INPUTS.json')
    assert inputs['source_policy_sha256'] == sha(source_path)
    assert inputs['protocol_sha256'] == sha(protocol_path) and inputs['training_indices'] == indices
    assert sha(ROOT / protocol['asset']['path']) == protocol['asset']['sha256']
    assert all(sha(ROOT / name) == digest for name, digest in protocol['physical_configs'].items())
    proposal_path = REPORTS / 'gain_body_after_prepose_queue02/ACTUAL_PROPOSALS.json'
    assert sha(proposal_path) == inputs['proposal_sha256']
    entry = next(e for e in read(run / 'metrics.json')['history'] if e['candidate_id'] == args.candidate_id)
    candidate = entry['candidate']
    assert candidate == read(proposal_path)['candidates'][args.candidate_id]
    suffix = f'{args.candidate_id:03d}'
    initial = read(run / f'initial_ground_states_{suffix}.json')
    rows = read(run / f'episodes_{suffix}.json')
    assert len(rows) == 128 and initial['all_clocks_zero']
    assert not np.asarray(initial['actual_environment_origins_m']).any()
    roots = np.asarray(initial['root_state_local'])
    assert roots.shape == (128, 13) and np.isfinite(roots).all()
    root_error = float(np.max(abs(roots[:, :3] - protocol['initialization']['root_position_m'])))
    assert root_error < 1e-5 and np.max(abs(roots[:, 3:7] - [1, 0, 0, 0])) < 1e-5
    assert np.max(abs(roots[:, 7:])) < 1e-5 and np.max(abs(np.asarray(initial['joint_velocity_rad_s']))) < 1e-5
    for lane, row in enumerate(rows):
        assert row['env_id'] == row['jump']['env_id'] == lane and row['candidate_id'] == args.candidate_id
        assert row['episode_index'] == initial['episode_ids'][lane]
        assert row['source_training_index'] == lane and row['case_id'] == cases[lane]['case_id'] and row['ground_init']
    summary = summarize_candidate(rows, protocol)
    assert summary == entry['summary'] and summary['numerical_failures'] == 0
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    controller = GainBodyAdaptationControllerV4(source, candidate['gains'], 2, 'cpu', candidate['body_deltas'])
    default, limits = [torch.tensor(inputs[k], dtype=torch.float32) for k in
                       ['default_joint_position_rad', 'joint_limits_rad']]
    public_path = run / f'public_and_actions_{suffix}.npz'
    with np.load(public_path) as arrays:
        public, actions, valid = [arrays[k].copy() for k in ['observations', 'actions', 'valid_first_episode']]
    maximum, previous = 0., None
    with torch.inference_mode():
        for obs, actual, keep in zip(public, actions, valid):
            observation = torch.from_numpy(obs)
            if previous is not None:
                reset = torch.where(observation[:, 775] < previous - 1e-5)[0]
                if len(reset):
                    controller.reset(reset)
            command = controller.action({'policy': observation}, default, limits).numpy()
            if keep.any():
                maximum = max(maximum, float(abs(command[keep] - actual[keep]).max()))
            previous = observation[:, 775].clone()
    assert maximum <= 5e-4
    held = read(REPORTS / 'origin0_residual_after_gain_body_queue01/PLAN.json')['source_sha256']
    assert all(sha(Path(path)) == digest for path, digest in held.items())
    native_sources = read(run / 'source_hashes.json')
    assert all(sha(ROOT / path) == digest for path, digest in native_sources.items())
    files = [source_path, protocol_path, proposal_path, ROOT / protocol['training_suite']['path'],
        run / 'INPUTS.json', run / 'source_hashes.json', public_path, Path(__file__)]
    files += [run / f'{name}_{suffix}.json' for name in ['episodes', 'initial_ground_states', 'contacts']]
    files.append(run / f'all_first_episodes_{suffix}.npz')
    counts = {key: sum(bool(row[key]) for row in rows) for key in
              ['had_contact', 'frame_hit', 'crossed_net', 'landed_legal_court', 'landed_target']}
    counts.update({key: sum(bool(row['jump'][key]) for row in rows) for key in
                  ['body_landing', 'stable_recovery', 'qualified_flight_seen', 'qualified_jump_landed']})
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_completed_candidate=entry,
        original128_rows_and_actual_native_episode_ids_checked=True,
        actual_initial_episode_indices=initial['episode_ids'], observed_counts=counts,
        maximum_initial_root_position_error_m=root_error, actual_valid_public_rows=int(valid.sum()),
        CPU_vs_actual_GPU_max_action_error=maximum, original_action_tolerance=5e-4,
        native_source_files_checked=len(native_sources), held482_unchanged=True,
        source_sha256={str(path): sha(path) for path in files},
        scope='One completed search candidate with original128 cases. This is not a completed8-way job, independent cold128 or formal A/B result; no later candidate outcome inferred.',
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False)
    output.mkdir(exist_ok=True)
    (output / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(candidate_id=args.candidate_id, summary=summary, observed_counts=counts,
        actual_valid_public_rows=int(valid.sum()), CPU_vs_actual_GPU_max_action_error=maximum)), flush=True)


if __name__ == '__main__':
    main()
