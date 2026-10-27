"""Verify a completed measured-swing cold128 source and replay real commands."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_measured_source(path, source, evaluation, root, protocol_sha256):
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
    from scripts.initialize_centered_policy_v4 import same_tree
    path, evaluation, root = Path(path), Path(evaluation), Path(root)
    main = root.parent / 'unitree_smash'
    factory = main / 'reports/G07_full_racket_v5/origin0_measured_swing_velocity_20261027/swing_controller.py'
    original = main / 'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/best_candidate.pt'
    protocol_path = root / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    assert source['schema'] == 'g07-measured-swing-contact-recovery-v5' and source['native_execution']
    assert source['protocol_sha256'] == sha(protocol_path) == protocol_sha256
    assert source['factory_sha256'] == sha(factory)
    assert same_tree(source['source_public_policy'], torch.load(original, map_location='cpu', weights_only=False))
    assert torch.equal(source['parameters'], source['source_public_policy']['parameters'])
    job, inputs, metrics, completed = [read(evaluation / name) for name in
        ['job.json', 'INPUTS.json', 'metrics.json', 'completed.json']]
    assert job['status'] == 'SUCCEEDED' and job['receipt_present'] and job['exit_code'] == 0
    assert 0 < job['charged_env_steps'] == completed['env_steps'] <= 102400
    assert inputs['mode'] == 'cold' and inputs['source_policy_sha256'] == sha(path)
    assert inputs['protocol_sha256'] == protocol_sha256 and inputs['training_indices'] == list(range(128))
    assert inputs['origin_spacing_m'] == 0. and inputs['reset_noise'] == 'none' and inputs['seed'] == 724201
    assert inputs['formal_cases_read'] is False
    assert len(metrics['history']) == 1
    protocol = read(protocol_path)
    protocol['_root'] = str(root)
    cases, indices = training_cases(root, protocol)
    rows = read(evaluation / 'episodes.json')
    summary = summarize_training(rows, cases, indices, protocol)
    actual = metrics['history'][0]['summary']
    for key in ['complete', 'airborne_contacts', 'conditional_recovery_count', 'numerical_failures',
                'actual_racket_contacts', 'crossed_net', 'landed_legal_court', 'landed_target', 'fault_counts']:
        assert summary[key] == actual[key]
    assert summary['numerical_failures'] == 0
    assert source['asset_sha256'] == protocol['asset']['sha256'] == sha(root / protocol['asset']['path'])
    assert all(sha(root / name) == digest for name, digest in protocol['physical_configs'].items())
    hashes = read(evaluation / 'execution_source_hashes.json')
    assert len(hashes) == 394 and all(sha(root / name) == digest for name, digest in hashes.items())
    selected = source['actual_native_search']['candidate_id']
    assert metrics['history'][0]['candidate_id'] == selected
    initial = read(evaluation / f'initial_ground_states_{selected:03d}.json')
    assert initial['all_clocks_zero'] and initial['episode_ids'] == [0] * 128
    assert initial['actual_environment_origins_m'] == [[0., 0., 0.]] * 128
    search = Path(source['actual_native_search']['output'])
    assert read(search / 'job.json')['status'] == 'SUCCEEDED'
    assert len(read(search / 'metrics.json')['history']) == 9
    assert sha(search / 'best_candidate.pt') == sha(path)
    dependencies = [path, factory, original, protocol_path, *[evaluation / name for name in
        ['job.json', 'INPUTS.json', 'metrics.json', 'episodes.json', 'completed.json',
         'execution_source_hashes.json', f'initial_ground_states_{selected:03d}.json',
         f'public_and_actions_{selected:03d}.npz']]]
    return dict(source_policy_sha256=sha(path), source_cold_summary=summary,
        selected_swing_candidate_id=selected, original_source_actors_and_normalizers_unchanged=True,
        source_sha256={str(p): sha(p) for p in dependencies},
        formal_cases_read=False, formal_acceptance=False)


def replay_selected_public(path, source, evaluation, root, early_factory):
    check = verify_measured_source(path, source, evaluation, root,
        sha(Path(root) / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'))
    inputs = read(Path(evaluation) / 'INPUTS.json')
    default, limits = [torch.tensor(inputs[name], dtype=torch.float32) for name in
                       ['default_joint_position_rad', 'joint_limits_rad']]
    selected = check['selected_swing_candidate_id']
    data_path = Path(evaluation) / f'public_and_actions_{selected:03d}.npz'
    with np.load(data_path) as data:
        public, actions, valid = [data[name].copy() for name in ['observations', 'actions', 'valid_first_episode']]
        indices = data['training_indices'].tolist()
    assert public.shape[1:] == (6, 794) and 0 < len(public) <= 800
    assert indices == [0, 29, 69, 100, 112, 123]
    controller = early_factory.EarlyLateralController(source, 6, 'cpu', gain=0., start_clock_s=2.)
    previous, maximum = None, 0.
    with torch.inference_mode():
        for value, native, active in zip(public, actions, valid):
            obs = torch.from_numpy(value)
            if previous is not None:
                ids = torch.where(obs[:, 775] < previous - 1e-5)[0]
                if len(ids):
                    controller.reset(ids)
            actual = controller.action({'policy': obs}, default, limits).numpy()
            if active.any():
                maximum = max(maximum, float(abs(actual[active] - native[active]).max()))
            previous = obs[:, 775].clone()
    assert maximum <= 5e-4
    # The full proposed gains were checked on the real completed reference.
    # After selecting the actually measured source, exercise each retained
    # controller on its own real public prefix through the preparation gate.
    samples = []
    for candidate in early_factory.candidates():
        if candidate['candidate_id'] not in [0, 1, 2, 3, 5, 6, 7]:
            continue
        controller = early_factory.EarlyLateralController(source, 6, 'cpu', candidate['gain'], candidate['start_clock_s'])
        with torch.inference_mode():
            for value in public[:251]:
                result = controller.action({'policy': torch.from_numpy(value)}, default, limits)
                assert result.shape == (6, 18) and torch.isfinite(result).all() and (abs(result) <= 1).all()
        samples.append(dict(candidate_id=candidate['candidate_id'], real_prefix_rows=min(251, len(public)) * 6, bounded=True))
    return dict(source_check=check, actual_valid_public_rows=int(valid.sum()),
        CPU_vs_selected_source_GPU_max_action_error=maximum, original_action_tolerance=5e-4,
        all_retained_candidates_on_selected_source_prefix=samples,
        native_execution=False, formal_acceptance=False)
