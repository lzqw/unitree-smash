"""Admit an actually completed early-lateral independent128 source."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_early_source(path, source, evaluation, root, protocol_sha256):
    from scripts.initialize_centered_policy_v4 import same_tree
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
    path, evaluation, root = Path(path), Path(evaluation), Path(root)
    main = root.parent / 'unitree_smash'
    reports = main / 'reports/G07_full_racket_v5'
    factory = reports / 'early_tangent_intercept_20261027/early_lateral_controller.py'
    protocol_path = root / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    assert source['schema'] == 'g07-public-early-lateral-intercept-v5' and source['native_execution']
    assert source['factory_sha256'] == sha(factory)
    assert source['protocol_sha256'] == sha(protocol_path) == protocol_sha256
    swing_path = reports / 'measured_swing9_n128_01/best_candidate.pt'
    swing_cold = reports / 'measured_swing9_best_cold128_01'
    swing = torch.load(swing_path, map_location='cpu', weights_only=False)
    assert same_tree(source['source_public_policy'], swing)
    support_path = reports / 'early_lateral_queue01/source_support.py'
    spec = importlib.util.spec_from_file_location('window_preserved_measured_admission', support_path)
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    previous = support.verify_measured_source(swing_path, swing, swing_cold, root, protocol_sha256)
    job, inputs, metrics, completed = [read(evaluation / name) for name in
        ['job.json', 'INPUTS.json', 'metrics.json', 'completed.json']]
    assert job['status'] == 'SUCCEEDED' and job['receipt_present'] and job['exit_code'] == 0
    assert 0 < job['charged_env_steps'] == completed['env_steps'] <= 102400
    assert inputs['mode'] == 'cold' and inputs['source_policy_sha256'] == sha(path)
    assert inputs['protocol_sha256'] == protocol_sha256 and inputs['training_indices'] == list(range(128))
    assert inputs['origin_spacing_m'] == 0. and inputs['reset_noise'] == 'none' and inputs['seed'] == 724201
    assert inputs['formal_cases_read'] is False and len(metrics['history']) == 1
    assert torch.equal(source['parameters'], swing['parameters']) and source['asset_sha256'] == swing['asset_sha256']
    protocol = read(protocol_path)
    protocol['_root'] = str(root)
    cases, indices = training_cases(root, protocol)
    summary = summarize_training(read(evaluation / 'episodes.json'), cases, indices, protocol)
    actual = metrics['history'][0]['summary']
    for key in ['complete', 'airborne_contacts', 'conditional_recovery_count', 'numerical_failures',
                'actual_racket_contacts', 'crossed_net', 'landed_legal_court', 'landed_target', 'fault_counts']:
        assert summary[key] == actual[key]
    assert summary['numerical_failures'] == 0
    selected = source['actual_native_search']['candidate_id']
    assert metrics['history'][0]['candidate_id'] == selected
    initial = read(evaluation / f'initial_ground_states_{selected:03d}.json')
    assert initial['all_clocks_zero'] and initial['episode_ids'] == [0]*128
    assert initial['actual_environment_origins_m'] == [[0., 0., 0.]]*128
    hashes = read(evaluation / 'execution_source_hashes.json')
    assert len(hashes) == 394 and all(sha(root / name) == digest for name, digest in hashes.items())
    search = Path(source['actual_native_search']['output'])
    assert search == reports / 'early_lateral7_n128_01'
    assert read(search / 'job.json')['status'] == 'SUCCEEDED'
    assert [r['candidate_id'] for r in read(search / 'metrics.json')['history']] == [0, 1, 2, 3, 5, 6, 7]
    assert sha(search / 'best_candidate.pt') == sha(path)
    dependencies = [path, factory, support_path, protocol_path, *[evaluation / name for name in
        ['job.json', 'INPUTS.json', 'metrics.json', 'episodes.json', 'completed.json',
         'execution_source_hashes.json', f'initial_ground_states_{selected:03d}.json',
         f'public_and_actions_{selected:03d}.npz']], *[search / name for name in
        ['job.json', 'metrics.json', 'completed.json', 'best_candidate.pt']]]
    expected = dict(previous['source_sha256'])
    expected.update({str(p): sha(p) for p in dependencies})
    return dict(source_policy_sha256=sha(path), source_cold_summary=summary,
        selected_early_candidate_id=selected, previous_measured_source_check=previous,
        source_actor_and_normalizer_and_physical_task_unchanged=True,
        source_sha256=expected, formal_cases_read=False, formal_acceptance=False)


def replay_selected_public(path, source, evaluation, root, factory):
    check = verify_early_source(path, source, evaluation, root,
        sha(Path(root) / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'))
    inputs = read(Path(evaluation) / 'INPUTS.json')
    defaults, limits = [torch.tensor(inputs[name], dtype=torch.float32) for name in
                       ['default_joint_position_rad', 'joint_limits_rad']]
    selected = check['selected_early_candidate_id']
    data_path = Path(evaluation) / f'public_and_actions_{selected:03d}.npz'
    with np.load(data_path) as data:
        public, actions, valid = [data[name].copy() for name in ['observations', 'actions', 'valid_first_episode']]
        assert data['training_indices'].tolist() == [0, 29, 69, 100, 112, 123]
    assert public.shape[1:] == (6, 794) and 0 < len(public) <= 800
    proposals = read(HERE.parent / 'windowed_swing_20261027/PROPOSAL.json')['candidates']
    results = []
    for candidate in proposals:
        controller = factory.WindowedSwingController(source, 6, 'cpu', candidate['joint_velocity_rad_s'])
        reference = factory.make_source(source, 6, 'cpu')
        previous = previous_target = None
        maximum = maximum_rate = 0.
        with torch.inference_mode():
            for value, native, active in zip(public, actions, valid):
                obs = torch.from_numpy(value)
                reset = torch.zeros(6, dtype=torch.bool)
                if previous is not None:
                    reset = obs[:, 775] < previous - 1e-5
                    ids = torch.where(reset)[0]
                    if len(ids):
                        controller.reset(ids)
                        reference.reset(ids)
                expected = reference.action({'policy': obs}, defaults, limits)
                actual = controller.action({'policy': obs}, defaults, limits)
                assert actual.shape == (6, 18) and torch.isfinite(actual).all() and (abs(actual) <= 1.).all()
                assert torch.equal(actual[:, :12], expected[:, :12])
                if candidate['candidate_id'] == 0:
                    assert torch.equal(actual, expected)
                    if active.any():
                        maximum = max(maximum, float(abs(actual.numpy()[active]-native[active]).max()))
                targets = limits[:, 0] + .5*(actual+1)*(limits[:, 1]-limits[:, 0])
                if previous_target is not None and (~reset).any():
                    maximum_rate = max(maximum_rate,
                        float(abs(targets[~reset, 12:]-previous_target[~reset, 12:]).max())/.01)
                previous, previous_target = obs[:, 775].clone(), targets.clone()
        assert maximum <= 5e-4 and maximum_rate <= 5.0005
        results.append(dict(candidate_id=candidate['candidate_id'], actual_valid_public_rows=int(valid.sum()),
            CPU_vs_selected_source_GPU_max_action_error=maximum if candidate['candidate_id'] == 0 else None,
            maximum_arm_target_rate_rad_s=maximum_rate, finite_bounded=True,
            same_input_leg_commands_unchanged=True))
    return dict(source_check=check, actual_selected_source_all7_public_replays=results,
        original_action_tolerance=5e-4, new_native_execution=False, formal_acceptance=False)
