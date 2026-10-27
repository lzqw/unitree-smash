"""Admit completed recovery-prior cold128 and check all3 actual-source replays."""
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


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_prior_source(path,source,evaluation,root,protocol_sha256):
    from scripts.initialize_centered_policy_v4 import same_tree
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases,summarize_training
    path,evaluation,root = Path(path),Path(evaluation),Path(root)
    reports = root.parent/'unitree_smash/reports/G07_full_racket_v5'
    factory_path = reports/'recovery_prior_native_20261027/prior_controller.py'
    protocol_path = root/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    assert source['schema'] == 'g07-shared-native-recovery-prior-v5' and source['native_execution']
    assert source['factory_sha256'] == sha(factory_path)
    assert source['protocol_sha256'] == sha(protocol_path) == protocol_sha256
    earlier_path = reports/'windowed_swing7_n128_01/best_candidate.pt'
    earlier_cold = reports/'windowed_swing7_best_cold128_01'
    earlier = torch.load(earlier_path,map_location='cpu',weights_only=False)
    assert same_tree(source['source_public_policy'],earlier)
    support_path = reports/'recovery_prior_queue01/source_support.py'
    support = load('air_preserved_window_admission',support_path)
    previous = support.verify_window_source(earlier_path,earlier,earlier_cold,root,protocol_sha256)
    seed_factory = load('air_admitted_prior_seed',factory_path)
    metadata,arrays = seed_factory.load_seed()
    assert same_tree(source['seed_metadata'],metadata)
    assert source['seed_arrays'].keys() == arrays.keys()
    assert all(np.array_equal(source['seed_arrays'][k],value) for k,value in arrays.items())
    assert source['seed_metadata_sha256'] == sha(reports/'recovery_motion_prior_20261027/SEED_RECEIPT.json')
    assert source['seed_table_sha256'] == metadata['table_sha256']
    job,inputs,metrics,completed = [read(evaluation/n) for n in ['job.json','INPUTS.json','metrics.json','completed.json']]
    assert job['status'] == 'SUCCEEDED' and job['receipt_present'] and job['exit_code'] == 0
    assert 0 < job['charged_env_steps'] == completed['env_steps'] <= 102400
    assert inputs['mode'] == 'cold' and inputs['source_policy_sha256'] == sha(path)
    assert inputs['protocol_sha256'] == protocol_sha256 and inputs['training_indices'] == list(range(128))
    assert inputs['origin_spacing_m'] == 0. and inputs['reset_noise'] == 'none' and inputs['seed'] == 724201
    assert inputs['formal_cases_read'] is False and len(metrics['history']) == 1
    assert torch.equal(source['parameters'],earlier['parameters']) and source['asset_sha256'] == earlier['asset_sha256']
    protocol = read(protocol_path);protocol['_root'] = str(root)
    cases,indices = training_cases(root,protocol)
    summary = summarize_training(read(evaluation/'episodes.json'),cases,indices,protocol)
    for key in ['complete','airborne_contacts','conditional_recovery_count','numerical_failures',
                'actual_racket_contacts','crossed_net','landed_legal_court','landed_target','fault_counts']:
        assert summary[key] == metrics['history'][0]['summary'][key]
    assert summary['numerical_failures'] == 0
    selected = source['actual_native_search']['candidate_id']
    assert metrics['history'][0]['candidate_id'] == selected
    initial = read(evaluation/f'initial_ground_states_{selected:03d}.json')
    assert initial['all_clocks_zero'] and initial['episode_ids'] == [0]*128
    assert initial['actual_environment_origins_m'] == [[0.,0.,0.]]*128
    hashes = read(evaluation/'execution_source_hashes.json')
    assert len(hashes) == 394 and all(sha(root/n) == h for n,h in hashes.items())
    search = Path(source['actual_native_search']['output'])
    assert search == reports/'recovery_prior9_n128_01'
    assert read(search/'job.json')['status'] == 'SUCCEEDED'
    assert [r['candidate_id'] for r in read(search/'metrics.json')['history']] == list(range(9))
    assert sha(search/'best_candidate.pt') == sha(path)
    files = [path,factory_path,support_path,protocol_path,*[evaluation/n for n in
        ['job.json','INPUTS.json','metrics.json','episodes.json','completed.json','execution_source_hashes.json',
         f'initial_ground_states_{selected:03d}.json',f'public_and_actions_{selected:03d}.npz']],
        *[search/n for n in ['job.json','metrics.json','completed.json','best_candidate.pt']]]
    expected = dict(previous['source_sha256']);expected.update({str(p):sha(p) for p in files})
    return dict(source_policy_sha256=sha(path),source_cold_summary=summary,selected_prior_candidate_id=selected,
        previous_window_source_check=previous,source_sha256=expected,source_weights_physics_and_cases_unchanged=True,
        formal_cases_read=False,formal_acceptance=False)


def replay_selected_public(path,source,evaluation,root,factory):
    check = verify_prior_source(path,source,evaluation,root,
        sha(Path(root)/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'))
    inputs = read(Path(evaluation)/'INPUTS.json')
    defaults,limits = [torch.tensor(inputs[n],dtype=torch.float32) for n in ['default_joint_position_rad','joint_limits_rad']]
    selected = check['selected_prior_candidate_id']
    with np.load(Path(evaluation)/f'public_and_actions_{selected:03d}.npz') as data:
        public,actions,valid = [data[n].copy() for n in ['observations','actions','valid_first_episode']]
        assert data['training_indices'].tolist() == [0,29,69,100,112,123]
    assert public.shape[1:] == (6,794) and 0 < len(public) <= 800
    results = []
    for candidate in factory.candidates():
        controller = factory.make_controller(source,6,'cpu',candidate['maximum_velocity_change_rad_s'])
        reference = factory.make_source(source,6,'cpu')
        previous = None;maximum = 0.;maximum_target_change = 0.
        for value,native,active in zip(public,actions,valid):
            obs = torch.from_numpy(value)
            if previous is not None:
                ids = torch.where(obs[:,775] < previous-1e-5)[0]
                if len(ids):controller.reset(ids);reference.reset(ids)
            expected = reference.action({'policy':obs},defaults,limits)
            actual = controller.action({'policy':obs},defaults,limits)
            assert actual.shape == (6,18) and torch.isfinite(actual).all() and (abs(actual)<=1).all()
            assert torch.equal(actual[:,12:],expected[:,12:])
            if candidate['candidate_id'] == 0:
                assert torch.equal(actual,expected)
                if active.any():maximum=max(maximum,float(abs(actual.numpy()[active]-native[active]).max()))
            else:
                mask = controller.last_diagnostic['active']
                assert torch.equal(actual[~mask],expected[~mask])
            target = limits[:,0]+.5*(actual+1)*(limits[:,1]-limits[:,0])
            base_target = limits[:,0]+.5*(expected+1)*(limits[:,1]-limits[:,0])
            maximum_target_change=max(maximum_target_change,float(abs(target-base_target).max()))
            previous=obs[:,775].clone()
        assert maximum<=5e-4 and maximum_target_change<=candidate['maximum_velocity_change_rad_s']*.03+1e-5
        results.append(dict(candidate_id=candidate['candidate_id'],actual_valid_public_rows=int(valid.sum()),
            CPU_vs_selected_source_GPU_max_action_error=maximum if candidate['candidate_id']==0 else None,
            maximum_leg_target_change_rad=maximum_target_change,arm_and_inactive_commands_unchanged=True))
    return dict(source_check=check,actual_selected_source_all3_public_replays=results,
        new_native_execution=False,formal_acceptance=False)
