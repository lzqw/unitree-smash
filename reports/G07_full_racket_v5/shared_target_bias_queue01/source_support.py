"""Admit the last completed real source, including legitimate skipped stages."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
# Order is newest to oldest. A skipped stage does not create a checkpoint.
STAGES = [
    ('stance_recovery_queue01', 'COMPLETED_STANCE_RECOVERY4_AND_INDEPENDENT_COLD128',
     'stance_recovery4', 'g07-public-stance-recovery-v5', None, 'verify_stance_source'),
    ('momentum_strike_queue01', 'COMPLETED_MOMENTUM_STRIKE4_AND_INDEPENDENT_COLD128',
     'momentum_strike4', 'g07-public-momentum-strike-v5', 'stance_recovery_queue01', 'verify_strike_source'),
    ('air_countermotion_queue01', 'COMPLETED_AIR_COUNTERMOTION3_AND_INDEPENDENT_COLD128',
     'air_countermotion3', 'g07-public-air-countermotion-v5', 'momentum_strike_queue01', 'verify_countermotion_source'),
    ('recovery_prior_queue01', 'COMPLETED_RECOVERY_PRIOR9_AND_INDEPENDENT_COLD128',
     'recovery_prior9', 'g07-shared-native-recovery-prior-v5', 'air_countermotion_queue01', 'verify_prior_source'),
    ('windowed_swing_queue01', 'COMPLETED_WINDOWED_SWING7_AND_INDEPENDENT_COLD128',
     'windowed_swing7', 'g07-windowed-swing-contact-recovery-v5', 'recovery_prior_queue01', 'verify_window_source'),
    ('early_lateral_queue01', 'COMPLETED_EARLY_LATERAL7_AND_INDEPENDENT_COLD128',
     'early_lateral7', 'g07-public-early-lateral-intercept-v5', 'windowed_swing_queue01', 'verify_early_source'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_completed_stage(states):
    """Pure gate; actual checkpoint/cold validation is a separate mandatory step."""
    for stage in STAGES:
        state = states[stage[0]]['state']
        if state == stage[1]:
            return stage
        if not state.startswith('SKIPPED_'):
            raise ValueError('Predecessor is not successfully terminal: '+stage[0]+' '+state)
    raise ValueError('No completed real source available; skipped stages cannot fabricate a policy')


def stage_paths(stage):
    return BASE/(stage[2]+'_n128_01/best_candidate.pt'), BASE/(stage[2]+'_best_cold128_01')


def verify_selected_source(path, source, evaluation, root, protocol_sha256):
    path, evaluation = Path(path), Path(evaluation)
    matches = [s for s in STAGES if stage_paths(s) == (path, evaluation)]
    assert len(matches) == 1, 'Require a canonical completed search and independent cold source'
    stage = matches[0]
    assert source['schema'] == stage[3] and source['native_execution']
    helper = (HERE/'stance_source_admission.py' if stage[4] is None
              else BASE/stage[4]/'source_support.py')
    check = getattr(load('target_actual_source_'+stage[0], helper), stage[5])(
        path, source, evaluation, root, protocol_sha256)
    selected = int(source['actual_native_search']['candidate_id'])
    result = dict(check)
    result.update(selected_candidate_id=selected, source_stage=stage[0],
                  source_policy_path=str(path), source_independent_evaluation=str(evaluation))
    result['source_sha256'] = dict(check['source_sha256'])
    result['source_sha256'].update({str(Path(__file__)): sha(Path(__file__)), str(helper): sha(helper)})
    return result


def replay_selected_public(path, source, evaluation, root, factory):
    from scripts.initialize_centered_policy_v4 import same_tree
    check = verify_selected_source(path, source, evaluation, root,
        sha(Path(root)/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'))
    saved = copy.deepcopy(source)
    selected = check['selected_candidate_id']
    inputs = read(Path(evaluation)/'INPUTS.json')
    defaults, limits = [torch.tensor(inputs[k], dtype=torch.float32) for k in
                        ['default_joint_position_rad', 'joint_limits_rad']]
    data_path = Path(evaluation)/f'public_and_actions_{selected:03d}.npz'
    with np.load(data_path) as data:
        public, actions, valid = [data[k].copy() for k in ['observations', 'actions', 'valid_first_episode']]
        assert data['training_indices'].tolist() == [0, 29, 69, 100, 112, 123]
    assert public.shape[1:] == (6, 794) and 0 < len(public) <= 800 and valid.any()
    results = []
    for candidate in factory.candidates():
        controller = factory.make_controller(source, 6, 'cpu', bias_world_m=candidate['bias_world_m'])
        baseline = factory.make_source(source, 6, 'cpu')
        previous = previous_target = None
        ever_active = torch.zeros(6, dtype=torch.bool)
        maximum = maximum_arm_rate = 0.
        active_rows = changed_rows = 0
        digest = hashlib.sha256()
        for value, native, keep in zip(public, actions, valid):
            obs = torch.from_numpy(value)
            untouched = obs.clone()
            reset = torch.zeros(6, dtype=torch.bool)
            if previous is not None:
                reset = obs[:, 775] < previous-1e-6
                ids = torch.where(reset)[0]
                if len(ids):
                    controller.reset(ids)
                    baseline.reset(ids)
                    ever_active[ids] = False
            expected = baseline.action({'policy': obs}, defaults, limits)
            original_offset = controller.reference_controller.target_offset_world
            actual = controller.action({'policy': obs}, defaults, limits)
            assert torch.equal(obs, untouched) and torch.isfinite(actual).all() and (abs(actual) <= 1).all()
            assert controller.reference_controller.target_offset_world is original_offset
            assert torch.equal(controller.reference_controller.target_offset_world, baseline.reference_controller.target_offset_world)
            assert torch.equal(controller.reference_controller.parameters, baseline.reference_controller.parameters)
            assert torch.equal(actual[:, :12], expected[:, :12])
            changed_rows += int((abs(actual-expected).amax(1) > 0).sum())
            if candidate['candidate_id'] == 0:
                assert torch.equal(actual, expected)
                if keep.any():
                    maximum = max(maximum, float(abs(actual.numpy()[keep]-native[keep]).max()))
            else:
                diagnostic = controller.last_diagnostic
                active = diagnostic['active']
                assert not bool((active & ((obs[:, 67] <= .5) | (obs[:, 773] >= .5) | (obs[:, 775] < 2.))).any())
                ever_active |= active
                active_rows += int(active.sum())
                assert torch.equal(actual[~ever_active], expected[~ever_active])
            target = limits[:, 0]+.5*(actual+1)*(limits[:, 1]-limits[:, 0])
            if previous is not None:
                advancing = (~reset) & (obs[:, 775] > previous+1e-6)
                if advancing.any():
                    dt = (obs[advancing, 775]-previous[advancing])[:, None]
                    maximum_arm_rate = max(maximum_arm_rate,
                        float((abs(target[advancing, 12:]-previous_target[advancing, 12:])/dt).max()))
            digest.update(actual.numpy().tobytes())
            previous, previous_target = obs[:, 775].clone(), target.clone()
        assert maximum <= .0005 and maximum_arm_rate <= 5.0005
        assert same_tree(source, saved)
        results.append(dict(candidate_id=candidate['candidate_id'], actual_public_rows=int(np.prod(public.shape[:2])),
            actual_valid_public_rows=int(valid.sum()), active_precontact_reference_rows=active_rows,
            changed_command_rows=changed_rows, CPU_vs_selected_source_GPU_max_action_error=maximum if candidate['candidate_id'] == 0 else None,
            maximum_arm_target_rate_rad_s=maximum_arm_rate, bounded_finite=True,
            leg_commands_prefix_and_source_weights_preserved=True,
            causal_plan_tensor_and_parameters_restored=True, command_stream_sha256=digest.hexdigest()))
    return dict(source_check=check, actual_selected_source_all5_public_checks=results,
        original_action_tolerance=.0005, source_data_sha256=sha(data_path),
        active_branch_coverage_reported_not_assumed=True, new_native_execution=False, formal_acceptance=False)
