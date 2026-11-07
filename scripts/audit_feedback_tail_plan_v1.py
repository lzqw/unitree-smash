"""Independent continuation replay used inside the unchanged full-panel audit."""
import hashlib
import json
from pathlib import Path
import numpy as np

from scripts.native_terminal_attitude_v1 import load_verified_policy
from scripts.feedback_tail_rollout_v1 import evaluate_feedback_tail


def load_audit_policy(controller, result, inputs, bundle, manifest):
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    validation = Path(result['feedback_validation'])
    if (result['planner_version'] != 'feedback_tail_candidate_selection_v1'
            or inputs['recovery_planner_version'] != result['planner_version']
            or str(validation) != inputs['feedback_validation']
            or sha(validation) != result['feedback_validation_sha256']
            or inputs['feedback_validation_sha256'] != sha(validation)
            or result['per_query_soft_budget_s'] != 300. or result['recursive_feasibility_proven']
            or result['task_thresholds_changed']
            or result['acceptance_rule'] != 'constrained_prefix_plus_explicit_stable_feedback_continuation'):
        raise ValueError('Different feedback-tail control contract')
    with np.load(Path(bundle)/manifest['bundle_name'], allow_pickle=False) as z:
        query = z['case_003_query_public'].copy()
    policy, _ = load_verified_policy(controller.qp.model, controller, query,
                                    Path(inputs['frictional_warm_plan']), validation)
    protocol_path = Path(__file__).resolve().parents[1]/'configs/evaluation_protocol_v2.json'
    protocol = json.loads(protocol_path.read_text())['recovery']
    return policy, protocol


def check_plan(controller, state, commands, spheres, policy, protocol, plan):
    decision = plan['decision']; tick = plan['tick']
    if (not decision['accepted'] or decision['observed_control_tick'] != tick
            or decision['planner_version'] != 'feedback_tail_candidate_selection_v1'
            or decision['acceptance_rule'] != 'constrained_prefix_plus_explicit_stable_feedback_continuation'
            or decision['recursive_feasibility_proven'] or decision['valid_expert_labels']
            or decision['horizon_steps'] != 80 or decision['selected'] not in ['shifted_previous_plan', 'state_feedback_seed']):
        raise ValueError('Invalid selected-plan provenance')
    for field, actual in [('initial_q', state.q), ('initial_v', state.v),
                          ('initial_contacts', state.contacts), ('previous_command', state.previous_command_target)]:
        if not np.array_equal(decision[field], actual):
            raise ValueError('Accepted plan did not use the actual query: '+field)
    records = decision['records']
    fine_record = records[-1]
    if (fine_record['name'] != decision['selected']
            or fine_record['stage'] != 'independent_fixed_prefix_fine'
            or not fine_record['verification']['accepted']):
        raise ValueError('No independently checked selected continuation')
    checked = evaluate_feedback_tail(controller, state, policy, spheres, controller.sequence_geometry,
        protocol, control_tick=tick, prefix_commands=commands, substeps=160, time_limit_s=300.)
    if (not checked['accepted'] or not np.array_equal(checked['commands'][:80], commands)
            or decision['verified_continuation_steps'] != checked['required_model_steps']):
        raise ValueError('Full declared continuation does not pass independent replay')
    all_saved = decision['predicted_states']+decision['continuation_states']
    if len(all_saved) != len(checked['predicted_states']):
        raise ValueError('Missing planned continuation states')
    for actual, saved in zip(checked['predicted_states'], all_saved):
        for name in ['q', 'v', 'contacts']:
            if not np.allclose(actual[name], saved[name], atol=1e-10, rtol=0):
                raise ValueError('Independent full-state prediction differs: '+name)
    if not np.allclose(checked['commands'][80:], decision['continuation_commands'], atol=1e-10, rtol=0):
        raise ValueError('Recorded tail actions differ from actual feedback replay')
    for name in ['required_model_steps', 'completed_model_steps', 'trailing_model_kinematic_window_s']:
        if checked[name] != decision['verification'][name]:
            raise ValueError('Misreported continuation result: '+name)
    for name, value in checked['terminal'].items():
        saved = decision['verification']['terminal'][name]
        if value is None or saved is None or not np.isclose(value, saved, atol=1e-10, rtol=0):
            raise ValueError('Misreported terminal condition: '+name)
    prediction = checked['predicted_states'][:80]
    summary = {k: v for k, v in checked.items() if k not in ['commands', 'predicted_states', 'records']}
    summary['same_full_continuation_replayed'] = True
    return summary, prediction
