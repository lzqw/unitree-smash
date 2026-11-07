"""Native adapter for the fully validated arm-margin plus attitude candidate.

This tests a continuous terminal controller, not an online replanning claim.
The parent actor, original arm correction and motor interface stay frozen.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from scripts.native_terminal_arm_margin_v1 import NativeTerminalArmMarginV1
from scripts.native_terminal_arm_margin_v1 import load_verified_policy as load_base
from scripts.terminal_attitude_feedback_v1 import TerminalAttitudeFeedbackV1

MODEL_NAME = 'planned_prefix_terminal_attitude_feedback_v1'


def read_validation(path):
    record = json.loads(Path(path).read_text())
    if (not record['model_candidate_accepted'] or record['failure'] is not None
            or record['executed_model_steps'] != 340 or record['needed_steps'] != 340
            or record['start_tick'] != 460 or record['case'] != 3
            or record['integration_substeps'] != 160 or record['native_continuation_executed']
            or record['valid_expert_labels'] or len(record['states']) != 340 or len(record['commands']) != 340
            or record['model_window']['longest_model_kinematic_window_s'] < .8
            or record['terminal']['attitude_margin_deg'] < 0 or record['terminal']['support_count'] < 1):
        raise ValueError('Require complete, constrained, stable-terminal model validation')
    for name, digest in record['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError('Changed attitude validation source ' + name)
    for key in ['base_validation', 'feedback_config']:
        path = Path(record[key+'_path'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record[key+'_sha256'] or record['source_sha256'].get(str(path.resolve())) != digest:
            raise ValueError('Unbound candidate input ' + key)
    if json.loads(Path(record['feedback_config_path']).read_text()) != record['feedback_parameters']:
        raise ValueError('Feedback parameters differ from the validated configuration')
    return record


def load_verified_policy(model, controller, query_public, warm_path, validation_path):
    record = read_validation(validation_path)
    base, old = load_base(model, controller, query_public, warm_path, record['base_validation_path'])
    for key in ['case', 'start_tick', 'needed_steps', 'selected_tail']:
        if record[key] != old[key]:
            raise ValueError('Changed base control provenance ' + key)
    if (np.max(abs(pin.difference(model.model, np.asarray(old['initial_q']), np.asarray(record['initial_q'])))) > 2e-5
            or np.max(abs(np.asarray(record['initial_v'])-np.asarray(old['initial_v']))) > 2e-4
            or not np.array_equal(record['initial_contacts'], old['initial_contacts'])
            or np.max(abs(np.asarray(record['actual_previous_command'])-np.asarray(old['actual_previous_command']))) > 5e-6):
        raise ValueError('Different actual initial query or command history')
    return TerminalAttitudeFeedbackV1(base, **record['feedback_parameters']), record


class NativeTerminalAttitudeV1(NativeTerminalArmMarginV1):
    def __init__(self, bundle, case, limits, policy_sha256, output, warm_plan, sphere_audit, validation_path):
        record = read_validation(validation_path)
        super().__init__(bundle, case, limits, policy_sha256, output, warm_plan, sphere_audit,
                         record['base_validation_path'])
        try:
            self.terminal_policy, record = load_verified_policy(self.model, self.controller, self.query_public,
                                                                warm_plan, validation_path)
            self.validation_path = str(Path(validation_path).resolve())
            self.validation_sha256 = hashlib.sha256(Path(validation_path).read_bytes()).hexdigest()
            self.feedback_parameters = dict(record['feedback_parameters'])
            self.expected_q = np.vstack([record['initial_q'], [s['q'] for s in record['states']]])
            self.expected_v = np.vstack([record['initial_v'], [s['v'] for s in record['states']]])
            self.expected_contacts = np.vstack([record['initial_contacts'], [s['contacts'] for s in record['states']]])
        except Exception:
            self.planner.close()
            raise

    def save(self):
        result = super().save()
        result.update(teacher_model=MODEL_NAME, feedback_parameters=self.feedback_parameters,
                      prediction_integration_substeps=160)
        path = self.output/'sequence_native_result.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)
        return result
