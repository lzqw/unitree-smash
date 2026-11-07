"""Native adapter for the resolution-verified coupled arm-margin candidate.

This is a planned prefix plus persistent feedback, not online replanning MPC.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

from scripts.native_terminal_support_v1 import NativeTerminalSupportV1
from scripts.native_terminal_support_v1 import load_verified_policy as load_base
from scripts.search_terminal_prefix_arm_margin_v1 import ArmPrefixCandidate

MODEL_NAME = 'planned_prefix_terminal_arm_margin_v1'
CORRECTION = dict(joint_index=13, amplitude_rad=-.006, start=15, end=45, profile='half_sine')


def read_validation(path):
    record=json.loads(Path(path).read_text())
    if (not record['model_candidate_accepted'] or record['failure'] is not None or
            record['executed_model_steps']!=340 or record['needed_steps']!=340 or
            record['start_tick']!=460 or record['integration_substeps']!=160 or
            record['arm_correction']!=CORRECTION or record['native_continuation_executed'] or
            record['valid_expert_labels'] or len(record['states'])!=340 or len(record['commands'])!=340 or
            record['model_window']['longest_model_kinematic_window_s']<.8):
        raise ValueError('Complete corrected model-only validation required')
    for name,digest in record['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Changed corrected validation input '+name)
    base=Path(record['base_validation_path'])
    digest=hashlib.sha256(base.read_bytes()).hexdigest()
    if digest!=record['base_validation_sha256'] or record['source_sha256'].get(str(base.resolve()))!=digest:
        raise ValueError('Different base validation')
    return record


def load_verified_policy(model,controller,query_public,warm_path,validation_path):
    record=read_validation(validation_path)
    base,old=load_base(model,controller,query_public,warm_path,record['base_validation_path'])
    for key in ['case','start_tick','needed_steps','selected_tail','initial_q','initial_v',
                'initial_contacts','actual_previous_command']:
        if record[key]!=old[key]:raise ValueError('Changed corrected initial condition or base controller '+key)
    policy=ArmPrefixCandidate(base,controller.qp.interface,CORRECTION['amplitude_rad'])
    policy.model=model
    return policy,record


class NativeTerminalArmMarginV1(NativeTerminalSupportV1):
    def __init__(self,bundle,case,limits,policy_sha256,output,warm_plan,sphere_audit,validation_path):
        record=read_validation(validation_path)
        super().__init__(bundle,case,limits,policy_sha256,output,warm_plan,sphere_audit,record['base_validation_path'])
        try:
            self.terminal_policy,record=load_verified_policy(self.model,self.controller,self.query_public,
                                                            warm_plan,validation_path)
            self.validation_path=str(Path(validation_path).resolve())
            self.validation_sha256=hashlib.sha256(Path(validation_path).read_bytes()).hexdigest()
            self.expected_q=np.vstack([record['initial_q'],[s['q'] for s in record['states']]])
            self.expected_v=np.vstack([record['initial_v'],[s['v'] for s in record['states']]])
            self.expected_contacts=np.vstack([record['initial_contacts'],[s['contacts'] for s in record['states']]])
        except Exception:
            self.planner.close()
            raise

    def save(self):
        result=super().save()
        result.update(teacher_model=MODEL_NAME,arm_correction=CORRECTION,
                      prediction_integration_substeps=160)
        path=self.output/'sequence_native_result.json';temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(path)
        return result
