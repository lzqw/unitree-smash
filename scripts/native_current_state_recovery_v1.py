"""Measured-state full-remainder replanning after a preserved native prefix.

An experimental, time-conditioned native interface, not a deployed risk gate.
Physics waits at a control boundary for the CPU service. If no verified plan
is available, this diagnostic aborts before dispatching a purported correction.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.native_replanned_suffix_v1 import NativeReplannedSuffixV1
from scripts.current_state_recovery_service_v1 import CurrentStateRecoveryServiceV1
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.solve_recorded_recovery_suffix_v1 import SuffixCommandCorrectionV1
from scripts.feedback_tail_rollout_v1 import terminal_measures,kinematically_stable
from scripts.audit_joint_sequence_proposal_v1 import plain


class NativeCurrentStateRecoveryV1(NativeReplannedSuffixV1):
    def __init__(self,bundle,case,limits,policy_sha256,output):
        super().__init__(bundle,case,limits,policy_sha256,output)
        cfg=self.manifest['current_state_planning']
        self.planning_ticks=cfg['control_ticks']
        if (self.planning_ticks!=sorted(set(self.planning_ticks)) or
                any(type(t) is not int or not self.suffix_start<t<800 for t in self.planning_ticks)
                or not 1<=len(self.planning_ticks)<=4):
            raise ValueError('Explicit bounded planning schedule required')
        self.planning_seconds=cfg['max_seconds_per_call']
        if not 120<=self.planning_seconds<=600:raise ValueError('Bounded planning time required')
        source=Path(self.manifest['replanned_suffix_definition']['model_search'])
        payload=json.loads(source.read_text())
        self.planner_context=source_context(payload['configuration']['source'])
        self.planner_pins=dict(payload['source_sha256'])
        for path in [source,Path(__file__).resolve(),Path(bundle)/'manifest.json']:
            self.planner_pins[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        self.feedback_base=self.suffix_feedback.base
        self.active_plan=None;self.service=None;self.planning_records=[]
        self.observed_stable_ticks=0;self.last_stability_tick=None
        self.protocol=json.loads(Path('configs/evaluation_protocol_v2.json').read_text())['recovery']

    def _save_planning(self):
        path=self.output/'current_state_planning.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(dict(calls=self.planning_records,
            observed_stable_ticks=self.observed_stable_ticks,
            sample_interval_s=.01,substep_stability_not_certified=True,
            native_dispatch=True,valid_expert_labels=False)),indent=2,allow_nan=False)+'\n')
        tmp.replace(path)

    def close(self):
        if self.service is not None:self.service.close();self.service=None

    def action(self,tick,public,student,finished,native_state=None):
        if tick>=self.suffix_start and not bool(finished[self.case]):
            if native_state is None:raise ValueError('Native planning requires current state')
            if self.last_stability_tick is not None and tick!=self.last_stability_tick+1:
                raise ValueError('Observed stability requires consecutive control samples')
            stable=(native_state.q[2]>=.12 and kinematically_stable(
                terminal_measures(self.controller,native_state),self.protocol))
            self.observed_stable_ticks=(self.observed_stable_ticks+1
                if stable and self.last_stability_tick is not None else 0)
            self.last_stability_tick=tick
            if tick in self.planning_ticks:
                if self.active_plan is None:raise ValueError('Missing admitted prefix/suffix plan')
                state=copy.deepcopy(native_state);state.previous_command_target=self.previous.copy()
                if self.service is None:
                    self.service=CurrentStateRecoveryServiceV1(self.planner_context[4],self.planner_context[3],
                        self.planner_pins,workers=4,max_seconds=self.planning_seconds)
                try:
                    answer=self.service.solve(state,tick,self.active_plan,
                        preceding_stable_ticks=self.observed_stable_ticks)
                    self.planning_records.append(answer);self._save_planning()
                    if not answer['accepted']:
                        raise ValueError('No verified current-state recovery; diagnostic stops before dispatch')
                    if not np.array_equal(answer['plan']['original_knots_rad'],self.active_plan['original_knots_rad']):
                        raise ValueError('Planner changed frozen reference origin')
                    self.active_plan=copy.deepcopy(answer['plan']);feedback=self.feedback_base
                    for patch in self.active_plan['corrections']:
                        feedback=SuffixCommandCorrectionV1(feedback,patch['control_tick'],
                            patch['previous_command'],patch['knots_rad'])
                    command,_=feedback.action(80+tick-280,state,self.previous.copy())
                    if not np.array_equal(command,answer['first_command']):
                        raise ValueError('Predicted and dispatched first command differ')
                    self.suffix_feedback=feedback
                except BaseException:self.close();raise
        result=super().action(tick,public,student,finished,native_state)
        if self.planning_records and self.records and self.records[-1]['control_tick']==tick:
            self.records[-1]['mode']='current_state_planned_feedback'
            self.feedback_records[-1]['feedback']=dict(kind='current_state_planned_feedback',
                latest_planning_tick=self.planning_records[-1]['control_tick'],online_mpc=True)
        if tick==self.suffix_start and self.suffix_query and self.suffix_query['passed']:
            definition=self.manifest['replanned_suffix_definition']
            self.active_plan=dict(original_knots_rad=definition['original_knots_rad'],corrections=[dict(
                control_tick=tick,previous_command=self.suffix_feedback.previous_at_start.copy(),
                knots_rad=definition['corrections_rad'])])
        if tick==799 or bool(finished[self.case]):self.close()
        return result

    def save(self):
        self.close();self._save_planning();result=super().save()
        result.update(controller_kind='measured_state_full_remainder_planning_with_feedback',
            online_mpc=bool(self.planning_records),planning_calls=len(self.planning_records),
            prediction_segments='Prefix/offline suffix plus current-state predictions in current_state_planning.json.',
            accepted_planning_calls=sum(r['accepted'] for r in self.planning_records),
            deployable_risk_trigger=False,real_time_guarantee=False,valid_expert_labels=False)
        path=self.output/'sequence_native_result.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n');tmp.replace(path)
        return result
