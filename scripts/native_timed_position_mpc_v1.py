"""Timed recovery with actual-state, full-remainder position-stop MPC.

The early reference is an explicitly experimental native-tested prefix.
Planning thereafter includes coupled joint stops and continuously applies
measured-state feedback between solves. No deployed risk trigger is claimed.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np

from scripts.native_timed_experimental_recovery_v1 import NativeTimedExperimentalRecoveryV1
from scripts.current_state_timed_recovery_service_v4 import CurrentStateTimedRecoveryServiceV4
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.native_current_state_recovery_v2 import canonical_state
from scripts.solve_recorded_recovery_suffix_v1 import SuffixCommandCorrectionV1
from scripts.audit_joint_sequence_proposal_v1 import plain


class NativeTimedPositionMPCV1(NativeTimedExperimentalRecoveryV1):
    def __init__(self,bundle,case,limits,policy_sha256,output):
        super().__init__(bundle,case,limits,policy_sha256,output)
        cfg=self.manifest['timed_position_stop_mpc']
        if cfg['execution_contract']!='timed_position_stop_mpc_v1':raise ValueError('Explicit MPC execution contract required')
        self.planning_ticks=cfg['control_ticks'];self.planning_seconds=cfg['max_seconds_per_call']
        if (self.planning_ticks!=sorted(set(self.planning_ticks)) or not 1<=len(self.planning_ticks)<=3
                or any(type(t) is not int or not 400<=t<=700 for t in self.planning_ticks)
                or not 120<=self.planning_seconds<=600):raise ValueError('Bounded complete-remainder planning schedule required')
        setup=json.loads(Path(cfg['source_plan']).read_text());parent=json.loads(Path(setup['source']).read_text())
        _,_,_,self.geometry,context=source_context(parent['configuration']['source'])
        context=list(copy.deepcopy(context));q=self.manifest['query']
        if context[0]!=self.manifest['configuration']:raise ValueError('MPC physical interface differs')
        context[1].update(initial_q=q['q'],initial_v=q['v'],initial_contacts=q['contacts'],actual_previous_command=q['target'],tick=280)
        with np.load(self.bundle/self.manifest['bundle_name'],allow_pickle=False) as z:
            context[4]=z['tracking_reference_q'].copy();context[5]=z['tracking_reference_v'].copy()
        context[6]=self.manifest['tracking_gains'];self.context=tuple(context)
        self.planner_pins=dict(self.manifest['source_sha256'])
        for path in [Path(bundle)/'manifest.json',Path(__file__).resolve()]:
            self.planner_pins[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        self.base_feedback=self.feedback;self.active_plan=copy.deepcopy(cfg['warm_plan'])
        for key in ['original_knots_rad','transition_knots_rad','early_timing_s']:
            if not np.array_equal(self.active_plan[key],self.manifest['proposal'][key]):raise ValueError('Warm plan changes reference '+key)
        if (self.active_plan['braking']!=self.manifest['proposal']['braking']
                or len(self.active_plan['corrections'])!=1
                or self.active_plan['corrections'][0]['control_tick']!=self.planning_ticks[0]):
            raise ValueError('One unexecuted first-call correction seed required')
        self.service=None;self.planning_records=[];self.representation_records=[]

    def close(self):
        if self.service is not None:self.service.close();self.service=None

    def _save_planning(self):
        path=self.output/'timed_position_stop_planning.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(dict(calls=self.planning_records,representations=self.representation_records,
            actual_native_queries=True,valid_expert_labels=False)),indent=2,allow_nan=False)+'\n');tmp.replace(path)

    def action(self,tick,public,student,finished,native_state=None):
        if tick>=self.planning_ticks[0] and not bool(finished[self.case]):
            if native_state is None:raise ValueError('Actual state required for MPC')
            native_state,errors=canonical_state(native_state,self.model)
            self.representation_records.append(dict(tick=tick,errors=errors))
            if tick in self.planning_ticks:
                state=copy.deepcopy(native_state);state.previous_command_target=self.previous.copy()
                try:
                    if self.service is None:
                        self.service=CurrentStateTimedRecoveryServiceV4(self.context,self.geometry,self.planner_pins,
                            workers=2,max_seconds=self.planning_seconds)
                    answer=self.service.solve(state,tick,self.active_plan)
                    self.planning_records.append(answer);self._save_planning()
                    if not answer['accepted']:raise ValueError('Current-state MPC failed full-remainder admission')
                    for key in ['original_knots_rad','transition_knots_rad','early_timing_s']:
                        if not np.array_equal(answer['plan'][key],self.active_plan[key]):raise ValueError('Planner changed original reference '+key)
                    if answer['plan']['braking']!=self.active_plan['braking']:raise ValueError('Original braking changed')
                    self.active_plan=copy.deepcopy(answer['plan']);feedback=self.base_feedback
                    for patch in self.active_plan['corrections']:
                        feedback=SuffixCommandCorrectionV1(feedback,patch['control_tick'],patch['previous_command'],patch['knots_rad'])
                    target,_=feedback.action(80+tick-280,state,self.previous.copy())
                    if not np.array_equal(target,answer['first_command']):raise ValueError('Native/MPC first command differs')
                    self.feedback=feedback
                except BaseException:self.close();raise
        result=super().action(tick,public,student,finished,native_state)
        if self.planning_records and self.records and self.records[-1]['tick']==tick:
            self.records[-1]['mode']='timed_position_stop_mpc_feedback'
        if tick==799 or bool(finished[self.case]):self.close()
        return result

    def save(self):
        self.close();self._save_planning();result=super().save()
        result.update(controller_kind='timed_position_stop_current_state_mpc_with_experimental_prefix',
            online_mpc=bool(self.planning_records),planning_calls=len(self.planning_records),
            accepted_planning_calls=sum(r['accepted'] for r in self.planning_records),
            real_time_guarantee=False,valid_expert_labels=False)
        path=self.output/'sequence_native_result.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n');tmp.replace(path)
        return result
