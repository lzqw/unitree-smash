"""Current-state full-remainder replanning of the complete foot-space controller.

Experimental time-conditioned entry, not a deployed risk gate. Simulation
waits at control boundaries for CPU planning. Every executed step still uses
measured-state feedback and actual command history, never a held first target.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from scripts.native_foot_space_recovery_v1 import NativeFootSpaceRecoveryV1,LoggedFeedback
from scripts.current_state_recovery_service_v3 import CurrentStateRecoveryServiceV3
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.recovery_family_queries_v1 import transfer_reference
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.native_current_state_recovery_v2 import canonical_state
from scripts.solve_recorded_recovery_suffix_v1 import SuffixCommandCorrectionV1
from scripts.feedback_tail_rollout_v1 import terminal_measures,kinematically_stable
from scripts.audit_joint_sequence_proposal_v1 import plain


class NativeFootSpaceMPCV1(NativeFootSpaceRecoveryV1):
    def __init__(self,bundle,case,limits,policy_sha256,output):
        super().__init__(bundle,case,limits,policy_sha256,output)
        cfg=self.manifest['foot_space_current_state_planning'];definition=self.manifest['foot_space_definition']
        self.planning_ticks=cfg['control_ticks'];self.planning_seconds=cfg['max_seconds_per_call']
        if (self.planning_ticks!=sorted(set(self.planning_ticks)) or
                not 1<=len(self.planning_ticks)<=4 or
                any(type(t) is not int or not self.start<t<800 for t in self.planning_ticks) or
                not 120<=self.planning_seconds<=600):
            raise ValueError('Explicit bounded actual-state planning schedule required')
        source=Path(cfg['model_search']);search=json.loads(source.read_text())
        setup_path=Path(cfg['source_plan']);setup=json.loads(setup_path.read_text())
        if (not search['verified'] or search['case']!=case or
                not np.array_equal(search['original_knots'],definition['original_knots']) or
                not np.array_equal(search['verified']['replay']['transition_knots'],definition['transition_knots']) or
                search['braking']!=definition['braking']):
            raise ValueError('Complete native controller and planner warm start differ')
        parent=json.loads(Path(setup['source']).read_text())
        _,_,_,self.geometry,context=source_context(parent['configuration']['source'])
        q=self.expected_q[0].copy();v=self.expected_v[0].copy()
        initial=PublicState(q,v,pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),
            q[self.model.q_indices].copy(),v[self.model.v_indices].copy(),self.expected_contacts[0].copy(),0.)
        initial.previous_command_target=self.decode(self.query_previous_action)
        self.context,_=transfer_reference(context,initial)
        with np.load(self.bundle/self.manifest['bundle_name'],allow_pickle=False) as z:
            if (not np.array_equal(self.context[4],z['tracking_reference_q']) or
                    not np.array_equal(self.context[5],z['tracking_reference_v'])):
                raise ValueError('Planning reference differs from native controller')
        self.planner_pins=dict(search['source_sha256'])
        for path in [source,setup_path,Path(bundle)/'manifest.json',Path(__file__).resolve()]:
            self.planner_pins[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        self.base_policy=self.feedback.policy
        self.active_plan=dict(contract='foot_space_current_state_plan_v1',
            original_knots_rad=definition['original_knots'],transition_knots_rad=definition['transition_knots'],
            braking=definition['braking'],corrections=[])
        self.service=None;self.planning_records=[];self.representation_records=[]
        self.stable_ticks=0;self.last_stability_tick=None
        self.protocol=json.loads(Path('configs/evaluation_protocol_v2.json').read_text())['recovery']

    def _save_planning(self):
        path=self.output/'foot_space_current_state_planning.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(dict(calls=self.planning_records,
            representation_records=self.representation_records,control_sampled_stable_ticks=self.stable_ticks,
            actual_native_queries=True,substep_stability_not_certified=True,valid_expert_labels=False)),
            indent=2,allow_nan=False)+'\n');tmp.replace(path)

    def close(self):
        if self.service is not None:self.service.close();self.service=None

    def action(self,tick,public,student,finished,native_state=None):
        if tick>=self.start and not bool(finished[self.case]):
            if native_state is None:raise ValueError('Actual measured planning state required')
            if self.last_stability_tick is not None and tick!=self.last_stability_tick+1:
                raise ValueError('Consecutive control-sampled stability history required')
            stable=(native_state.q[2]>=.12 and kinematically_stable(
                terminal_measures(self.controller,native_state),self.protocol))
            self.stable_ticks=self.stable_ticks+1 if stable and self.last_stability_tick is not None else 0
            self.last_stability_tick=tick
            if tick>=self.planning_ticks[0]:
                native_state,errors=canonical_state(native_state,self.model)
                self.representation_records.append(dict(control_tick=tick,redundant_field_errors=errors))
            if tick in self.planning_ticks:
                state=copy.deepcopy(native_state);state.previous_command_target=self.previous.copy()
                try:
                    if self.service is None:
                        self.service=CurrentStateRecoveryServiceV3(self.context,self.geometry,self.planner_pins,
                            workers=2,max_seconds=self.planning_seconds)
                    answer=self.service.solve(state,tick,self.active_plan,preceding_stable_ticks=self.stable_ticks)
                    self.planning_records.append(answer);self._save_planning()
                    if not answer['accepted']:
                        raise ValueError('No verified current-state recovery; stop before applying an unverified correction')
                    for key in ['original_knots_rad','transition_knots_rad']:
                        if not np.array_equal(answer['plan'][key],self.active_plan[key]):
                            raise ValueError('Planner changed the frozen reference origin or transition')
                    if answer['plan']['braking']!=self.active_plan['braking']:
                        raise ValueError('Planner changed the original braking policy')
                    self.active_plan=copy.deepcopy(answer['plan']);feedback=self.base_policy
                    for patch in self.active_plan['corrections']:
                        feedback=SuffixCommandCorrectionV1(feedback,patch['control_tick'],
                            patch['previous_command'],patch['knots_rad'])
                    target,_=feedback.action(80+tick-280,state,self.previous.copy())
                    if not np.array_equal(target,answer['first_command']):
                        raise ValueError('Predicted and recomputed current command differ')
                    self.feedback=LoggedFeedback(feedback)
                except BaseException:
                    self.close();raise
        result=super().action(tick,public,student,finished,native_state)
        if self.planning_records and self.records and self.records[-1]['control_tick']==tick:
            self.records[-1]['mode']='foot_space_current_state_planned_feedback'
        if tick==799 or bool(finished[self.case]):self.close()
        return result

    def save(self):
        self.close();self._save_planning();result=super().save()
        result.update(controller_kind='foot_space_current_state_full_remainder_planning_with_feedback',
            online_mpc=bool(self.planning_records),planning_calls=len(self.planning_records),
            accepted_planning_calls=sum(r['accepted'] for r in self.planning_records),
            deployable_risk_trigger=False,real_time_guarantee=False,valid_expert_labels=False,
            nominal_error_reference='Initial offline plan; later predictions stored in foot_space_current_state_planning.json')
        path=self.output/'sequence_native_result.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n');tmp.replace(path)
        return result
