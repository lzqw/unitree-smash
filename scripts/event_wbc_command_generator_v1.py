"""Causal event-task to per-node leg references through PushOffSupportQPV1."""
import copy
import numpy as np
from scripts.push_off_support_qp_v1 import PushOffSupportQPV1

class EventWBCCommandGeneratorV1:
    """Generate leg corrections by solving the existing support WBC at each node.

    The generator rolls a private state copy from the same initial state. It
    never writes a future base state or force into the candidate. Failed QP
    nodes return zero correction and are recorded for diagnostics.
    """
    def __init__(self, interface, *, substeps=160):
        self.interface=interface; self.substeps=int(substeps)
        if self.substeps<1: raise ValueError('Positive WBC prediction substeps required')
        self.last_diagnostics=[]
    def __call__(self,event,base):
        r=base.rollout; model=r.model; predictor=r.predictor; state=copy.deepcopy(r.initial)
        qp=PushOffSupportQPV1(self.interface); qp.model=model
        n=len(r.commands); corrections=np.zeros((n,12)); diagnostics=[]
        delta_com=np.asarray(event['com_acceleration_delta_world_m_s2'],float)
        delta_ang=np.array([0.,float(event['body_pitch_rate_delta_rad_s']),0.])
        for node,reference in enumerate(r.commands):
            try:
                before=model.snapshot(state.q,state.v)
                baseline=predictor(model,self.interface,copy.deepcopy(state),reference.copy(),duration_s=.01,substeps=self.substeps,enforce_joint_stops=True)
                after=model.snapshot(baseline['state'].q,baseline['state'].v)
                plan=qp.propose(state,reference.copy(),(baseline['state'].v-state.v)/.01,
                    (after['com_velocity_world']-before['com_velocity_world'])/.01,delta_com,delta_ang,
                    phase='PUSH_OFF',desired_angular_momentum_rate_world=delta_ang)
                if plan.get('accepted'):
                    corrections[node]=np.asarray(plan['joint_position_target'])[:12]-reference[:12]
                    chosen=predictor(model,self.interface,copy.deepcopy(state),np.asarray(plan['joint_position_target']),duration_s=.01,substeps=self.substeps,enforce_joint_stops=True)
                    state=chosen['state']
                else:
                    state=baseline['state']
                diagnostics.append(dict(node=node,accepted=bool(plan.get('accepted')),rejection=plan.get('rejection'),solver_status=plan.get('solver_status')))
            except (ValueError,np.linalg.LinAlgError) as exc:
                diagnostics.append(dict(node=node,accepted=False,rejection='MODEL_UNCERTAIN:'+str(exc))); state=baseline['state'] if 'baseline' in locals() else state
        self.last_diagnostics=diagnostics
        return corrections
