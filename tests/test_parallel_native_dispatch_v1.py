"""Exercise extracted native dispatch without starting Isaac or a GPU job."""
import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.prepare_parallel_landing_entry_v1 import transform
from scripts.native_parallel_landing_configuration_v1 import configuration_from_controller
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.landing_full_command_interval_mpc_v1 import LandingFullCommandIntervalMPCV1
from scripts.landing_feasible_command_cost_mpc_v1 import LandingFeasibleCommandCostMPCV1
from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
from scripts.whole_body_support_qp_v1 import PositionPDInterface, SupportSettings

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('PARALLEL_ENTRY_TEST_SOURCE',
    ROOT/'scripts/evaluate_shared_target_ground64_persistent_recovery_v1.py')).read_text()


@pytest.mark.parametrize('cls',[LandingFullCommandIntervalMPCV1,LandingFeasibleCommandCostMPCV1])
def test_worker_configuration_keeps_native_contract(cls):
    interface=PositionPDInterface(np.tile([-2.,2.],(18,1)),np.r_[np.full(12,25.),np.full(6,80.)],
        np.r_[np.full(12,.5),np.full(6,4.)],np.r_[np.full(12,23.5),np.full(6,100.)],
        np.r_[np.full(12,30.),np.full(6,5.)],.65)
    original=cls(interface,qp_settings=SupportSettings(nonfoot_clearance_weight=5.),
                 settings=LandingMPCSettingsV2(horizon_steps=40),arm_target_rate_rad_s=10.)
    original.set_posture_target(np.zeros(18))
    cfg=configuration_from_controller(original)
    assert build_controller(cfg).contract()==original.contract()
    original.qp.substeps=32
    with pytest.raises(ValueError,match='Undeclared'):
        configuration_from_controller(original)


class FakeMPC:
    def __init__(self):
        self.observations=[];self.solves=[];self.resets=[]
        self.settings=SimpleNamespace(control_dt_s=.01)
        self.qp=SimpleNamespace(interface=SimpleNamespace(normalized_action=lambda x:x,arm_rate=10.))

    def admit_command_target(self,state,reference):return np.clip(reference,-1.,1.)
    def normalized_command_target(self,*args):raise AssertionError('not called')
    def observe(self,state,reference,*,lane,control_tick):self.observations.append((lane,control_tick))
    def reset(self,lane=None):self.resets.append(lane)
    def solve(self,state,reference,*,lane,control_tick,force):
        self.solves.append((lane,control_tick,force))
        if lane==1:return dict(accepted=False,horizon=[],risk=dict(trigger=True))
        target=reference+.01
        return dict(accepted=True,horizon=[dict(accepted=True,joint_position_target=target,
            normalized_action=target,contacts=state.contacts)],risk=dict(trigger=True),
            initial_risk_score=3.,first_step_risk_score=1.)


class FakePool:
    def __init__(self,fail=False):self.controller=FakeMPC();self.batches=[];self.fail=fail
    def batch(self,requests):
        self.batches.append(requests)
        if self.fail:raise RuntimeError('worker failed')
        result=[]
        for r in requests:
            if r['operation']=='reset':self.controller.reset(lane=r['lane']);result.append(None);continue
            self.controller.observe(r['state'],r['reference'],lane=r['lane'],control_tick=r['control_tick'])
            result.append(self.controller.solve(r['state'],r['reference'],lane=r['lane'],
                control_tick=r['control_tick'],force=r['force']))
        return result


def methods(source):
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and
             any(isinstance(m,ast.FunctionDef) and m.name=='apply_landing_mpc' for m in n.body))
    selected=[n for n in cls.body if isinstance(n,ast.FunctionDef) and
              n.name in ['apply_landing_mpc','finish_landing_mpc_episodes','reset']]
    ns=dict(torch=torch,np=np,LandingSupportMPCV3=FakeMPC)
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),
                 '<extracted-native-methods>','exec'),ns)
    return ns


class ContactCounter:
    def __init__(self):self.count=0
    def update(self,flags,force):self.count+=1;return flags
    def reset(self):self.count=0


def policy(pool=None):
    p=SimpleNamespace()
    # Flags default to disabled; explicit arrays/state below exercise the
    # actual dispatch body, command arbitration, records, and lifecycle.
    for n in ast.walk(ast.parse(SOURCE)):
        if isinstance(n,ast.Attribute) and n.attr.startswith('landing_mpc_'):
            setattr(p,n.attr,False)
    count=5
    p.landing_mpc=FakeMPC();p._landing_solver_pool=pool
    p.landing_mpc_cases={0,1,2,3};p.landing_mpc_command_history_interface=True
    p.landing_mpc_touchdown_recovery=True;p.landing_mpc_force_post_touchdown=True
    p.landing_mpc_min_risk_score_drop=.05;p.landing_mpc_min_start_risk_score=2.
    p.landing_mpc_logs=[]
    p._landing_mpc_finished_first_episode=torch.tensor([False,False,False,True,False])
    p._landing_mpc_hold_remaining=torch.zeros(count,dtype=torch.long)
    p._landing_mpc_held_action=torch.zeros((count,18))
    p._landing_mpc_latched_post_touchdown=torch.zeros(count,dtype=torch.bool)
    p._landing_mpc_latched_age=torch.zeros(count,dtype=torch.long)
    p._landing_mpc_mode_mismatch=torch.zeros(count,dtype=torch.long)
    p._landing_mpc_expected_contacts=np.zeros((count,4),bool)
    p._landing_mpc_last_target=np.full((count,18),np.nan)
    p._landing_mpc_last_predicted=np.full((count,18),np.nan)
    p._landing_mpc_prediction_tick=np.full(count,-1)
    p._effort_override=torch.zeros((count,18));p._effort_override_mask=torch.zeros(count,dtype=torch.bool)
    p._landing_mpc_contact_state_lane=[ContactCounter() for _ in range(count)]
    p._landing_mpc_native_env=SimpleNamespace(episode_index=torch.zeros(count),
        targets=torch.zeros((count,18)),joint_ids=list(range(18)),
        robot=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros((count,18)))))
    state=SimpleNamespace(contacts=np.ones(4,bool),geometric_contacts=np.ones(4,bool),
        measured_force_world=np.tile([0.,0.,50.],(4,1)),joint_position=np.zeros(18),joint_velocity=np.zeros(18))
    p._landing_mpc_model=SimpleNamespace(state_from_native=lambda *args:copy.deepcopy(state))
    p.reactive_vhsip=None;p.base=SimpleNamespace(reset=lambda ids:None)
    p.mixer=None;p.memory_conditioned=False
    return p


@pytest.mark.parametrize('parallel',[False,True])
@pytest.mark.parametrize('exact_passthrough',[False,True])
def test_native_commands_contact_ticks_and_early_returns_match(parallel,exact_passthrough):
    original=methods(SOURCE);candidate=methods(transform(SOURCE))
    left=policy();right=policy(FakePool() if parallel else None)
    left.landing_mpc_exact_student_passthrough=exact_passthrough
    right.landing_mpc_exact_student_passthrough=exact_passthrough
    raw=torch.zeros((5,794));raw[:,774]=1.;raw[:,775]=3.
    allowed=torch.tensor([True,True,False,True,True])
    limits=torch.tensor([[-1.,1.]]*18);default=torch.zeros(18)
    for tick in [300,301,302]:
        raw[:,775]=tick*.01
        action=torch.full((5,18),.2)
        expected=original['apply_landing_mpc'](left,dict(policy=raw),default,limits,action.clone(),eligible_mask=allowed)
        actual=candidate['apply_landing_mpc'](right,dict(policy=raw),default,limits,action.clone(),eligible_mask=allowed)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert left.landing_mpc_logs==right.landing_mpc_logs
    assert [c.count for c in right._landing_mpc_contact_state_lane]==[3,3,0,0,0]
    owner=right._landing_solver_pool.controller if parallel else right.landing_mpc
    assert owner.observations==left.landing_mpc.observations
    assert owner.solves==left.landing_mpc.solves
    if parallel:
        assert right.landing_mpc.observations==[] and right.landing_mpc.solves==[]
        assert len(right._landing_solver_pool.batches)==3
    candidate['finish_landing_mpc_episodes'](right,torch.tensor([True,False,False,True,False]))
    assert owner.resets==[0]
    candidate['reset'](right,torch.tensor([1]))
    assert owner.resets==[0,1]
    candidate['reset'](right,None)
    assert set(owner.resets[2:])==({0,1,2,3} if parallel else {None})


def test_worker_failure_aborts_native_batch_instead_of_silent_fallback():
    p=policy(FakePool(fail=True));raw=torch.zeros((5,794));raw[:,774]=1.;raw[:,775]=3.
    with pytest.raises(RuntimeError,match='worker failed'):
        methods(transform(SOURCE))['apply_landing_mpc'](p,dict(policy=raw),torch.zeros(18),
            torch.tensor([[-1.,1.]]*18),torch.zeros((5,18)))
