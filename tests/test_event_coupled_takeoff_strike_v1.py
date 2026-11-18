import numpy as np,pytest
from quadsmash.control.event_coupled_takeoff_strike_v1 import EventCoupledTakeoffStrikePlannerV1

def test_planner_uses_rollout_state_and_fail_closed():
 seen=[]
 def rollout(t):
  seen.append(t.copy());return dict(path_valid=True,contact_position_error_m=.002,normal_error=.1,velocity_error_m_s=.7,finite_mass_status='FINITE_MASS_VERIFIED')
 p=EventCoupledTakeoffStrikePlannerV1(rollout_fn=rollout,bounds=np.array([[0,1],[0,1]]),max_candidates=2)
 r=p.search([np.array([.2,.3]),np.array([.4,.5])]);assert len(r['accepted'])==0 and len(r['failed'])==2 and len(seen)==2

def test_hard_gates_reject_without_mutating_state():
 state=np.array([.2,.3])
 def rollout(t): return dict(path_valid=True,contact_position_error_m=.004,normal_error=.01,velocity_error_m_s=.1,finite_mass_status='FINITE_MASS_VERIFIED')
 r=EventCoupledTakeoffStrikePlannerV1(rollout_fn=rollout,bounds=np.array([[0,1],[0,1]])).search([state]);assert r['accepted']==[] and r['failed'][0]['result']['contact_position_error_m']==.004;np.testing.assert_array_equal(state,[.2,.3])

def test_unverified_mass_never_passes():
 def rollout(t): return dict(path_valid=True,contact_position_error_m=.001,normal_error=.01,velocity_error_m_s=.1,finite_mass_status='EFFECTIVE_MASS_UNVERIFIED')
 r=EventCoupledTakeoffStrikePlannerV1(rollout_fn=rollout,bounds=np.array([[0,1]])).search([np.array([.5])]);assert r['failed'][0]['result']

def test_optimize_updates_parameters_and_keeps_best_trace():
 seen=[]
 def rollout(t):
  seen.append(t.copy());return dict(path_valid=True,contact_position_error_m=float(abs(t[0])+.001),normal_error=.1,velocity_error_m_s=.7,finite_mass_status='FINITE_MASS_VERIFIED',contact_feasible=True,legal_return_predicted=True,attack5_predicted=True,target_landing_predicted=True)
 p=EventCoupledTakeoffStrikePlannerV1(rollout_fn=rollout,bounds=np.array([[-1,1]]),max_candidates=1);r=p.optimize([np.array([.5])],iterations=2);assert r['calls']>1 and r['best'][0]['cost']<100
