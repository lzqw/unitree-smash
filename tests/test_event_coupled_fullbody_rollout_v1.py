import numpy as np
from scripts.event_coupled_fullbody_rollout_v1 import EventCoupledFullBodyRolloutV1

def test_parameter_contract_requires_24d_and_rejects_nonfinite():
 class R:
  def __call__(self,x):return dict(path_valid=False)
 obj=object.__new__(EventCoupledFullBodyRolloutV1);obj.rollout=R()
 try:obj(np.zeros(12));raise AssertionError
 except ValueError:pass
 try:obj(np.full(24,np.nan));raise AssertionError
 except ValueError:pass

def test_impact_builder_controls_mass_and_task_evidence():
 class R:
  def __call__(self,x): return dict(path_valid=True,terminal_errors={'position_m':.001,'normal':.01,'velocity_m_s':.1},trajectory=[{'q':[1]}])
 from scripts.event_coupled_fullbody_rollout_v1 import EventCoupledFullBodyRolloutV1
 obj=object.__new__(EventCoupledFullBodyRolloutV1);obj.rollout=R();obj.impact_builder=lambda t,x:dict(finite_mass_status='FINITE_MASS_VERIFIED',contact_feasible=True,legal_return_predicted=True,attack5_predicted=True,target_landing_predicted=True)
 r=obj([0]*24);assert r['finite_mass_status']=='FINITE_MASS_VERIFIED'
