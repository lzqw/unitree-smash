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
 obj=object.__new__(EventCoupledFullBodyRolloutV1);obj.rollout=R();obj.impact_builder=lambda *args,**kwargs:dict(finite_mass_status='FINITE_MASS_VERIFIED',contact_feasible=True,legal_return_predicted=True,attack5_predicted=True,target_landing_predicted=True)
 r=obj([0]*24);assert r['finite_mass_status']=='FINITE_MASS_VERIFIED'

def test_impact_evidence_cannot_overwrite_rollout_gate():
 class R:
  def __call__(self,x): return dict(path_valid=True,terminal_errors={'position_m':.001,'normal':.01,'velocity_m_s':.1},trajectory=[{'q':[1]}])
 obj=object.__new__(EventCoupledFullBodyRolloutV1);obj.rollout=R();obj.impact_builder=lambda *a,**k:dict(path_valid=False)
 try:
  obj([0]*24);raise AssertionError
 except ValueError:
  pass

def test_event_layer_changes_joint_parameters_and_calls_wbc_hook():
 from scripts.event_coupled_fullbody_rollout_v1 import EventCoupledFullBodyEventRolloutV1
 seen=[]
 class Base:
  class R:
   commands=np.zeros((3,18));
  rollout=R()
  def __call__(self,x):
   seen.append(np.asarray(x).copy())
   return dict(path_valid=True,terminal_errors={'position_m':.1,'normal':.2,'velocity_m_s':1.},trajectory=[])
 base=Base()
 obj=EventCoupledFullBodyEventRolloutV1(base,wbc_event_fn=lambda event,_: np.ones(12)*event['com_acceleration_delta_world_m_s2'][2])
 x=np.zeros(32);x[4]=.5
 r=obj(x)
 assert r['wbc_event_source']=='event_wbc_callback' and len(seen)==1
 assert not np.array_equal(seen[0][:12],np.zeros(12))
