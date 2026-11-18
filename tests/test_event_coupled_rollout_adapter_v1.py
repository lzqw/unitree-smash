import numpy as np
from scripts.event_coupled_rollout_adapter_v1 import EventCoupledRolloutAdapterV1

def test_adapter_exports_only_real_terminal_errors():
 def rollout(theta):
  return dict(path_valid=True,terminal_errors={'position_m':.002,'normal':.1,'velocity_m_s':.7},trajectory=[{'q':[1]}],final_impact={'x':1})
 r=EventCoupledRolloutAdapterV1(rollout,finite_mass_status='FINITE_MASS_VERIFIED')(np.array([.2]))
 assert r['contact_position_error_m']==.002 and r['finite_mass_status']=='FINITE_MASS_VERIFIED';assert r['teacher_eligible'] is False

def test_invalid_path_fails_closed():
 r=EventCoupledRolloutAdapterV1(lambda _:dict(path_valid=False))(np.zeros(1));assert r['path_valid'] is False and 'contact_position_error_m' not in r
