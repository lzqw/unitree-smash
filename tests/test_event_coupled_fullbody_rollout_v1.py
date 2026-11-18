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
