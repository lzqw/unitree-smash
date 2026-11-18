import numpy as np,pytest
from quadsmash.control.contact_reachability_atlas_v1 import ContactReachabilityAtlasV1

def atlas():
 def fk(q):
  return q[:3],np.array([1.,0,0]),np.c_[np.eye(3),np.zeros((3,3))],np.zeros((3,6))
 return ContactReachabilityAtlasV1(kinematics_fn=fk,lower=-np.ones(6),upper=np.ones(6),speed=np.ones(6)*5)

def test_atlas_is_read_only_and_sorted():
 a=atlas();r=a.evaluate([dict(time_s=.33,position_m=[.1,.2,.3],normal=[1,0,0],velocity_m_s=[1,0,0],q_seed_rad=np.zeros(6))]);assert r['kinematic_only'] and not r['teacher_eligible'];assert r['candidates'][0]['fullbody_verified'] is False

def test_atlas_rejects_incomplete_or_nonfinite_candidates():
 with pytest.raises(ValueError):atlas().evaluate([dict(time_s=.33,position_m=[0,0,0],normal=[1,0,0],velocity_m_s=[1,0,0])])
 with pytest.raises(ValueError):atlas().evaluate([dict(time_s=.33,position_m=[np.nan,0,0],normal=[1,0,0],velocity_m_s=[1,0,0],q_seed_rad=np.zeros(6))])
