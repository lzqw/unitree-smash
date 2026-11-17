import numpy as np
from quadsmash.control.contact_pose_reachability_v1 import diagnose_contact_pose


def test_diagnostic_distinguishes_pose_and_speed_limits_without_relaxing():
    def kin(q):return q[:3],np.array([0.,0.,1.]),np.c_[np.eye(3),np.zeros((3,3))],np.zeros((3,6))
    result=diagnose_contact_pose(kin,np.zeros(6),np.full(6,-1.),np.ones(6),np.ones(6),
        np.array([.2,0.,0.]),np.array([0.,0.,1.]),np.array([4.,0.,0.]))
    r=result['stages']['position_normal']
    assert r['position_error_m']<1e-6
    assert r['first_failed_check']=='velocity'
    assert r['velocity_error_m_s']>2.9
    assert min(r['joint_limit_margin_rad'])>=0
    assert not result['global_infeasibility_proven']
