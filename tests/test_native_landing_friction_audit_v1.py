import numpy as np
from scripts.audit_native_landing_friction_v1 import interval_balance


def test_measured_impulses_close_momentum_with_gravity():
    normal=np.array([[0.,0.,3.],[0.,0.,1.]])
    friction=np.array([[.4,-.8,0.],[-.1,.2,0.]])
    before=np.array([1.,2.,3.]);mass=20.
    after=before+normal.sum(0)+friction.sum(0)+[0.,0.,-mass*9.81*.01]
    result=interval_balance(before,after,mass,normal,friction)
    assert np.linalg.norm(result['measured_force_balance_residual_ns'])<1e-12
    assert np.linalg.norm(result['reversed_friction_sign_residual_ns'])>1.


def test_normal_only_telemetry_does_not_explain_friction():
    result=interval_balance([0.,0.,0.],[1.,0.,-.981],10.,np.zeros((4,3)),np.zeros((4,3)))
    assert np.allclose(result['measured_force_balance_residual_ns'],[1.,0.,0.])
