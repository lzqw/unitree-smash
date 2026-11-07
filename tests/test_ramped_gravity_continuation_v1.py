"""A future equilibrium reference must not bypass the actual arm slew."""
from types import SimpleNamespace
import numpy as np
import pytest
import scripts.audit_ramped_gravity_continuation_v1 as ramp


def test_future_target_uses_two_legal_ticks_and_keeps_actual_history():
    interface=SimpleNamespace(limits=np.tile([-2.,2.],(18,1)),arm_rate=10.)
    c=SimpleNamespace(qp=SimpleNamespace(interface=interface),settings=SimpleNamespace(control_dt_s=.01))
    s=SimpleNamespace(previous_command_target=np.zeros(18));target=np.zeros(18);target[12:14]=[.116,-.13]
    commands=ramp.rate_admitted_seed(c,s,target,40)
    np.testing.assert_allclose(commands[0,12:14],[.1,-.1])
    np.testing.assert_allclose(commands[1],target)
    assert np.max(abs(np.diff(np.vstack([s.previous_command_target,commands]),axis=0)[:,12:]))<=.10000001
    np.testing.assert_array_equal(s.previous_command_target,np.zeros(18))


def test_future_reference_exception_restores_real_admission(monkeypatch):
    admission=lambda q,dt:(np.zeros(18),np.ones(18))
    interface=SimpleNamespace(limits=np.tile([-2.,2.],(18,1)),admitted_target_bounds=admission)
    c=SimpleNamespace(qp=SimpleNamespace(interface=interface));calls=[]
    def solve(*args):
        calls.append(1)
        if len(calls)==1:return dict(solved=False)
        raise RuntimeError('diagnostic failure')
    monkeypatch.setattr(ramp,'gravity_target',solve)
    with pytest.raises(RuntimeError):ramp.future_equilibrium_reference(c,None)
    assert interface.admitted_target_bounds is admission
