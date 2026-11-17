import numpy as np
import pytest
from quadsmash.control.push_off_window_v1 import PushOffWindowV1


def test_release_latches_even_if_contact_returns():
    w=PushOffWindowV1(29)
    assert w.step(0,np.array([1,0,0,0],bool))['enabled']
    assert not w.step(1,np.zeros(4,bool))['enabled']
    r=w.step(2,np.ones(4,bool));assert not r['enabled'] and r['phase']=='FLIGHT_OBSERVED'


def test_support_deadline_and_recovery_always_override():
    w=PushOffWindowV1(3)
    assert w.step(0,np.ones(4,bool))['enabled']
    assert w.step(1,np.ones(4,bool))['enabled']
    assert w.step(2,np.ones(4,bool))['phase']=='CONTACT_DEADLINE'
    assert w.step(3,np.ones(4,bool),recovery_owned=True)['phase']=='RECOVERY_OWNED'
    assert not w.step(4,np.ones(4,bool))['enabled']


def test_partial_reset_does_not_clear_other_window_or_mutate_contacts():
    a=PushOffWindowV1(10);b=PushOffWindowV1(10);contacts=np.ones(4,bool)
    r=a.step(0,contacts,recovery_owned=True);r['contacts'][:]=False
    b.step(0,contacts,recovery_owned=True);a.reset()
    assert a.step(0,contacts)['enabled']
    assert not b.step(1,contacts)['enabled']
    assert contacts.all()


def test_invalid_or_skipped_state_fails():
    w=PushOffWindowV1(5)
    with pytest.raises(ValueError):w.step(1,np.ones(4,bool))
    with pytest.raises(ValueError):w.step(0,np.ones(4))
