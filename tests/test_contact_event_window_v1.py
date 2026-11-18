import numpy as np
import pytest
from scripts.contact_event_window_v1 import hold_extension


def test_hold_keeps_entire_history_and_does_not_alias():
    commands=np.arange(54,dtype=float).reshape(3,18);ref=np.arange(24,dtype=float).reshape(4,6)
    extended,reference=hold_extension(commands,ref,7)
    np.testing.assert_array_equal(extended[:3],commands)
    np.testing.assert_array_equal(reference[:4],ref)
    np.testing.assert_array_equal(extended[3:],np.repeat(commands[-1:],4,axis=0))
    np.testing.assert_array_equal(reference[4:],np.repeat(ref[-1:],4,axis=0))
    extended[:]=0;reference[:]=0
    assert commands[-1,-1]==53 and ref[-1,-1]==23


def test_no_unbounded_or_partial_history_extension():
    commands=np.zeros((3,18));ref=np.zeros((4,6))
    for count in (2,8,4.5):
        with pytest.raises(ValueError):hold_extension(commands,ref,count)
    with pytest.raises(ValueError):hold_extension(commands,ref[:-1],5)
    commands[0,0]=np.nan
    with pytest.raises(ValueError):hold_extension(commands,ref,5)
