import numpy as np
import pytest
from scripts.audit_native_early_one_step_v1 import check_alignment


def test_alignment_requires_actual_consecutive_command_history():
    arrays={k:np.zeros((3,n)) for k,n in [('q',25),('v',24),('previous_target_rad',18),('final_target_rad',18)]}
    arrays['final_target_rad'][:]=np.arange(3)[:,None]
    arrays['previous_target_rad'][1:]=arrays['final_target_rad'][:-1]
    records=[{'control_tick':t} for t in [280,281,282]]
    assert check_alignment(arrays,records)==[280,281,282]
    arrays['previous_target_rad'][2,0]+=.01
    with pytest.raises(ValueError,match='dispatched command'):check_alignment(arrays,records)
    arrays['previous_target_rad'][1:]=arrays['final_target_rad'][:-1]
    records[2]['control_tick']=284
    with pytest.raises(ValueError,match='Contiguous'):check_alignment(arrays,records)
