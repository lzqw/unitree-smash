import copy
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.refine_contact_event_v1 import retimed_metadata


def fixture():
    court=dict(net_width_m=6.,opponent_x_limits_m=[0,7],singles_width_m=5.,target_x_limits_m=[2,5],target_y_half_width_m=1.)
    forward=dict(outgoing_velocity=[6.,0.,-2.],landing_position=[3.,0.,.025],net_clearance=.2,crossing_position=[0.,0.,2.],contact_feasible=True,legal_return_predicted=True,attack5_predicted=True,target_landing_predicted=True)
    impact=SimpleNamespace(physics={'collision_radius_m':.025},court=court,evaluate=lambda *a:copy.deepcopy(forward))
    row=dict(time_s=.33,ball_world_m=[1,0,2],incoming_world_m_s=[-3,0,-1],analytic_goal=dict(accepted=True,normal_world=[1,0,0],racket_velocity_world_m_s=[2,0,-1]))
    q=np.zeros(25);q[6]=1.;q[:3]=[.1,.2,.3]
    return dict(contact_point_local=[.0006,0,0]),row,q,impact,forward


def test_goal_retransport_preserves_source_and_requires_real_forward_task():
    original,row,q,impact,forward=fixture();m=retimed_metadata(original,row,q,impact)
    assert m['contact_index']==33 and m['contact_time_s']==.33
    np.testing.assert_allclose(m['target_position_world_m'],[.9,-.2,1.7])
    assert m['analytic_goal_valid'] and not m['robot_target_verified']
    assert 'contact_index' not in original
    forward['outgoing_velocity']=[4.0,0,-1.1]
    with pytest.raises(ValueError,match='Forward task'):retimed_metadata(original,row,q,impact)


def test_missing_or_invalid_event_cannot_reuse_accepted_label():
    original,row,q,impact,forward=fixture();row['analytic_goal']['normal_world']=[2,0,0]
    with pytest.raises(ValueError,match='Unit'):retimed_metadata(original,row,q,impact)
    original,row,q,impact,forward=fixture();forward.pop('attack5_predicted')
    with pytest.raises(ValueError,match='Forward task'):retimed_metadata(original,row,q,impact)
    original,row,q,impact,forward=fixture();row['time_s']=.331
    with pytest.raises(ValueError,match='Control-grid'):retimed_metadata(original,row,q,impact)
