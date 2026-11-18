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


def test_old_event_claims_are_nested_not_relabelled_as_current():
    original,row,q,impact,forward=fixture()
    original.update(attack_geometry={'predicted_contact_height':99},achieved_position_current_frame_m=[9,9,9],task_feasible_velocity_retarget=True)
    m=retimed_metadata(original,row,q,impact)
    assert 'attack_geometry' not in m and 'task_feasible_velocity_retarget' not in m
    assert m['seed_metadata']['attack_geometry']['predicted_contact_height']==99
    assert m['robot_target_verified'] is False and m['retimed_task_margins']['valid']


def continuation_fixture():
    from scripts.refine_contact_event_v1 import continued_event_reference
    metadata=dict(contact_time_s=.02,contact_index=2,target_position_world_m=np.array([1,2,3]))
    reference=np.full((3,6),.1);trace=[]
    for k in range(2):
        trace.append(dict(node=k,q=np.r_[np.zeros(7),np.full(6,.12)],planned_arm_reference_rad=reference[k+1].copy(),arm_tracking_error_rad=.02,final_target_rad=np.zeros(18)))
    record=dict(metadata=copy.deepcopy(metadata),event_time_s=.02,selected_index=0,selected_arm_reference_rad=reference,trials=[dict(result=dict(path_valid=True,trajectory=trace))])
    return continued_event_reference,record,metadata,reference


def test_continuation_preserves_history_and_storage_isolation():
    fn,record,metadata,reference=continuation_fixture()
    trace,commands,restored=fn(record,metadata,reference[0],np.arange(7,13))
    np.testing.assert_array_equal(restored,reference)
    trace[0]['q'][:]=99;commands[:]=99;restored[:]=99
    assert record['trials'][0]['result']['trajectory'][0]['q'][0]==0
    assert reference[0,0]==.1


def test_continuation_rejects_target_partial_or_reference_drift():
    for mode in ('target','partial','reference'):
        fn,record,metadata,reference=continuation_fixture()
        if mode=='target':metadata['target_position_world_m'][0]+=1
        if mode=='partial':record['trials'][0]['result']['path_valid']=False
        if mode=='reference':record['selected_arm_reference_rad'][-1,0]+=.01
        with pytest.raises(ValueError):fn(record,metadata,np.full(6,.1),np.arange(7,13))


def test_projected_goal_rechecks_task_and_preserves_event_and_history():
    import json, copy
    from pathlib import Path
    from scripts.refine_contact_event_v1 import projected_event_metadata
    from quadsmash.control.impact_outcome_evaluator_v1 import ImpactOutcomeEvaluator
    from quadsmash.physics.full_racket_v4 import load_racket
    record=json.loads(Path('reports/tracking_model_reserve_20261118/RESULTS.json').read_text())
    projection=json.loads(Path('reports/task_goal_projection_20261118/RESULTS.json').read_text())
    q=json.loads(Path('reports/push_off_task_20261118/DIAGNOSED.json').read_text())['rows'][0]['initial_q']
    m=record['metadata'];before=copy.deepcopy(m)
    impact=ImpactOutcomeEvaluator(m['task_physics'],m['task_court'],load_racket())
    updated=projected_event_metadata(m,projection,q,impact)
    assert m==before
    for key in ('contact_index','contact_time_s','target_position_world_m','incoming_velocity_planning','contact_point_local','task_physics','task_court'):
        assert updated[key]==m[key]
    assert updated['retimed_task_margins']['valid']
    assert not updated['robot_target_verified']
    bad=copy.deepcopy(projection);bad['ball_world_m'][2]+=.001
    with pytest.raises(ValueError,match='causal ball'):projected_event_metadata(m,bad,q,impact)
    bad=copy.deepcopy(projection);bad['selected']['racket_velocity_world_m_s']=bad['incoming_world_m_s']
    with pytest.raises(ValueError,match='forward task'):projected_event_metadata(m,bad,q,impact)
    bad=copy.deepcopy(projection);bad['contact_time_s']+=.01
    with pytest.raises(ValueError,match='event'):projected_event_metadata(m,bad,q,impact)
