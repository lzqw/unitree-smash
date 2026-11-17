import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from quadsmash.control.action_conditioned_base_v1 import ActionConditionedBaseV1


def fixture():
    q=np.zeros(25);q[:3]=[3.,-2.,1.];q[3:7]=Rotation.from_rotvec([.2,.3,-.4]).as_quat()
    v=np.arange(24)*.01;targets=np.arange(36).reshape(2,18)*.01
    rows=[]
    for i in range(2):
        after=q.copy();after[:3]+=[.1*(i+1),-.2,.3]
        after[3:7]=Rotation.from_rotvec([.4,-.2,.1*(i+1)]).as_quat()
        rows.append(dict(node=i,elapsed_s=(i+1)*.01,q=after,v=v*(i+2),final_target_rad=targets[i].copy()))
    return q,v,targets,rows


def test_nonzero_pose_and_body_twist_transport_and_copies():
    q,v,targets,rows=fixture();p=ActionConditionedBaseV1(q,v,targets,rows)
    initial=Rotation.from_quat(q[3:7]).as_matrix();future=Rotation.from_quat(rows[-1]['q'][3:7]).as_matrix()
    out=p(.02)
    np.testing.assert_allclose(out['translation_current_m'],initial.T@(rows[-1]['q'][:3]-q[:3]))
    np.testing.assert_allclose(out['rotation_current_from_future'],initial.T@future)
    np.testing.assert_allclose(out['linear_velocity_current_m_s'],initial.T@future@rows[-1]['v'][:3])
    np.testing.assert_allclose(out['angular_velocity_current_rad_s'],initial.T@future@rows[-1]['v'][3:6])
    assert not out['execution_valid'] and out['requires_full_revalidation']
    saved=out['translation_current_m'].copy();out['translation_current_m'][:]=99;rows[-1]['q'][:]=77
    np.testing.assert_array_equal(p(.02)['translation_current_m'],saved)


@pytest.mark.parametrize('time',[-.01,.015,.03,float('nan')])
def test_no_unpredicted_time_or_interpolation(time):
    p=ActionConditionedBaseV1(*fixture())
    with pytest.raises(ValueError):p(time)


@pytest.mark.parametrize('changed',['q','v','arm','leg'])
def test_changed_state_or_either_command_group_invalidates_reference(changed):
    q,v,u,rows=fixture();p=ActionConditionedBaseV1(q,v,u,rows);p.validate_context(q,v,u)
    if changed=='q':q[0]+=.001
    elif changed=='v':v[1]+=.001
    elif changed=='arm':u[0,17]+=.001
    else:u[0,0]+=.001
    with pytest.raises(ValueError):p.validate_context(q,v,u)


def test_missing_nodes_or_command_mismatch_rejected():
    q,v,u,rows=fixture();rows[-1]['node']=4
    with pytest.raises(ValueError):ActionConditionedBaseV1(q,v,u,rows)
    q,v,u,rows=fixture();rows[0]['final_target_rad'][0]+=.1
    with pytest.raises(ValueError):ActionConditionedBaseV1(q,v,u,rows)
