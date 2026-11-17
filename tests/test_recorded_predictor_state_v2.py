from pathlib import Path
import numpy as np
from scripts.compare_recorded_predictors_v2 import measured_state
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1,rotation_wxyz


def test_native_state_adapter_preserves_joint_mapping_link_twist_and_history_isolation():
    source=Path('reports/task_constrained_hit_20261118/attempts/calibration01/native/control_state.npz')
    with np.load(source,allow_pickle=False) as data:
        model=LoadedPinocchioModelV1();s=measured_state(model,data,249,0);other=measured_state(model,data,249,1)
        np.testing.assert_array_equal(s.q[model.q_indices],data['joint_position'][249,0])
        np.testing.assert_array_equal(s.v[model.v_indices],data['joint_velocity'][249,0])
        np.testing.assert_array_equal(s.v[:6],data['public'][249,0,:6])
        np.testing.assert_allclose(s.root_rotation_world,rotation_wxyz(data['root_quaternion'][249,0]))
        previous=s.previous_command_target.copy();other_previous=other.previous_command_target.copy()
        s.previous_command_target[:]=123.;s.contacts[:]=False;s.q[:]=0
        np.testing.assert_array_equal(data['previous_target'][249,0],previous)
        np.testing.assert_array_equal(other.previous_command_target,other_previous)
        assert np.linalg.norm(other.q)>0
