import numpy as np
import pytest
from scripts.offline_reference_bundle_v1 import write_candidate,load_candidate,verify_replay_handoff


def arguments():
    return dict(final_targets_rad=np.zeros((3,18)),prior_final_target_rad=np.zeros(18),
        state_q=np.r_[np.zeros(6),1.,np.zeros(18)],state_v=np.zeros(24),control_tick=10,contact_index=3,
        limits=np.array([[-1.,1.]]*18),input_hashes={'source':'a'*64},controller_parameters={'mode':'frozen'},
        prediction_contract={'version':'diagnostic'},cpu_prediction_valid=False,state_source='native')


def test_bundle_never_certifies_native_and_checks_command_history(tmp_path):
    path=tmp_path/'candidate';a=arguments();meta=write_candidate(path,**a)
    assert meta['PHYSX_REFERENCE_VALID']=='NOT_RUN' and not meta['FULLBODY_PREDICTION_VALID']
    kwargs=dict(current_q=a['state_q'],current_v=a['state_v'],previous_final_target_rad=a['prior_final_target_rad'],input_hashes=a['input_hashes'])
    np.testing.assert_array_equal(verify_replay_handoff(path,**kwargs),a['final_targets_rad'])
    kwargs['previous_final_target_rad']=np.ones(18)*3e-6
    with pytest.raises(ValueError,match='prior_final'):verify_replay_handoff(path,**kwargs)
    with pytest.raises(FileExistsError):write_candidate(path,**a)


def test_incomplete_or_nonfinite_sequence_refused(tmp_path):
    a=arguments();a['final_targets_rad'][1,0]=np.nan
    with pytest.raises(ValueError):write_candidate(tmp_path/'bad',**a)
    assert not (tmp_path/'bad').exists()


def test_tamper_and_reconstructed_state_are_not_native_evidence(tmp_path):
    a=arguments();a['state_source']='public_reconstruction';path=tmp_path/'candidate';write_candidate(path,**a)
    with pytest.raises(ValueError,match='native state'):
        verify_replay_handoff(path,current_q=a['state_q'],current_v=a['state_v'],previous_final_target_rad=a['prior_final_target_rad'],input_hashes=a['input_hashes'])
    with (path/'commands_and_state.npz').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='digest'):load_candidate(path)
