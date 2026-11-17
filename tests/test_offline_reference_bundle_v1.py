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


def test_final_adoption_gate_refuses_physically_valid_but_unreturned_candidates():
    from scripts.offline_reference_bundle_v1 import require_final_trajectory_admission
    phys=dict(accepted=True,leg_reference='causal_frozen_base',
              replayed_target_sequence_rad=[[0.]*18])
    plan=dict(accepted=True,physical_evaluator=True,final_contact_position_error_m=.0008,
              final_impact_checks=dict(contact_feasible=True,legal_return_predicted=True,
                attack5_predicted=True,target_landing_predicted=True))
    assert require_final_trajectory_admission(plan,phys)
    for key in ('contact_feasible','legal_return_predicted','attack5_predicted',
                'target_landing_predicted'):
        bad=dict(plan,final_impact_checks=dict(plan['final_impact_checks'],**{key:False}))
        with pytest.raises(ValueError,match='Final physical-terminal analytic'):
            require_final_trajectory_admission(bad,phys)
    with pytest.raises(ValueError,match='3mm'):
        require_final_trajectory_admission(
            dict(plan,final_contact_position_error_m=.0031),phys)
    with pytest.raises(ValueError,match='planner and physical'):
        require_final_trajectory_admission(
            dict(plan,accepted=False),phys)
    with pytest.raises(ValueError,match='planner and physical'):
        require_final_trajectory_admission(
            plan,dict(phys,accepted=False))
    with pytest.raises(ValueError,match='Missing full 18D'):
        require_final_trajectory_admission(
            dict(plan,physical_evaluator=False),phys)


def test_writer_itself_requires_final_admission_and_exact_sequence(tmp_path):
    a=arguments();a['cpu_prediction_valid']=True
    with pytest.raises(ValueError,match='planner and physical'):
        write_candidate(tmp_path/'missing',**a)
    assert not (tmp_path/'missing').exists()
    plan=dict(accepted=True,physical_evaluator=True,final_contact_position_error_m=.0008,
        final_impact_checks=dict(contact_feasible=True,legal_return_predicted=True,attack5_predicted=True,target_landing_predicted=True))
    phys=dict(accepted=True,leg_reference='causal_frozen_base',replayed_target_sequence_rad=a['final_targets_rad'].tolist())
    meta=write_candidate(tmp_path/'valid',**a,planner_record=plan,physical_result=phys)
    assert meta['final_cpu_admission']['final_contact_position_error_m']==.0008
    assert meta['PHYSX_REFERENCE_VALID']=='NOT_RUN'
    phys['replayed_target_sequence_rad'][0][0]=1e-3
    with pytest.raises(ValueError,match='differ'):
        write_candidate(tmp_path/'wrong',**a,planner_record=plan,physical_result=phys)


def test_no_custom_contact_limit_can_relax_final_gate():
    from scripts.offline_reference_bundle_v1 import require_final_trajectory_admission
    with pytest.raises(ValueError,match='3mm'):
        require_final_trajectory_admission({}, {},contact_limit_m=.1)
