"""Synthetic report integrity tests, never native performance evidence."""
import json
from pathlib import Path
import importlib.util
import pytest

SPEC=importlib.util.spec_from_file_location('sm_comparison',
    Path(__file__).resolve().parents[1]/'tools/compare_supermimic_source_evaluation.py')
MODULE=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


def fixture(tmp_path):
    run=tmp_path/'run'; old=tmp_path/'old';run.mkdir();old.mkdir()
    def write(p,x):p.write_text(json.dumps(x))
    original=[dict(training_index=i,air=i<74,recovery=i<34,
                   ball5_except_recovery=i<53,complete5=i<27) for i in range(128)]
    rows=[dict(env_id=i,source_case_index=i,episode_index=0,
               qualified_airborne_string_contact=i<74,recovery_success=i<34,
               ball_side_complete_except_recovery=i<53,complete5=i<27,
               head_contact=None,independent_nonfoot_contact=None) for i in range(128)]
    write(old/'metrics.json',dict(airborne_contacts=74,complete5=27,complete5_case_outcomes=original))
    write(old/'INPUTS.json',dict(policy_sha256='synthetic'))
    write(run/'metrics.json',dict(evaluated_checkpoint_sha256='synthetic',actor_updated=False,
        num_envs=128,control_ticks=800,qualified_airborne_string_contact=74,
        ball_side_complete_except_recovery=53,complete5=27,conditional_recovery_count=34,
        conditional_recovery_denominator=74,numerical_failures=0,task_contract={}))
    write(run/'completed.json',dict(control_ticks=800,episodes=128))
    write(run/'episodes_summary.json',rows)
    write(run/'command_consistency.json',dict(checked_native_ticks=800,tolerance_rad=2e-6,
        max_final_target_error_rad=0,phase_samples={str(i):1 for i in range(5)}))
    (tmp_path/'exit_code.txt').write_text('0\n')
    return run,old,rows


def test_unknown_contact_evidence_not_converted_to_clean_success(tmp_path):
    run,old,_=fixture(tmp_path)
    result=MODULE.compare(run,old)
    assert result['no_contact_recovery']==dict(head_contact=None,independent_nonfoot_contact=None)
    assert result['fixed_contact_recovery']['denominator']==74
    assert result['recorded_outcomes_and_commands_match']


def test_same_total_different_cases_is_not_reproduction(tmp_path):
    run,old,rows=fixture(tmp_path)
    rows[0]['recovery_success']=rows[0]['complete5']=False
    rows[40]['recovery_success']=rows[40]['complete5']=True
    (run/'episodes_summary.json').write_text(json.dumps(rows))
    result=MODULE.compare(run,old)
    assert not result['recorded_outcomes_and_commands_match']
    assert result['fixed_contact_recovery']['lost']==[0]
    assert result['fixed_contact_recovery']['added']==[40]
    assert result['fixed_contact_recovery']['current']==34


def test_no_completed_marker_refuses_partial_score(tmp_path):
    run,old,_=fixture(tmp_path)
    (run/'completed.json').unlink()
    with pytest.raises(FileNotFoundError):MODULE.compare(run,old)


def test_cleanup_failure_not_counted_as_clean_completed_execution(tmp_path):
    run,old,_=fixture(tmp_path)
    (tmp_path/'exit_code.txt').write_text('139\n')
    result=MODULE.compare(run,old)
    assert result['matched_outcome_sets']
    assert not result['recorded_outcomes_and_commands_match']


def test_inconsistent_success_components_rejected(tmp_path):
    run,old,rows=fixture(tmp_path)
    rows[0]['recovery_success']=False
    (run/'episodes_summary.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='contradicts'):
        MODULE.compare(run,old)
