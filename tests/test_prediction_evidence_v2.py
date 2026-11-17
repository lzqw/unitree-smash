from scripts.summarize_prediction_evidence_v2 import summarize


def test_missing_nonfinite_and_violating_diagnostics_never_become_admission():
    row = dict(method='model', lane=0, tick=3, solved=True, racket_position_error_m=.004,
               diagnostics={'joint_position_violation_rad': .008, 'motor_torque_violation_nm': float('nan')})
    record = dict(summary={'model': {}}, steps=80, start_tick=3, mode='open_loop', rows=[row])
    result = summarize(record)
    group = result['groups'][0]
    assert group['requested'] == 80 and group['solved'] == 1
    assert group['physical_admission'] == 'NOT_EVALUATED'
    assert group['contact_point_accuracy'] == 'NOT_MEASURED'
    assert group['diagnostic_evidence']['motor_torque_violation_nm']['missing_or_nonfinite_ticks'] == [3]
    assert group['diagnostic_evidence']['foot_penetration_m']['missing_or_nonfinite_ticks'] == [3]
    assert group['diagnostic_evidence']['joint_position_violation_rad']['first_above_display_threshold']['value'] == .008
    assert not result['valid_teacher_reference']
