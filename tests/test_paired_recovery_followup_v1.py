import pytest
from scripts.run_paired_recovery_learning_followup_v1 import validate_teacher_off_command


def command():
    return ['python3','scripts/run_gpu_job.py','--max-seconds','3600','--env-steps-cap','102400','--',
            'bash','scripts/python.sh','eval.py','--output','run','--device','cuda:0',
            '--learned-recovery-command-head','head.pt']


def test_candidate_must_be_teacher_off_and_use_the_declared_head():
    validate_teacher_off_command(command(), 'head.pt', 'run')
    for flag in ['--learned-recovery-mpc-correction','--landing-mpc-force-post-touchdown']:
        with pytest.raises(ValueError): validate_teacher_off_command(command()+[flag], 'head.pt', 'run')
    with pytest.raises(ValueError): validate_teacher_off_command(command(), 'wrong.pt', 'run')


def test_guard_and_gpu_are_required():
    c=command();c[c.index('cuda:0')]='cuda:1'
    with pytest.raises(ValueError): validate_teacher_off_command(c,'head.pt','run')
    with pytest.raises(ValueError): validate_teacher_off_command(command()[7:],'head.pt','run')
