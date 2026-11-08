import numpy as np
import pytest

from scripts.event_recovery_manager_v1 import EventRecoveryManagerV1,RecoveryCommandArbiterV1


def sample(manager,t,**changes):
    data=dict(episode=0,time_s=t,had_racket_contact=False,released=False,
        racket_touching=False,separated_now=False,ground_normal_force_n=np.zeros(4),
        roll_pitch_rad=np.zeros(2),root_velocity_world=np.zeros(3),
        root_angular_velocity_world=np.zeros(3),joint_velocity=np.zeros(18))
    data.update(changes)
    return manager.update(**data)


def test_release_requires_current_separation_not_future_outcome_or_clock():
    m=EventRecoveryManagerV1()
    assert sample(m,3.)['phase']=='PREFIX'
    assert sample(m,3.01,had_racket_contact=True,released=True,racket_touching=True,separated_now=False)['phase']=='PREFIX'
    assert sample(m,3.02,had_racket_contact=True,released=True,separated_now=True)['phase']=='AIR_PREPARE'
    assert sample(m,3.03)['owns_final_recovery_command']


def test_loss_of_ground_does_not_release_control_and_light_contact_is_retained():
    m=EventRecoveryManagerV1()
    r=sample(m,.4,had_racket_contact=True,released=True,separated_now=True,
        ground_normal_force_n=[2.,0,0,0])
    assert r['phase']=='BUFFER_LOAD' and r['ground_contacts'][0] and not r['currently_loaded'][0]
    r=sample(m,.41,ground_normal_force_n=[.7,0,0,0])
    assert r['ground_contacts'][0]
    r=sample(m,.42)
    assert r['phase']=='AIR_PREPARE' and r['owns_final_recovery_command']
    r=sample(m,.43,ground_normal_force_n=[0,2.,0,0])
    assert r['phase']=='BUFFER_LOAD'


def test_stability_does_not_exit_and_nonfoot_contact_resets_window():
    m=EventRecoveryManagerV1()
    sample(m,0.,had_racket_contact=True,released=True,separated_now=True,ground_normal_force_n=np.ones(4)*20)
    for i in range(1,101):r=sample(m,i*.01,ground_normal_force_n=np.ones(4)*20)
    assert r['phase']=='SUSTAIN' and r['owns_final_recovery_command']
    assert r['control_sample_stable_duration_s']>.8
    r=sample(m,1.01,ground_normal_force_n=np.ones(4)*20,nonfoot_contact_now=True)
    assert r['phase']=='BUFFER_LOAD' and r['control_sample_stable_duration_s']==0.
    r=sample(m,0.,episode=1)
    assert r['phase']=='PREFIX' and r['release_time_s'] is None


def test_no_skipped_time_counted_as_continuous_stability():
    m=EventRecoveryManagerV1();sample(m,0.)
    with pytest.raises(ValueError):sample(m,.5)


def test_controller_owns_total_command_with_original_bounds_only():
    limits=np.tile([-2.,2.],(18,1));arb=RecoveryCommandArbiterV1(limits)
    previous=np.zeros(18);student=np.full(18,-1.);controller=np.full(18,1.)
    out,record=arb.select(student,controller,previous,owns_recovery=True)
    assert np.array_equal(out[:12],controller[:12])
    assert np.allclose(out[12:],.1)
    assert not record['student_residual_added_after_controller']
    other,_=arb.select(-student,controller,previous,owns_recovery=True)
    assert np.array_equal(other,out)
    prefix,_=arb.select(student,controller,previous,owns_recovery=False)
    assert np.array_equal(prefix[:12],student[:12])
