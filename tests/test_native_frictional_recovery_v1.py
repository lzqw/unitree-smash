"""Dispatch/tamper tests with a synthetic solver; these do not run native physics."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_native_recovery_sequence_replay_v1 import setup_replay
from scripts.native_frictional_recovery_v1 import NativeFrictionalRecoveryV1
from scripts.audit_native_frictional_recovery_v1 import audit_dispatch


def recorded_fixture(tmp_path):
    c,public,student,finished,state=setup_replay(tmp_path)
    c.__class__=NativeFrictionalRecoveryV1;c.steps=40;c.start=760;c.query_public[775]=7.6
    c.last_tick=-1;c.previous=None;c.plan=None;c.plan_start=None;c.plan_contacts=None
    c.planning_failed=False;c.plans=[];c.teacher_steps=0;c.fallback_steps=0
    c.controller=SimpleNamespace(qp=SimpleNamespace(interface=SimpleNamespace(limits=c.limits,arm_rate=10.)))
    c.warm_commands=np.vstack([c.commands,c.commands]);c.predicted_states=None;c.executed_prediction_errors=[]
    c.warm_plan_sha256='test-warm';c.sphere_audit_sha256='test-sphere'
    def solve(observed,reference,warm):
        if not observed.contacts.any():return dict(accepted=False,commands=None,wall_seconds=.001)
        commands=warm.copy();commands[:,0]+=.001
        return dict(accepted=True,commands=commands,initial_q=observed.q.copy(),initial_v=observed.v.copy(),
            previous_command=observed.previous_command_target.copy(),wall_seconds=.001,
            predicted_states=[dict(q=observed.q.tolist(),v=observed.v.tolist()) for _ in range(80)])
    c.planner=SimpleNamespace(solve=solve,close=lambda:None);raw=[];final=[]
    for tick in range(800):
        public[15,775]=tick*.01;current=None
        if tick>=760:
            current=state(tick-760)
            if tick>=763:current.contacts[0]=not c.expected_contacts[0,0]
            if tick>=780:current.contacts[:]=False
        raw.append(student.numpy().copy());final.append(c.action(tick,public,student,finished,current).numpy().copy())
    result=c.save()
    with np.load(c.output/'sequence_native_trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    return result,trace,np.stack(raw),np.stack(final),c.limits


def test_feedback_and_latched_fallback_are_separately_verified(tmp_path):
    args=recorded_fixture(tmp_path);report=audit_dispatch(*args)
    assert report['teacher_steps']==20 and report['unverified_fallback_steps']==20
    assert report['replans']==4 and report['accepted_replans']==3
    assert report['maximum_command_mapping_error_rad']<1e-6


@pytest.mark.parametrize('tamper',['observation','history','false_teacher','wrong_plan','other_lane','missing_contact_replan'])
def test_corrupted_feedback_evidence_is_rejected(tmp_path,tamper):
    result,trace,raw,final,limits=recorded_fixture(tmp_path)
    if tamper=='observation':result['planning_records'][1]['observed_v'][0]+=.1
    elif tamper=='history':result['planning_records'][1]['actual_previous_target_rad'][12]+=.01
    elif tamper=='false_teacher':result['records'][20]['mode']='frictional_receding_support_sequence'
    elif tamper=='wrong_plan':result['planning_records'][0]['decision']['commands'][0][0]+=.01
    elif tamper=='other_lane':final[799,0,0]+=.01
    elif tamper=='missing_contact_replan':result['planning_records'].pop(1)
    with pytest.raises(ValueError):audit_dispatch(result,trace,raw,final,limits)
