from types import SimpleNamespace
import torch
from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4


def fixture():
    c=object.__new__(FullRacketRallyControllerV4)
    c.contact_first_planner=SimpleNamespace(config=SimpleNamespace(dt_s=.01,horizon_steps=4,handoff_duration_s=.02))
    c.contact_first_plan_valid=torch.ones(2,dtype=torch.bool)
    c.contact_first_joint_valid=torch.ones(2,dtype=torch.bool)
    c.contact_first_recovery_owned=torch.tensor([False,True])
    c.contact_first_plan_q=torch.zeros(2,4,6)
    c.contact_first_plan_start_time=torch.zeros(2)
    c.contact_first_plan_contact_time=torch.full((2,),.03)
    c.contact_first_joint_targets=torch.arange(144).reshape(2,4,18).float()/200
    c.contact_first_execution_log=[]
    public=torch.zeros(2,794);public[:,775]=.01
    limits=torch.tensor([[-1.,1.]]*18)
    return c,public,limits


def test_exact_verified_full_command_and_recovery_exclusion():
    c,public,limits=fixture();base=torch.full((2,18),-.2)
    result=c._overlay_contact_first_plan(base.clone(),public,limits)
    torch.testing.assert_close(result[0],c.contact_first_joint_targets[0,1])
    torch.testing.assert_close(result[1],base[1])
    assert c.contact_first_execution_log[0]['lanes']==[0]
    assert max(c.contact_first_execution_log[0]['round_trip_error_rad'])<2e-6


def test_joint_prefix_never_executes_unvalidated_terminal_or_handoff_blend():
    c,public,limits=fixture();base=torch.full((2,18),-.2)
    for time in (.03,.04,.06):
        public[:,775]=time
        torch.testing.assert_close(c._overlay_contact_first_plan(base.clone(),public,limits),base)
