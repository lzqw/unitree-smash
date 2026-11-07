import pytest
import torch
from quadsmash.control.learned_recovery_command_v1 import NativeRecoveryCommandAdapter
from quadsmash.control.teacher_corrected_recovery_command_v1 import teacher_corrected_recovery_command


class InspectingHead(torch.nn.Module):
    def __init__(self):
        super().__init__();self.inputs=[]
    def forward(self,x):
        self.inputs.append(x.clone())
        return torch.full((len(x),18),.01,dtype=x.dtype,device=x.device)


def setup():
    head=InspectingHead();student=NativeRecoveryCommandAdapter(head,2,'cpu',torch.tensor([[-2.,2.]]*18))
    public=torch.zeros((2,794));base=torch.zeros((2,18))
    teacher_corrected_recovery_command(student,public,base,lambda *_:pytest.fail('Teacher before eligibility'))
    public[:,775]=.01;public[0,773]=1
    teacher_corrected_recovery_command(student,public,base,lambda *_:pytest.fail('Teacher while airborne'))
    public[:,775]=.02;public[:,773]=0;public[:,774]=1
    return student,public,base


def test_final_command_is_remembered_and_proposal_is_preserved():
    student,public,base=setup();calls=[]
    def correct(proposal,active):
        calls.append(proposal.clone());assert active.tolist()==[True,False]
        proposal[0,:12]=.08;proposal[0,12:]=.04
        return proposal
    final,pair=teacher_corrected_recovery_command(student,public,base,correct)
    torch.testing.assert_close(pair['student_command'][0],torch.full((18,),.02))
    assert pair['teacher_changed'].tolist()==[True,False]
    assert len(calls)==1
    torch.testing.assert_close(student.previous,final)
    public[:,775]=.03
    teacher_corrected_recovery_command(student,public,base,lambda proposal,active:proposal)
    torch.testing.assert_close(student.head.inputs[-1][:,812:830],final[:1])


@pytest.mark.parametrize('mode',['inactive','slew','nan'])
def test_invalid_teacher_never_returns_an_executable_command(mode):
    student,public,base=setup()
    def invalid(proposal,active):
        if mode=='inactive':proposal[1,0]=.1
        elif mode=='slew':proposal[0,12]=.2
        else:proposal[0,0]=float('nan')
        return proposal
    with pytest.raises(ValueError):teacher_corrected_recovery_command(student,public,base,invalid)


def test_episode_reset_removes_previous_hit_and_teacher_eligibility():
    student,public,base=setup()
    teacher_corrected_recovery_command(student,public,base,lambda proposal,active:proposal)
    student.reset();public.zero_()
    final,pair=teacher_corrected_recovery_command(student,public,base,lambda *_:pytest.fail('Leaked eligibility'))
    assert not pair['active'].any()
    torch.testing.assert_close(final,base)
