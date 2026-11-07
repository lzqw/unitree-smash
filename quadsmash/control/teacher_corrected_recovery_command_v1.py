"""Single final-command arbitration for learned recovery followed by MPC.

The student proposes once. The controller sees that proposal and may correct
causally eligible recovery lanes. Both future command histories use the final
executed command. This deterministic collection path is not a PPO buffer.
"""
import torch


@torch.inference_mode()
def teacher_corrected_recovery_command(student, public, base, correct):
    previous=student.previous.clone()
    proposal=student.action(public,base)
    active=student.active.clone()
    final=correct(proposal.clone(),active.clone()) if bool(active.any()) else proposal.clone()
    if (not isinstance(final,torch.Tensor) or final.shape!=proposal.shape or
        final.dtype!=proposal.dtype or final.device!=proposal.device or
        not torch.isfinite(final).all() or torch.any(abs(final)>1.+1e-6)):
        raise ValueError('Finite same-interface final command required')
    if not torch.equal(final[~active],proposal[~active]):
        raise ValueError('Teacher changed a lane before causal recovery eligibility')
    width=student.limits[:,1]-student.limits[:,0]
    arm_step=abs(final[:,12:]-previous[:,12:])*width[12:]/2.
    if bool(active.any()) and torch.any(arm_step[active]>student.arm_rate*student.dt+5e-6):
        raise ValueError('Teacher violated actual final-command arm slew')
    student.previous.copy_(final)
    return final,dict(student_command=proposal.clone(),previous_command=previous,
        teacher_command_delta=final-proposal,
        teacher_changed=(abs(final-proposal)*width/2.).amax(dim=1)>1e-5,
        active=active,learned_delta=student.last_delta.clone())
