import pytest
import torch

from quadsmash.control.full_racket_residual_v4 import ResidualActionMixerV4
from quadsmash.rl.supermimic_execution_preview_v1 import MixerMemory, ResidualExecutionPreviewV1


def inputs(n=4):
    limits = torch.tensor([[-2., 2.]]*18)
    default = torch.zeros(18)
    public = torch.zeros(n, 794)
    return default, limits, public


@pytest.mark.parametrize('initial_zero', [False, True])
def test_multistep_clipping_clock_restart_and_partial_reset_match_native(initial_zero):
    torch.manual_seed(14)
    source = ResidualActionMixerV4(4, 'cpu', [.2]*18)
    model = ResidualExecutionPreviewV1([.2]*18)
    memory = MixerMemory.capture(source)
    default, limits, public = inputs()
    for tick in range(12):
        public[:, 775] = tick*.01
        if tick == 4:
            # Repeated clock: no additional slew step.
            public[:, 775] = .03
        if tick == 7:
            source.reset(torch.tensor([1, 3]))
            memory = memory.reset(torch.tensor([1, 3]))
        if tick == 9:
            public[2, 775] = 0.  # native implicit time rollback reset
        base = torch.rand(4, 18)*1.8-.9
        residual = torch.randn(4, 18)*1.5
        if initial_zero and tick < 2:
            residual.zero_()
        elif tick in (3, 8):
            residual.zero_()  # zero AFTER nonzero must still honor history
        public[:, 21:27] = torch.randn(4, 6)*.1
        saved = [x.clone() for x in memory.__dict__.values()]
        preview = model(base, residual, public, default, limits, memory)
        actual = source.action(base, residual, public, default, limits)
        torch.testing.assert_close(preview.final_normalized, actual, atol=2e-6, rtol=0)
        torch.testing.assert_close(preview.requested_normalized, source.last_requested, atol=2e-6, rtol=0)
        target = limits[:, 0]+.5*(actual+1)*(limits[:, 1]-limits[:, 0])
        torch.testing.assert_close(preview.final_joint_targets, target, atol=2e-6, rtol=0)
        for old, value in zip(saved, memory.__dict__.values()):
            torch.testing.assert_close(old, value, atol=0, rtol=0)
        for expected, value in zip(MixerMemory.capture(source).__dict__.values(), preview.next_memory.__dict__.values()):
            torch.testing.assert_close(expected, value, atol=2e-6, rtol=0)
        memory = preview.next_memory


def test_candidate_order_does_not_pollute_history_or_actual_mixer():
    source = ResidualActionMixerV4(4, 'cpu', [.2]*18)
    model = ResidualExecutionPreviewV1([.2]*18)
    memory = MixerMemory.capture(source)
    default, limits, public = inputs()
    base = torch.zeros(4, 18)
    wanted = torch.full_like(base, .01)
    first = model(base, wanted, public, default, limits, memory)
    model(base, torch.ones_like(base), public, default, limits, memory)
    second = model(base, wanted, public, default, limits, memory)
    torch.testing.assert_close(first.final_joint_targets, second.final_joint_targets, atol=0, rtol=0)
    assert not source.arm_rate.initialized.any()
    assert not source.nonzero_seen.any()


def test_gradient_reaches_residual_and_previous_final_command():
    source = ResidualActionMixerV4(4, 'cpu', [.2]*18)
    model = ResidualExecutionPreviewV1([.2]*18)
    default, limits, public = inputs()
    residual = torch.full((4, 18), .03, requires_grad=True)
    first = model(torch.zeros_like(residual), residual, public, default, limits, MixerMemory.capture(source))
    public = public.clone(); public[:, 775] = .01
    second = model(torch.ones_like(residual)*.8, torch.ones_like(residual), public, default, limits, first.next_memory)
    grad = torch.autograd.grad(second.final_joint_targets[:, 12:].sum(), residual)[0]
    torch.testing.assert_close(grad[:, 12:], torch.full((4, 6), .2), atol=2e-6, rtol=0)
    assert torch.isfinite(grad).all()


def test_reject_changed_control_dt():
    with pytest.raises(ValueError, match='original mixer control period'):
        ResidualExecutionPreviewV1([.2]*18, control_dt_s=.02)
