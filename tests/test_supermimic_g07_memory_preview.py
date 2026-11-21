"""Ownership/gradient tests, not native policy performance evidence."""
from types import SimpleNamespace, MethodType
import pytest
import torch

from quadsmash.control.full_racket_arm_target_rate_v4 import ArmTargetRateV4
from quadsmash.rl.supermimic_g07_memory_preview_v1 import clone_execution_graph


def test_arm_history_parity_over_multiple_clipped_and_unclipped_steps():
    native=ArmTargetRateV4(3,'cpu',5.)
    branch=clone_execution_graph(native,learning=True)
    targets=[];outputs=[]
    for i in range(5):
        desired=torch.full((3,6),.01 if i==0 else .4,requires_grad=True)
        time=torch.full((3,),i*.01)
        observed=torch.zeros(3,6)
        expected=native(desired.detach(),observed,time)
        branch=clone_execution_graph(branch,learning=True)
        actual=branch(desired,observed,time)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        targets.append(desired);outputs.append(actual)
    gradients=torch.autograd.grad(sum(x.sum() for x in outputs),targets)
    assert all(torch.isfinite(x).all() for x in gradients)
    assert gradients[0].abs().sum()>0


def test_partial_reset_is_branch_local_and_preserves_other_lanes():
    rate=ArmTargetRateV4(3,'cpu',5.)
    rate(torch.ones(3,6),torch.zeros(3,6),torch.zeros(3))
    a=clone_execution_graph(rate,learning=True)
    b=clone_execution_graph(rate,learning=True)
    a.reset(torch.tensor([1]))
    assert rate.initialized.all() and b.initialized.all()
    assert a.initialized.tolist()==[True,False,True]
    t=torch.full((3,),.01)
    actual=a(torch.ones(3,6),torch.full((3,6),.5),t)
    other=b(torch.ones(3,6),torch.full((3,6),.5),t)
    torch.testing.assert_close(actual[[0,2]],other[[0,2]],rtol=0,atol=0)
    assert not torch.equal(actual[1],other[1])


def test_bound_callbacks_and_mutable_aliases_rebind_to_branch():
    obj=SimpleNamespace(state=torch.ones(2),plan=[])
    def call(self):self.plan.append('called');return self.state
    obj.callback=MethodType(call,obj)
    root=SimpleNamespace(first=obj,alias=obj)
    branch=clone_execution_graph(root,learning=True)
    assert branch.first is branch.alias
    assert branch.first.callback.__self__ is branch.first
    branch.first.callback()
    assert obj.plan==[] and branch.first.plan==['called']


def test_float_plan_rows_keep_aliases_and_branch_local_mutations():
    row = [float(i) for i in range(794)]
    log = {'used_public_observations': [row, row], 'mixed': [0., {'row': row}]}
    branch = clone_execution_graph(log, learning=True)
    cloned = branch['used_public_observations'][0]
    assert cloned == row and cloned is not row
    assert cloned is branch['used_public_observations'][1] is branch['mixed'][1]['row']
    cloned[0] = 1.
    assert row[0] == 0.


def test_fixed_budget_command_uses_each_records_post_base_mixer_memory():
    from quadsmash.control.full_racket_residual_v4 import ResidualActionMixerV4
    from quadsmash.rl.supermimic_g07_memory_preview_v1 import G07MemoryPreviewV1

    count = 3
    default = torch.zeros(18)
    limits = torch.tensor([[-2., 2.]]*18)
    obs = torch.zeros(count, 794)
    obs[:,775] = .3
    mixer = ResidualActionMixerV4(count,'cpu',[.2]*18)
    base = SimpleNamespace(offset=torch.tensor(.1), mixer=mixer)

    def action(self, observation, default, limits):
        self.mixer.arm_rate.initialized.fill_(True)
        self.mixer.arm_rate.clock.fill_(.29)
        self.mixer.arm_rate.target.fill_(self.offset)
        return self.offset.expand(len(observation['policy']),18)

    base.action = MethodType(action, base)
    memory = SimpleNamespace(base=base, mixer=mixer)
    executor = G07MemoryPreviewV1(memory)
    for offset in (.1, -.2):
        base.offset.fill_(offset)
        command, before = executor.prepare_fixed_command(obs, default, limits, memory)
        for request in (torch.zeros(count,18), torch.full((count,18),.8), torch.full((count,18),-.3)):
            preview, _, _ = executor.forward(obs,request,default,limits,memory)
            actual = executor.mixer(command,request,obs,default,limits,before)
            torch.testing.assert_close(actual.final_normalized,preview.final_normalized,atol=0,rtol=0)
            torch.testing.assert_close(actual.final_joint_targets,preview.final_joint_targets,atol=0,rtol=0)
        assert not memory.mixer.arm_rate.initialized.any()


def test_candidate_scores_advance_independent_controller_histories():
    from quadsmash.control.full_racket_residual_v4 import ResidualActionMixerV4
    from quadsmash.rl.supermimic_g07_memory_preview_v1 import G07MemoryPreviewV1

    mixer = ResidualActionMixerV4(2, 'cpu', [.2] * 18)
    base = SimpleNamespace(history=[])

    def action(self, observation, default, limits):
        self.history.append(len(self.history) + 1)
        return default.expand(2, -1) + .03 * self.history[-1]

    base.action = MethodType(action, base)
    original = SimpleNamespace(base=base, mixer=mixer)
    executor = G07MemoryPreviewV1(original)
    default = torch.zeros(18)
    limits = torch.tensor([[-2., 2.]] * 18)
    obs = torch.zeros(2, 794)
    for value in (.1, -.2, .1):
        step = executor.private_scoring_step(original)
        reference = original
        request = torch.full((2, 18), value)
        with pytest.raises(RuntimeError, match='disabled gradients'):
            step(obs, request, default, limits)
        with torch.no_grad():
            for tick in range(4):
                obs[:, 775] = .3 + tick * .01
                expected, reference, _ = executor.forward(obs, request, default, limits, reference)
                result, branch, _ = step(obs, request, default, limits)
                torch.testing.assert_close(result.final_joint_targets, expected.final_joint_targets, rtol=0, atol=0)
                assert branch.base.history == reference.base.history
        assert original.base.history == []
        assert not original.mixer.arm_rate.initialized.any()
