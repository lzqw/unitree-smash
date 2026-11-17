from types import SimpleNamespace
import pytest
from quadsmash.control.go2_walk_transfer import Go2WalkTransfer
from quadsmash.control.go2_standing_transfer import Go2StandingTransfer
from quadsmash.control.piecewise_affine import PiecewiseAffinePolicy
from scripts.audit_teacher_action_sources_v1 import (
    learned_action_sources, require_no_known_learned_action_sources,
)


def test_nested_frozen_base_sources_are_detected_without_forward_or_weight_loading():
    # Do not load weights or run an actor just to identify its actual class.
    teacher = SimpleNamespace(walk=object.__new__(Go2WalkTransfer),
        standing=object.__new__(Go2StandingTransfer),
        strike=object.__new__(PiecewiseAffinePolicy))
    controller = SimpleNamespace(actor=None, frozen_public_policy=SimpleNamespace(teacher=teacher))
    controller.cycle = controller
    sources = learned_action_sources(controller)
    assert {r['path'] for r in sources} == {
        'controller.frozen_public_policy.teacher.walk.actor',
        'controller.frozen_public_policy.teacher.standing.actor',
        'controller.frozen_public_policy.teacher.strike'}
    with pytest.raises(RuntimeError, match='reachable learned action sources'):
        require_no_known_learned_action_sources(controller)


def test_analytic_object_is_not_rejected_by_actor_like_attribute_name():
    controller = SimpleNamespace(actor_configuration={'enabled': False}, analytic=lambda x:x)
    assert learned_action_sources(controller) == []


def test_frozen_base_counting_preserves_outputs_and_detects_parameter_mutation():
    import torch
    from scripts.audit_teacher_action_sources_v1 import FrozenBaseInferenceAudit
    modules=[torch.nn.Linear(3,2) for _ in range(3)]
    teacher=SimpleNamespace(walk=SimpleNamespace(actor=modules[0]),
        standing=SimpleNamespace(actor=modules[1]),strike=modules[2])
    x=torch.ones(1,3); expected=[m(x).detach().clone() for m in modules]
    audit=FrozenBaseInferenceAudit(teacher)
    for call,want in zip((teacher.walk.actor,teacher.standing.actor,teacher.strike),expected):
        torch.testing.assert_close(call(x),want)
    assert audit.report()['frozen_base_forward_counts']==dict(walk=1,standing=1,strike=1)
    assert all(not p.requires_grad for m in modules for p in m.parameters())
    with torch.no_grad(): modules[0].weight.add_(1.)
    with pytest.raises(RuntimeError,match='changed'):audit.report()
