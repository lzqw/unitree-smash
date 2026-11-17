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
