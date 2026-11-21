"""Explicit training stages for the physical SuperMimic loop.

The stage is part of the checkpoint contract.  It controls which gradients
may reach the actor while keeping the shared world model trainable.
"""

from dataclasses import dataclass


STAGES = (
    "wm_calibrate",
    "shadow_policy",
    "precontact_only",
    "contact_detached",
    "research_contact",
    "full_contact",
)

CONTACT_BPTT_MODES = ("blocked", "detached", "precontact", "enabled")


@dataclass(frozen=True)
class TrainingStage:
    name: str
    contact_bptt_mode: str
    actor_updates_allowed: bool
    promotable: bool

    def manifest(self):
        return {
            "stage": self.name,
            "contact_bptt_mode": self.contact_bptt_mode,
            "actor_updates_allowed": self.actor_updates_allowed,
            "promotable": self.promotable,
        }


def resolve_training_stage(name):
    if name not in STAGES:
        raise ValueError(f"Unknown SuperMimic training stage: {name}")
    if name == "wm_calibrate":
        return TrainingStage(name, "blocked", False, False)
    if name == "shadow_policy":
        return TrainingStage(name, "detached", True, False)
    if name == "precontact_only":
        return TrainingStage(name, "precontact", True, False)
    if name == "contact_detached":
        return TrainingStage(name, "detached", True, False)
    return TrainingStage(name, "enabled", True, name == "full_contact")


def validate_contact_bptt_mode(mode):
    if mode not in CONTACT_BPTT_MODES:
        raise ValueError(f"Unknown contact BPTT mode: {mode}")
    return mode
