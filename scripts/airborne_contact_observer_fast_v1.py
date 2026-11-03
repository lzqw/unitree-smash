"""Deferred-I/O contact observer for the native airborne paired diagnostic.

The measured force/contact semantics are inherited unchanged.  Only the
per-event JSON rewrite is deferred until ``save`` so the native comparison is
not dominated by serializing the complete event history after every event.
"""

import torch

from scripts.strike_ground_contact_observer_v1 import StrikeGroundContactObserver
from scripts.standing_contact_telemetry_v1 import StandingContactTelemetry


class DeferredStandingContactTelemetry(StandingContactTelemetry):
    """Keep all native event fields but defer the growing JSON archive."""

    def on_physics(self, env):
        original_atomic = self._atomic
        self._atomic = lambda *_args, **_kwargs: None
        try:
            return super().on_physics(env)
        finally:
            self._atomic = original_atomic


class DeferredStrikeGroundContactObserver(StrikeGroundContactObserver):
    """Same observer contract as v1 with deferred event-file writes."""

    def __init__(self, env, output):
        self.telemetry = DeferredStandingContactTelemetry(env, output)
        self.seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.callbacks = 0
        self.last_frame = None
        self.telemetry.metadata['sampling_scope'] = (
            'First nonfoot contact events only; no continuous force archive')
        self.telemetry._atomic('contact_telemetry_metadata.json', self.telemetry.metadata)
        self.telemetry._atomic('first_nonfoot_contact_details.json', [])
