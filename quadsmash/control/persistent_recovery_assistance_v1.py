"""Keep explicit recovery settings effective after nested controller resets.

Install around the complete source controller, before any landing/support
wrapper. Only declared parameter columns are changed; untouched lanes retain
their current plans during a partial reset. This is an explicit new execution
branch, not a change to historical policy playback.
"""
import math

from .shared_target_assistance_v6 import RECOVERY_FIELDS


SCHEMA = 'reset-persistent-recovery-assistance-v1'


class PersistentRecoveryAssistanceV1:
    def __init__(self, source, settings):
        if not isinstance(settings, dict) or set(settings) - set(RECOVERY_FIELDS):
            raise ValueError('Unknown recovery assistance settings')
        checked = {}
        for name, value in settings.items():
            _, low, high = RECOVERY_FIELDS[name]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not low <= value <= high):
                raise ValueError('Invalid recovery assistance: ' + name)
            checked[name] = float(value)
        self.source = source
        self.settings = checked
        reference = source.reference_controller
        if checked and (reference.parameters.ndim != 2 or reference.parameters.shape[1] < 37):
            raise ValueError('Require recovery37 parameter layout')
        self._apply(None)

    @property
    def reference_controller(self):
        return self.source.reference_controller

    @property
    def logs(self):
        return self.source.logs

    @logs.setter
    def logs(self, value):
        self.source.logs = value

    def _apply(self, ids):
        reference = self.reference_controller
        for name, value in self.settings.items():
            index = RECOVERY_FIELDS[name][0]
            if index is None:
                reference.teacher.leg_recovery_blend.seconds = value
            elif ids is None:
                reference.parameters[:, index] = value
            else:
                reference.parameters[ids, index] = value

    def reset(self, ids=None):
        self.source.reset(ids)
        self._apply(ids)

    def action(self, observation, default, limits):
        return self.source.action(observation, default, limits)
