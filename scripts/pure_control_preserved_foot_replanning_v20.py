"""Retain the verified v15 support controller while adding causal plan reuse.

The MRO deliberately puts V15 between V17's unplanned super().command and
V14. Uncovered states therefore retain V15's attained soft footholds. V17's
replanned child remains the same V15 controller used by prior feedback tests.
No case/time gate, physical limit, blend, or success criterion is introduced.
"""
from scripts.pure_control_preserved_stance_replanning_v17 import PureControlPreservedStanceReplanningV17
from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15


class PureControlPreservedFootReplanningV20(
        PureControlPreservedStanceReplanningV17, PureControlStanceFootReferenceV15):
    def command(self, state, phase, time_s):
        target, diagnostic = super().command(state, phase, time_s)
        diagnostic.update(controller='pure_control_preserved_foot_replanning_v20',
            unplanned_prefix_controller='pure_control_stance_foot_reference_v15')
        return target, diagnostic
