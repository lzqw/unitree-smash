"""Causal state-triggered reuse of an offline optimized recovery sequence.

The verified v14 stance controller owns the unchanged recovery prefix. After its reference ends, a locally
covered actual state can select a new full18 plan and continuing support
feedback. No TRAIN ID, fixed episode second, outcome label or future native
state is used by this controller. This is not online sequence optimization.
"""
from scripts.pure_control_settled_stance_v14 import PureControlSettledStanceV14
from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15
from scripts.recovery_reference_library_v2 import RecoveryReferenceLibraryV2


class PureControlPreservedStanceReplanningV17(PureControlSettledStanceV14):
    def __init__(self,interface,nominal,library,replanning_library):
        super().__init__(interface,nominal,library)
        self.replanning_library=RecoveryReferenceLibraryV2(replanning_library,self.model)
        self.replanned=None;self.replanning_score=None;self.replanning_time=None

    def command(self,state,phase,time_s):
        # The horizon boundary belongs to the selected plan, not to a TRAIN
        # clock. Uncovered states continue through the original controller.
        reference_finished=(self.started is not None and self.chosen is not None and
            int(round((float(time_s)-self.started)/.01))>=len(self.chosen['trajectory']))
        if self.replanned is None and reference_finished:
            candidate,score=self.replanning_library.match(state)
            if candidate is not None:
                child=PureControlStanceFootReferenceV15(self.interface,self.nominal.copy(),self.library.path)
                child.chosen=candidate;child.selection_done=True;child.selection_score=score
                self.replanned=child;self.replanning_score=score;self.replanning_time=float(time_s)
        if self.replanned is None:
            target,diagnostic=super().command(state,phase,time_s)
        else:
            target,diagnostic=self.replanned.command(state,phase,time_s)
        diagnostic.update(controller='pure_control_preserved_stance_replanning_v17',
            replanned=self.replanned is not None,replanning_state_match_score=self.replanning_score,
            replanning_time_s=self.replanning_time,
            replanning_source_key=(self.replanned.chosen['source_key'] if self.replanned is not None else None),
            state_matched_model_plan_not_online_sequence_optimization=True,
            native_replanning_success=False)
        return target,diagnostic
