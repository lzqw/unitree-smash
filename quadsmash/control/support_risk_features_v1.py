"""Causal proprioceptive features for student support-risk diagnostics.

No case identity, clock, ball target, outcome or future frame is an input.
The candidate action is the original student's current final normalized target.
"""
import numpy as np


def support_risk_features(public, student_action):
    public=np.asarray(public);action=np.asarray(student_action)
    if (public.ndim!=2 or public.shape[1]!=794 or action.shape!=(len(public),18)
            or not np.isfinite(public).all() or not np.isfinite(action).all()):
        raise ValueError('Finite public794 and current student command18 required')
    current=public[:,:61]
    past=public[:,690:751]  # tenth previous 69D frame, ending at foot flags
    return np.concatenate((current,(current-past)/.1,action,public[:,776:794]),axis=1).astype(np.float64)
