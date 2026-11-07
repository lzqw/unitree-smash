"""Execute distinct predicted states and fixed feedback-reference states.

Predicted states diagnose model error. Explicit reference states determine the
feedback command. They need not coincide after optimizing the joint trajectory.
"""
import json
from pathlib import Path
import numpy as np
from scripts.native_reference_tracking_v1 import NativeReferenceTrackingV1
from scripts.landing_reference_tracking_v1 import LandingReferenceTrackingV1


def load_tracking_reference(bundle, manifest, steps):
    definition = manifest['reference_tracking_definition']
    if definition.get('state_reference_contract') != 'explicit_frozen_tracking_reference_v1':
        raise ValueError('Explicit feedback-state reference required; no prediction fallback')
    with np.load(Path(bundle)/manifest['bundle_name'], allow_pickle=False) as data:
        q = data[definition['q_key']].copy()
        v = data[definition['v_key']].copy()
    if (q.shape != (steps,25) or v.shape != (steps,24)
            or not np.isfinite(q).all() or not np.isfinite(v).all()):
        raise ValueError('Invalid explicit feedback reference')
    return q, v


class NativeReferenceTrackingV2(NativeReferenceTrackingV1):
    def __init__(self, bundle, case, limits, policy_sha256, output):
        super().__init__(bundle, case, limits, policy_sha256, output)
        q, v = load_tracking_reference(bundle, self.manifest, self.steps)
        # Replace before any action; retain the exact whole-body base and
        # original outer attitude feedback used by the model evaluator.
        base = self.reference_tracking.base
        self.reference_tracking = LandingReferenceTrackingV1(base, q, v,
            **self.manifest['reference_tracking_definition']['gains'])
        self.feedback.base = self.reference_tracking

    def save(self):
        result = super().save()
        result['tracking_reference_separate_from_prediction'] = True
        result['state_reference_contract'] = 'explicit_frozen_tracking_reference_v1'
        path = self.output/'sequence_native_result.json'; temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n'); temp.replace(path)
        return result
