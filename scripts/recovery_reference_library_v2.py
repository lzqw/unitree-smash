"""Variable-duration model references with the original causal state matcher."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.recovery_reference_library_v1 import RecoveryReferenceLibraryV1
from scripts.loaded_pinocchio_model_v1 import PublicState


class RecoveryReferenceLibraryV2(RecoveryReferenceLibraryV1):
    def __init__(self, path, model):
        self.path = Path(path)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        record = json.loads(self.path.read_text())
        self.model = model
        if (record['schema'] != 'model-recovery-reference-library-v2' or record['native_success']
                or record['valid_expert_labels'] or record['control_dt_s'] != .01):
            raise ValueError('Explicit original-interval model-candidate provenance required')
        self.records = record['references']
        for row in self.records:
            if not 1 <= len(row['trajectory']) <= 800 or row['native_success']:
                raise ValueError('Invalid model reference horizon')
            for k, r in enumerate([row['query'], *row['trajectory']]):
                if k and r['step'] != k - 1:
                    raise ValueError('Non-contiguous model reference')
                for name, width in [('q', 25), ('v', 24), ('target', 18), ('contacts', 4)]:
                    if np.shape(r[name]) != (width,) or not np.isfinite(r[name]).all():
                        raise ValueError('Malformed model state or command')

    def state(self, row, index):
        r = row['query'] if index < 0 else row['trajectory'][min(index, len(row['trajectory']) - 1)]
        q, v = np.asarray(r['q'], float), np.asarray(r['v'], float)
        state = PublicState(q=q.copy(), v=v.copy(), root_rotation_world=pin.Quaternion(q[3:7]).matrix(),
            root_position_world=q[:3].copy(), joint_position=q[self.model.q_indices].copy(),
            joint_velocity=v[self.model.v_indices].copy(), contacts=np.asarray(r['contacts'], bool),
            rotation_projection_error=0.)
        state.previous_command_target = np.asarray(r['target'], float).copy()
        return state
