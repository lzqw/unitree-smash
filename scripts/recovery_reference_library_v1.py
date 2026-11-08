"""Match only locally covered causal release states to immutable model plans."""
import hashlib,json
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.loaded_pinocchio_model_v1 import PublicState


class RecoveryReferenceLibraryV1:
    def __init__(self,path,model):
        self.path=Path(path);self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        record=json.loads(self.path.read_text());self.model=model
        if record['schema']!='model-recovery-reference-library-v1' or record['native_success'] or record['valid_expert_labels']:
            raise ValueError('Explicit model-candidate provenance required')
        self.records=record['references']
        for row in self.records:
            if len(row['trajectory'])!=60 or row['native_success']:raise ValueError('Wrong reference horizon')
            for k,r in enumerate(row['trajectory']):
                if r['step']!=k or np.shape(r['q'])!=(25,) or np.shape(r['v'])!=(24,) or np.shape(r['target'])!=(18,):
                    raise ValueError('Malformed model trajectory')
                if not all(np.isfinite(r[x]).all() for x in ['q','v','target']):raise ValueError('Nonfinite reference')

    def match(self,state):
        matches=[]
        for row in self.records:
            source=row['query'];dq=pin.difference(self.model.model,np.asarray(source['q']),state.q)
            dv=state.v-np.asarray(source['v']);du=state.previous_command_target-np.asarray(source['target'])
            # A local initial-state coverage test, not a physical limit or a
            # stability guarantee. No nearest-plan extrapolation outside it.
            normalized=np.r_[dq[:3]/.01,dq[3:6]/.03,dq[6:]/.03,dv[:6]/.10,dv[6:]/.20,du/.03]
            score=float(np.max(abs(normalized)))
            if score<=1.:matches.append((score,row))
        if not matches:return None,None
        score,row=min(matches,key=lambda pair:pair[0]);return row,score

    def state(self,row,index):
        r=row['query'] if index<0 else row['trajectory'][min(index,59)]
        q=np.asarray(r['q'],float);v=np.asarray(r['v'],float)
        state=PublicState(q=q.copy(),v=v.copy(),root_rotation_world=pin.Quaternion(q[3:7]).matrix(),
            root_position_world=q[:3].copy(),joint_position=q[self.model.q_indices].copy(),
            joint_velocity=v[self.model.v_indices].copy(),contacts=np.asarray(r['contacts'],bool),rotation_projection_error=0.)
        state.previous_command_target=np.asarray(r['target'],float).copy();return state
