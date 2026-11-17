"""Offline FK attribution and centroidal momentum errors on recorded states.

Hybrid configurations are mathematical FK probes only, never candidate states
or controller inputs. Native labels use the same authored model at recorded
native q/v; they are not independent PhysX force/momentum measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.audit_prediction_contact_geometry_v2 import racket_contact_state


def attribution(model,predicted_q,native_q,offset):
    qp=np.asarray(predicted_q,float);qn=np.asarray(native_q,float)
    qb=qn.copy();qb[:7]=qp[:7]
    qj=qn.copy();qj[7:]=qp[7:]
    zero=np.zeros(model.model.nv)
    pn=racket_contact_state(model,qn,zero,offset)[0]
    pp=racket_contact_state(model,qp,zero,offset)[0]
    pb=racket_contact_state(model,qb,zero,offset)[0]
    pj=racket_contact_state(model,qj,zero,offset)[0]
    # Symmetric two-factor attribution; the vector sum is exactly pp-pn.
    base=.5*((pb-pn)+(pp-pj));joints=.5*((pj-pn)+(pp-pb))
    return dict(base_effect_world_m=base.tolist(),joint_effect_world_m=joints.tolist(),
        total_error_world_m=(pp-pn).tolist(),base_effect_norm_m=float(np.linalg.norm(base)),
        joint_effect_norm_m=float(np.linalg.norm(joints)),total_error_norm_m=float(np.linalg.norm(pp-pn)),
        reconstruction_error_m=float(np.linalg.norm(base+joints-(pp-pn))))


def momentum(model,q,v):
    pin.ccrba(model.model,model.data,np.asarray(q),np.asarray(v))
    return model.data.hg.vector.copy()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--contact-offset',nargs=3,type=float,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior attribution')
    record=json.loads(a.input.read_text());model=LoadedPinocchioModelV1();rows=[]
    for r in record['rows']:
        if not r['solved']:continue
        hp=momentum(model,r['predicted_q'],r['predicted_v'])
        hn=momentum(model,r['native_q'],r['native_v'])
        rows.append(dict(lane=r['lane'],tick_after=r['tick']+1,
            **attribution(model,r['predicted_q'],r['native_q'],a.contact_offset),
            predicted_momentum_world=hp.tolist(),native_state_momentum_world=hn.tolist(),
            linear_momentum_error_kg_m_s=float(np.linalg.norm(hp[:3]-hn[:3])),
            angular_momentum_error_kg_m2_s=float(np.linalg.norm(hp[3:]-hn[3:])),
            predicted_feet=r['predicted_contacts'],native_feet=r['native_contacts']))
    a.output.write_text(json.dumps(dict(rows=rows,contact_offset=a.contact_offset,
        source_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
        angular_momentum_reference='World axes, about whole-body COM',
        hybrid_states_used_only_for_fk_attribution=True,controller_input=False,native_force_measured=False),indent=2,allow_nan=False))


if __name__=='__main__':main()
