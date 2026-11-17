"""Compare native/predicted authored ground witnesses; never certify a reference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1


def racket_contact_state(model,q,v,offset):
    pin.computeJointJacobians(model.model,model.data,q)
    pin.updateFramePlacements(model.model,model.data)
    fid=model.body_frames['Racket'];pose=model.data.oMf[fid]
    lever=pose.rotation@np.asarray(offset)
    J=pin.getFrameJacobian(model.model,model.data,fid,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    # Pinocchio data is mutable workspace: no returned direction may alias it.
    return pose.translation+lever,pose.rotation[:,0].copy(),(J[:3]-pin.skew(lever)@J[3:])@v


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction',type=Path,required=True)
    p.add_argument('--geometry',type=Path,default=Path('configs/go2_piper_authored_ground_geometry_v1.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--contact-offset',type=float,nargs=3,help='Explicit frozen local point; no implicit geometry change')
    args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve previous evidence')
    record=json.loads(args.prediction.read_text())
    model=LoadedPinocchioModelV1();geometry=AuthoredGroundClearance(args.geometry,model,margin_m=0.)
    rows=[]
    for r in record['rows']:
        if not r['solved']:continue
        row=dict(lane=r['lane'],tick_after=r['tick']+1,method=r['method'],
            predicted=geometry.minimum_witness(np.asarray(r['predicted_q'])),
            native=geometry.minimum_witness(np.asarray(r['native_q'])))
        if args.contact_offset is not None:
            prediction=racket_contact_state(model,np.asarray(r['predicted_q']),np.asarray(r['predicted_v']),args.contact_offset)
            truth=racket_contact_state(model,np.asarray(r['native_q']),np.asarray(r['native_v']),args.contact_offset)
            row['contact_point_error_m']=float(np.linalg.norm(prediction[0]-truth[0]))
            row['normal_vector_error']=float(np.linalg.norm(prediction[1]-truth[1]))
            row['contact_velocity_error_m_s']=float(np.linalg.norm(prediction[2]-truth[2]))
        rows.append(row)
    args.output.write_text(json.dumps(dict(rows=rows,source_sha256=hashlib.sha256(args.prediction.read_bytes()).hexdigest(),
        geometry_sha256=geometry.sha256,meaning='Authored geometric witnesses, not contact-force measurements',
        contact_offset=args.contact_offset,
        point_truth_semantics='FK and Jacobian evaluated at recorded native q/v, not an independent PhysX contact measurement',
        teacher_reference_valid=False),indent=2,allow_nan=False))


if __name__=='__main__':main()
