"""Compare predicted/native authored hip/mount boundary overlap in radians.

This is a geometric diagnostic, not native collision-force evidence and not
a changed motor limit or physical admission tolerance.
"""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.hip_mount_collision_response_v1 import HipMountCollisionResponseV1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior evidence')
    record=json.loads(a.input.read_text());model=LoadedPinocchioModelV1()
    response=HipMountCollisionResponseV1(model);rows=[]
    for r in record['rows']:
        if not r['solved']:continue
        for joint,bound,lower in response.constraints:
            index=model.q_indices[joint];sign=-1 if lower else 1
            rows.append(dict(lane=r['lane'],tick_after=r['tick']+1,joint=model.joint_order[joint],
                boundary_rad=bound,predicted_overlap_rad=sign*(r['predicted_q'][index]-bound),
                native_overlap_rad=sign*(r['native_q'][index]-bound)))
    a.output.write_text(json.dumps(dict(rows=rows,source_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
        positive_means_authored_overlap=True,native_force_measured=False,teacher_admission=False),indent=2))


if __name__=='__main__':main()
