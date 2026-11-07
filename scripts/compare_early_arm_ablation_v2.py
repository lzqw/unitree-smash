"""Same-state same-leg-trajectory arm ablation; model execution only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.search_early_landing_trajectory_v1 import evaluate
from scripts.search_whole_body_early_landing_v2 import WholeBodyEarlyReference
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--search',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh paired output required')
    read=lambda f:json.loads(Path(f).read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    record=read(a.search)
    for f,h in record['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Changed trajectory-search source '+f)
    cfg=record['configuration'];query=read(cfg['query']);manifest=read(Path(cfg['bundle'])/'manifest.json')
    controller=build_controller(manifest['configuration']);model=controller.qp.model
    q=np.asarray(query['initial_q']);v=np.asarray(query['initial_v'])
    initial=PublicState(q,v,pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),q[model.q_indices].copy(),
        v[model.v_indices].copy(),np.asarray(query['initial_contacts'],bool),0.)
    initial.previous_command_target=np.asarray(query['actual_previous_command'])
    candidate=min(record['fine_candidates'],key=lambda x:x['fine']['rank'])
    knots=np.asarray(candidate['knots_rad']);zero=knots.copy();zero[:,12:]=0.
    geometry=AuthoredGroundClearance(ROOT/'configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.)
    spheres=read(cfg['sphere_audit'])['spheres'];protocol=read(ROOT/'configs/evaluation_protocol_v2.json')['recovery']
    rows=[];maximum=None
    for name,parameters in [('selected_full_body',knots),('same_legs_zero_arm_offsets',zero)]:
        result=evaluate(controller,initial,parameters,spheres,geometry,protocol,substeps=160,
            deadline=time.monotonic()+300,save=True,policy_factory=WholeBodyEarlyReference,control_tick=record['tick'])
        if name=='selected_full_body':
            saved=candidate['fine']
            if result['failure']!=saved['failure'] or result['completed_model_steps']!=saved['completed_model_steps']:
                raise ValueError('Full-body repeat changed the stored result')
            errors=[float(np.max(abs(np.asarray(x[k])-np.asarray(y[k]))))
                for x,y in zip(result['trajectory'],saved['trajectory']) for k in ['target','q','v']]
            maximum=max(errors,default=0.)
            if maximum>1e-12:raise ValueError('Full-body replay differs')
        compact={k:v for k,v in result.items() if k!='trajectory'}
        rows.append(dict(mode=name,knots_rad=parameters,result=compact));print(json.dumps(plain(rows[-1])),flush=True)
    for f,h in record['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Changed source during ablation '+f)
    output=dict(scope=__doc__,tick=record['tick'],case=record['case'],rows=rows,
        replay_maximum_difference=maximum,source_sha256={str(a.search.resolve()):sha(a.search),str(Path(__file__).resolve()):sha(__file__)},
        search_source_sha256=record['source_sha256'],native_execution=False,valid_expert_labels=False,
        limitation='Conditional comparison at one selected trajectory; not a controlled comparison of search algorithms or native success rates.')
    a.output.write_text(json.dumps(plain(output),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
