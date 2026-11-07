"""Independently recheck geometry-constrained plans; no native success labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.landing_joint_sequence_proposal_v1 import verify_sequence
from scripts.landing_sequential_recovery_search_v1 import rollout_for_search


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True);p.add_argument('--original-bundle',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous export')
    result=json.loads(a.result.read_text())
    for name,digest in result['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Result source changed '+name)
    original=json.loads((a.original_bundle/'manifest.json').read_text());cfg=original['configuration']
    archive=a.original_bundle/original['bundle_name']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=original['bundle_sha256']:raise ValueError('Original query changed')
    limits=np.asarray(cfg['interface']['limits']);arrays={};rows=[]
    originals={r['case']:r for r in original['rows']}
    with np.load(archive,allow_pickle=False) as z:
        for row in result['records']:
            if not row['accepted']:continue
            case=row['case'];key=f'case_{case:03d}';c=build_controller(cfg)
            c.sequence_geometry=AuthoredGroundClearance(a.geometry,c.qp.model,margin_m=0.)
            state=c.qp.model.state_from_public(z[key+'_query_public'],np.asarray(cfg['posture_target']))
            state.previous_command_target=limits[:,0]+.5*(z[key+'_previous_action']+1.)*np.diff(limits).ravel()
            reference=limits[:,0]+.5*(z[key+'_query_student_action']+1.)*np.diff(limits).ravel()
            commands=np.asarray(row['commands']);checked=verify_sequence(c,state,commands)
            replay=rollout_for_search(c,state,commands,reference)
            if not checked['accepted'] or not replay['physically_feasible'] or not replay['terminal_ok']:
                raise ValueError('Candidate failed independent export check')
            profile=replay['profile']
            for suffix in ['query_public','previous_action','query_student_action']:
                arrays[key+'_'+suffix]=z[key+'_'+suffix].copy()
            arrays.update({key+'_commands_rad':commands,
                key+'_predicted_q':np.stack([state.q]+[n['following'].q for n in profile]),
                key+'_predicted_v':np.stack([state.v]+[n['following'].v for n in profile]),
                key+'_predicted_contacts':np.stack([state.contacts]+[n['following'].contacts for n in profile])})
            rows.append(dict(originals[case],terminal_risk=checked['terminal_risk'],
                maximum_physical_violation=replay['maximum_physical_violation'],
                maximum_ground_violation_m=max(r['physical']['authored_ground_clearance_m'] for r in replay['checks'])))
    if not rows:raise ValueError('No candidates passed independent check')
    a.output.mkdir(parents=True);bundle=a.output/'sequences.npz';np.savez_compressed(bundle,**arrays)
    files=[Path(__file__),a.result,a.original_bundle/'manifest.json',archive,a.geometry,
        ROOT/'configs/go2_piper_racket_landing_model_v1.json']
    files += [Path(m.__file__) for n,m in sys.modules.items() if n.startswith('scripts.') and getattr(m,'__file__',None)]
    manifest=dict(scope=__doc__,native_execution=False,valid_expert_labels=False,
        baseline=original['baseline'],policy_sha256=original['policy_sha256'],configuration=cfg,
        control_dt_s=.01,rows=rows,bundle_name=bundle.name,
        bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        ground_geometry_sha256=hashlib.sha256(a.geometry.read_bytes()).hexdigest(),
        requires=['Original from-ground prefix and query/previous-command match.',
            'Actual dispatched sequence, contact and state response verification.',
            'Continued feedback and original full-episode recovery assessment.',
            'No self-collision or cooked-contact certificate from this export.'],
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(cases=[r['case'] for r in rows],native_execution=False,valid_expert_labels=False)))

if __name__=='__main__':main()
