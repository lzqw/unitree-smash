"""Recheck every continuous block before exporting a model-only recovery plan."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.landing_joint_sequence_proposal_v1 import verify_sequence
from scripts.landing_sequential_recovery_search_v1 import rollout_for_search
from scripts.audit_microstep_recovery_v1 import characterize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['result','original-bundle','geometry','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous export')
    result=json.loads(a.result.read_text())
    for name,digest in result['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Result source changed '+name)
    m=json.loads((a.original_bundle/'manifest.json').read_text());cfg=m['configuration']
    archive=a.original_bundle/m['bundle_name']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=m['bundle_sha256']:raise ValueError('Changed archived queries')
    metrics_path=Path(m['baseline'])/'metrics.json';metrics=json.loads(metrics_path.read_text())
    outcomes={r['training_index']:r for r in metrics['complete5_case_outcomes']}
    originals={r['case']:r for r in m['rows']};limits=np.asarray(cfg['interface']['limits'])
    arrays={};rows=[]
    with np.load(archive,allow_pickle=False) as z:
        for record in result['records']:
            row=record.get('continuation_check',record)
            if not row['after']['accepted'] or row['verified_continuation_s']<=0:continue
            case=row['case'];key=f'case_{case:03d}';c=build_controller(cfg)
            c.sequence_geometry=AuthoredGroundClearance(a.geometry,c.qp.model,margin_m=0.)
            state=c.qp.model.state_from_public(z[key+'_query_public'],np.asarray(cfg['posture_target']))
            state.previous_command_target=limits[:,0]+.5*(z[key+'_previous_action']+1.)*np.diff(limits).ravel()
            reference=limits[:,0]+.5*(z[key+'_query_student_action']+1.)*np.diff(limits).ravel()
            commands=[row['commands']]+[t['commands'] for t in row['continuation'] if t['after']['accepted'] and t['after']['joint_settling_diagnostic']]
            qs=[state.q.copy()];vs=[state.v.copy()];contacts=[state.contacts.copy()];checks=[]
            for block,sequence in enumerate(commands):
                # Every block starts from the exact previous replay's endpoint,
                # including command history and predicted contact additions.
                sequence=np.asarray(sequence);checked=verify_sequence(c,state,sequence)
                replay=rollout_for_search(c,state,sequence,reference)
                if not checked['accepted'] or not replay['physically_feasible']:
                    raise ValueError(f'Continuous recheck failed case {case} block {block}')
                for n in replay['profile']:
                    qs.append(n['following'].q.copy());vs.append(n['following'].v.copy());contacts.append(n['following'].contacts.copy())
                state=copy.deepcopy(replay['profile'][-1]['following']);reference=state.previous_command_target.copy()
                if max(abs(state.joint_velocity))>.5 or abs((state.root_rotation_world@state.v[:3])[2])>.1:
                    raise ValueError('Exported endpoint no longer satisfies settling diagnostic')
                checks.append(dict(block=block,terminal_risk=checked['terminal_risk'],
                    maximum_physical_violation=replay['maximum_physical_violation'],
                    maximum_ground_violation_m=max(r['physical']['authored_ground_clearance_m'] for r in replay['checks'])))
            held,_=characterize(c,state,np.tile(reference,(40,1)),reference)
            for suffix in ['query_public','previous_action','query_student_action']:arrays[key+'_'+suffix]=z[key+'_'+suffix].copy()
            arrays.update({key+'_commands_rad':np.concatenate(commands),key+'_predicted_q':np.stack(qs),
                key+'_predicted_v':np.stack(vs),key+'_predicted_contacts':np.stack(contacts)})
            rows.append(dict(case=case,query_tick=originals[case]['query_tick'],horizon_steps=40*len(commands),
                original_outcome=outcomes[case],ball5_except_recovery=outcomes[case]['ball5_except_recovery'],
                original_complete5=outcomes[case]['complete5'],checks=checks,held_next_0p4s=held))
    if not rows:raise ValueError('No independently verified continued plans')
    a.output.mkdir(parents=True);bundle=a.output/'sequences.npz';np.savez_compressed(bundle,**arrays)
    files=[Path(__file__),a.result,a.original_bundle/'manifest.json',archive,a.geometry,metrics_path]
    files += [Path(m.__file__) for n,m in sys.modules.items() if n.startswith('scripts.') and getattr(m,'__file__',None)]
    manifest=dict(scope=__doc__,native_execution=False,valid_expert_labels=False,
        baseline=m['baseline'],policy_sha256=m['policy_sha256'],configuration=cfg,control_dt_s=.01,
        rows=rows,bundle_name=bundle.name,bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        ground_geometry_sha256=hashlib.sha256(a.geometry.read_bytes()).hexdigest(),
        requires=['Original ground-start prefix and actual query match.','Actual execution/model-error audit.',
            'Continued state feedback through the original episode.','Self-collision and native cooked-contact validation.'],
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(case=r['case'],horizon_steps=r['horizon_steps'],ball5_except_recovery=r['ball5_except_recovery'],
        held_next_0p4s_accepted=r['held_next_0p4s']['accepted']) for r in rows]))

if __name__=='__main__':main()
