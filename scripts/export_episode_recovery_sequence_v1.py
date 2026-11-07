"""Independently replay a nominal continuation through the real 8 s boundary."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.landing_joint_sequence_proposal_v1 import verify_sequence
from scripts.landing_sequential_recovery_search_v1 import rollout_for_search
from scripts.extend_recovery_to_episode_v1 import kinematic_window


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['result','prefix-bundle','geometry','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous export')
    result=json.loads(a.result.read_text());m=json.loads((a.prefix_bundle/'manifest.json').read_text())
    if not result['model_plan_reaches_original_end']:raise ValueError('Incomplete nominal recovery')
    for name,digest in result['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed extension source '+name)
    source=Path(result['arrays'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=result['arrays_sha256']:raise ValueError('Changed extended arrays')
    original=a.prefix_bundle/m['bundle_name']
    if hashlib.sha256(original.read_bytes()).hexdigest()!=m['bundle_sha256']:raise ValueError('Changed original prefix')
    case=result['case'];key=f'case_{case:03d}';cfg=m['configuration'];limits=np.asarray(cfg['interface']['limits'])
    with np.load(source,allow_pickle=False) as z:
        commands=z['commands_rad'].copy();expected_q=z['predicted_q'].copy();expected_v=z['predicted_v'].copy();expected_contacts=z['predicted_contacts'].copy()
    if len(commands)!=800-result['query_tick']:raise ValueError('Must cover exactly the original episode remainder')
    with np.load(original,allow_pickle=False) as z:arrays={key+'_'+suffix:z[key+'_'+suffix].copy() for suffix in ['query_public','previous_action','query_student_action']}
    c=build_controller(cfg);state=c.qp.model.state_from_public(arrays[key+'_query_public'],np.asarray(cfg['posture_target']))
    state.previous_command_target=limits[:,0]+.5*(arrays[key+'_previous_action']+1.)*np.diff(limits).ravel()
    reference=limits[:,0]+.5*(arrays[key+'_query_student_action']+1.)*np.diff(limits).ravel()
    states=[copy.deepcopy(state)];checks=[];lengths=[]
    for start in range(0,len(commands),40):
        chunk=commands[start:start+40];local=copy.deepcopy(cfg);local['settings']['horizon_steps']=len(chunk)
        c=build_controller(local);c.sequence_geometry=AuthoredGroundClearance(a.geometry,c.qp.model,margin_m=0.)
        check=verify_sequence(c,state,chunk);replay=rollout_for_search(c,state,chunk,reference)
        if not check['accepted'] or not replay['physically_feasible']:raise ValueError('Continuous original-end replay failed')
        states.extend(copy.deepcopy(n['following']) for n in replay['profile']);state=copy.deepcopy(states[-1]);reference=state.previous_command_target.copy()
        checks.append(dict(start_tick=result['query_tick']+start,steps=len(chunk),terminal_risk=check['terminal_risk'],
            maximum_physical_violation=replay['maximum_physical_violation'],
            maximum_ground_violation_m=max(r['physical']['authored_ground_clearance_m'] for r in replay['checks'])))
        lengths.append(len(chunk))
    for i,s in enumerate(states):
        if np.max(abs(pin.difference(c.qp.model.model,expected_q[i],s.q)))>1e-8 or np.max(abs(expected_v[i]-s.v))>1e-8 or not np.array_equal(s.contacts,expected_contacts[i]):
            raise ValueError('Independent end-to-end prediction changed')
    protocol_path=ROOT/'configs/evaluation_protocol_v2.json';protocol=json.loads(protocol_path.read_text())['recovery']
    window=kinematic_window(c.qp.model,states[1:],protocol,.01)
    if window['longest_model_kinematic_window_s']<protocol['continuous_stable_window_s']-1e-6:
        raise ValueError('No original-threshold kinematic stable window in model')
    arrays.update({key+'_commands_rad':commands,key+'_predicted_q':np.stack([s.q for s in states]),
        key+'_predicted_v':np.stack([s.v for s in states]),key+'_predicted_contacts':np.stack([s.contacts for s in states])})
    a.output.mkdir(parents=True);archive=a.output/'sequences.npz';np.savez_compressed(archive,**arrays)
    original_row=next(r for r in m['rows'] if r['case']==case)
    files=[Path(__file__),a.result,source,a.prefix_bundle/'manifest.json',original,a.geometry,protocol_path]
    files += [Path(m.__file__) for n,m in sys.modules.items() if n.startswith('scripts.') and getattr(m,'__file__',None)]
    manifest=dict(scope=__doc__,native_execution=False,valid_expert_labels=False,
        baseline=m['baseline'],policy_sha256=m['policy_sha256'],configuration=cfg,control_dt_s=.01,
        rows=[dict(case=case,query_tick=result['query_tick'],horizon_steps=len(commands),block_lengths=lengths,
            original_outcome=original_row['original_outcome'],ball5_except_recovery=original_row['ball5_except_recovery'],
            original_complete5=False,checks=checks,model_kinematic_window=window)],
        bundle_name=archive.name,bundle_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        ground_geometry_sha256=hashlib.sha256(a.geometry.read_bytes()).hexdigest(),
        full_episode_nominal_continuation=True,measured_feedback=False,
        requires=['Ground-start native query match.','Actual contact, geometry and response validation.',
            'Full task outcome under original scoring.','Feedback correction for deviations and perturbations.'],
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(case=case,steps=len(commands),model_window_s=window['longest_model_kinematic_window_s'],native_execution=False)))

if __name__=='__main__':main()
