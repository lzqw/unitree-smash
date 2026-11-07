#!/usr/bin/env python3
"""Export independently rechecked model plans for subsequent native replay.

This exports no expert-success label. Recorded query matching is an explicit
diagnostic, not a deployable case-independent trigger or a stability result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.landing_joint_sequence_proposal_v1 import verify_sequence
from scripts.landing_sequential_recovery_search_v1 import rollout_for_search


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--restoration-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve existing replay artifact')
    data=json.loads(a.restoration_report.read_text())
    for name,digest in data['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Restoration source changed: '+name)
    seed_path=Path(data['seed_report']);seed=json.loads(seed_path.read_text())
    baseline=Path(seed['baseline']);cfg=seed['configuration'];limits=np.asarray(cfg['interface']['limits'])
    rows=[];arrays={};files=[Path(__file__),a.restoration_report,seed_path]
    for row in data['records']:
        if not row['accepted']:continue
        case,tick=row['case'],row['query_tick'];key=f'case_{case:03d}'
        trace=baseline/'fixed_policy_traces'/f'{key}.npz';files.append(trace)
        with np.load(trace,allow_pickle=False) as z:
            public=unpack_public(z['packed_public'])[tick];action=z['dispatched_actions'][tick-1:tick+1]
        c=build_controller(cfg);state=c.qp.model.state_from_public(public,np.asarray(cfg['posture_target']))
        state.previous_command_target=limits[:,0]+.5*(action[0]+1.)*np.diff(limits).ravel()
        reference=limits[:,0]+.5*(action[1]+1.)*np.diff(limits).ravel()
        commands=np.asarray(row['commands'],dtype=np.float64)
        checked=verify_sequence(c,state,commands)
        if not checked['accepted']:raise ValueError('Previously accepted candidate failed independent recheck')
        predicted=rollout_for_search(c,state,commands,reference)
        if not predicted['physically_feasible'] or not predicted['terminal_ok']:
            raise ValueError('Search and strict trajectory checks disagree: '+json.dumps(dict(case=case,stored_max=row['maximum_physical_violation'],actual_max=predicted['maximum_physical_violation'],physical=predicted['physically_feasible'],terminal=predicted['terminal_ok'],risk=predicted['terminal_risk'])))
        profile=predicted['profile']
        arrays.update({key+'_commands_rad':commands,key+'_query_public':public,
            key+'_previous_action':action[0],key+'_query_student_action':action[1],
            key+'_predicted_q':np.stack([state.q]+[n['following'].q for n in profile]),
            key+'_predicted_v':np.stack([state.v]+[n['following'].v for n in profile]),
            key+'_predicted_contacts':np.stack([state.contacts]+[n['following'].contacts for n in profile])})
        rows.append(dict(case=case,query_tick=tick,horizon_steps=len(commands),
            newly_model_feasible=row['newly_model_feasible'],ball5_except_recovery=row['ball5_except_recovery'],
            original_complete5=row['complete5'],terminal_risk=checked['terminal_risk'],
            maximum_physical_violation=predicted['maximum_physical_violation']))
    if not rows:raise ValueError('No strictly checked command sequences to export')
    a.output.mkdir(parents=True);bundle=a.output/'sequences.npz';np.savez_compressed(bundle,**arrays)
    files += [Path(m.__file__) for n,m in sys.modules.items() if n.startswith('scripts.') and getattr(m,'__file__',None)]
    result=dict(scope=__doc__,native_execution=False,valid_expert_labels=False,
        baseline=str(baseline),policy_sha256=json.loads((baseline/'INPUTS.json').read_text())['policy_sha256'],
        configuration=cfg,control_dt_s=.01,rows=rows,bundle_name=bundle.name,
        bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        requires=['From-ground actual prefix, no state teleportation.',
            'Check actual query state and previous dispatched command against stored query.',
            'Record final dispatched commands, actual state and contact evolution over planned horizon.',
            'Continue to original episode end; no success from model or animation.',
            'Subsequent sustained feedback recovery remains unverified.'],
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(sequences=len(rows),cases=[r['case'] for r in rows],native_execution=False)))


if __name__=='__main__':main()
