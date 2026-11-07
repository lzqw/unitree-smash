#!/usr/bin/env python3
"""Compare measured recovery after the first changed teacher command.

Matched historical baseline is observational evidence, not an isolated command
counterfactual. Uses actual recorded root poses; does not integrate a model.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','baseline','integrity','output']:
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior evidence')
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    integrity=json.loads(a.integrity.read_text())
    if not integrity['passed'] or not integrity['all_baseline_ball5_prefixes_equal']:
        raise ValueError('Require verified causal dispatch and ball-qualified prefixes')
    if Path(integrity['run']).resolve()!=a.run.resolve() or Path(integrity['baseline']).resolve()!=a.baseline.resolve():
        raise ValueError('Different run pair')
    for path,digest in integrity['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed source '+path)
    read=lambda root,name:json.loads((root/name).read_text())
    before={x['training_index']:x for x in read(a.baseline,'metrics.json')['complete5_case_outcomes']}
    after={x['training_index']:x for x in read(a.run,'metrics.json')['complete5_case_outcomes']}
    native=read(a.run,'native_command_inputs.json');limits=np.asarray(native['joint_limits_rad'])
    with np.load(a.run/'causal_teacher_commands.npz',allow_pickle=False) as z:
        base=z['base_command'].copy();final=z['final_command'].copy();first=z['first_episode'].copy()
    command_delta=.5*(limits[:,1]-limits[:,0])*(final-base)
    sources=[Path(__file__),a.integrity,a.run/'causal_teacher_commands.npz',a.run/'native_command_inputs.json']
    poses={};root_ids={}
    for name,path in [('teacher',a.run),('baseline',a.baseline)]:
        geo=read(path,'geometry.json');root_ids[name]=geo['body_names'].index('base')
        with np.load(path/'all_first_episodes.npz',allow_pickle=False) as z:
            poses[name]={k:z[k].copy() for k in ['link_poses','episode_clock_s','valid_first_episode']}
        sources += [path/'geometry.json',path/'all_first_episodes.npz',path/'metrics.json']
    records=read(a.run,'landing_mpc_records.json');sources.append(a.run/'landing_mpc_records.json')
    bycase=defaultdict(dict)
    for r in records:bycase[r['case']][r['control_tick']]=r
    rows=[]
    for case,b in before.items():
        if not b['ball5_except_recovery']:continue
        changed=np.flatnonzero(np.any(final[:,case]!=base[:,case],axis=1)&first[:,case])
        group=('retained' if after[case]['complete5'] else 'lost') if b['complete5'] else (
            'rescued' if after[case]['complete5'] else 'unresolved')
        row=dict(case=case,group=group,first_changed_tick=int(changed[0]) if len(changed) else None)
        if not len(changed):rows.append(row);continue
        onset=int(changed[0]);stop=min(onset+20,len(final))
        delta=command_delta[onset:stop,case][first[onset:stop,case]]
        material=np.flatnonzero((abs(command_delta[:,case]).max(axis=1)>5e-6)&first[:,case])
        applied=[t for t,r in bycase[case].items() if r.get('applied_first_step')]
        row.update(first_200ms_change_l2_mean_rad=float(np.linalg.norm(delta,axis=1).mean()),
                   first_above_mapping_tolerance_tick=int(material[0]) if len(material) else None,
                   first_applied_qp_tick=min(applied) if applied else None,
                   first_200ms_only_subtolerance_changes=bool(abs(delta).max()<=5e-6),
                   first_200ms_change_max_rad=float(abs(delta).max()),
                   first_200ms_leg_change_l2_mean_rad=float(np.linalg.norm(delta[:,:12],axis=1).mean()),
                   first_200ms_arm_change_l2_mean_rad=float(np.linalg.norm(delta[:,12:],axis=1).mean()),
                   first_changed_plan=bycase[case].get(onset,{}).get('rejection'),checkpoints=[])
        public={}
        for name,path in [('teacher',a.run),('baseline',a.baseline)]:
            f=path/'fixed_policy_traces'/f'case_{case:03d}.npz';sources.append(f)
            with np.load(f,allow_pickle=False) as z:public[name]=unpack_public(z['packed_public'])
        for offset in [0,10,20,40]:
            tick=2*((onset+offset+1)//2);item=dict(tick=tick,seconds_after_first_change=.01*(tick-onset))
            for name in poses:
                snap=poses[name];idx=tick//2
                if idx>=len(snap['link_poses']) or tick>=len(public[name]) or not snap['valid_first_episode'][idx,case]:
                    item[name]=None;continue
                if abs(float(snap['episode_clock_s'][idx,case])-.01*tick)>2e-6:
                    raise ValueError('Pose clock mismatch')
                obs=public[name][tick]
                item[name]=dict(root_height_m=float(snap['link_poses'][idx,case,root_ids[name],2]),
                    angular_speed_rad_s=float(np.linalg.norm(obs[3:6])),
                    tilt_rad=float(np.arccos(np.clip(-obs[8],-1.,1.))),
                    force_observed_feet=int((obs[57:61]>.5).sum()))
            row['checkpoints'].append(item)
        rows.append(row)
    summary={}
    for group in ['retained','lost','rescued','unresolved']:
        subset=[r for r in rows if r['group']==group];changed=[r for r in subset if r['first_changed_tick'] is not None]
        summary[group]=dict(cases=len(subset),changed_cases=len(changed),
            command_change_l2_mean_rad=float(np.mean([r['first_200ms_change_l2_mean_rad'] for r in changed])) if changed else None)
    result=dict(scope=__doc__,new_native_execution=False,causal_effect_of_single_command_established=False,
        command_mapping_tolerance_rad=5e-6,tolerance_is_not_a_physical_harmlessness_guarantee=True,
        run=str(a.run.resolve()),baseline=str(a.baseline.resolve()),summary=summary,cases=rows,
        source_sha256={str(f.resolve()):sha(f) for f in sources})
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
