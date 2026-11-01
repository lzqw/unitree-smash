#!/usr/bin/env python3
"""Independently recompute a completed native paired standing probe."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def audit(source):
    source=Path(source)
    inputs=json.loads((source/'INPUTS.json').read_text())
    metrics=json.loads((source/'metrics.json').read_text())
    job=json.loads((source/'job.json').read_text())
    if job['status']!='SUCCEEDED' or job['exit_code']!=0:raise ValueError('Require completed native job')
    admission=json.loads((source/'native_admission.json').read_text())
    if not admission['passed'] or not admission['native_execution']:raise ValueError('Native admission missing')
    protocol=json.loads((ROOT/'configs/evaluation_protocol_v2.json').read_text())
    threshold=protocol['recovery']
    if metrics['support_stability_thresholds']!=threshold:raise ValueError('Original thresholds changed')
    with np.load(source/'physics_support_metrics.npz') as data:
        frames=data['physics_frame'];x=data['values'];dt=float(data['physics_dt_s'])
    if x.ndim!=3 or x.shape[1:]!=(12,12) or not np.isfinite(x).all() or np.any(np.diff(frames)<=0):
        raise ValueError('Invalid native physics metric stream')
    if dt!=inputs['physics_dt_s']:raise ValueError('Physics timestep changed')
    run=np.zeros(12);longest=run.copy();body=run.copy();tilt=run.copy();bad=np.zeros(12,dtype=bool)
    for i,row in enumerate(x):
        elapsed=0. if not i else float(frames[i]-frames[i-1])*dt
        live=row[:,10]>.5;bad|=live&(row[:,7]>threshold['maximum_body_ground_force_n'])
        stable=(live&(row[:,8]>.5)&(~bad)&(row[:,6]>=.12)&
            (abs(row[:,1:3]).max(1)<=np.deg2rad(threshold['maximum_abs_roll_pitch_deg']))&
            (row[:,3]<=threshold['maximum_horizontal_root_speed_m_s'])&
            (row[:,4]<=threshold['maximum_root_angular_speed_rad_s'])&
            (row[:,5]<=threshold['maximum_abs_joint_velocity_rad_s']))
        if not np.array_equal(stable,row[:,11]>.5):raise ValueError('Published native stability flag differs from raw criteria')
        run=np.where(stable,run+elapsed,0.);longest=np.maximum(longest,run)
        body=np.where(live,np.maximum(body,row[:,7]),body)
        tilt=np.where(live,np.maximum(tilt,abs(row[:,1:3]).max(1)),tilt)
    jumps={}
    for row in json.loads((source/'jump_episode_records.json').read_text()):jumps.setdefault(row['env_id'],row)
    if set(jumps)!=set(range(12)):raise ValueError('First-episode evidence missing')
    plans=json.loads((source/'qp_plans.json').read_text());rows=[]
    for i in range(12):
        expected=metrics['rows'][i];terminated=bool(jumps[i]['terminated'])
        passed=bool(longest[i]>=threshold['continuous_stable_window_s']-1e-6 and not bad[i] and not terminated)
        if abs(longest[i]-expected['longest_support_stability_s'])>1e-3:raise ValueError('Continuous support window disagrees')
        if abs(body[i]-expected['maximum_body_ground_force_n'])>1e-4:raise ValueError('Body force maximum disagrees')
        if abs(np.rad2deg(tilt[i])-expected['maximum_abs_roll_pitch_deg'])>1e-3:raise ValueError('Tilt maximum disagrees')
        selected=[p for p in plans if p['env_id']==i]
        if (passed!=expected['support_stability_only'] or terminated!=expected['terminated_early'] or
            len(selected)!=expected['qp_attempts'] or sum(p['accepted'] for p in selected)!=expected['qp_accepted']):
            raise ValueError('Independent paired classification differs')
        rows.append(dict(env_id=i,mode=expected['mode'],support_stability_only=passed,
                         longest_s=float(longest[i]),terminated_early=terminated))
    with np.load(source/'actual_first_episode_streams.npz') as stream:
        obs=stream['observations'];action=stream['actions'];valid=stream['valid_first_episode']
        if obs.shape[:2]!=action.shape[:2] or action.shape[1:]!=(12,18) or obs.shape[2]!=794:
            raise ValueError('Actual control stream shape differs')
        if not np.isfinite(obs).all() or not np.isfinite(action).all() or abs(action).max()>1.000001:
            raise ValueError('Invalid actual control stream')
        if not np.all(action[:,:6]==action[0:1,:6]):raise ValueError('Baseline is not fixed original posture')
        if metrics['actual_control_env_steps']!=len(obs)*12:raise ValueError('Interaction accounting mismatch')
        for i in range(6):
            if not np.array_equal(obs[0,i],obs[0,i+6]):raise ValueError('Initial paired public observations differ')
        for plan in plans:
            tick,lane=plan['tick'],plan['env_id']
            if lane<6 or not valid[tick,lane] or not (obs[tick,lane,57:61]>.5).any():raise ValueError('Support was invoked without an observed active contact')
    counts={mode:sum(r['support_stability_only'] for r in rows if r['mode']==mode)
            for mode in ['original_pd','whole_body_qp']}
    if counts['original_pd']!=metrics['original_pd_stable'] or counts['whole_body_qp']!=metrics['whole_body_qp_stable']:
        raise ValueError('Group counts differ')
    evidence={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in
        ['INPUTS.json','native_admission.json','metrics.json','job.json','completed.json',
         'physics_support_metrics.npz','actual_first_episode_streams.npz','jump_episode_records.json','qp_plans.json']}
    return dict(source=str(source),scope='Paired standing support only, not landing or strike success',
        counts=counts,rows=rows,physics_samples=len(x),numerical_failures=metrics['numerical_failures'],
        evidence_sha256=evidence,native_execution=True,formal_acceptance=False,goal_complete=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve previous independent result')
    result=audit(args.source);args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(args.output)
    print(json.dumps({k:result[k] for k in ['counts','physics_samples','numerical_failures','scope']}))


if __name__=='__main__':main()
