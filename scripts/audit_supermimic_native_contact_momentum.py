#!/usr/bin/env python3
"""Compare observed shuttle motion to the forces and lever arms reported on each loaded PhysX substep."""
import argparse
import json
from pathlib import Path

import torch
from audit_supermimic_native_window_response import complete_window
from evaluate_supermimic_contact_rollout import sha
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch,_same_contract


def momentum_increment(center,points,delta_v,inertia_per_mass):
    return delta_v.sum(0),torch.linalg.cross(points-center,delta_v).sum(0)/inertia_per_mass


def trace_errors(trace,physics,spin_cap):
    dt=trace['physics_dt_s']; mass=physics['mass_kg']
    inertia=.4*physics['collision_radius_m']**2
    shuttle=trace['tensors']['shuttle_state_w']
    rows=[]
    for i,(normal,friction) in enumerate(zip(trace['manifold_frames'][1:],trace['racket_friction_frames'][1:],strict=True)):
        loaded=normal['normal_force_n'].abs()>physics['contact_force_threshold_n']
        f_loaded=friction['force_w_n'].norm(dim=-1)>physics['contact_force_threshold_n']
        lanes=torch.cat((normal['lane'][loaded & (normal['filter_index']==0)],friction['lane'][f_loaded])).unique()
        for lane in lanes.tolist():
            hit=loaded & (normal['lane']==lane)
            if ((normal['filter_index']!=0)&hit).any():
                rows.append(dict(lane=lane,substep=i,other_contact=True))
                continue
            f=f_loaded & (friction['lane']==lane)
            s,after=shuttle[i:i+2,lane].double()
            points=torch.cat((normal['point_w'][hit],friction['point_w_m'][f])).double()
            force=torch.cat((normal['normal_w'][hit]*normal['normal_force_n'][hit,None],friction['force_w_n'][f])).double()
            imp=force*dt/mass
            normal_count=int(hit.sum())
            dv,dw=momentum_increment(s[:3],points,imp,inertia)
            _,dw_end=momentum_increment(after[:3],points,imp,inertia)
            wind=s.new_tensor(physics['wind_m_s'])
            rel=s[7:10]-wind
            free_dv=(s.new_tensor([0.,0.,-9.81])-physics['quadratic_drag_k_per_m']*rel.norm()*rel)*dt
            native_dv=after[7:10]-s[7:10]-free_dv
            pre_spin=s[10:13]
            restricted=pre_spin*(spin_cap/pre_spin.norm().clamp_min(1.e-12)).clamp(max=1.)
            native_dw=after[10:13]-restricted
            rows.append(dict(lane=lane,substep=i,other_contact=False,normal_points=normal_count,friction_points=int(f.sum()),
                native_dv=native_dv.tolist(),reported_dv=dv.tolist(),native_dw=native_dw.tolist(),reported_dw=dw.tolist(),
                reported_dw_end_center=dw_end.tolist(),normal_dw=momentum_increment(s[:3],points[:normal_count],imp[:normal_count],inertia)[1].tolist(),
                pre_spin_norm=float(pre_spin.norm()),post_spin_norm=float(after[10:13].norm())))
    return rows


def summarize(rows):
    selected=[r for r in rows if not r['other_contact']]
    stats=dict(racket_frames=len(rows),isolated_frames=len(selected),other_contact_frames=len(rows)-len(selected))
    if not selected:
        return stats
    for k,raw in [('dv','reported_dv'),('dw','reported_dw'),('dw_end_center','reported_dw_end_center')]:
        truth=torch.tensor([r['native_'+k[:2]] for r in selected])
        model=torch.tensor([r[raw] for r in selected])
        err=model-truth
        stats[k]=dict(component_rmse=float(err.square().mean().sqrt()),reference_component_rms=float(truth.square().mean().sqrt()),
                      norm_error_p95=float(err.norm(dim=1).quantile(.95)))
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','response','output'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    status=json.loads((a.collection/'status.json').read_text());r=json.loads(a.response.read_text())
    if status['status']!='COLLECTION_FINISHED' or sha(a.collection/'status.json')!=r['status_sha256']:
        p.error('Changed native collection')
    config=Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json'
    physics=json.loads(config.read_text())
    runs={}
    for item in status['completed']:
        d=a.collection/item['spec'];b=item['branch']
        if sha(d/b/'native_probe.pt')!=item['probe_sha256']:
            raise ValueError('Changed native branch')
        runs[b]=_load_branch(d,b)
    _same_contract(runs)
    frames={}
    for key,run in runs.items():
        observed=[]
        for offset,tr in enumerate(complete_window(run['raw'],run['spec'],160)):
            for row in trace_errors(tr,physics,100.):
                observed.append(dict(row,lane=run['spec']['lanes'][row['lane']],tick=run['spec']['action_tick']+offset))
        frames[key]=observed
    stable={c['lane'] for c in r['cases'] if c['group']=='racket_stable'}
    report=dict(branches={b:{'all':summarize(obs),'racket_stable':summarize([x for x in obs if x['lane'] in stable])}
                            for b,obs in frames.items()},frames=frames,physics_sha256=sha(config),response_sha256=sha(a.response),
                scope='Observed racket-only substeps. Solid-sphere inertia, configured free-flight forces and measured 100 rad/s pre-solve spin limit. Duplicate nominal runs reported separately.')
    with a.output.open('x') as out:
        json.dump(report,out,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
