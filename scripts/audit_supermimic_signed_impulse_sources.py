#!/usr/bin/env python3
"""Separate signed normal, simultaneous friction and friction-only errors using native substep recordings."""
import argparse
import json
from pathlib import Path

import torch
from audit_supermimic_compound_impact_teacher import predict
from audit_supermimic_paired_body_transfer import response_metrics
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1, sha
from quadsmash.rl.supermimic_racket_material_contract_v1 import validate_contract


@torch.no_grad()
def audit(directory, split, prior, cfg, material, mass):
    report=json.loads((directory/(split+'.json')).read_text())
    if report['split']!=split or report['design_sha256']!=sha(directory/'design.json'):
        raise ValueError('Different launch-family split')
    results={}
    for seed in sorted({r['seed'] for r in report['samples']}):
        rows=[r for r in report['samples'] if r['seed']==seed]
        ids=[r['lane'] for r in rows]
        branches={}
        for branch in ('nominal_a','nominal_b','plus','minus'):
            path=directory/branch/f'spec_{seed}'/'trace.pt'
            if sha(path)!=report['archives'][str(path)]:
                raise ValueError('Native substep trace changed')
            tr=torch.load(path,mmap=True,weights_only=True)['raw']['substep_trace']
            t=tr['tensors']
            scale=tr['physics_dt_s']/mass
            normal=t['racket_force_w'][1:,ids]
            friction=t['racket_friction_force_w'][1:,ids]
            loaded=normal.norm(dim=-1,keepdim=True)>.001
            model=predict(tr,ids,prior,material,cfg,compound=True)
            model_n=model['normal'].sum(0)
            model_t=model['tangent'].sum(0)
            real_n=normal.sum(0)*scale
            real_t=friction.sum(0)*scale
            branches[branch]=dict(native=t['shuttle_state_w'][-1,ids,7:10],native_normal=real_n,
                native_all=real_n+real_t,native_normal_loaded_friction=real_n+(loaded*friction).sum(0)*scale,
                predicted_normal_native_friction=model_n+real_t,native_normal_predicted_friction=real_n+model_t,
                predicted_both=model_n+model_t)
        def difference(key,left='minus',right='plus'):
            return branches[right][key]-branches[left][key]
        delta={k:difference(k) for k in branches['plus']}
        delta['repeat']=difference('native','nominal_a','nominal_b')
        for i,row in enumerate(rows):
            for group in ('all',row['kind']):
                results.setdefault(group,[]).append({k:x[i] for k,x in delta.items()})
    summary={}
    for group,rows in results.items():
        data={k:torch.stack([r[k] for r in rows]) for k in rows[0]}
        summary[group]={k:response_metrics(data['native'],v,data['repeat'])
                        for k,v in data.items() if k not in ('native','repeat')}
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('collection','material','calibration','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(3)
    cfg=json.loads(a.calibration.read_text())
    provenance=json.loads((a.collection/'nominal_a/provenance.json').read_text())
    teacher=Path(provenance['args'][provenance['args'].index('--teacher')+1])
    if sha(teacher)!=provenance['teacher_sha256']:
        raise ValueError('Changed collector asset ancestry')
    asset=torch.load(teacher,mmap=True,weights_only=True)['task_contract']['asset']['sha256']
    material=torch.tensor(validate_contract(json.loads(a.material.read_text()),asset)['effective'])
    physics=json.loads((Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json').read_text())
    prior=BallImpulsePriorV1().eval()
    results={s:audit(a.collection,s,prior,cfg,material,physics['mass_kg']) for s in ('train','development')}
    out=dict(results=results,scope='Native incoming state supplied each substep; native impulse substitutions for attribution',
        calibration_sha256=sha(a.calibration),material_sha256=sha(a.material),final_holdout_opened=False,
        policy_updates=0,contact_gradients_admitted=False)
    with a.output.open('x') as f:
        json.dump(out,f,indent=2,allow_nan=False)
        f.write('\n')


if __name__=='__main__':
    main()
