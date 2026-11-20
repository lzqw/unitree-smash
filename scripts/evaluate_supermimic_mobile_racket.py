#!/usr/bin/env python3
"""Test the reaction of a free articulated racket in a multi-shape contact solve."""
import argparse
import json
from pathlib import Path

import torch
from fit_supermimic_racket_material_response import PairedMaterialFit
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1, sha
from quadsmash.rl.supermimic_articulated_inertia_v1 import ArticulatedInertiaV1
from quadsmash.rl.supermimic_mobile_racket_contact import local_racket_mobility
from quadsmash.rl.supermimic_racket_material_contract_v1 import validate_contract


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('wide','calibration','material','readout','spin-limit','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    read=lambda path:json.loads(path.read_text())
    cfg,fit,spin,design=map(read,(a.calibration,a.readout,a.spin_limit,a.wide/'design.json'))
    if (fit['calibration_sha256']!=sha(a.calibration) or fit['material_sha256']!=sha(a.material)
            or fit['split_hashes']['train']!=sha(a.wide/'train.json') or fit['final_holdout_opened']):
        raise ValueError('Material fit belongs to a different configuration')
    cfg=dict(cfg,fitted=dict(cfg['fitted'],pre_solve_spin_limit_rad_s=spin['limit_rad_s']))
    prov=read(a.wide/'nominal_a/provenance.json')
    path=Path(prov['args'][prov['args'].index('--teacher')+1])
    if sha(path)!=design['source_teacher_sha256'] or sha(path)!=prov['teacher_sha256']:
        raise ValueError('Source articulation changed')
    teacher=torch.load(path,mmap=True,weights_only=True)
    state=teacher['raw']['start']['before']['physical']
    asset=teacher['task_contract']['asset']['sha256']
    inertia=ArticulatedInertiaV1(asset).eval()
    mobility=local_racket_mobility(inertia,state['joint_position'].double(),state['root_quaternion_w'].double()).float()
    if not torch.isfinite(mobility).all() or torch.linalg.eigvalsh(mobility.double()).amin()<=0:
        raise ValueError('Invalid articulated inverse mass')
    mat=validate_contract(read(a.material),asset)
    table=torch.tensor(mat['effective']);strings=torch.tensor([n.startswith('Collider/strip_') for n in mat['component_names']])
    coefs=torch.tensor(fit['selected']);prior=BallImpulsePriorV1().eval()
    physics=Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json'
    mass=read(physics)['mass_kg']
    out={}
    # Calibrate the fraction of the authored articulation's reaction on the training families.
    # Revert the corresponding portion of the effective normal scaling to avoid counting it twice.
    def score(panel,batch,w):
        panel.settings=dict(cfg,fitted=dict(cfg['fitted'],normal_impulse_scale=(1-w)*cfg['fitted']['normal_impulse_scale']+w))
        panel.mobility=[w*b for b in batch] if w else None
        panel.shuttle_mass=mass
        return panel.evaluate(coefs)[1]
    for split in ('train','development'):
        panel=PairedMaterialFit(a.wide,split,prior,cfg,table,strings)
        batch=[]
        for seed in sorted(int(k) for k,s in design['assignments'].items() if s==split):
            record_path=a.wide/'nominal_a'/f'spec_{seed}'/'trace.pt'
            if sha(record_path)!=spin['source_sha256'][str(record_path)]:raise ValueError('Changed native case')
            packet=torch.load(record_path,mmap=True,weights_only=True)
            batch.append(mobility[packet['parent_lane']])
        weights=(0.,.25,.5,.75,1.) if split=='train' else sorted({0.,chosen,1.})
        metrics={w:score(panel,batch,w) for w in weights}
        if split=='train':
            keep=[w for w,r in metrics.items() if r['absolute_rmse']<=metrics[0.]['absolute_rmse']
                  and r['all_signed']['absolute_rmse']<=metrics[0.]['all_signed']['absolute_rmse']]
            chosen=min(keep,key=lambda w:metrics[w]['stable_signed']['absolute_rmse'])
        out[split]=metrics
    with a.output.open('x') as f:
        json.dump(dict(results=out,chosen_train_weight=chosen,readout_sha256=sha(a.readout),spin_limit_sha256=sha(a.spin_limit),
                       calibration_sha256=sha(a.calibration),physical_config_sha256=sha(physics),teacher_sha256=sha(path),
                       scope='Initial fixture joints and measured racket motion; free spatial inertia transported through substeps',
                       contact_gradients_admitted=False,policy_updates=0),f,indent=2,allow_nan=False)


if __name__=='__main__':main()
