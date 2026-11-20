#!/usr/bin/env python3
"""Check a reduced temporal constraint solve against the frozen compound-impact baseline."""
import argparse
import json
from pathlib import Path

import torch
from fit_supermimic_racket_material_response import PairedMaterialFit
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1,sha
from quadsmash.rl.supermimic_racket_material_contract_v1 import validate_contract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('wide','calibration','material','readout','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--fit-patch-inertia',action='store_true',help='Fit separate effective tangential inertias for one and multiple active shapes on calibration only')
    a=p.parse_args();torch.set_num_threads(2)
    read=lambda path:json.loads(path.read_text())
    fit=read(a.readout)
    cfg=read(a.calibration)
    if (fit['calibration_sha256']!=sha(a.calibration) or fit['material_sha256']!=sha(a.material)
            or any(fit['split_hashes'][s]!=sha(a.wide/(s+'.json')) for s in ('train','development'))):
        raise ValueError('Material fit does not match the native family and solver')
    prov=read(a.wide/'nominal_a/provenance.json')
    teacher=Path(prov['args'][prov['args'].index('--teacher')+1])
    if sha(teacher)!=prov['teacher_sha256']:raise ValueError('Changed geometry source')
    asset=torch.load(teacher,mmap=True,weights_only=True)['task_contract']['asset']['sha256']
    mat=validate_contract(read(a.material),asset);table=torch.tensor(mat['effective'])
    strings=torch.tensor([n.startswith('Collider/strip_') for n in mat['component_names']])
    coefs=torch.tensor(fit['selected'])
    prior=BallImpulsePriorV1()
    scores={}
    for split in ('train','development'):
        panel=PairedMaterialFit(a.wide,split,prior,cfg,table,strings)
        scores[split]={}
        def score(mode,ratios=None):
            settings=dict(cfg['fitted'])
            if mode=='temporal':
                settings.update(separation_velocity_gain=0.,temporal_solver=dict(position_iterations=8,
                                velocity_iterations=2,bounce_speed=.05))
            elif ratios is not None:
                settings['patch_inertia']=ratios
            panel.settings=dict(cfg,fitted=settings)
            return panel.evaluate(coefs)[1]
        anchor=score('compound')
        scores[split]['compound']=anchor
        if not a.fit_patch_inertia:
            scores[split]['temporal']=score('temporal')
            continue
        if split=='train':
            chosen=(.4,.4)
            trials={}
            def objective(v):
                return v['stable_signed']['absolute_rmse']/anchor['stable_signed']['absolute_rmse'] + 1-v['stable_signed']['cosine_p10']
            for index in (0,1):
                eligible=[(objective(score('inertia',chosen)),chosen)]
                for value in (.2,.3,.4,.5,.65,.8,1.):
                    ratio=list(chosen)
                    ratio[index]=value
                    ratio=tuple(ratio)
                    v=score('inertia',ratio)
                    trials[str(ratio)]=v
                    if (v['absolute_rmse']<=anchor['absolute_rmse'] and
                            v['all_signed']['absolute_rmse']<=anchor['all_signed']['absolute_rmse']):
                        eligible.append((objective(v),ratio))
                chosen=min(eligible,key=lambda v:v[0])[1]
            scores[split]['train_candidates']=trials
        scores[split]['fitted_patch_inertia']=dict(ratios=chosen,metrics=score('inertia',chosen))
    with a.output.open('x') as f:
        json.dump(dict(results=scores,model='spherical shuttle against prescribed racket surfaces',
                       calibration_sha256=sha(a.calibration),coefficient_fit_sha256=sha(a.readout),
                       material_sha256=sha(a.material),native_material_changed=False,final_holdout_opened=False,
                       actor_updates=0,contact_gradients_admitted=False),f,indent=2,allow_nan=False)
        f.write('\n')


if __name__=='__main__':main()
