#!/usr/bin/env python3
"""Calibrate causal friction strength on measured racket paths using absolute and signed native loads."""
import argparse
import json
from pathlib import Path

import torch
from scipy.optimize import lsq_linear

from audit_supermimic_paired_body_transfer import response_metrics
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1,sha,tick
from fit_supermimic_racket_material_response import PairedMaterialFit
from quadsmash.rl.supermimic_racket_material_contract_v1 import validate_contract


def fit_strength(design, target, stable):
    """Linearize on the baseline trajectory, with equal weight per absolute and paired objective."""
    change=design[1]-design[0]
    labels=target[1]-target[0]
    x,y=[],[]
    for inputs,outputs in ((design,target),(change,labels),(change[stable],labels[stable])):
        if outputs.numel()==0:
            raise ValueError('No native samples for a required fit objective')
        normalizer=outputs.numel()**.5
        x.append(inputs.reshape(-1,2)/normalizer)
        y.append(outputs.flatten()/normalizer)
    solution=lsq_linear(torch.cat(x).double().numpy(),torch.cat(y).double().numpy(),bounds=(0.,1.))
    if not solution.success:raise RuntimeError(solution.message)
    return solution.x.tolist()


@torch.no_grad()
def trace_strength(dataset, root, split, material, mass, coefficients=None):
    seeds=sorted(int(s) for s,k in json.loads((root/'design.json').read_text())['assignments'].items() if k==split)
    design=[];observed=[];endpoints=[];errors=[]
    for branch in ('minus','plus'):
        parts=[];labels=[];out=[]
        for seed,(initial,outgoing,path),spin in zip(seeds,dataset.streams[branch],dataset.spins[branch],strict=True):
            trace=torch.load(root/branch/f'spec_{seed}/trace.pt',weights_only=True,mmap=True)['raw']['substep_trace']
            forces=trace['tensors']['racket_friction_force_w']
            dt=trace['physics_dt_s']
            if len(forces)!=1+dataset.prior.steps or abs(dt-dataset.prior.sub_dt)>1.e-10:
                raise ValueError('Native friction labels have a different substep time base')
            labels.append(forces[1:].sum(0)*dt/mass)
            events={}
            fitted=dict(dataset.settings['fitted'],friction_strength=coefficients)
            ball,_=tick(dataset.prior,initial,[v[0] for v in path],[v[-1] for v in path],fitted,
                        dataset.settings['solver']['pair_contact_offset_m'],measured_path=[v[:-1] for v in path],
                        initial_spin=spin,racket_material=material,compound_racket=True,event_trace=events,
                        trace_normal_impulse=True)
            tangent=events['tangent_impulse_dv']
            load=events['normal_impulse_dv'].norm(dim=-1,keepdim=True)
            parts.append(torch.stack(((tangent/(1+load)).sum(0),(tangent*(load/(1+load))).sum(0)),-1))
            out.append(ball[:,3:])
            errors.append(ball[:,3:]-outgoing)
        design.append(torch.cat(parts));observed.append(torch.cat(labels));endpoints.append(torch.cat(out))
    x,y=torch.stack(design),torch.stack(observed)
    predicted=x.sum(-1)
    delta=endpoints[1]-endpoints[0]
    return x,y,dict(absolute_ball_rmse=float(torch.cat(errors).square().mean().sqrt()),
        absolute_friction_rmse=float((predicted-y).square().mean().sqrt()),
        signed_friction_rmse=float((predicted.diff(dim=0)-y.diff(dim=0)).square().mean().sqrt()),
        stable_ball=response_metrics(dataset.native[dataset.stable],delta[dataset.stable],dataset.repeat[dataset.stable]),
        all_ball=response_metrics(dataset.native,delta,dataset.repeat))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('collection','material','calibration','output'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();torch.set_num_threads(2)
    read=lambda p:json.loads(p.read_text())
    prov=read(a.collection/'nominal_a/provenance.json')
    teacher=Path(prov['args'][prov['args'].index('--teacher')+1])
    if sha(teacher)!=prov['teacher_sha256']:raise ValueError('Changed native fixture')
    asset=torch.load(teacher,mmap=True,weights_only=True)['task_contract']['asset']['sha256']
    m=validate_contract(read(a.material),asset)
    material=torch.tensor(m['effective'])
    strings=torch.tensor([name.startswith('Collider/strip_') for name in m['component_names']])
    cfg=read(a.calibration);prior=BallImpulsePriorV1().eval()
    physics=Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json'
    if sha(physics)!=prior.preview.geometry_hashes['configs/shuttle_physics_v2.json']:
        raise ValueError('Physics hash differs')
    mass=read(physics)['mass_kg']
    report={};coef=None
    for split in ('train','development'):
        dataset=PairedMaterialFit(a.collection,split,prior,cfg,material,strings)
        x,y,baseline=trace_strength(dataset,a.collection,split,material,mass)
        if split=='train':coef=fit_strength(x,y,dataset.stable)
        _,_,candidate=trace_strength(dataset,a.collection,split,material,mass,coef)
        report[split]=dict(split_sha256=sha(a.collection/(split+'.json')),baseline=baseline,candidate=candidate)
    a0,a1=report['development']['baseline'],report['development']['candidate']
    checks=dict(absolute=a1['absolute_ball_rmse']<=a0['absolute_ball_rmse'],
                signed=a1['stable_ball']['absolute_rmse']<a0['stable_ball']['absolute_rmse'],
                strength=a1['signed_friction_rmse']<a0['signed_friction_rmse'],
                direction=a1['stable_ball']['cosine_median']>=.8 and a1['stable_ball']['cosine_p10']>=.5,
                gain=.5<=a1['stable_ball']['gain_median']<=2.)
    report.update(coefficients=coef,checks=checks,eligible_for_reserved=all(checks.values()),
                  calibration_sha256=sha(a.calibration),material_sha256=sha(a.material),physics_sha256=sha(physics),
                  final_holdout_opened=False,policy_updates=0,contact_gradients_admitted=False,
                  scope='Measured racket path, free-running shuttle. Native friction supplies training labels only.')
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)


if __name__=='__main__':main()
