#!/usr/bin/env python3
"""Compare native G07 request derivatives across amplitudes at the registered end of the held-action window."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn.functional import cosine_similarity
from audit_supermimic_paired_body_transfer import response_metrics
from evaluate_supermimic_contact_rollout import sha


def compare(reports):
    first = reports[0]
    for result in reports[1:]:
        for key in ('ticks','request_directions','prefix_sha256','task_contract','body_sha256',
                    'calibration_sha256','material_sha256'):
            if result[key] != first[key]:
                raise ValueError('Changed across perturbation scales: '+key)
    eps = [x['epsilon_request'] for x in reports]
    if eps[0] <= 0 or eps != [eps[0],eps[0]/2,eps[0]/4]:
        raise ValueError('Expected three successive factor-of-two request amplitudes')
    ids = [c['lane'] for c in first['cases']]
    if len(set(ids)) != len(ids) or any([c['lane'] for c in r['cases']] != ids for r in reports):
        raise ValueError('Different or duplicated case identities')
    end = str(first['ticks'][-1])
    def vector(level,key):
        return torch.tensor([case['ticks'][end][key] for case in reports[level]['cases']])
    topology = torch.tensor([all(r['cases'][i]['group']=='racket_stable' and
                       r['cases'][i]['signatures']['nominal_a']==first['cases'][i]['signatures']['nominal_a']
                       for r in reports) for i in range(len(ids))])
    for i in (1,2):
        torch.testing.assert_close(vector(0,'native_nominal'),vector(i,'native_nominal'),atol=0,rtol=0)
    central = [vector(i,'native')/(2*e) for i,e in enumerate(eps)]
    smooth = topology.clone()
    for left,right in [(central[0],central[1]),(central[1],central[2]),
                       (vector(2,'native_left')/eps[2],vector(2,'native_right')/eps[2])]:
        gain = right.norm(dim=-1) / left.norm(dim=-1).clamp_min(1e-12)
        smooth &= (cosine_similarity(left,right)>=.8) & (gain>=.5) & (gain<=2.)
    # Same native-only observability rules as the surface-intervention study.
    smooth &= vector(0,'native').norm(dim=-1)>.01
    for i in range(3):
        smooth &= vector(i,'native').norm(dim=-1)>3*vector(i,'native_repeat').norm(dim=-1)
    masks = dict(all=torch.ones_like(smooth),timing_and_manifold_stable=topology,
                 native_convergent_stable=smooth,stable_without_native_convergence=topology & ~smooth)
    stats = {}
    for name,mask in masks.items():
        cohort = dict(case_ids=[lane for lane,take in zip(ids,mask) if take],count=int(mask.sum()))
        if mask.any():
            for i,level in enumerate(('wide','half','quarter')):
                noise = vector(i,'native_repeat')[mask]/(2*eps[i])
                cohort[level] = {model:response_metrics(central[i][mask],vector(i,model)[mask]/(2*eps[i]),noise)
                                 for model in ('measured_racket','body_racket')}
        stats[name]=cohort
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('wide','half','quarter','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    paths=[a.wide,a.half,a.quarter]
    reports=[json.loads(f.read_text()) for f in paths]
    result=dict(groups=compare(reports),scale_reports_sha256=[sha(f) for f in paths],
                units='ball velocity change per normalized requested action along each registered direction',
                scope='development system identification',policy_updates=0,contact_gradients_admitted=False)
    with a.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')


if __name__=='__main__':
    main()
