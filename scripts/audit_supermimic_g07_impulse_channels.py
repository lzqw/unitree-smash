#!/usr/bin/env python3
"""Separate contact geometry, normal load and friction errors in a full native G07 command window."""
import argparse
import json
from pathlib import Path

import torch
from audit_supermimic_contact_normal_transfer import replay_geometry
from audit_supermimic_native_window_response import complete_window
from audit_supermimic_paired_body_transfer import response_metrics
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1,sha
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch,_same_contract
from quadsmash.rl.supermimic_racket_material_contract_v1 import validate_contract


@torch.no_grad()
def measure(run,prior,material,settings,mass):
    history=complete_window(run['raw'],run['spec'],prior.steps)
    sums={}
    for tr in history:
        t=tr['tensors']
        model,_=replay_geometry(tr,prior,material,settings,separate_impulses=True)
        phys=tr['physics_dt_s']/mass
        model['measured_normal_impulse']=phys*t['racket_force_w'][1:].sum(0)
        model['measured_friction_impulse']=phys*t['racket_friction_force_w'][1:].sum(0)
        for k,v in model.items():
            sums[k]=sums[k]+v if k in sums else v
    return sums


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','response','shapes','material','calibration','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    read=lambda path:json.loads(path.read_text())
    result=read(a.response);shapes=read(a.shapes);status=read(a.collection/'status.json')
    if (status['status']!='COLLECTION_FINISHED' or result['status_sha256']!=sha(a.collection/'status.json')
            or result['material_sha256']!=sha(a.material) or result['calibration_sha256']!=sha(a.calibration)
            or shapes['source_sha256']!=sha(a.response)):
        raise ValueError('Inconsistent capture or model settings')
    cfg=read(a.calibration);prior=BallImpulsePriorV1().eval()
    material=torch.tensor(validate_contract(read(a.material),result['task_contract']['asset']['sha256'])['effective'])
    physics=Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json'
    if sha(physics)!=prior.preview.geometry_hashes['configs/shuttle_physics_v2.json']:
        raise ValueError('Shuttle physics changed')
    mass=read(physics)['mass_kg']
    runs={}
    for row in status['completed']:
        b=row['branch'];f=a.collection/row['spec']
        if sha(f/b/'native_probe.pt')!=row['probe_sha256']:raise ValueError('Changed signed G07 branch')
        runs[b]=_load_branch(f,b)
    _same_contract(runs)
    cases=result['cases']
    if [c['lane'] for c in cases]!=[c['lane'] for c in shapes['cases']]:
        raise ValueError('Different release case identities')
    branches={b:measure(run,prior,material,cfg,mass) for b,run in runs.items()}
    final=str(result['ticks'][-1])
    native,noise=(torch.tensor([c['ticks'][final][k] for c in cases]) for k in ('native','native_repeat'))
    change={k:branches['plus'][k]-branches['minus'][k] for k in branches['plus']}
    normal,friction=change['measured_normal_impulse'],change['measured_friction_impulse']
    # Preserve actual signed drag and any external response for channel substitution.
    other=native-normal-friction
    predictors={'native_racket_impulses':normal+friction,'native_outside_racket':other}
    for mode in ('analytic_at_loaded_times','native_normal','native_normal_and_point','native_normal_point_separation'):
        n,f=(change[mode+'_'+s] for s in ('normal','tangent'))
        predictors[mode+'_model']=other+n+f
        predictors[mode+'_true_normal']=other+normal+f
        predictors[mode+'_true_friction']=other+n+friction
    stats={}
    for name in ('all','count_stable','resolved_same_shape','unresolved_count_stable','toi_switch','manifold_switch','no_contact'):
        keep=torch.tensor([name=='all' or name==c['group'] or (c['group']=='racket_stable' and (name=='count_stable' or
                           name=='resolved_same_shape' and s['shape_assignment']=='consistent' or
                           name=='unresolved_count_stable' and s['shape_assignment']=='unresolved'))
                           for c,s in zip(cases,shapes['cases'],strict=True)])
        if keep.any():
            stats[name]={k:response_metrics(native[keep],v[keep],noise[keep]) for k,v in predictors.items()}
    with a.output.open('x') as f:
        json.dump(dict(groups=stats,raw_signed={k:v.tolist() for k,v in change.items()},lanes=[x['lane'] for x in cases],
                       response_sha256=sha(a.response),shapes_sha256=sha(a.shapes),physics_sha256=sha(physics),
                       scope='Post-hoc signed impulse replacements using measured incoming states and native load times',
                       contact_gradients_admitted=False),f,indent=2,allow_nan=False)
        f.write('\n')


if __name__=='__main__':main()
