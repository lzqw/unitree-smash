#!/usr/bin/env python3
"""Transfer the frozen fixture material calibration to actual G07 commands and measured racket paths."""
import argparse
import json
from pathlib import Path

import torch

from audit_supermimic_native_window_response import BRANCHES, measure
from audit_supermimic_paired_body_transfer import response_metrics
from audit_supermimic_surface_derivative_scales import read_material
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1, PhysicalWorldV2, sha
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch, _same_contract
from quadsmash.rl.supermimic_racket_material_contract_v1 import validate_contract


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('collection','response','fixture','material','readout','calibration','body','output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--split-racket-path',action='store_true',
                        help='Propagate separate shuttle states for measured racket pose or twist on the frozen robot trajectory')
    parser.add_argument('--compare-rotation',action='store_true',
                        help='Compare rotation interpolation with the same material and recorded native action')
    a=parser.parse_args()
    torch.set_num_threads(2)
    read=lambda path: json.loads(path.read_text())
    baseline=read(a.response)
    status_path=a.collection/'status.json'
    status=read(status_path)
    for key,path in [('status',status_path),('material',a.material),('calibration',a.calibration),('body',a.body)]:
        if baseline[key+'_sha256'] != sha(path):
            raise ValueError('Baseline input changed: '+key)
    completed=status['completed']
    if (status['status']!='COLLECTION_FINISHED' or len(completed)!=4 or
            {r['branch'] for r in completed}!=set(BRANCHES) or len({r['spec'] for r in completed})!=1):
        raise ValueError('Expected four completed signed runs for the same G07 specification')
    runs={}
    for row in completed:
        folder=a.collection/row['spec']
        branch=row['branch']
        if sha(folder/branch/'native_probe.pt')!=row['probe_sha256']:
            raise ValueError('Changed G07 native probe')
        runs[branch]=_load_branch(folder,branch)
    _same_contract(runs)
    run=runs['nominal_a']
    task,spec=run['task_contract'],run['spec']
    cases=baseline['cases']
    if (baseline['task_contract']!=task or spec['lanes']!=[c['lane'] for c in cases]
            or spec['request_directions']!=baseline['request_directions'] or baseline['ticks']!=list(range(spec['action_tick'],spec['tick']+1))):
        raise ValueError('G07 panel or command window differs')
    state=torch.load(a.body,map_location='cpu',weights_only=True)
    if state['source_manifest']['task_contract']!=task or set(spec['lanes']) & set(state['calibration']['train_lanes']):
        raise ValueError('Different body dynamics or held-out panel')
    suite=Path(task['suite']['path'])
    if sha(suite)!=task['suite']['sha256']:
        raise ValueError('Changed incoming launches')
    launches=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in read(suite)['cases']])[spec['lanes']]
    authored=torch.tensor(validate_contract(read(a.material),task['asset']['sha256'])['effective'])
    prov=read(a.fixture/'nominal_a/provenance.json')
    fitted=read_material(a.fixture,prov,a.material,a.readout,a.calibration)
    base=read(a.calibration)
    prior=BallImpulsePriorV1().eval()
    body=PhysicalWorldV2(task['asset']['sha256']).eval()
    body.load_state_dict(state['model'])
    end=str(baseline['ticks'][-1])
    native=torch.tensor([c['ticks'][end]['native'] for c in cases])
    repeat=torch.tensor([c['ticks'][end]['native_repeat'] for c in cases])
    groups={'all':torch.ones(len(cases),dtype=torch.bool)}
    for group in sorted({c['group'] for c in cases}):
        groups[group]=torch.tensor([c['group']==group for c in cases])
    results={}
    variants=[('authored',authored,False),('fixture_fit',fitted,False),('fixture_fit_with_floor',fitted,True)]
    if a.compare_rotation:
        variants=[('authored',authored,False),('authored_hermite',authored,False)]
    for mode,table,floor in variants:
        cfg=dict(base,fitted=dict(base['fitted'],articulation_friction_floor=floor,racket_rotation_hermite=mode=='authored_hermite'))
        branches={b:measure(runs[b],body,prior,table,cfg,launches,split_racket_path=a.split_racket_path)[-1] for b in ('minus','plus')}
        def signed(field):
            return branches['plus'][field]-branches['minus'][field]
        torch.testing.assert_close(signed('native'),native,rtol=0,atol=0)
        scores={}
        fields=['measured_racket','body_racket']
        if a.split_racket_path:fields.extend(('native_pose','native_twist','native_endpoints'))
        for field in fields:
            delta=signed(field)
            if mode=='authored' and field in ('measured_racket','body_racket'):
                torch.testing.assert_close(delta,torch.tensor([c['ticks'][end][field] for c in cases]),rtol=1e-5,atol=2e-6)
            scores[field]=dict(by_group={k:response_metrics(native[m],delta[m],repeat[m]) for k,m in groups.items()},
                               signed_ball_response_mps=delta.tolist())
        results[mode]=scores
    with a.output.open('x') as f:
        json.dump(dict(results=results,lanes=spec['lanes'],response_sha256=sha(a.response),
                       readout_sha256=sha(a.readout),fixture_train_sha256=sha(a.fixture/'train.json'),
                       scope='Frozen contact candidates transferred to development G07 command sequences',
                       contact_gradients_admitted=False,actor_updates=0),f,indent=2,allow_nan=False)
        f.write('\n')


if __name__=='__main__':
    main()
