#!/usr/bin/env python3
"""Isolate incoming shuttle integration from the contact solve on held G07 command sequences."""
import argparse
import copy
import json
from pathlib import Path

import torch

from audit_supermimic_native_window_response import complete_window
from audit_supermimic_paired_body_transfer import response_metrics
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1, sha, tick
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch, _same_contract
from quadsmash.rl.supermimic_racket_material_contract_v1 import validate_contract


def reset_frames(cases, steps):
    """Use the same boundary for each signed pair, before its first native load in any branch."""
    first = [min((i for sig in case['signatures'].values() for i in sig[3]),default=-1) for case in cases]
    first = torch.tensor(first)
    control = torch.where(first < 0, -1, (first // steps) * steps)
    return torch.stack((torch.full_like(first,-1), control, first))


@torch.no_grad()
def rollout(traces, reset, prior, cfg, material):
    fields = ('shuttle_state_w','racket_position_w','racket_quaternion_w','racket_linear_velocity_w','racket_angular_velocity_w')
    full = [torch.cat([trace['tensors'][key][:-1] for trace in traces]) for key in fields]
    shuttle, *motion = full
    horizon, lanes = shuttle.shape[:2]
    reset = reset.flatten()
    if reset.shape!=(3*lanes,) or (reset>=horizon).any():
        raise ValueError('Invalid paired reset layout')
    def incoming(i):
        return torch.cat((shuttle[i,:,:3],shuttle[i,:,7:10]),-1)
    ball = incoming(0).repeat(3,1)
    spin = shuttle[0,:,10:13].repeat(3,1)
    predicted_toi = torch.full_like(reset, -1)
    incoming_error = torch.zeros(3*lanes,2)
    boundaries = sorted({0,horizon,*reset[reset>=0].tolist()})
    for start,stop in zip(boundaries,boundaries[1:]):
        source = (reset==start).nonzero().flatten()
        native = source%lanes
        truth = incoming(start)[native]
        error = (ball[source]-truth).reshape(-1,2,3).norm(dim=-1)
        incoming_error[source] = error
        ball[source] = truth
        spin[source] = shuttle[start,native,10:13]
        # Clear predictions before the injected state for the two diagnostic interventions.
        predicted_toi[source] = -1
        part = copy.copy(prior)
        part.steps = stop-start
        part.dt = prior.sub_dt*part.steps
        path = [stream[start:stop].repeat(1,3,1) for stream in motion]
        events = {}
        ball,_,spin = tick(part,ball,None,None,cfg['fitted'],cfg['solver']['pair_contact_offset_m'],
                           initial_spin=spin,return_spin=True,racket_material=material,
                           measured_path=path,compound_racket=True,event_trace=events)
        hit = events['racket']>=0
        new = (predicted_toi<0)&hit.any(0)
        predicted_toi[new] = start+hit[:,new].long().argmax(0)
    return ball[:,3:].reshape(3,lanes,3),predicted_toi.reshape(3,lanes),incoming_error.reshape(3,lanes,2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('collection','response','calibration','material','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    torch.set_num_threads(2)
    read=lambda path:json.loads(path.read_text())
    response=read(a.response)
    status=read(a.collection/'status.json')
    if (status['status']!='COLLECTION_FINISHED' or sha(a.collection/'status.json')!=response['status_sha256']
            or sha(a.calibration)!=response['calibration_sha256'] or sha(a.material)!=response['material_sha256']):
        raise ValueError('Native acquisition or physics inputs have changed')
    runs={}
    for row in status['completed']:
        folder=a.collection/row['spec']
        branch=row['branch']
        if sha(folder/branch/'native_probe.pt')!=row['probe_sha256']:
            raise ValueError('Changed native branch')
        runs[branch]=_load_branch(folder,branch)
    _same_contract(runs)
    cases=response['cases']
    if [case['lane'] for case in cases]!=runs['nominal_a']['spec']['lanes']:
        raise ValueError('Different case indices')
    prior=BallImpulsePriorV1().eval()
    frame=reset_frames(cases,prior.steps)
    cfg=read(a.calibration)
    mat=torch.tensor(validate_contract(read(a.material),response['task_contract']['asset']['sha256'])['effective'])
    branches={}
    for b in ('plus','minus'):
        windows=complete_window(runs[b]['raw'],runs[b]['spec'],prior.steps)
        branches[b]=rollout(windows,frame,prior,cfg,mat)
    tick_id=str(response['ticks'][-1])
    pred=branches['plus'][0]-branches['minus'][0]
    vector=lambda k:torch.tensor([r['ticks'][tick_id][k] for r in cases])
    torch.testing.assert_close(pred[0],vector('measured_racket'),rtol=1e-5,atol=5e-5)
    native,repeat=vector('native'),vector('native_repeat')
    metrics={}
    modes=('initial_shuttle','native_at_control_start','native_at_substep_start')
    for name in ['all',*sorted({r['group'] for r in cases})]:
        mask=torch.tensor([name=='all' or name==r['group'] for r in cases])
        metrics[name]={m:response_metrics(native[mask],pred[i,mask],repeat[mask]) for i,m in enumerate(modes)}
    for j,case in enumerate(cases):
        case.pop('ticks')
        case['responses']={m:pred[i,j].tolist() for i,m in enumerate(modes)}
        case['reset_physics_frames']=frame[:,j].tolist()
        case['branches']={b:dict(first_model_load=rows[1][:,j].tolist(),
                           pre_reset_error_position_m_velocity_mps=rows[2][:,j].tolist()) for b,rows in branches.items()}
    with a.output.open('x') as out:
        json.dump(dict(groups=metrics,cases=cases,source_sha256=sha(a.response),modes=modes,
                       scope='Measured racket on every substep; observed shuttle injected once for attribution',
                       optimizer_updates=0,contact_admitted=False),out,indent=2,allow_nan=False)
        out.write('\n')


if __name__=='__main__':
    main()
