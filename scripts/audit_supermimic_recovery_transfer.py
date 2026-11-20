#!/usr/bin/env python3
"""Compare frozen robot dynamics at measured post-strike support transitions."""
import argparse
import json
from pathlib import Path

import torch

from scripts.evaluate_supermimic_fresh_wm_panel import check_panel, digest
from quadsmash.rl.supermimic_ball_residual_v2 import load_ball_residual
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_physical_dataset_v1 import valid_windows
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from quadsmash.rl.supermimic_physical_world_v1 import PhysicalWorldV1
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2


def event_starts(data):
    """Use a common 64-step first-episode window for all reported horizons."""
    valid = valid_windows(data['episode'],data['terminated'],data['timeout'],64)
    valid &= data['episode'][:len(valid)]==data['episode'][0]
    support = data['feet'].bool().any(-1)
    next_support = data['next_feet'].bool().any(-1)
    starts = {'touchdown':[], 'subsequent_support_loss':[]}
    for lane in range(support.shape[1]):
        same_episode = data['episode'][:,lane]==data['episode'][0,lane]
        hits = (data['contact'][:,lane] & same_episode).nonzero().flatten()
        if not len(hits):
            continue
        ticks = torch.arange(len(support))
        landing = (same_episode & (ticks>=hits[0]) & ~support[:,lane] & next_support[:,lane]).nonzero().flatten()
        if not len(landing):
            continue
        t=int(landing[0])
        if t<len(valid) and valid[t,lane]:
            starts['touchdown'].append((t,lane))
        loss=(same_episode & (ticks>t) & support[:,lane] & ~next_support[:,lane]).nonzero().flatten()
        if len(loss) and loss[0]<len(valid) and valid[loss[0],lane]:
            starts['subsequent_support_loss'].append((int(loss[0]),lane))
    return starts


@torch.no_grad()
def evaluate(model, ball, data, ids, launch):
    ticks,lanes=torch.tensor(ids).T
    state,feet,present=(data[k][ticks,lanes] for k in ('state','feet','present'))
    metrics={}
    for h in range(1,65):
        t=ticks+h-1
        args=(state,data['targets'][t,lanes],data['previous_action'][t,lanes],
              feet.float(),present,data['elapsed'][t,lanes],launch[lanes])
        if ball is None:
            state,logits,present=model(*args)
            feet=logits.sigmoid()
        else:
            state,feet,present=ball_transition(model,ball,*args)
        if h not in (1,4,16,32,64):
            continue
        target=data['next_state'][t,lanes]
        diff=state-target
        diff[:,3:7]=torch.where((state[:,3:7]*target[:,3:7]).sum(1,keepdim=True)<0,
                                -state[:,3:7]-target[:,3:7],diff[:,3:7])
        metrics[h]=dict(body_rmse=float(diff[:,:49].square().mean().sqrt()),
                        root_angular_mae=float(diff[:,10:13].abs().mean()),
                        root_height_mae=float(diff[:,2].abs().mean()),
                        foot_contact_brier=float((feet-data['next_feet'][t,lanes].float()).square().mean()))
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('panel','capture','panel-actor','anchor','physical','ball','output'):
        parser.add_argument('--'+k,required=True,type=Path)
    args=parser.parse_args()
    torch.set_num_threads(3)
    root=Path(__file__).resolve().parents[1]
    panel=json.loads(args.panel.read_text())
    check_panel(panel,root/'configs/g07_full_racket_eval_v4.json',root,[args.anchor,args.panel_actor])
    prospective = panel.get('prospective_body_comparison')
    if prospective and str(args.physical.resolve()) not in prospective['bodies']:
        raise ValueError('Requested Recovery checkpoint was not declared before collection')
    measured=json.loads((args.capture/'metrics.json').read_text())
    data_path=args.capture/'native.pt'
    if (digest(data_path)!=measured['data_sha256'] or digest(args.panel)!=measured['panel_sha256']
            or measured['weights'] != panel['weights'] or measured['ticks']!=800):
        raise ValueError('Native panel identity or duration mismatch')
    actor=torch.load(args.panel_actor,weights_only=False,mmap=True,map_location='cpu')
    if tensor_digest(actor['actor'].items()) != measured['actor_sha256']:
        raise ValueError('Native recording came from a different Actor')
    ck=torch.load(args.physical,weights_only=True,map_location='cpu')
    if ck['source_manifest']['task_contract']!=actor['metadata']['task_contract']:
        raise ValueError('Different task for physical model and captured policy')
    evidence=actor['metadata']['physical_imagination']['checkpoint_sha256']
    ball_checkpoint=torch.load(args.ball,map_location='cpu',weights_only=True)
    if digest(args.ball)!=evidence['ball']:
        from quadsmash.rl.supermimic_ball_residual_v2 import verify_event_calibration
        if prospective and prospective.get('ball_candidates',{}).get(str(args.ball.resolve()))!=digest(args.ball):
            raise ValueError('Unregistered ball checkpoint on prospective Recovery evaluation')
        verify_event_calibration(ball_checkpoint,evidence['ball'],digest(args.physical),ck)
    if (digest(args.physical)!=evidence['physical'] and
            ck.get('calibration',{}).get('source_sha256')!=evidence['physical']):
        raise ValueError('Frozen physical model ancestry or ball head mismatch')
    data=torch.load(data_path,map_location='cpu',weights_only=True,mmap=True)
    groups=event_starts(data)
    cases=panel['cases']
    launches=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in cases])
    anchor=PhysicalWorldV1().eval()
    anchor.load_state_dict(torch.load(args.anchor,map_location='cpu',weights_only=True)['model'])
    body=PhysicalWorldV2(ck['source_manifest']['task_contract']['asset']['sha256']).eval()
    body.load_state_dict(ck['model'])
    ball=load_ball_residual(ball_checkpoint).eval()
    groups_report={}
    for group,ids in groups.items():
        row=dict(cases=[dict(tick=t,case=cases[lane]['case_id']) for t,lane in ids],count=len(ids))
        if ids:
            row.update(anchor=evaluate(anchor,None,data,ids,launches),body=evaluate(body,ball,data,ids,launches))
        groups_report[group]=row
    result=dict(groups=groups_report,native_panel_sha256=digest(args.panel),source_sha256=measured['data_sha256'],
                physical_sha256=digest(args.physical),anchor_sha256=digest(args.anchor),ball_sha256=digest(args.ball),
                scope='forward contact and body prediction at first measured post-hit touchdown and subsequent support loss; no optimizer steps or gradient receipt')
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
