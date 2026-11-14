#!/usr/bin/env python3
"""Read-only contact and impact diagnostics of one frozen physical model."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from quadsmash.rl.supermimic_physical_world_v1 import PhysicalWorldV1
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from scripts.fit_supermimic_physical_world import model_args


def classification(prob,target):
    p=prob.flatten();y=target.flatten().bool();pred=p>=.5
    tp=int((pred&y).sum());tn=int((~pred&~y).sum())
    fp=int((pred&~y).sum());fn=int((~pred&y).sum())
    return dict(count=len(p),positive_fraction=float(y.float().mean()),probability_mean=float(p.mean()),
        probability_p05=float(torch.quantile(p,.05)),probability_p95=float(torch.quantile(p,.95)),
        brier=float((p-y.float()).square().mean()),accuracy=float((pred==y).float().mean()),
        true_positive=tp,true_negative=tn,false_positive=fp,false_negative=fn,
        recall_positive=tp/max(1,tp+fn),recall_negative=tn/max(1,tn+fp))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);device='cuda:0'
    checkpoint=torch.load(args.run/'world.pt',map_location=device,weights_only=True)
    dataset=args.run/'physical_dataset.pt'
    if not dataset.exists():dataset=Path(json.loads((args.run/'source.json').read_text())['dataset_path'])
    data={k:v.to(device) for k,v in torch.load(dataset,map_location='cpu',weights_only=True)['tensors'].items()}
    split=json.loads((args.run/'split.json').read_text())
    separate=checkpoint['schema']==PhysicalWorldV2.schema
    if separate:model=PhysicalWorldV2(checkpoint['source_manifest']['task_contract']['asset']['sha256'])
    elif checkpoint['schema']==PhysicalWorldV1.schema:model=PhysicalWorldV1()
    else:raise ValueError('Unknown physical model schema')
    model=model.to(device);model.load_state_dict(checkpoint['model'],strict=True);model.eval()
    cases=json.loads(Path(checkpoint['source_manifest']['task_contract']['suite']['path']).read_text())['cases']
    launch=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in cases],device=device)
    pairs=torch.tensor(split['validation_selected'],device=device);t,l=pairs[:,0],pairs[:,1]
    outcomes={}
    with torch.no_grad():
        state,_,_,feet,present,elapsed,launch_batch=model_args(data,pairs,launch)
        for k in range(64):
            state,logits,present=model(state,data['final_joint_targets'][t+k,l],data['observation'][t+k,l,-18:],
                                       feet,present,elapsed,launch_batch)
            feet=logits.sigmoid();elapsed=elapsed+.01
            if k+1 not in (1,16,64):continue
            error=(state[:,49:52]-data['next_state'][t+k,l,49:52]).abs().mean(-1)
            indices=t[:,None]+torch.arange(k+1,device=device)[None]
            delta=data['next_event_state'][indices,l[:,None]]-data['event_state'][indices,l[:,None]]
            # Only recorded first legal-contact/frame events, not all substep impacts.
            hit=(delta[:,:,[1,6]]>.5).any(-1).any(-1)
            outcomes[str(k+1)]=dict(foot=classification(feet,data['next_foot_contact'][t+k,l]),
                ball_by_recorded_first_contact_event={str(flag):dict(count=int((hit==flag).sum()),
                    position_component_mae_m=float(error[hit==flag].mean()) if (hit==flag).any() else None)
                    for flag in (False,True)})
    # Separate loss gradient contributions; no optimizer step or actor access.
    pairs=pairs[:128];t,l=pairs[:,0],pairs[:,1]
    state,_,_,feet,present,elapsed,launch_batch=model_args(data,pairs,launch)
    dynamics=state.new_zeros(());contact=state.new_zeros(())
    for k in range(16):
        state,logits,present=model(state,data['final_joint_targets'][t+k,l],data['observation'][t+k,l,-18:],
                                   feet,present,elapsed,launch_batch,**({'detach_contacts':True} if separate else {}))
        dynamics=dynamics+model.state_loss(state,data['next_state'][t+k,l])/16
        if separate:
            native_pairs=torch.stack((t+k,l),-1)
            probability=model.contact_probabilities(*model_args(data,native_pairs,launch))
            contact=contact+F.binary_cross_entropy(probability,data['next_foot_contact'][t+k,l].float())/16
        else:contact=contact+F.binary_cross_entropy_with_logits(logits,data['next_foot_contact'][t+k,l].float())/16
        feet=logits.sigmoid();elapsed=elapsed+.01
    parameters=list(model.parameters())
    def gradient_norm(loss):
        values=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
        return float(torch.sqrt(sum(v.square().sum() for v in values if v is not None)))
    result=dict(scope='Frozen model diagnosis, held-out recorded-control windows, no parameter update',
        outcomes=outcomes,dynamics_loss=float(dynamics.detach()),contact_bce=float(contact.detach()),
        dynamics_gradient_norm=gradient_norm(dynamics),weighted_contact_gradient_norm=gradient_norm(contact if separate else .1*contact),
        contact_optimizer_separate=separate,contact_gradient_coefficient=1. if separate else .1,
        actor_updates=0,world_updates_added=0)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
