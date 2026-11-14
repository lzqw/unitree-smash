#!/usr/bin/env python3
"""One fixed GPU fit of object-separated causal ball response; no actor updates."""
import argparse, hashlib, json, subprocess, time, traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2, MODE_NAMES, response_mode_labels
from scripts.fit_supermimic_physical_world import write_json
from scripts.fit_supermimic_ball_residual import validate


def verified(path, sha):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
        raise ValueError('Changed artifact: '+str(path))
    return path


@torch.no_grad()
def diagnostics(model, features, target, labels, present, masks, ids):
    probability,response=model.components(features[ids]); delta,_=model(features[ids],present[ids])
    prediction=probability.argmax(-1);truth=labels[ids]
    confusion=torch.bincount(truth*5+prediction,minlength=25).reshape(5,5)
    result={'mode_names':MODE_NAMES,'confusion_true_rows':confusion.cpu().tolist(),
            'accuracy':float((prediction==truth).float().mean()),'groups':{}}
    for name,mask in masks.items():
        sel=mask[ids];e=(delta-target[ids])[sel];n=int(sel.sum())
        correct=truth[sel];selected_response=response[sel]
        oracle=torch.zeros_like(e)
        nonzero=correct>0
        oracle[nonzero]=selected_response[nonzero,correct[nonzero]-1]
        result['groups'][name]={'count':n,
            'mean_mode_probabilities':probability[sel].mean(0).cpu().tolist() if n else None,
            'selected_modes':torch.bincount(prediction[sel],minlength=5).cpu().tolist(),
            'position_mae_m':float(e[:,:3].abs().mean()) if n else None,
            'velocity_mae_m_s':float(e[:,3:].abs().mean()) if n else None,
            'oracle_mode_velocity_mae_m_s':float((oracle-target[ids][sel])[:,3:].abs().mean()) if n else None}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-receipt',type=Path,required=True)
    p.add_argument('--trace-receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=600)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'startup.json',dict(argv={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty_diff=subprocess.check_output(['git','diff'],text=True)))
    try:
        if not torch.cuda.is_available():raise RuntimeError('GPU required')
        if a.updates<1:raise ValueError('Positive updates required')
        torch.set_num_threads(2);torch.manual_seed(971);begin=time.perf_counter()
        receipt=json.loads(a.source_receipt.read_text());source=verified(receipt['checkpoint'],receipt['checkpoint_sha256'])
        info=json.loads((source.parent/'source.json').read_text())
        dataset=verified(info['dataset_path'],info['receipt']['dataset_sha256'])
        trace_receipt=json.loads(a.trace_receipt.read_text())
        residual_path=verified(trace_receipt['residuals'],trace_receipt['residuals_sha256'])
        raw=torch.load(dataset,map_location='cpu',weights_only=True)['tensors']
        residuals=torch.load(residual_path,map_location='cpu',weights_only=True)
        c=torch.load(source,map_location='cpu',weights_only=True)
        if c['schema']!=PhysicalWorldV2.schema:raise ValueError('Expected physical v2')
        base=PhysicalWorldV2(c['source_manifest']['task_contract']['asset']['sha256']).cuda()
        base.load_state_dict(c['model'],strict=True)
        for par in base.parameters():par.requires_grad_(False)
        split=json.loads((source.parent/'split.json').read_text())
        train=torch.tensor(split['training_lanes']);val=torch.tensor(split['validation_lanes'])
        if set(train.tolist())&set(val.tolist()):raise ValueError('Split overlap')
        count,lanes=raw['state'].shape[:2]
        flat=torch.arange(count*lanes).reshape(count,lanes)
        train_ids=flat[:,train].flatten().cuda();val_ids=flat[:,val].flatten().cuda()
        data={k:v.cuda() for k,v in raw.items()}
        with torch.no_grad():
            features=torch.cat([ball_geometry_features(s,base.geometry) for s in data['state'].flatten(0,1).split(2048)])
        target=residuals['ball_residual'].flatten(0,1).cuda()
        if target.shape!=(count*lanes,6):raise ValueError('Residual shape mismatch')
        masks={k:v.flatten().cuda() for k,v in residuals['masks'].items()}
        present=data['ball_present'].flatten()
        labels=response_mode_labels(masks,target,present)
        model=BallResidualV2().cuda()
        model.normalize_from_training(features[train_ids],target[train_ids],labels[train_ids])
        gate_opt=torch.optim.Adam(model.gate.parameters(),lr=2e-4)
        response_opt=torch.optim.Adam(model.response.parameters(),lr=2e-4)
        pools={i:train_ids[labels[train_ids]==i] for i in range(5)}
        if any(len(v)==0 for v in pools.values()):raise ValueError('Missing training mode; do not invent data')
        new_ground=train_ids[(labels[train_ids]==2)&masks['new_ground'][train_ids]]
        rng=torch.Generator(device='cuda').manual_seed(971)
        def sample(ids,n):return ids[torch.randint(len(ids),(n,),device='cuda',generator=rng)]
        def balanced():
            # Equal mode exposure; ground includes first-impact and later residuals.
            rows=[sample(pools[i],128) for i in range(5)]
            if len(new_ground):rows[2]=torch.cat((sample(new_ground,64),sample(pools[2],64)))
            return torch.cat(rows)
        suite=c['source_manifest']['task_contract']['suite'];suitepath=verified(suite['path'],suite['sha256'])
        cases=json.loads(suitepath.read_text())['cases']
        launch=torch.tensor([x['launch']['position_m']+x['launch']['velocity_mps'] for x in cases],device='cuda')
        validation=torch.tensor(split['validation_selected'],device='cuda')
        write_json(a.output/'inputs.json',dict(source_receipt=receipt,trace_receipt=trace_receipt,
            dataset=str(dataset),split=split,mode_names=MODE_NAMES,pool_sizes={MODE_NAMES[k]:len(v) for k,v in pools.items()},
            native_new_ground_training=len(new_ground),sampling='128 per mode; ground 64 first-event + 64 all-ground',
            label_priority='racket > ground > net > other_residual > free',
            labels='Recorded first events plus ongoing-ground/net residual proxies, not full contact truth',
            learning_rate=2e-4,actor_updates=0,critic_updates=0,robot_updates=0))
        torch.cuda.synchronize();setup_s=time.perf_counter()-begin;begin=time.perf_counter()
        with (a.output/'training.jsonl').open('w',buffering=1) as log:
            for step in range(1,a.updates+1):
                ids=balanced(); x=model.normalized(features[ids])
                gate_loss=F.cross_entropy(model.gate(x),labels[ids])
                gate_opt.zero_grad(set_to_none=True);gate_loss.backward()
                gg=torch.nn.utils.clip_grad_norm_(model.gate.parameters(),10.,error_if_nonfinite=True);gate_opt.step()
                ids=balanced();ids=ids[labels[ids]>0];mode=labels[ids]-1
                responses=model.response(model.normalized(features[ids])).reshape(-1,4,6)
                expected=target[ids]/model.residual_scale[mode]
                predicted=responses[torch.arange(len(ids),device='cuda'),mode]
                response_loss=F.mse_loss(predicted,expected)
                if not torch.isfinite(response_loss+gate_loss):raise FloatingPointError('Nonfinite loss')
                response_opt.zero_grad(set_to_none=True);response_loss.backward()
                rg=torch.nn.utils.clip_grad_norm_(model.response.parameters(),10.,error_if_nonfinite=True);response_opt.step()
                row=dict(update=step,gate_loss=float(gate_loss),response_loss=float(response_loss),
                    gate_gradient_before_clip=float(gg),response_gradient_before_clip=float(rg))
                log.write(json.dumps(row)+'\n')
                if step==1 or step%50==0:print(json.dumps(row),flush=True)
        torch.cuda.synchronize();fit_s=time.perf_counter()-begin
        write_json(a.output/'one_step_diagnostics.json',{
            'train':diagnostics(model,features,target,labels,present,masks,train_ids),
            'validation':diagnostics(model,features,target,labels,present,masks,val_ids)})
        write_json(a.output/'rollout_validation.json',validate(base,model,data,validation,launch))
        torch.save(dict(schema=model.schema,model=model.state_dict(),gate_optimizer=gate_opt.state_dict(),
            response_optimizer=response_opt.state_dict(),updates=a.updates,source_receipt=receipt),a.output/'ball.pt')
        write_json(a.output/'completed.json',dict(gate_updates=a.updates,response_updates=a.updates,
            actor_updates=0,critic_updates=0,robot_updates=0,setup_s=setup_s,fit_s=fit_s,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),scope='Offline recorded-control model fit; no native score'))
    except BaseException as e:
        (a.output/'failure_traceback.txt').write_text(traceback.format_exc())
        write_json(a.output/'failure.json',dict(type=type(e).__name__,message=str(e)));raise


if __name__=='__main__':main()
