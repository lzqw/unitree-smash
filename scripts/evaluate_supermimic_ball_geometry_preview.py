#!/usr/bin/env python3
"""Evaluate one fixed causal geometry mask with frozen models; no fitting."""
import argparse,hashlib,json,subprocess,time,traceback
from pathlib import Path
import torch
from quadsmash.rl.supermimic_ball_geometry_preview_v1 import BallGeometryPreviewV1,GeometrySupportedBallV1
from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2,response_mode_labels
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from scripts.fit_supermimic_ball_residual_v2 import verified,diagnostics
from scripts.fit_supermimic_ball_residual import validate
from scripts.fit_supermimic_physical_world import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ball-receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'startup.json',dict(args={k:str(v) for k,v in vars(a).items()},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty_diff=subprocess.check_output(['git','diff'],text=True)))
    try:
        torch.set_num_threads(2);torch.manual_seed(971);begin=time.perf_counter()
        receipt=json.loads(a.ball_receipt.read_text());path=verified(receipt['checkpoint'],receipt['checkpoint_sha256'])
        inputs=json.loads((path.parent/'inputs.json').read_text());src=inputs['source_receipt']
        physical_path=verified(src['checkpoint'],src['checkpoint_sha256'])
        info=json.loads((physical_path.parent/'source.json').read_text())
        dataset=verified(inputs['dataset'],info['receipt']['dataset_sha256'])
        raw=torch.load(dataset,map_location='cpu',weights_only=True)['tensors']
        tr=inputs['trace_receipt'];res=torch.load(verified(tr['residuals'],tr['residuals_sha256']),map_location='cpu',weights_only=True)
        physical=torch.load(physical_path,map_location='cpu',weights_only=True)
        base=PhysicalWorldV2(physical['source_manifest']['task_contract']['asset']['sha256']).cuda()
        base.load_state_dict(physical['model'],strict=True)
        ball=BallResidualV2().cuda();c=torch.load(path,map_location='cpu',weights_only=True)
        if c['schema']!=ball.schema:raise ValueError('Unexpected ball schema')
        ball.load_state_dict(c['model'],strict=True)
        preview=BallGeometryPreviewV1().cuda();model=GeometrySupportedBallV1(ball,preview)
        for par in list(base.parameters())+list(model.parameters()):par.requires_grad_(False)
        data={k:v.cuda() for k,v in raw.items()};count,lanes=raw['state'].shape[:2]
        train=torch.tensor(inputs['split']['training_lanes']);val=torch.tensor(inputs['split']['validation_lanes'])
        flat=torch.arange(count*lanes).reshape(count,lanes)
        ids={'train':flat[:,train].flatten().cuda(),'validation':flat[:,val].flatten().cuda()}
        with torch.no_grad():
            features=torch.cat([ball_geometry_features(s,base.geometry) for s in data['state'].flatten(0,1).split(2048)])
            target=res['ball_residual'].flatten(0,1).cuda();masks={k:v.flatten().cuda() for k,v in res['masks'].items()}
            present=data['ball_present'].flatten();labels=response_mode_labels(masks,target,present)
            summary={};support_summary={}
            for split,selected in ids.items():
                summary[split]=diagnostics(model,features,target,labels,present,masks,selected)
                support,gap=preview(features[selected]);groups={}
                for i,key in enumerate(('new_racket','new_ground','new_net')):
                    sel=masks[key][selected];g=gap[sel,i];s=support[sel,i]
                    groups[key]=dict(count=len(g),mean_support=float(s.mean()),
                        support_at_least_half=int((s>=.5).sum()),
                        gap_m_p10_p50_p90=[float(torch.quantile(g,q)) for q in (.1,.5,.9)])
                support_summary[split]=groups
                print(json.dumps({'split':split,'support':groups}),flush=True)
            write_json(a.output/'one_step_diagnostics.json',summary)
            write_json(a.output/'support_diagnostics.json',support_summary)
            suite=physical['source_manifest']['task_contract']['suite'];suitepath=verified(suite['path'],suite['sha256'])
            cases=json.loads(suitepath.read_text())['cases']
            launch=torch.tensor([x['launch']['position_m']+x['launch']['velocity_mps'] for x in cases],device='cuda')
            result=validate(base,model,data,torch.tensor(inputs['split']['validation_selected'],device='cuda'),launch)
            write_json(a.output/'rollout_validation.json',result)
        # Explicit gradient check on frozen model parameters, outside no_grad.
        probe=features[ids['validation'][:8]].clone().requires_grad_()
        delta,_=model(probe,torch.ones(8,device='cuda',dtype=torch.bool))
        grad=torch.autograd.grad(delta.sum(),probe)[0]
        if not torch.isfinite(grad).all():raise FloatingPointError('Nonfinite input gradient')
        write_json(a.output/'inputs.json',dict(ball_receipt=receipt,physical_receipt=src,source_inputs=inputs,
            geometry_hashes=preview.geometry_hashes,surrogate_softness_m=preview.softness_m,
            motion_approximation='racket constant current relative velocity; ground/net ballistic; 161 time samples; other residual unmasked',
            gradients_finite=True,input_gradient_abs_sum=float(grad.abs().sum())))
        torch.cuda.synchronize()
        write_json(a.output/'completed.json',dict(all_optimizer_updates=0,actor_updates=0,critic_updates=0,
            wall_seconds=time.perf_counter()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            scope='One frozen offline candidate, no native physics modification or performance evaluation'))
    except BaseException as e:
        (a.output/'failure_traceback.txt').write_text(traceback.format_exc())
        write_json(a.output/'failure.json',dict(type=type(e).__name__,message=str(e)));raise


if __name__=='__main__':main()
