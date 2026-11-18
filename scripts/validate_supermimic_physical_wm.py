#!/usr/bin/env python3
"""Compare frozen WM candidates on identical held-out native episodes, no training."""
import argparse,hashlib,json
from pathlib import Path
import torch
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop
from quadsmash.rl.supermimic_physical_dataset_v1 import valid_windows
from quadsmash.rl.supermimic_wm_validation_v1 import validate_windows,command_sensitivity,validation_stages,event_aligned_windows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','physical-checkpoint','ball-checkpoint','task-checkpoint','safety-mapping','packed-data','split-manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--candidate',action='append',default=[],help='NAME=training checkpoint, frozen world state')
    p.add_argument('--action-ball-checkpoint',type=Path,
        help='Optional offline-fitted swept-ball candidate; NOT native action-response certification')
    p.add_argument('--g07-execution-root',type=Path,
        help='Explicit root containing the measured G07 execution contract and diagnostics')
    p.add_argument('--device',default='cpu');p.add_argument('--max-windows',type=int,default=128)
    a=p.parse_args()
    if a.max_windows<1:raise ValueError('max-windows must be positive')
    if a.output.exists():raise FileExistsError(a.output)
    split=json.loads(a.split_manifest.read_text());train=set(split['train_lanes']);val=set(split['validation_lanes'])
    if not train or not val or train&val:raise ValueError('Explicit disjoint whole-case split required')
    data=torch.load(a.packed_data,map_location='cpu',weights_only=True)['tensors']
    if min(train|val)<0 or max(train|val)>=data['episode'].shape[1]:raise ValueError('Split lane outside archive')
    valid=valid_windows(data['episode'],data['terminated'],data['timeout'],64)
    if 'valid_lengths' in data: valid &= torch.arange(len(valid))[:,None]+64<=data['valid_lengths'][None]
    allowed=torch.zeros(valid.shape[1],dtype=torch.bool);allowed[list(val)]=True;valid&=allowed[None]
    # Fixed time/lane selection independent of model outcome.
    pairs=valid.nonzero()[::64][:a.max_windows]
    if not len(pairs):raise ValueError('No held-out 64-transition windows')
    t,lanes=pairs.unbind(1);at=t[None]+torch.arange(64)[:,None]
    window={k:v[at,lanes[None]].to(a.device) for k,v in data.items() if v.ndim>=2 and v.shape[:2]==data['episode'].shape}
    _,adapter,_,launch,_=load_physical_loop(archive=a.archive,physical_checkpoint=a.physical_checkpoint,
        ball_checkpoint=a.ball_checkpoint,task_checkpoint=a.task_checkpoint,safety_mapping=a.safety_mapping,
        action_ball_checkpoint=a.action_ball_checkpoint,device=a.device,
        execution_root=a.g07_execution_root)
    baseline={k:v.detach().clone() for k,v in adapter.state_dict().items()}
    results={}; candidates=[('initial',None)]+[x.split('=',1) for x in a.candidate]
    aligned=event_aligned_windows(data,val,maximum=a.max_windows)
    if len({x[0] for x in candidates})!=len(candidates):raise ValueError('Duplicate candidate name')
    for name,path in candidates:
        adapter.load_state_dict(baseline,strict=True)
        if path:
            payload=torch.load(path,map_location=a.device,weights_only=False)
            # Training checkpoints wrap the adapter under ``world``.  The
            # bounded native-response fitter writes a standalone world.pt
            # with ``model`` instead.  Accept both explicit formats while
            # refusing an unrecognised artifact, so validation cannot silently
            # inspect the wrong state dictionary.
            if isinstance(payload,dict) and 'world' in payload:
                world_state=payload['world']
            elif isinstance(payload,dict) and payload.get('schema') in (
                    'physical-imagination-adapter-v1', 'physical-imagination-adapter-v2') and 'model' in payload:
                world_state=payload['model']
            elif isinstance(payload,dict) and 'model' in payload:
                world_state=payload['model']
            else:
                raise ValueError('Candidate checkpoint has no explicit world/model state')
            if payload.get('schema') == 'physical55-separated-contact-transitions-v2':
                adapter.physical_model.load_state_dict(world_state,strict=True)
            else:
                adapter.load_state_dict(world_state,strict=True)
        adapter.eval()
        results[name]=dict(checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else adapter.manifest['checkpoint_sha256'],
                           errors=validate_windows(adapter,window,launch[lanes.to(a.device)]),
                           command_sensitivity=command_sensitivity(adapter,window,launch[lanes.to(a.device)]))
        groups={}
        for stage,mask in validation_stages(window).items():
            if stage=='all':continue
            groups[stage]=dict(windows=int(mask.sum()),errors=None)
            if mask.any():
                subset={k:v[:,mask] for k,v in window.items()}
                groups[stage]['errors']=validate_windows(adapter,subset,launch[lanes.to(a.device)][mask])
        results[name]['native_stage_groups']=groups
        aligned_results={}
        for offset,group in aligned.items():
            starts=group['starts'];record={k:v for k,v in group.items() if k!='starts'}
            record.update(windows=len(starts),starts=starts.tolist(),errors=None)
            if len(starts):
                times,ids=starts.unbind(1);indices=times[None]+torch.arange(64)[:,None]
                subset={k:v[indices,ids[None]].to(a.device) for k,v in data.items() if v.ndim>=2 and v.shape[:2]==data['episode'].shape}
                record['errors']=validate_windows(adapter,subset,launch[ids.to(a.device)])
            aligned_results[offset]=record
        results[name]['event_aligned']=aligned_results
    output=dict(results=results,window_starts=pairs.tolist(),split=split,
        packed_sha256=hashlib.sha256(a.packed_data.read_bytes()).hexdigest(),native_simulation=False,
        limitations=['Split manifest must match fitting provenance; this script cannot certify unseen training data.',
                     'Recorded final-command autoregression is not closed-loop policy evaluation.',
                     'No PhysX action perturbation evidence; native success is not inferred.'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2)+'\n')
if __name__=='__main__':main()
