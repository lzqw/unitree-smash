#!/usr/bin/env python3
"""Saved-state CPU execution parity and input-gradient audit; no Isaac steps."""
import argparse
import hashlib
import json
import traceback
from pathlib import Path
import torch

from quadsmash.control.g07_shared_target_residual_v1 import make_g07_residual_controller
from quadsmash.rl.supermimic_g07_memory_preview_v1 import (
    G07MemoryPreviewV1, restore_captured_execution, clone_execution_graph)
from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1


def freeze_modules(obj, seen=None):
    seen=set() if seen is None else seen
    if id(obj) in seen:return
    seen.add(id(obj))
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        for p in obj.parameters():p.requires_grad_(False)
    elif isinstance(obj,dict):
        for v in obj.values():freeze_modules(v,seen)
    elif isinstance(obj,(list,tuple)):
        for v in obj:freeze_modules(v,seen)
    elif hasattr(obj,'__dict__') and not isinstance(obj,(type,torch.Tensor)):
        import types
        if isinstance(obj,(types.ModuleType,types.FunctionType,types.MethodType)):return
        for v in vars(obj).values():freeze_modules(v,seen)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ticks',type=int,nargs='+',default=[0,190,200,260,300,400,700])
    p.add_argument('--steps',type=int,default=1)
    p.add_argument('--device',default='cpu')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    manifest=json.loads((args.archive/'manifest.json').read_text())
    suite_path=Path(__file__).resolve().parents[1]/manifest['task_contract']['suite']['path']
    if hashlib.sha256(suite_path.read_bytes()).hexdigest()!=manifest['task_contract']['suite']['sha256']:
        raise ValueError('Known launch suite hash mismatch')
    suite=json.loads(suite_path.read_text())
    cases=suite['cases'] if isinstance(suite,dict) else suite
    launch=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in cases])
    source=Path(manifest['source_checkpoint'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=manifest['source_sha256']:
        raise ValueError('Source hash mismatch')
    controller=make_g07_residual_controller(torch.load(source,weights_only=False),manifest['num_envs'],
        torch.device(args.device),'/home/autodrive/LZQW/unitree_smash')
    freeze_modules(controller)
    preview=G07MemoryPreviewV1(controller)
    preview.mixer.to(args.device)
    results=[]
    for tick in args.ticks:
        if tick//32 >= len(manifest['chunks']):
            raise ValueError(f'Tick {tick} is not yet in the committed archive manifest')
        chunk=args.archive/manifest['chunks'][tick//32]['file']
        row=torch.load(chunk,map_location='cpu',weights_only=True)['rows'][tick%32]
        full_state=PhysicalImaginationStateV1.from_record(row,executor_template=controller,
                                                        launch=launch,device=args.device)
        memory=full_state.executor_memory
        old=clone_execution_graph(memory)
        obs=row['observation'].to(args.device).clone();action=row['requested_action'].to(args.device).clone()
        default=row['before']['physical']['joint_default'][0].to(args.device)
        limits=row['before']['auxiliary']['command_limits'].to(args.device)
        expected=old.apply_residual({'policy':obs},action,default,limits)
        obs.requires_grad_(True);action.requires_grad_(True)
        candidate,branch,base=preview.forward(obs,action,default,limits,memory)
        error=float((candidate.final_normalized-expected).abs().max())
        target=limits[:,0]+.5*(expected+1)*(limits[:,1]-limits[:,0])
        target_error=float((candidate.final_joint_targets-target).abs().max())
        recorded_target_error=float((candidate.final_joint_targets-row['final_joint_targets'].to(args.device)).abs().max())
        if args.device.startswith('cuda'):
            torch.testing.assert_close(candidate.final_joint_targets,row['final_joint_targets'].to(args.device),atol=2e-6,rtol=0)
        torch.testing.assert_close(candidate.final_joint_targets,target,atol=2e-6,rtol=0)
        objective=candidate.final_joint_targets.square().mean()
        # Synthetic clock continuation tests autograd ownership only. It is
        # explicitly NOT a physical rollout, state prediction or task score.
        continuing=branch
        for step in range(1,args.steps):
            next_obs=obs.clone()
            next_obs[:,775]=obs[:,775]+.01*step
            next_obs[:,776:794]=candidate.final_normalized
            candidate,continuing,_=preview.forward(next_obs,action,default,limits,continuing)
            objective=objective+candidate.final_joint_targets.square().mean()
        obs_grad,action_grad=torch.autograd.grad(objective,(obs,action))
        if not torch.isfinite(obs_grad).all() or not torch.isfinite(action_grad).all():
            raise ValueError('Nonfinite execution gradient')
        result=dict(tick=tick,command_error=error,target_error=target_error,
                    observation_gradient_norm=float(obs_grad.norm()),action_gradient_norm=float(action_grad.norm()),
                    base_requires_grad=base.requires_grad,synthetic_memory_steps=args.steps,
                    device=args.device,preview_vs_recorded_gpu_target_error=recorded_target_error)
        results.append(result)
        print(json.dumps(result),flush=True)
        (args.output/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    (args.output/'completed.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
