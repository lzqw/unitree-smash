#!/usr/bin/env python3
"""Read-only dual-head gradient diagnostic on native starts. Never steps an optimizer.

Uses a truncated task return with ZERO bootstrap, explicitly not a trained
critic objective or a physical finite-difference counterfactual.
"""
import argparse,json,time,traceback
from pathlib import Path
import torch
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop,tensor_digest,set_actor_trainable,APPROACH_NAMES,RECOVERY_NAMES
from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
from quadsmash.rl.supermimic_horizon_learner_v2 import frozen
from quadsmash.rl.supermimic_timebase_v2 import Timebase,bptt_return
from quadsmash.rl.supermimic_policy_execution_v1 import policy_forward,recovery_gate,contract


def read_tick(archive,manifest,tick):
    offset=0
    for chunk in manifest['chunks']:
        if offset<=tick<offset+chunk['ticks']:
            return torch.load(archive/chunk['file'],map_location='cpu',weights_only=True)['rows'][tick-offset]
        offset+=chunk['ticks']
    raise ValueError('Native tick absent from archive')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('archive','physical-checkpoint','ball-checkpoint','task-checkpoint','safety-mapping','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--approach-tick',type=int,default=270);p.add_argument('--recovery-tick',type=int,default=460)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(20261117)
    stage='LOAD';results={}
    try:
        actor,adapter,template,launch,manifest=load_physical_loop(**{k:v for k,v in vars(a).items() if k not in ('output','approach_tick','recovery_tick')})
        if manifest.get('policy_execution')!=contract():raise ValueError('Archive is not the current causal execution contract')
        actor.eval();adapter.eval()
        actor_before=tensor_digest(actor.state_dict().items());world_before=tensor_digest(adapter.state_dict().items())
        if manifest.get('actor_state_sha256')!=actor_before:raise ValueError('Archive actor identity mismatch')
        cfg=Timebase(**json.loads(Path('configs/supermimic_task_horizon_v2.json').read_text()))
        for head,tick,names in (('recovery',a.recovery_tick,RECOVERY_NAMES),('approach',a.approach_tick,APPROACH_NAMES)):
            stage=head.upper();set_actor_trainable(actor,names);actor.zero_grad(set_to_none=True)
            row=read_tick(a.archive,manifest,tick)
            state=PhysicalImaginationStateV1.from_record(row,executor_template=template,launch=launch,device='cuda:0')
            rewards=[];dones=[];commands=[];gates=[];started=time.perf_counter()
            with frozen(adapter):
                for step in range(cfg.imagination_steps):
                    gate=(recovery_gate(state.task_memory['rally'],state.observation).float() if head=='recovery' else actor.approach_gate(state.observation).reshape(-1))
                    gates.append(gate.detach())
                    action=policy_forward(actor,state.observation,state.task_memory['rally'],sample=False)[0]
                    if step==0:torch.testing.assert_close(action,row['requested_action'].cuda(),atol=2e-6,rtol=0)
                    state,reward,done,detail=adapter(state,action)
                    if step==0:torch.testing.assert_close(detail['final_joint_targets'],row['final_joint_targets'].cuda(),atol=2e-6,rtol=0)
                    rewards.append(reward);dones.append(done);commands.append(detail['final_joint_targets'])
                    if step%8==0:print(json.dumps(dict(head=head,imagination_step=step+1,active_lanes=int((gate>.25).sum()))),flush=True)
                mask=(torch.stack(gates)>.25).any(0).detach()
                if not mask.any():raise ValueError(head+' has no effective gate in the fixed 64-step probe')
                ret=bptt_return(torch.stack(rewards),torch.stack(dones),torch.zeros_like(rewards[-1]),cfg.gamma)
                objective=-ret.reshape(-1)[mask].mean()
                # Autograd sensitivity of final executed targets, not residual alone.
                parameters=[p for p in actor.parameters() if p.requires_grad]
                sensitivity=torch.autograd.grad(torch.stack(commands)[:,mask].square().mean(),parameters,retain_graph=True,allow_unused=True)
                command_gradient=sum(float(g.detach().square().sum()) for g in sensitivity if g is not None)**.5
                objective.backward()
            gradients={n:dict(norm=None if p.grad is None else float(p.grad.norm()),finite=p.grad is not None and bool(torch.isfinite(p.grad).all())) for n,p in actor.named_parameters() if p.requires_grad}
            finite=all(r['finite'] for r in gradients.values())
            nonzero=any((r['norm'] or 0)>0 for r in gradients.values())
            if tensor_digest(actor.state_dict().items())!=actor_before or tensor_digest(adapter.state_dict().items())!=world_before:
                raise RuntimeError('Read-only probe mutated parameters')
            results[head]=dict(tick=tick,steps=cfg.imagination_steps,gate_active_lane_steps=int((torch.stack(gates)>.25).sum()),
                finite_task_gradient=finite,nonzero_task_gradient=nonzero,final_command_gradient_norm=command_gradient,
                task_gradients=gradients,seconds=time.perf_counter()-started,actor_updates=0,world_updates=0,critic_updates=0)
            (a.output/'progress.json').write_text(json.dumps(results,indent=2))
            if not finite or not nonzero or command_gradient<=0:raise ValueError(head+' gradient probe failed')
            del rewards,dones,commands,state,objective,ret,sensitivity
        report=dict(results=results,actor_sha256=actor_before,world_sha256=world_before,weights_unchanged=True,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),optimizer_steps=0,
            scope='native-start model gradient only; zero-bootstrap truncated task objective',
            physical_direction_consistency='unverified: no same-prefix native +/- perturbation',native_performance_improvement='not tested')
        (a.output/'result.json').write_text(json.dumps(report,indent=2))
    except BaseException as e:
        (a.output/'failure_traceback.txt').write_text(traceback.format_exc())
        (a.output/'failure.json').write_text(json.dumps(dict(stage=stage,type=type(e).__name__,message=str(e),completed_heads=results,optimizer_steps=0),indent=2))
        raise

if __name__=='__main__':main()
