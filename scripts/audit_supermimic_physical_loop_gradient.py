#!/usr/bin/env python3
"""Actual loaded physical-loop BPTT audit, without changing any policy weight."""
import argparse,json,time,traceback
from pathlib import Path
import torch
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop,tensor_digest,APPROACH_NAMES
from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
from quadsmash.rl.supermimic_horizon_learner_v2 import frozen
from quadsmash.rl.supermimic_timebase_v2 import Timebase,bptt_return
from quadsmash.rl.supermimic_v1 import MLP


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','physical-checkpoint','ball-checkpoint','task-checkpoint','safety-mapping','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--tick',type=int,default=240)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(20261114)
    model_args={k:v for k,v in vars(args).items() if k not in ('output','tick')}
    stage='LOAD'
    try:
        actor,adapter,template,launch,manifest=load_physical_loop(**model_args)
        (args.output/'model_manifest.json').write_text(json.dumps(adapter.manifest,indent=2)+'\n')
        chunk=args.archive/manifest['chunks'][args.tick//32]['file']
        row=torch.load(chunk,map_location='cpu',weights_only=True)['rows'][args.tick%32]
        state=PhysicalImaginationStateV1.from_record(row,executor_template=template,launch=launch,device='cuda:0')
        initial_obs=state.observation.clone()
        before_model=tensor_digest(adapter.state_dict().items());before_actor=tensor_digest(actor.state_dict().items())
        context=adapter.critic_input(state)
        if not torch.isfinite(context).all():raise ValueError('Nonfinite contextual critic input')
        critic=MLP(context.shape[-1],(512,256,128),1).cuda()
        config=Timebase(**json.loads(Path('configs/supermimic_task_horizon_v2.json').read_text()))
        rewards=[];dones=[];commands=[];step_records=[];started=time.perf_counter()
        stage='ROLLOUT'
        with frozen(adapter,critic):
            for i in range(config.imagination_steps):
                action=actor(state.observation,sample=False)[0]
                if i==0:
                    torch.testing.assert_close(action,row['requested_action'].cuda(),atol=2e-6,rtol=0)
                state,reward,done,detail=adapter(state,action)
                if i==0:
                    torch.testing.assert_close(detail['final_joint_targets'],row['final_joint_targets'].cuda(),atol=2e-6,rtol=0)
                rewards.append(reward);dones.append(done);commands.append(detail['final_joint_targets'])
                context=adapter.critic_input(state)
                if not torch.isfinite(context).all():raise ValueError('Nonfinite critic input at step '+str(i))
                step_records.append(dict(step=i+1,reward_mean=float(reward.detach().mean()),
                    done_mean=float(done.detach().mean()),observation_max=float(state.observation.detach().abs().max())))
                if i%8==0:print(json.dumps(step_records[-1]),flush=True)
            ret=bptt_return(torch.stack(rewards),torch.stack(dones),critic(context),config.gamma)
            stage='BACKWARD'
            (-ret.mean()).backward()
        gradients={name:dict(norm=float(param.grad.norm()) if param.grad is not None else None,
                             finite=bool(torch.isfinite(param.grad).all()) if param.grad is not None else None)
                   for name,param in actor.named_parameters() if param.requires_grad}
        if not all(v['finite'] for v in gradients.values()):raise ValueError('Missing or nonfinite actor gradients')
        if not any(v['norm']>0 for v in gradients.values()):raise ValueError('All approach gradients are zero')
        assert before_model==tensor_digest(adapter.state_dict().items()),'World parameters changed in BPTT'
        assert before_actor==tensor_digest(actor.state_dict().items()),'Audit unexpectedly updated actor'
        assert all(p.grad is None for p in adapter.parameters()),'Frozen model accumulated parameter gradients'
        torch.testing.assert_close(initial_obs,row['observation'].cuda(),atol=0,rtol=0)
        result=dict(steps=config.imagination_steps,imagination_s=config.imagination_s,actor_updates=0,
                    native_steps=0,imagined_return_mean=float(ret.detach().mean()),gradients=gradients,
                    model_unchanged=True,actor_unchanged=True,critic_context_width=context.shape[-1],
                    elapsed_seconds=time.perf_counter()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                    step_records=step_records)
        (args.output/'completed.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    except BaseException as error:
        text=traceback.format_exc();(args.output/'failure_traceback.txt').write_text(text)
        (args.output/'failure.json').write_text(json.dumps(dict(stage=stage,type=type(error).__name__,message=str(error)),indent=2)+'\n')
        raise


if __name__=='__main__':main()
