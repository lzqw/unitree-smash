#!/usr/bin/env python3
"""Measure policy changes through the frozen executor at identical archived control states."""
import argparse
import copy
import json
from pathlib import Path

import torch

from scripts.index_supermimic_success_archive import archive_actor_sha256
from quadsmash.rl.supermimic_paired_evaluation_v1 import parameter_delta
from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest
from quadsmash.rl.supermimic_physical_start_cache_v1 import load_start_cache, APPROACH_TICKS, RECOVERY_TICKS
from quadsmash.rl.supermimic_policy_execution_v1 import policy_forward, recovery_gate, contract
from scripts.evaluate_supermimic_fresh_wm_panel import digest


def preview(policy, executor, state):
    obs = state.observation
    rally = state.task_memory['rally']
    req = policy_forward(policy,obs,rally,sample=False)[0]
    gate = recovery_gate(rally,obs)
    actor = policy.actor
    z = actor.normalizer(obs)
    mean = actor.base(z) + gate[:,None] * actor.recovery(z) * actor.recovery_scale + actor.gate(obs) * actor.approach(z) * actor.approach_scale
    torch.testing.assert_close(req,mean.clamp(-1,1),rtol=0,atol=0)
    context = state.clock_goal_context
    command, _, _ = executor.forward(obs,req,context['joint_default'][0],context['limits'],state.executor_memory)
    return mean, req, command.final_normalized, command.final_joint_targets


@torch.no_grad()
def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('archive','safety-mapping','after','output'):
        cli.add_argument('--'+name,type=Path,required=True)
    cli.add_argument('--before',type=Path,help='Defaults to the hash-verified original archive Actor')
    a = cli.parse_args()
    torch.set_num_threads(3)
    def checkpoint(path):
        return torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    after = checkpoint(a.after)
    meta = after['metadata']
    wm = meta['physical_imagination']
    paths = wm['checkpoint_paths']
    initial, adapter, template, launch, manifest = load_physical_loop(archive=a.archive,physical_checkpoint=paths['physical'],
        ball_checkpoint=paths['ball'],task_checkpoint=paths['task'],action_ball_checkpoint=paths.get('action_ball'),
        safety_mapping=a.safety_mapping,execution_root=wm['g07_execution_root'],device='cpu')
    if wm['checkpoint_sha256'] != adapter.manifest['checkpoint_sha256'] or manifest['policy_execution']!=contract():
        raise ValueError('Changed physical dependencies or original action contract')
    def load_actor(ck,policy):
        metadata = ck['metadata']
        if metadata['policy_execution']!=contract() or metadata['task_contract']!=manifest['task_contract'] or metadata['physical_imagination']['g07_execution_root']!=wm['g07_execution_root']:
            raise ValueError('Checkpoint has different execution or task')
        policy.load_state_dict(ck['actor'],strict=True)
    first = initial.eval()
    second = copy.deepcopy(first)
    if a.before is not None:
        load_actor(checkpoint(a.before),first)
    elif tensor_digest(first.state_dict().items())!=archive_actor_sha256(manifest):
        raise ValueError('The initial actor differs from the actor that collected the archive')
    load_actor(after,second)
    states=load_start_cache(a.archive,manifest)
    fields=('request','final_normalized','final_target_rad')
    result={}
    for head,ticks in [('approach',APPROACH_TICKS),('recovery',RECOVERY_TICKS)]:
        parts={key:[] for key in (*fields,'kl','gate','approach_gate')}
        for tick in ticks:
            state=PhysicalImaginationStateV1.from_record(states[tick],executor_template=template,launch=launch,device='cpu')
            x,*u=preview(first,adapter.executor,state.branch())
            y,*v=preview(second,adapter.executor,state.branch())
            l0,l1=(p.log_std.clamp(-8,1).double() for p in (first,second))
            diff=2*(l0-l1)
            parts['kl'].append(.5*(torch.expm1(diff)-diff+(x.double()-y.double()).square()*(-2*l1).exp()))
            parts['gate'].append(recovery_gate(state.task_memory['rally'],state.observation))
            parts['approach_gate'].append(first.approach_gate(state.observation).reshape(-1))
            for k,old,new in zip(fields,u,v,strict=True):
                parts[k].append(new.double()-old.double())
        parts={k:torch.cat(v) for k,v in parts.items()}
        gate=parts['gate']
        approach_gate=parts['approach_gate']
        groups={}
        for group,mask in [('all',torch.ones_like(gate)),('recovery_active',gate),('recovery_inactive',~gate),
                           ('approach_zero',approach_gate==0),('approach_up_to_quarter',(approach_gate>0)&(approach_gate<=.25)),
                           ('approach_above_quarter',approach_gate>.25)]:
            stats=dict(rows=int(mask.sum()))
            if mask.any():
                stats['gaussian_kl_per_dim']=float(parts['kl'][mask].mean())
                for k in fields:
                    vec=parts[k][mask]
                    stats[k]=dict(rms=float(vec.square().mean().sqrt()),max_abs=float(vec.abs().max()),
                                  changed_rows=int(vec.ne(0).any(-1).sum()))
            groups[group]=stats
        result[head]=dict(ticks=ticks,groups=groups)
    orig,end=first.state_dict(),second.state_dict()
    if set(orig)!=set(end):raise ValueError('Incompatible actor parameters')
    report=dict(groups=result,parameter_delta=parameter_delta(orig,end),
                changed_tensors=[k for k in orig if not torch.equal(orig[k],end[k])],
                before_actor_sha256=tensor_digest(orig.items()),after_actor_sha256=tensor_digest(end.items()),
                before_checkpoint_sha256=digest(a.before) if a.before else manifest['source_sha256'],
                after_checkpoint_sha256=digest(a.after),archive_manifest_sha256=digest(a.archive/'manifest.json'),
                scope='Both policies and G07 previews use the same recorded state, command limits, latch and controller memory in each lane')
    with a.output.open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False)
        f.write('\n')


if __name__=='__main__':main()
