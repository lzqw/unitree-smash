#!/usr/bin/env python3
"""Replay recorded native task calls and reward; no physics or optimizers."""
import argparse,json,hashlib
from pathlib import Path
import torch
from quadsmash.physics.native import ContactEpisodes
from quadsmash.physics.rally_rules_v4 import RallyRulesV4
from quadsmash.tasks.stand.contact_flight_recovery_v5 import ContactFlightRecoveryV5
from quadsmash.rl.supermimic_native_sequence_v1 import replay_task_sequence
from quadsmash.rl.supermimic_physical_reward_v1 import native_reward_from_record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-chunks',type=int);p.add_argument('--max-rows',type=int)
    p.add_argument('--start-chunk',type=int,default=0)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    manifest=json.loads((args.archive/'manifest.json').read_text());cfg=manifest['task_replay_config']
    n=manifest['num_envs'];racket=manifest['body_names'].index('Racket')
    rally=RallyRulesV4(n,'cpu',cfg['court'],cfg['radius'])
    recovery=ContactFlightRecoveryV5(n,'cpu',cfg['recovery_protocol'])
    contact=ContactEpisodes(n,'cpu',cfg['contact_threshold'],cfg['contact_separation'])
    total=0;maximum=0.;calls={};chunks=[]
    for entry in manifest['chunks'][args.start_chunk:][:args.max_chunks]:
        path=args.archive/entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Chunk hash mismatch')
        for row in torch.load(path,map_location='cpu',weights_only=True)['rows'][:args.max_rows]:
            initial=row['before'];final=row['after']
            for obj,values in ((rally,initial['task']),
                (recovery,initial['auxiliary']['contact_recovery']),
                (contact,initial['auxiliary']['contact_episodes'])):
                for name,value in values.items():setattr(obj,name,value.clone())
            replay_task_sequence(row['task_sequence'],rally,recovery,contact)
            for obj,values in ((rally,final['task']),
                (recovery,final['auxiliary']['contact_recovery']),
                (contact,final['auxiliary']['contact_episodes'])):
                for name,value in values.items():
                    torch.testing.assert_close(getattr(obj,name),value,atol=0,rtol=0,equal_nan=True,
                        msg=lambda msg:f'tick={initial["control_tick"]} field={type(obj).__name__}.{name}: '+msg)
            reward,paid=native_reward_from_record(row,racket,manifest['control_dt_s'])
            torch.testing.assert_close(paid,final['event_state'],atol=0,rtol=0)
            torch.testing.assert_close(reward,row['reward'],atol=1e-5,rtol=0)
            maximum=max(maximum,float((reward-row['reward']).abs().max()))
            for name,stream in row['task_sequence']['streams'].items():calls[name]=calls.get(name,0)+stream['count']
            total+=1
        chunks.append(entry)
        result=dict(checked_ticks=total,checked_transitions=total*n,max_reward_error=maximum,
                    reward_atol=1e-5,event_state_atol=0,calls=calls,chunks=chunks,
                    partial=args.start_chunk>0 or args.max_chunks is not None or args.max_rows is not None,
                    actor_updates=0,native_steps=0)
        (args.output/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('checked_ticks','max_reward_error','calls')}),flush=True)
    (args.output/'completed.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
