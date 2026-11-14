#!/usr/bin/env python3
"""Pack current/next physical and named task supervision; never reset targets."""
import argparse,hashlib,json
from pathlib import Path
import torch
from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical
from quadsmash.rl.supermimic_task_memory_schema_v1 import TaskMemorySchemaV1


def task_memory(state):
    return dict(rally=state['task'],recovery=state['recovery'],auxiliary=state['auxiliary'],paid_events=state['event_state'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    manifest=json.loads((args.archive/'manifest.json').read_text())
    schema=template=None;records=[]
    returns=torch.zeros(manifest['num_envs'],dtype=torch.float64)
    first_done=torch.zeros(manifest['num_envs'],dtype=torch.bool)
    for entry in manifest['chunks']:
        path=args.archive/entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Chunk hash mismatch')
        rows=torch.load(path,map_location='cpu',weights_only=True)['rows']
        for row in rows:
            before,after=row['before'],row['after']
            if not torch.equal(before['episode_index'],after['episode_index']):raise ValueError('Reset target leak')
            b,a=task_memory(before),task_memory(after)
            if schema is None:
                schema=TaskMemorySchemaV1.from_native(b,manifest['num_envs'])
                template=b
            values,valid=schema.encode(b);next_values,next_valid=schema.encode(a)
            record=dict(state=encode_physical(before['physical']),next_state=encode_physical(after['physical']),
                values=values,valid=valid,next_values=next_values,next_valid=next_valid,
                event_state=b['paid_events'],next_event_state=a['paid_events'],
                observation=row['observation'],requested_action=row['requested_action'],
                targets=row['final_joint_targets'],final_action=row['final_action'],
                previous_action=before['physical']['final_action'],feet=before['physical']['foot_contact'],
                next_feet=after['physical']['foot_contact'],present=before['physical']['ball_present'],
                elapsed=before['physical']['elapsed_s'],episode=before['episode_index'],
                terminated=row['terminated'],timeout=row['timeout'],reward=row['reward'],
                torque=after['auxiliary']['reward_inputs']['applied_torque_all'],
                complete_reward=row['reward_components']['complete5_terminal']>0)
            records.append({k:v.clone() for k,v in record.items()})
            first=~first_done
            returns[first]+=row['reward'][first].double()
            first_done|=row['terminated']|row['timeout']
        print(json.dumps(dict(packed_ticks=len(records),last_chunk=entry['file'])),flush=True)
    data={key:torch.stack([r[key] for r in records]) for key in records[0]}
    package=dict(schema='physical-loop-supervision-v1',task_schema=schema.manifest(),template=template,
                 source_manifest=manifest,source_archive=str(args.archive),tensors=data,
                 first_returns=returns,first_done=first_done)
    destination=args.output/'data.pt';torch.save(package,destination)
    result=dict(path=str(destination),sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                ticks=len(records),transitions=len(records)*manifest['num_envs'],
                first_completed=int(first_done.sum()),native_return_mean=float(returns.mean()),
                task_width=schema.width,task_fields=len(schema.fields),optimizer_updates=0)
    (args.output/'completed.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
