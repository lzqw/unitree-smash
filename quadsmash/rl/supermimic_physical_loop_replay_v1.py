"""Native supervision windows and complete CURRENT imagination starts."""
import torch
from .supermimic_physical_dataset_v1 import encode_physical,valid_windows
from .supermimic_physical_world_v2 import sample_event_windows
from .supermimic_physical_capture_v1 import tensor_tree_copy


def row_tensors(row,schema):
    def memory(record):
        return dict(rally=record['task'],recovery=record['recovery'],auxiliary=record['auxiliary'],paid_events=record['event_state'])
    b,a=row['before'],row['after'];bv,bm=schema.encode(memory(b));av,am=schema.encode(memory(a))
    return dict(state=encode_physical(b['physical']),next_state=encode_physical(a['physical']),
        values=bv,valid=bm,next_values=av,next_valid=am,event_state=b['event_state'],next_event_state=a['event_state'],
        targets=row['final_joint_targets'],previous_action=b['physical']['final_action'],
        feet=b['physical']['foot_contact'],next_feet=a['physical']['foot_contact'],present=b['physical']['ball_present'],
        elapsed=b['physical']['elapsed_s'],episode=b['episode_index'],terminated=row['terminated'],timeout=row['timeout'],
        torque=a['auxiliary']['reward_inputs']['applied_torque_all'],complete_reward=row['reward_components']['complete5_terminal']>0)


def current_start(row):
    # Drop ALL future labels, native future commands and event traces.
    return tensor_tree_copy({key:row[key] for key in ('before','executor_before','observation')},cpu=True)


class PhysicalLoopReplay:
    def __init__(self,package,schema,*,capacity_ticks=1856):
        self.schema=schema;self.count=len(package['tensors']['state']);self.capacity=capacity_ticks
        required=('state','next_state','values','valid','next_values','next_valid','event_state','next_event_state',
                  'targets','previous_action','feet','next_feet','present','elapsed','episode','terminated','timeout','torque','complete_reward')
        self.data={key:torch.empty((capacity_ticks,*package['tensors'][key].shape[1:]),dtype=package['tensors'][key].dtype)
                   for key in required}
        for key,tensor in self.data.items():tensor[:self.count].copy_(package['tensors'][key])
        self.generator=torch.Generator().manual_seed(20261114)
        self.new_starts=[]

    def append(self,row):
        if self.count>=self.capacity:raise RuntimeError('Finite replay capacity reached; no hidden reset/discard')
        record=row_tensors(row,self.schema)
        for key,tensor in self.data.items():tensor[self.count].copy_(record[key].detach().cpu())
        self.count+=1
        elapsed=row['before']['physical']['elapsed_s']
        # State sampling preference, never a runtime control gate/case list.
        if ((elapsed>=2.2)&(elapsed<=2.8)).any():
            self.new_starts.append(current_start(row))
            self.new_starts=self.new_starts[-8:]

    def window(self,steps=16,batch=128,iteration=0,device='cuda:0'):
        d={k:v[:self.count] for k,v in self.data.items()}
        valid=valid_windows(d['episode'],d['terminated'],d['timeout'],steps)
        pools={'uniform':valid.nonzero()}
        for i in range(8):
            event=(d['next_event_state'][...,i]>d['event_state'][...,i]).unfold(0,steps,1).any(-1)&valid
            if event.any():pools[str(i)]=event.nonzero()
        chosen,counts=sample_event_windows(pools,batch,self.generator,iteration)
        t,l=chosen.unbind(1);at=t[None]+torch.arange(steps)[:,None]
        window={k:v[at,l[None]].to(device) for k,v in d.items()}
        return window,l.to(device),counts
