"""Explicit physical imagination state; no archived future controls/events.

NaN timestamps remain NaN in state. A critic representation uses value/valid
pairs so an unset timestamp is not silently made a real zero timestamp.
This schema is independent of model prediction error and native scoring.
"""
from dataclasses import dataclass
from typing import Any
import torch

from .supermimic_physical_dataset_v1 import encode_physical
from .supermimic_imagination_contract_v1 import require_physical_start
from .supermimic_g07_memory_preview_v1 import clone_execution_graph
from .supermimic_physical_capture_v1 import restore_archive_execution


def tensor_copy(tree,device):
    if isinstance(tree,torch.Tensor):return tree.to(device).clone()
    if isinstance(tree,dict):return {k:tensor_copy(v,device) for k,v in tree.items()}
    if isinstance(tree,list):return [tensor_copy(v,device) for v in tree]
    return tree


@dataclass(frozen=True)
class PhysicalImaginationStateV1:
    schema = 'physical-executor-task-history-clock-v1'
    physical: torch.Tensor
    executor_memory: Any
    task_memory: dict
    observation_history: torch.Tensor
    observation: torch.Tensor
    clock_goal_context: dict

    @classmethod
    def from_record(cls,row,*,executor_template,launch,device='cpu'):
        before=row['before']
        require_physical_start(before,row['executor_before'])
        # The input record is a CURRENT start. Never consult its after/base/
        # final targets. Native substep sequences are supervision, not inputs.
        p=tensor_copy(before['physical'],device)
        n=len(p['joint_position'])
        if launch.shape!=(n,6):raise ValueError('Known causal launch input must be [batch,6]')
        current_history=before['history'].to(device).clone()
        observation=row['observation'].to(device).clone()
        torch.testing.assert_close(current_history.flatten(1),observation[:,:759],atol=0,rtol=0)
        memory=dict(rally=tensor_copy(before['task'],device),
                    recovery=tensor_copy(before['recovery'],device),
                    auxiliary=tensor_copy(before['auxiliary'],device),
                    paid_events=before['event_state'].to(device).clone())
        context={key:value for key,value in p.items() if key.startswith('goal_')}
        context.update(joint_default=p['joint_default'],
            limits=before['auxiliary']['command_limits'].to(device).clone(),
            launch=launch.to(device).clone(),elapsed=p['elapsed_s'],
            ball_present=p['ball_present'],foot_contact=p['foot_contact'],
            final_action=p['final_action'],episode_index=before['episode_index'].to(device).clone(),
            history_tick=before['history_tick'].to(device).clone())
        if '_actor_lane_mask' in row:
            context['actor_lane_mask']=row['_actor_lane_mask'].to(device).bool().clone()
        executor=restore_archive_execution(executor_template,row['executor_before'],device=device)
        return cls(encode_physical(p),executor,memory,current_history,observation,context)

    def branch(self):
        device=self.physical.device
        return type(self)(self.physical.clone(),clone_execution_graph(self.executor_memory),
            tensor_copy(self.task_memory,device),self.observation_history.clone(),
            self.observation.clone(),tensor_copy(self.clock_goal_context,device))


def finite_value_and_validity(value):
    if not isinstance(value,torch.Tensor):raise TypeError('Context must use explicit tensors')
    valid=torch.isfinite(value) if value.is_floating_point() else torch.ones_like(value,dtype=torch.bool)
    numeric=torch.where(valid,value,torch.zeros_like(value)).float()
    return torch.cat((numeric.reshape(len(value),-1),valid.float().reshape(len(value),-1)),-1)
