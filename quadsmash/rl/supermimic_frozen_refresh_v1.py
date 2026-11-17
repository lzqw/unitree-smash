"""Collect the first complete frozen-policy episode per lane, without forced reset.

Partial suffixes and later prefixes may train WM but never become MC labels.
Only a bounded number of full G07 starts is retained (one per calibration stage).
"""
import torch
from .supermimic_physical_capture_v1 import tensor_tree_copy
from .supermimic_physical_loop_replay_v1 import current_start, row_tensors

class FrozenRefresh:
    def __init__(self,lanes,episode_ticks,gamma,policy_id,schema,stage_ticks=(0,96,192,288,384,480,576,672)):
        self.lanes=lanes; self.episode_ticks=episode_ticks; self.gamma=gamma
        self.policy_id=policy_id; self.schema=schema; self.stages=stage_ticks
        self.ids=torch.full((lanes,),-1,dtype=torch.long)
        self.lengths=torch.zeros(lanes,dtype=torch.long);self.complete=torch.zeros(lanes,dtype=torch.bool)
        self.rewards=torch.zeros(episode_ticks,lanes,dtype=torch.float64)
        self.dones=torch.zeros(episode_ticks,lanes,dtype=torch.bool)
        self.starts={};self.compact=None;self.calls=0

    def append(self,row):
        b=row['before']; episode=b['episode_index'].detach().cpu().long()
        elapsed=b['elapsed_tick'].detach().cpu().long(); done=(row['terminated']|row['timeout']).detach().cpu().bool()
        new=(self.ids<0)&(elapsed==0);self.ids[new]=episode[new]
        active=(self.ids==episode)&~self.complete
        if ((self.ids>=0)&~self.complete&~active).any():raise RuntimeError('Frozen episode changed without terminal evidence')
        if not torch.equal(elapsed[active],self.lengths[active]):raise RuntimeError('Frozen refresh lost or duplicated a control tick')
        if (elapsed[active]>=self.episode_ticks).any():raise RuntimeError('Frozen episode exceeded task horizon')
        record=row_tensors(row,self.schema)
        if self.compact is None:
            self.compact={k:torch.zeros((self.episode_ticks,*v.shape),dtype=v.dtype) for k,v in record.items()}
        lanes=torch.where(active)[0]; steps=elapsed[lanes]
        for k,v in record.items(): self.compact[k][steps,lanes]=v.detach().cpu()[lanes]
        self.rewards[steps,lanes]=row['reward'].detach().cpu()[lanes].double()
        self.dones[steps,lanes]=done[lanes]
        for stage in self.stages:
            mask=active&(elapsed==stage)
            if mask.any() and stage not in self.starts:
                self.starts[stage]=(dict(current_start(row),_actor_lane_mask=mask.clone()),mask.clone())
        self.lengths[active]+=1; self.complete|=active&done; self.calls+=1

    def result(self):
        if not self.complete.all(): raise RuntimeError('Incomplete frozen episodes: '+str(torch.where(~self.complete)[0].tolist()))
        rtg=torch.zeros_like(self.rewards); running=torch.zeros(self.lanes,dtype=torch.float64)
        for t in range(self.episode_ticks-1,-1,-1):
            active=t<self.lengths
            running=torch.where(active,self.rewards[t]+self.gamma*running*(~self.dones[t]),torch.zeros_like(running))
            rtg[t]=running
        return rtg,dict(policy_id=self.policy_id,complete_lanes=int(self.complete.sum()),
                        episode_ids=self.ids.tolist(),lengths=self.lengths.tolist(),native_ticks=self.calls,
                        returns=self.rewards.sum(0).tolist(),stage_ticks=list(self.starts),
                        scope='first complete episode starting after policy freeze, per lane')

    def packed_complete(self):
        self.result()
        # Each column is one complete episode. Pad only AFTER its terminal;
        # builders must honor valid_windows/terminal boundaries and never use
        # padded tails as new training windows.
        return dict(self.compact,reward=self.rewards,valid_lengths=self.lengths.clone())
