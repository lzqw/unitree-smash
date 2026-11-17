"""Collect the first complete frozen-policy episode per lane, without forced reset.

Partial suffixes and later prefixes may train WM but never become MC labels.
Only a bounded number of full G07 starts is retained (all eligible lanes at each calibration stage; optionally spooled to disk).
"""
from pathlib import Path
import torch
from .supermimic_physical_capture_v1 import tensor_tree_copy
from .supermimic_physical_loop_replay_v1 import current_start, row_tensors

class FrozenRefresh:
    def __init__(self,lanes,episode_ticks,gamma,policy_id,schema,stage_ticks=(0,96,192,288,384,480,576,672),snapshot_dir=None,start_encoder=None):
        self.lanes=lanes; self.episode_ticks=episode_ticks; self.gamma=gamma
        self.policy_id=policy_id; self.schema=schema; self.stages=stage_ticks
        self.ids=torch.full((lanes,),-1,dtype=torch.long)
        self.lengths=torch.zeros(lanes,dtype=torch.long);self.complete=torch.zeros(lanes,dtype=torch.bool)
        self.rewards=torch.zeros(episode_ticks,lanes,dtype=torch.float64)
        self.dones=torch.zeros(episode_ticks,lanes,dtype=torch.bool)
        self.starts={};self.compact=None;self.calls=0;self.start_encoder=start_encoder
        self.snapshot_dir=Path(snapshot_dir) if snapshot_dir is not None else None
        if self.snapshot_dir is not None:self.snapshot_dir.mkdir(parents=True,exist_ok=False)
        self.stage_seen={stage:torch.zeros(lanes,dtype=torch.bool) for stage in stage_ticks}

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
        snapshot=None
        for stage in self.stages:
            mask=active&(elapsed==stage)&~self.stage_seen[stage]
            if mask.any():
                if snapshot is None:
                    snapshot=current_start(row)
                    if self.start_encoder is not None:
                        encoded=self.start_encoder(snapshot)
                        snapshot=tensor_tree_copy(encoded,cpu=True) if isinstance(encoded,dict) else {'critic_input':encoded.detach().cpu()}
                    elif self.snapshot_dir is not None:
                        from .supermimic_physical_run_guard_v1 import atomic_torch_save
                        path=self.snapshot_dir/f'tick_{self.calls:05d}.pt'
                        atomic_torch_save(snapshot,path);snapshot=str(path)
                self.starts.setdefault(stage,[]).append((snapshot,mask.clone()))
                self.stage_seen[stage]|=mask
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
                        stage_coverage={str(t):int(m.sum()) for t,m in self.stage_seen.items()},
                        scope='first complete episode starting after policy freeze, per lane')

    def packed_complete(self):
        self.result()
        # Each column is one complete episode. Pad only AFTER its terminal;
        # builders must honor valid_windows/terminal boundaries and never use
        # padded tails as new training windows.
        return dict(self.compact,reward=self.rewards,valid_lengths=self.lengths.clone())

    def calibration_rows(self):
        """Read one complete execution graph at a time; never merge G07 trees."""
        self.result()
        for stage in sorted(self.starts):
            for record,mask in self.starts[stage]:
                row=torch.load(record,map_location='cpu',weights_only=True) if isinstance(record,str) else record
                yield stage,dict(row,_actor_lane_mask=mask.clone()),mask
