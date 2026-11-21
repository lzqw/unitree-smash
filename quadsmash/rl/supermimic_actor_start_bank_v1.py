"""Bounded full-executor snapshots; future outcome labels are sampling-only."""
import torch

CATEGORIES=('gate_onset','pre_contact','failure_pre','success_near')
SCHEDULE=('gate_onset',)*5+('pre_contact',)*7+('failure_pre',)*5+('success_near',)*3

class ActorStartBank:
    schema='causal-full-executor-start-bank-v1'
    def __init__(self,capacity=32,seed=20261117):
        self.capacity=capacity;self.entries=[];self.seen=0;self.last_gate={};self.last_tick={}
        self.generator=torch.Generator().manual_seed(seed)
        self.sampling_history=[]
    def append(self,start,metadata,*,stream,policy_version,native_tick):
        if 'approach_gate' not in metadata:return
        gate=metadata['approach_gate'].detach().cpu().reshape(-1)
        pre=metadata['had_contact'].detach().cpu().reshape(-1)<.5
        prior=self.last_gate.get(stream,torch.zeros_like(gate));self.last_gate[stream]=gate.clone()
        onset=pre&(((gate>0)&(gate<=.25))|((gate>.25)&(prior<=.25)))
        active=pre&(gate>.25)
        # Native collection runs under inference mode.  Tensors created by
        # boolean operations there are inference tensors and cannot later be
        # updated when terminal labels arrive.  Materialize ordinary CPU
        # tensors for the mutable mask state kept by this bank.
        def mutable_mask(value):
            # ``empty`` created while an outer inference_mode is active is
            # itself an inference tensor.  Explicitly leave that mode for
            # this small metadata allocation so delayed labels can mutate it.
            with torch.inference_mode(False):
                out=torch.empty(value.shape,dtype=torch.bool)
                out.copy_(value.to(dtype=torch.bool))
            return out
        onset=mutable_mask(onset)
        active=mutable_mask(active)
        if not (onset|active).any():return
        if not onset.any() and native_tick-self.last_tick.get(stream,-100)<4:return
        self.last_tick[stream]=native_tick;self.seen+=1
        entry=dict(start=start,stream=int(stream),policy_version=int(policy_version),native_tick=int(native_tick),
                   masks=dict(gate_onset=onset,pre_contact=active,
                              failure_pre=mutable_mask(torch.zeros_like(active)),
                              success_near=mutable_mask(torch.zeros_like(active))))
        # Reservoir avoids "last 8 lanes" and retains pre-event snapshots until
        # terminal labels arrive. No snapshot is synthesized by mixing G07 trees.
        if len(self.entries)<self.capacity:self.entries.append(entry)
        else:
            slot=int(torch.randint(self.seen,(1,),generator=self.generator))
            if slot<self.capacity:self.entries[slot]=entry
        # Gate edge bookkeeping is needed only for the current stream.
        self.last_gate={stream:self.last_gate[stream]};self.last_tick={stream:self.last_tick[stream]}
    def label_terminal(self,record,stream):
        if record.get('scoring_schema')!='historical-independent-complete5-v1':raise ValueError('Start outcome requires independent native terminal scoring')
        if record.get('ball_side_complete_except_recovery') is None:raise ValueError('Missing ball-side terminal label')
        lane=int(record['env_id']);episode=int(record['episode_index'])
        for entry in self.entries:
            if entry['stream']!=stream:continue
            before=entry['start']['before']
            if int(before['episode_index'][lane])!=episode:continue
            eligible=entry['masks']['pre_contact'][lane]
            name='success_near' if record['ball_side_complete_except_recovery'] else 'failure_pre'
            target=entry['masks'][name]
            # Some snapshots are created inside the native collector's
            # inference_mode.  Normalize the destination before the delayed
            # terminal-label write, otherwise PyTorch rejects the in-place
            # update even though the mask is logically mutable metadata.
            if target.is_inference():
                with torch.inference_mode(False):
                    mutable=torch.empty(target.shape,dtype=torch.bool)
                    mutable.copy_(target)
                entry['masks'][name]=target=mutable
            target[lane]=bool(eligible)
    def sample(self, attempt, *, eligible_lanes=None):
        requested=SCHEDULE[(attempt-1)%len(SCHEDULE)]
        checks={}
        categories=dict.fromkeys((requested,'pre_contact','gate_onset','failure_pre','success_near'))
        for actual in categories:
            entries=[entry for entry in self.entries if entry['masks'][actual].any()]
            if not entries:
                continue
            if eligible_lanes is None:
                order=[int(torch.randint(len(entries),(1,),generator=self.generator))]
            else:
                order=torch.randperm(len(entries),generator=self.generator).tolist()
            for index in order:
                entry=entries[index]
                mask=entry['masks'][actual]
                if eligible_lanes is not None:
                    key=id(entry)
                    if key not in checks:
                        checks[key]=eligible_lanes(entry['start']).detach().reshape(-1).cpu()
                    valid=checks[key]
                    if valid.dtype!=torch.bool or valid.shape!=mask.shape:
                        raise ValueError('Start eligibility must identify each original lane')
                    mask=mask & valid
                    if not mask.any():
                        continue
                row=dict(entry['start'],_actor_lane_mask=mask.clone())
                report=dict(requested=requested,actual=actual,shortfall=int(actual!=requested),
                            effective_lanes=int(mask.sum()),policy_version=entry['policy_version'],
                            native_tick=entry['native_tick'],stream_id=entry['stream'],outcomes_are_sampling_only=True)
                if eligible_lanes is not None:
                    report['records_checked_for_objective']=len(checks)
                self.sampling_history.append(report)
                return row,report
        report=dict(requested=requested,actual=None,shortfall=1)
        if eligible_lanes is not None:
            report['records_checked_for_objective']=len(checks)
        self.sampling_history.append(report)
        return None,report
    def sampling_report(self, since=0):
        rows=self.sampling_history[since:]
        return dict(samples=len(rows),actual_counts={c:sum(r['actual']==c for r in rows) for c in CATEGORIES},
            requested_counts={c:sum(r['requested']==c for r in rows) for c in CATEGORIES},
            shortfall=sum(r['shortfall'] for r in rows),
            unlabeled_lane_snapshots=sum(int((e['masks']['pre_contact']&~e['masks']['failure_pre']&~e['masks']['success_near']).sum()) for e in self.entries))
    def state_dict(self):
        return dict(schema=self.schema,capacity=self.capacity,entries=self.entries,seen=self.seen,
                    sampling_history=self.sampling_history,last_gate=self.last_gate,last_tick=self.last_tick,rng=self.generator.get_state())
    def load_state_dict(self,state):
        if state['schema']!=self.schema or state['capacity']!=self.capacity:raise ValueError('Actor start bank schema/capacity mismatch')
        if len(state['entries'])>self.capacity:raise ValueError('Actor start bank exceeds capacity')
        self.entries=state['entries'];self.seen=int(state['seen']);self.last_gate=state['last_gate'];self.last_tick=state['last_tick']
        self.sampling_history=list(state.get('sampling_history',[]))
        self.generator.set_state(state['rng'].cpu())
