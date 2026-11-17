"""Event-aligned success references, never future observations or native scores.

Schema v2 rejects v1 banks: v1 used absolute time and did not prove terminals.
Discrete retrieval is detached; task-space FK remains differentiable.
"""
from dataclasses import dataclass, asdict
import hashlib
import math
import torch
from .supermimic_physical_dataset_v1 import decode_physical


def task_space_features(state, geometry, mode):
    p=decode_physical(state)
    fk=geometry(p['joint_position'],p['joint_velocity'],p['root_position_w'],p['root_quaternion_w'],
                p['root_link_velocity_w'],p['root_angular_velocity_w'])
    rid=geometry.ids['Racket']
    normal=fk['rotation'][:,rid,:,0]  # string plane is YZ in the pinned asset
    orientation=fk['rotation'][:,geometry.root_id,:,2]
    if mode=='approach':
        return torch.cat(((fk['position'][:,rid]-p['ball_position_w'])/.2,
                          (fk['velocity'][:,rid]-p['ball_velocity_w'])/5.,normal,
                          orientation,fk['center_velocity']/2.,p['root_angular_velocity_w']/5.),-1)
    if mode=='recovery':
        return torch.cat((orientation,fk['center_velocity']/2.,p['root_angular_velocity_w']/5.,
                          (fk['feet']-fk['center'][:,None]).flatten(1)/.4),-1)
    raise ValueError('Unknown tracking head: '+mode)


@dataclass
class TaskSpaceReferenceBankV1:
    entries: list
    geometry_sha256: str
    source_sha256: str
    match_distance_threshold: float = 1.0
    max_per_head: int = 128
    schema: str = 'success-guided-task-space-reference-v2'

    def __post_init__(self):
        if not math.isfinite(self.match_distance_threshold) or self.match_distance_threshold<=0:
            raise ValueError('Tracking requires a finite positive matching threshold')
        if self.max_per_head<1: raise ValueError('Reference capacity must be positive')

    def to(self,device):
        return type(self)([{k:v.to(device) if torch.is_tensor(v) else v for k,v in e.items()} for e in self.entries],
                          self.geometry_sha256,self.source_sha256,self.match_distance_threshold,self.max_per_head)

    @property
    def manifest(self):
        return dict(schema=self.schema,geometry_sha256=self.geometry_sha256,source_sha256=self.source_sha256,
                    approach_count=sum(e['mode']=='approach' for e in self.entries),
                    recovery_count=sum(e['mode']=='recovery' for e in self.entries),
                    match_distance_threshold=self.match_distance_threshold,max_per_head=self.max_per_head,
                    protocol_verified=bool(self.entries) and all(e.get('score_schema')=='historical-independent-complete5-v1' for e in self.entries),
                    native_contact_certification=False,alignment='current causal event estimate / current-state phase match')

    def payload(self): return asdict(self)

    @classmethod
    def from_payload(cls,payload):
        if payload.get('schema')!='success-guided-task-space-reference-v2':
            raise ValueError('Reference schema requires terminal-verified event-aligned v2 bank')
        return cls(**payload)

    @classmethod
    def from_packed(cls,tensors,geometry,*,source_sha256='',policy_id='unknown',match_distance_threshold=1.,horizon=None,terminal_records=None):
        if horizon is not None: raise ValueError('Do not truncate reference episodes')
        for name in ('state','next_state','event_state','next_event_state','episode','elapsed','terminated','timeout','feet','next_feet'):
            if name not in tensors: raise ValueError('Reference evidence missing: '+name)
        labels=None
        if terminal_records is not None:
            labels={}
            for record in terminal_records:
                key=(int(record['env_id']),int(record['episode_index']))
                if key in labels:raise ValueError('Duplicate reference terminal identity')
                if record.get('scoring_schema')!='historical-independent-complete5-v1':raise ValueError('Reference requires original independent task scoring')
                labels[key]=record
        entries=[]; d=tensors; T,N=d['episode'].shape
        for lane in range(N):
            start=0
            for end in torch.where(d['terminated'][:,lane]|d['timeout'][:,lane])[0].tolist():
                same=bool((d['episode'][start:end+1,lane]==d['episode'][end,lane]).all())
                if not same or float(d['elapsed'][start,lane])>.011:
                    start=end+1; continue
                final=d['next_event_state'][end,lane]
                label=None if labels is None else labels.get((lane,int(d['episode'][end,lane])))
                if labels is not None and label is None:raise ValueError('Missing terminal score for complete reference episode')
                times=torch.cat((d['elapsed'][start:end+1,lane],d['elapsed'][end:end+1,lane]+.01))
                seq=torch.cat((d['state'][start:end+1,lane],d['next_state'][end:end+1,lane])).detach()
                ev=torch.cat((d['event_state'][start:end+1,lane],final[None]))
                feet=torch.cat((d['feet'][start:end+1,lane],d['next_feet'][end:end+1,lane])).bool().any(-1)
                contact=torch.where(ev[:,1]>.5)[0]
                td=torch.where((~feet[:-1])&feet[1:]&(ev[:-1,0]>.5))[0]+1
                for mode,ok,anchor in (
                    ('approach',bool((final[1:4]>.5).all() and final[7]<.5),int(contact[0]) if len(contact) else None),
                    ('recovery',bool(final[4]>.5 and final[5]<.5 and final[7]<.5),int(td[0]) if len(td) else None)):
                    if not ok or anchor is None: continue
                    if label is not None:
                        required='ball_side_complete_except_recovery' if mode=='approach' else 'recovery_success'
                        if label.get(required) is not True:continue
                        if mode=='recovery' and label.get('body_ground_seen') is not False:continue
                    features=task_space_features(seq,geometry,mode).detach()
                    # Match within the same event phase. Terminal features are
                    # offline reference labels; never supplied to current actor.
                    choices=torch.arange(max(0,anchor-64),anchor+1,device=seq.device) if mode=='approach' else torch.arange(anchor,len(seq),device=seq.device)
                    digest=hashlib.sha256(seq.cpu().contiguous().numpy().tobytes()+mode.encode()).hexdigest()
                    entries.append(dict(mode=mode,features=features,times=times.detach(),anchor_time=float(times[anchor]),
                                        choices=choices,policy_id=str(policy_id),lane_id=lane,episode_id=int(d['episode'][end,lane]),
                                        digest=digest,source_sha256=source_sha256,terminal_verified=True,
                                        score_schema=label['scoring_schema'] if label else 'event-flags-only'))
                start=end+1
        return cls(entries,geometry.model_sha256,source_sha256,match_distance_threshold)

    def extend(self,other):
        if other.geometry_sha256!=self.geometry_sha256: raise ValueError('Reference geometry differs')
        seen=set(); kept=[]
        for e in reversed(self.entries+other.entries):
            if e['digest'] in seen: continue
            if sum(x['mode']==e['mode'] for x in kept)>=self.max_per_head: continue
            seen.add(e['digest']);kept.append(e)
        self.entries=list(reversed(kept))

    def bind(self,initial_physical,mode,*,geometry,elapsed,event_time=None):
        current=task_space_features(initial_physical,geometry,mode).detach()
        candidates=[e for e in self.entries if e['mode']==mode]
        if not candidates:return None
        best=current.new_full((len(current),),float('inf')); chosen=torch.zeros(len(current),dtype=torch.long,device=current.device)
        aligned=current.new_zeros(len(current))
        for i,e in enumerate(candidates):
            ids=e['choices'].to(current.device); ts=e['times'].to(current.device); fs=e['features'].to(current.device)
            distance=(current[:,None]-fs[ids][None]).square().mean(-1)
            if event_time is not None:
                relative=elapsed.reshape(-1)-event_time.reshape(-1)
                known=torch.isfinite(relative)
                compatible=(relative[:,None]-(ts[ids]-e['anchor_time'])[None]).abs()<=.02
                distance=distance.masked_fill(known[:,None]&~compatible,float('inf'))
            value,idx=distance.min(-1); better=value<best
            best=torch.where(better,value,best); chosen=torch.where(better,i,chosen)
            aligned=torch.where(better,ts[ids[idx]],aligned)
        # Pack once per rollout. Per-step interpolation stays batched on the
        # device (no per-lane int(cuda_tensor) synchronizations inside BPTT).
        length=max(len(e['times']) for e in candidates)
        time_table=current.new_full((len(candidates),length),float('inf'))
        feature_table=current.new_zeros(len(candidates),length,current.shape[-1])
        sizes=torch.tensor([len(e['times']) for e in candidates],device=current.device)
        for i,e in enumerate(candidates):
            size=len(e['times']);time_table[i,:size]=e['times'].to(current)
            feature_table[i,:size]=e['features'].to(current)
        return dict(indices=chosen,matched=torch.isfinite(best)&(best<=self.match_distance_threshold),
                    times=time_table[chosen],features=feature_table[chosen],lengths=sizes[chosen],
                    distance=best,offset=aligned-elapsed.detach().reshape(-1))

    def loss(self,predicted_physical,elapsed,reference,mode,geometry=None,valid_mask=None,weights=None,return_components=False):
        actual=task_space_features(predicted_physical,geometry,mode)
        if reference is None:
            zero=actual.sum()*0.
            return (zero,zero.detach(),actual.new_zeros(len(actual))) if return_components else zero
        times,features,lengths=reference['times'],reference['features'],reference['lengths']
        rows=torch.arange(len(actual),device=actual.device)
        query=elapsed.detach().reshape(-1)+reference['offset']
        query=torch.minimum(torch.maximum(query,times[:,0]),times[rows,lengths-1])
        hi=torch.searchsorted(times.contiguous(),query[:,None].contiguous()).flatten().clamp_min(1)
        hi=torch.minimum(hi,lengths-1);lo=hi-1
        alpha=((query-times[rows,lo])/(times[rows,hi]-times[rows,lo]).clamp_min(1e-8))[:,None]
        target=(1-alpha)*features[rows,lo]+alpha*features[rows,hi]
        mask=reference['matched'].clone().to(actual.device)
        if valid_mask is not None:mask=mask&valid_mask.to(mask.device)
        w=mask.to(actual.dtype)
        if weights is not None:w=w*weights.detach().reshape(-1)
        numerator=((actual-target.detach()).square().mean(-1)*w).sum()
        denominator=w.sum()
        if return_components:return numerator,denominator,w
        return numerator/denominator.clamp_min(1e-8)
