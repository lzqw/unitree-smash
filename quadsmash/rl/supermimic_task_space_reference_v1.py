"""Event-aligned success references, never future observations or native scores.

Schema v2 rejects v1 banks: v1 used absolute time and did not prove terminals.
Discrete retrieval is detached; task-space FK remains differentiable.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import torch

from .supermimic_terminal_scoring import SCORING_SCHEMAS
from .supermimic_physical_dataset_v1 import decode_physical


def task_space_features(state, geometry, mode, *, fk=None):
    p=decode_physical(state)
    if fk is None:
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
    excluded_lanes: tuple = ()
    execution_contract_sha256: str | None = None

    def __post_init__(self):
        if not math.isfinite(self.match_distance_threshold) or self.match_distance_threshold<=0:
            raise ValueError('Tracking requires a finite positive matching threshold')
        if self.max_per_head<1: raise ValueError('Reference capacity must be positive')
        self.excluded_lanes=tuple(sorted(set(self.excluded_lanes)))
        self.assert_excludes(self.excluded_lanes)

    def assert_excludes(self,lanes):
        """Never silently reuse a holdout trajectory as an Actor reference."""
        excluded=set(lanes)
        if not excluded:return
        if any('lane_id' not in e for e in self.entries):
            raise ValueError('Reference lane identity missing; cannot verify holdout independence')
        if any(e['lane_id'] in excluded for e in self.entries):
            raise ValueError('Reference bank contains heldout lane trajectories')

    def to(self,device):
        return type(self)([{k:v.to(device) if torch.is_tensor(v) else v for k,v in e.items()} for e in self.entries],
                          self.geometry_sha256,self.source_sha256,self.match_distance_threshold,self.max_per_head,
                          excluded_lanes=self.excluded_lanes,
                          execution_contract_sha256=self.execution_contract_sha256)

    @property
    def manifest(self):
        return dict(schema=self.schema,geometry_sha256=self.geometry_sha256,source_sha256=self.source_sha256,
                    approach_count=sum(e['mode']=='approach' for e in self.entries),
                    recovery_count=sum(e['mode']=='recovery' for e in self.entries),
                    match_distance_threshold=self.match_distance_threshold,max_per_head=self.max_per_head,
                    excluded_lanes=list(self.excluded_lanes),
                    execution_contract_sha256=self.execution_contract_sha256,
                    protocol_verified=bool(self.entries) and all(e.get('score_schema') in SCORING_SCHEMAS for e in self.entries),
                    native_contact_certification=False,alignment='current causal event estimate / current-state phase match')

    def payload(self): return asdict(self)

    def content_sha256(self):
        from .supermimic_physical_loop_loading_v1 import tensor_digest

        metadata = [self.manifest]
        tensors = []
        for index, entry in enumerate(self.entries):
            metadata.append({key: value for key, value in entry.items() if not torch.is_tensor(value)})
            tensors.extend((f'{index}/{key}', value) for key, value in entry.items() if torch.is_tensor(value))
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
        digest.update(tensor_digest(tensors).encode())
        return digest.hexdigest()

    @classmethod
    def from_payload(cls,payload):
        if payload.get('schema')!='success-guided-task-space-reference-v2':
            raise ValueError('Reference schema requires terminal-verified event-aligned v2 bank')
        return cls(**payload)

    @classmethod
    def from_packed(cls,tensors,geometry,*,source_sha256='',policy_id='unknown',match_distance_threshold=1.,horizon=None,terminal_records=None,excluded_lanes=(),execution_contract_sha256=None,use_substep_toi=False):
        if horizon is not None: raise ValueError('Do not truncate reference episodes')
        for name in ('state','next_state','event_state','next_event_state','episode','elapsed','terminated','timeout','feet','next_feet'):
            if name not in tensors: raise ValueError('Reference evidence missing: '+name)
        labels=None
        if terminal_records is not None:
            labels={}
            for record in terminal_records:
                key=(int(record['env_id']),int(record['episode_index']))
                if key in labels:raise ValueError('Duplicate reference terminal identity')
                if record.get('scoring_schema') not in SCORING_SCHEMAS:raise ValueError('Reference requires original independent task scoring')
                labels[key]=record
        entries=[]; d=tensors; T,N=d['episode'].shape
        excluded=set(excluded_lanes)
        if any(isinstance(x,bool) or not isinstance(x,int) or x<0 or x>=N for x in excluded):
            raise ValueError('Invalid excluded reference lane')
        for lane in range(N):
            if lane in excluded:continue
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
                    toi_data = {}
                    if mode == 'approach':
                        from .supermimic_toi_racket_objective import racket_features
                        contact_state = seq[anchor:anchor+1]
                        kin = racket_features(contact_state, geometry).detach()[0]
                        toi_data['toi_racket'] = torch.cat((kin[:3] - contact_state[0,49:52], kin[3:]))
                        if use_substep_toi:
                            # Tracking can use the success trajectory even if exact contact telemetry is missing.
                            toi_data = {}
                            samples = d.get('native_toi_racket')
                            if anchor > 0 and samples is not None:
                                sample = samples[start+anchor-1, lane]
                                if not sample.isnan().all():
                                    if (sample.shape != (13,) or not sample.isfinite().all()
                                            or not times[anchor-1] < sample[0] <= times[anchor]+1e-6
                                            or abs(sample[7:10].norm()-1.) > 2e-4):
                                        raise ValueError('Invalid native first-string substep target')
                                    toi_data = dict(toi_time=float(sample[0]), toi_racket=sample[1:].detach(),
                                                    toi_source='native-first-airborne-string-substep')
                    choices=torch.arange(max(0,anchor-64),anchor+1,device=seq.device) if mode=='approach' else torch.arange(anchor,len(seq),device=seq.device)
                    digest=hashlib.sha256(seq.cpu().contiguous().numpy().tobytes()+mode.encode()).hexdigest()
                    if 'toi_time' in toi_data:
                        from .supermimic_native_pair_capture_v1 import tree_digest
                        toi_data['trajectory_digest'] = digest
                        digest = tree_digest((digest, toi_data['toi_racket'], toi_data['toi_time'], toi_data['toi_source']))
                    entries.append(dict(**toi_data,mode=mode,features=features,times=times.detach(),anchor_time=float(times[anchor]),
                                        choices=choices,policy_id=str(policy_id),lane_id=lane,episode_id=int(d['episode'][end,lane]),
                                        digest=digest,source_sha256=source_sha256,terminal_verified=True,
                                        score_schema=label['scoring_schema'] if label else 'event-flags-only'))
                start=end+1
        return cls(entries,geometry.model_sha256,source_sha256,match_distance_threshold,excluded_lanes=tuple(excluded),
                   execution_contract_sha256=execution_contract_sha256)

    def extend(self,other):
        if other.geometry_sha256!=self.geometry_sha256: raise ValueError('Reference geometry differs')
        if self.execution_contract_sha256 != other.execution_contract_sha256:
            raise ValueError('Reference execution contract differs')
        excluded=set(self.excluded_lanes)|set(other.excluded_lanes)
        self.assert_excludes(excluded);other.assert_excludes(excluded)
        seen=set(); kept=[]
        for e in reversed(self.entries+other.entries):
            trajectory = e.get('trajectory_digest', e['digest'])
            if trajectory in seen: continue
            if sum(x['mode']==e['mode'] for x in kept)>=self.max_per_head: continue
            seen.add(trajectory);kept.append(e)
        self.entries=list(reversed(kept))
        self.excluded_lanes=tuple(sorted(excluded))

    @torch.no_grad()
    def extend_resolved_approach(self, resolved, geometry, court, *, source_sha256):
        from .supermimic_physical_dataset_v1 import encode_physical

        device = next(geometry.buffers()).device
        entries = []
        for (stream, lane, episode), (frames, terminal) in resolved.items():
            if lane in self.excluded_lanes or not terminal['ball_side_complete_except_recovery']:
                continue
            motion = (terminal.get('native_contact') or {}).get('first_approach_contact_motion')
            flight = terminal.get('native_flight') or {}
            landing = flight.get('first_landing_m')
            required = ('racket_center_m', 'racket_linear_velocity_mps', 'face_normal',
                        'racket_angular_velocity_rad_s', 'ball_position_m', 'incoming_ball_velocity_mps')
            if not motion or not all(name in motion for name in required) or not landing:
                continue
            xmin, xmax = court['target_x_limits_m']
            inside = min(landing[0] - xmin, xmax - landing[0], court['target_y_half_width_m'] - abs(landing[1]))
            if not math.isfinite(inside) or inside < .10:
                continue
            tensors = [torch.as_tensor(motion[name], device=device, dtype=torch.float32) for name in required]
            center, linear, normal, angular, ball, incoming = tensors
            target = torch.cat((center - ball, linear, normal, angular))
            arrival = terminal['contact_time_s']
            if not target.isfinite().all() or not incoming.isfinite().all() or not math.isfinite(arrival):
                continue
            states, times = [], []
            for frame in frames:
                physical = frame['start']['before']['physical']
                states.append(encode_physical({k: v[lane:lane+1] for k, v in physical.items()}).to(device))
                times.append(float(physical['elapsed_s'][lane]))
            order = sorted(range(len(times)), key=times.__getitem__)
            states = torch.cat([states[i] for i in order])
            times = torch.tensor([times[i] for i in order], device=device)
            event_source = (terminal.get('behavior_at_terminal') or {}).get('actor_sha256') or source_sha256
            digest = hashlib.sha256(f'{event_source}:{stream}:{lane}:{episode}'.encode()).hexdigest()
            entries.append(dict(mode='approach', lane_id=lane, episode_id=episode,
                                features=task_space_features(states, geometry, 'approach'), times=times,
                                choices=torch.arange(len(times), device=device), anchor_time=arrival,
                                toi_time=arrival, toi_racket=target, incoming_velocity=incoming,
                                outgoing_velocity=flight.get('outgoing_velocity_mps'), landing_position=landing,
                                contact_position=motion['point_m'], source_sha256=event_source,
                                policy_id=f'{stream}:{frames[-1]["policy_version"]}', digest=digest,
                                score_schema='approach_ball_only_v1', terminal_verified=True))
        self.extend(type(self)(entries, self.geometry_sha256, source_sha256, self.match_distance_threshold,
                               excluded_lanes=self.excluded_lanes,
                               execution_contract_sha256=self.execution_contract_sha256))
        return len(entries)

    def bind(self,initial_physical,mode,*,geometry,elapsed,event_time=None,require_toi=False):
        current=task_space_features(initial_physical,geometry,mode).detach()
        candidates=[e for e in self.entries if e['mode']==mode]
        if require_toi:
            candidates=[e for e in candidates if 'toi_racket' in e]
            if not candidates:
                raise ValueError('Rebuild Approach references from native episodes with racket angular velocity')
        if not candidates:return None
        best=current.new_full((len(current),),float('inf')); chosen=torch.zeros(len(current),dtype=torch.long,device=current.device)
        aligned=current.new_zeros(len(current))
        anchors = [(e.get('toi_time', e['anchor_time']) if require_toi else e['anchor_time']) for e in candidates]
        for i,e in enumerate(candidates):
            ids=e['choices'].to(current.device); ts=e['times'].to(current.device); fs=e['features'].to(current.device)
            distance=(current[:,None]-fs[ids][None]).square().mean(-1)
            if require_toi and 'incoming_velocity' in e:
                velocity = e['incoming_velocity'].to(current)
                delta = (initial_physical[:, 52:55] - velocity).square().mean(-1) / 25.
                distance = distance + .2 * delta[:, None]
            if event_time is not None:
                relative=elapsed.reshape(-1)-event_time.reshape(-1)
                known=torch.isfinite(relative)
                compatible=(relative[:,None]-(ts[ids]-anchors[i])[None]).abs()<=.02
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
        anchor_table=current.new_full((len(candidates),),float('nan'))
        for i,e in enumerate(candidates):
            size=len(e['times']);time_table[i,:size]=e['times'].to(current)
            feature_table[i,:size]=e['features'].to(current)
            anchor_table[i]=float(anchors[i])
        anchor_time=anchor_table[chosen]
        anchor_index=(time_table[chosen]-anchor_time[:,None]).abs().argmin(-1)
        anchor_features=feature_table[chosen,anchor_index]
        toi_data = {'toi_racket':torch.stack([e['toi_racket'].to(current) for e in candidates])[chosen],
                    'measured_interior_success':torch.tensor(
                        ['landing_position' in e for e in candidates], device=current.device)[chosen]} if require_toi else {}
        return dict(**toi_data,reference_digests=[e['digest'] for e in candidates],indices=chosen,matched=torch.isfinite(best)&(best<=self.match_distance_threshold),
                    times=time_table[chosen],features=feature_table[chosen],lengths=sizes[chosen],
                    distance=best,offset=aligned-elapsed.detach().reshape(-1),
                    anchor_time=anchor_time,anchor_features=anchor_features)

    def toi_loss(self, predicted_physical, reference, mode, geometry, *, valid_mask=None, weights=None,
                 return_components=False):
        """Loss against the event-aligned racket state at the verified TOI.

        Reference retrieval remains detached. FK of the predicted physical
        state stays differentiable, so this objective can guide Approach while
        the ball/contact branch is detached.
        """
        actual=task_space_features(predicted_physical,geometry,mode)
        target=reference.get('anchor_features')
        if target is None:
            zero=actual.sum()*0.
            return (zero,zero.detach(),actual.new_zeros(len(actual))) if return_components else zero
        target=target.to(actual)
        mask=reference['matched'].to(actual.device)
        if valid_mask is not None: mask=mask&valid_mask.to(mask.device)
        w=mask.to(actual.dtype)
        if weights is not None:w=w*weights.detach().reshape(-1)
        # The approach feature layout is relative position, relative velocity,
        # racket normal, orientation, root velocity and root angular velocity.
        scales=actual.new_tensor([.05]*3+[1.]*3+[.2]*3+[.2]*3+[2.]*3+[5.]*3)
        value=(((actual-target.detach())/scales).square().mean(-1)*w).sum()
        if return_components:return value,w.sum(),w
        return value/w.sum().clamp_min(1e-8)

    def loss(self,predicted_physical,elapsed,reference,mode,geometry=None,valid_mask=None,weights=None,return_components=False,*,fk=None):
        actual=task_space_features(predicted_physical,geometry,mode,fk=fk)
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
