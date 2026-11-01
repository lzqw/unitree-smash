"""Preserve independently labelled curriculum successes during future TRAIN.

Observability only: this recorder receives already-dispatched actions and
closed native rally/jump records. It changes no action, reward, physics, policy
input or original task label. Original8 and explicit curriculum outcomes stay
separate. Existing live producers are never modified or relabelled in place.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from quadsmash.rl.episode_trace_v4 import EpisodeTraceRecorderV4, trace_settings, unpack_public
from scripts.export_complete_demonstrations_v1 import classify

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='native-training-curriculum-episode-trace-v1'
PROTOCOL=ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'


def retention_settings(value):
    keys={'minimum_speed_mps','max_curriculum_success_traces','max_recovery_only_traces'}
    if not isinstance(value,dict) or set(value)!=keys:
        raise ValueError('Declare the complete explicit curriculum retention settings')
    speed=value['minimum_speed_mps']
    if isinstance(speed,bool) or not isinstance(speed,(int,float)) or not math.isfinite(speed) or not 0<speed<=8:
        raise ValueError('Explicit curriculum speed must be positive and at most original8')
    for key in ['max_curriculum_success_traces','max_recovery_only_traces']:
        if type(value[key]) is not int or value[key]<0:raise ValueError('Nonnegative integer retention quotas required')
    if not value['max_curriculum_success_traces']:raise ValueError('At least one curriculum success must be retainable')
    return dict(value)


class CurriculumTraceSelection:
    """Independent labels and explicit finite storage quotas, never a policy."""
    def __init__(self,settings,retention,protocol=None):
        self.settings=trace_settings(settings);self.retention=retention_settings(retention)
        self.protocol=json.loads(PROTOCOL.read_text()) if protocol is None else dict(protocol)
        self.protocol['_root']=str(ROOT)
        self.observed=Counter();self.quota_selected=Counter();self.retained=Counter();self.skipped=Counter()

    def choose(self,row,*,first_predeclared=False):
        label=classify(row,self.protocol,self.retention['minimum_speed_mps'])
        original=label['original_outcomes']
        kinds=dict(original_complete8=bool(original['complete'] and row['numerical_failures']==0),
            curriculum_complete=bool(label['curriculum_complete']),
            recovery_only=bool(original['conditional_recovery'] and not label['curriculum_complete']
                               and row['numerical_failures']==0))
        kinds['failure']=not any(kinds.values())
        quotas=dict(original_complete8=self.settings['max_success_traces'],
            curriculum_complete=self.retention['max_curriculum_success_traces'],
            recovery_only=self.retention['max_recovery_only_traces'],failure=self.settings['max_failure_traces'])
        self.observed['episodes']+=1
        reasons=['first_episode_of_predeclared_lane'] if first_predeclared else []
        for kind,eligible in kinds.items():
            self.observed[kind]+=int(eligible)
            if eligible and self.quota_selected[kind]<quotas[kind]:
                self.quota_selected[kind]+=1;reasons.append(kind+'_until_declared_limit')
        if reasons:
            self.retained['episodes']+=1
            for kind,value in kinds.items():self.retained[kind]+=int(value)
        else:
            for kind,value in kinds.items():self.skipped[kind]+=int(value)
        return label,kinds,reasons


class CurriculumEpisodeTraceRecorderV1(EpisodeTraceRecorderV4):
    def __init__(self,output,count,max_steps,body_names,settings,provenance,*,retention):
        self.selection=CurriculumTraceSelection(settings,retention)
        self.protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
        if provenance.get('protocol_sha256')!=self.protocol_sha256:
            raise ValueError('Trace source must use the unchanged original protocol')
        if provenance.get('asset_sha256')!=self.selection.protocol['asset']['sha256']:
            raise ValueError('Trace source belongs to a different loaded asset')
        super().__init__(output,count,max_steps,body_names,settings,provenance)

    def statistics(self):
        result=super().statistics()
        actual_retained=dict(episodes=len(self.saved),
            original_complete8=sum(x.get('full_expert_under_original_task',False) for x in self.saved),
            curriculum_complete=sum(x.get('full_expert_under_curriculum',False) for x in self.saved),
            recovery_only=sum(x.get('recovery_reference_only',False) for x in self.saved))
        result.update(schema=SCHEMA,independent_label_protocol_sha256=self.protocol_sha256,
            curriculum_selection_settings=self.selection.retention,
            independently_observed_outcomes=dict(self.selection.observed),
            selection_quota_counts=dict(self.selection.quota_selected),
            selection_policy_accepted_outcomes=dict(self.selection.retained),
            retained_outcomes=actual_retained,
            quota_skipped_outcomes=dict(self.selection.skipped),
            original_speed_threshold_mps=8.,
            labels_are_separate_from_original_complete_smash=True)
        return result

    def finish(self,rows):
        if self.disabled_error is not None:return
        changed=False
        for row in rows:
            lane,episode=row['env_id'],row['episode_index'];key=(lane,episode)
            if not 0<=lane<self.count or self.keys[lane]!=episode or self.lengths[lane]<1:
                raise ValueError('Closed native rally has no matching ground-start trace')
            original_row_bytes=json.dumps(row,sort_keys=True).encode()
            label,kinds,reasons=self.selection.choose(row,first_predeclared=(
                lane in self.settings['first_episode_lanes'] and self.closed_per_lane[lane]==0))
            self.success_selected=self.selection.quota_selected['original_complete8']
            self.failure_selected=self.selection.quota_selected['failure']
            if reasons:
                length=int(self.lengths[lane])
                data={name:array[lane,:length].copy() for name,array in self.arrays.items()}
                restored=unpack_public(data['packed_public'])
                public_sha=hashlib.sha256(restored.tobytes()).hexdigest()
                if public_sha!=self.public_hashes[lane].hexdigest():raise ValueError('Full public SHA mismatch')
                data['episode_clock_s']=data['packed_public'][:,85].copy()
                # Every positive event must occur within the actually retained
                # command interval. No short prefix can masquerade as an episode.
                endpoint=float(data['episode_clock_s'][-1])+.01
                if kinds['curriculum_complete']:
                    association=row['contact_flight_recovery']
                    earliest_stable_end=association['contact_flight_landing_s']+association['recovery_s']
                    if max(row['first_landing_time_s'],row['release_time_s'],row['crossing_time_s'],
                           earliest_stable_end)>endpoint+.002:
                        raise ValueError('Retained trace ends before its reported complete events')
                path=self.output/f'lane{lane:04d}_episode{episode:06d}.npz'
                if path.exists() or path.with_suffix('.json').exists():raise ValueError('Preserve previous trace')
                temp=path.with_suffix('.npz.tmp')
                with temp.open('xb') as f:np.savez_compressed(f,**data)
                temp.replace(path)
                meta=dict(schema=SCHEMA,data_file=path.name,data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    env_id=lane,episode_index=episode,source_training_index=row['source_training_index'],
                    native_execution=bool(self.provenance.get('native_source_confirmed',False)),
                    source_kind=self.provenance.get('source_kind','CPU replay; no new native execution'),
                    samples=length,control_dt_s=.01,full_public794_sha256=public_sha,
                    full_public_reconstruction_bitwise_verified=True,selection_reasons=reasons,
                    actor_iterations_encountered=np.unique(data['learner_iteration']).tolist(),
                    policy_scope='Actual changing-policy trace; no fixed-policy performance inherited.',
                    raw_episode=row,raw_episode_sha256=hashlib.sha256(original_row_bytes).hexdigest(),
                    actual_contact_events=self.contacts.get(key,[]),
                    original_outcomes=label['original_outcomes'],
                    original_complete_smash_field=bool(row['complete_smash']),
                    full_expert_under_original_task=kinds['original_complete8'],
                    full_expert_under_curriculum=kinds['curriculum_complete'],
                    recovery_reference_only=kinds['recovery_only'],
                    curriculum_minimum_speed_mps=self.selection.retention['minimum_speed_mps'],
                    curriculum_independent_label=label,protocol_sha256=self.protocol_sha256,
                    formal_cases_read=False,formal_acceptance=False,goal_complete=False)
                target=path.with_suffix('.json');temporary=target.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(meta,indent=2)+'\n');temporary.replace(target)
                self.saved.append({name:meta[name] for name in ['data_file','env_id','episode_index','source_training_index',
                    'samples','data_sha256','full_public794_sha256','selection_reasons','actor_iterations_encountered',
                    'full_expert_under_original_task','full_expert_under_curriculum','recovery_reference_only']})
                changed=True
            if json.dumps(row,sort_keys=True).encode()!=original_row_bytes:
                raise ValueError('Recorder changed the original episode result')
            self.closed_per_lane[lane]+=1;self.closed_observed+=1;self.lengths[lane]=0;self.keys[lane]=-1
            self.packer.keys[lane]=-1;self.packer.history[lane]=0.
            self.public_hashes[lane]=hashlib.sha256();self.contacts.pop(key,None)
        if changed:self.flush()
