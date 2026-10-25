"""Lossless TRAIN episode traces; read-only observation/action/pose recording.

The794D public input has690 repeated history values. Store its current69D
frame plus35D tail, verify the full input at every step, and reconstruct it
exactly for saved episodes. No physics state or policy input is modified.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA='native-training-episode-trace-v1'


def trace_settings(value):
    if value is None:return None
    keys={'max_success_traces','max_failure_traces','first_episode_lanes','max_buffer_bytes'}
    if not isinstance(value,dict) or set(value)!=keys:raise ValueError('Explicit episode trace settings required')
    for key in ['max_success_traces','max_failure_traces','max_buffer_bytes']:
        if type(value[key]) is not int or value[key]<=0:raise ValueError('Positive integer trace limits required')
    lanes=value['first_episode_lanes']
    if (not isinstance(lanes,list) or len(set(lanes))!=len(lanes)
            or any(type(lane)is not int or lane<0 for lane in lanes)):
        raise ValueError('Trace lanes must be distinct nonnegative integers')
    return dict(value,first_episode_lanes=list(lanes))


def unpack_public(packed):
    if packed.ndim!=2 or packed.shape[1]!=104 or packed.dtype!=np.float32:
        raise ValueError('Require lossless float32 current-frame/tail values')
    result=np.empty((len(packed),794),dtype=np.float32);history=np.zeros((10,69),dtype=np.float32)
    for index,row in enumerate(packed):
        result[index,:69]=row[:69];result[index,69:759]=history.reshape(-1);result[index,759:]=row[69:]
        history[1:]=history[:-1].copy();history[0]=row[:69]
    return result


class PublicHistoryPacker:
    def __init__(self,count):
        self.history=np.zeros((count,10,69),dtype=np.float32)
        self.keys=np.full(count,-1,dtype=np.int64)

    def reset(self):
        self.history.fill(0.);self.keys.fill(-1)

    def pack(self,public,episode_ids):
        if public.shape!=(len(self.keys),794) or public.dtype!=np.float32 or not np.isfinite(public).all():
            raise ValueError('Require actual finite float32 public794 observations')
        ids=np.asarray(episode_ids,dtype=np.int64)
        if ids.shape!=self.keys.shape:raise ValueError('One actual episode index per lane required')
        changed=ids!=self.keys;self.history[changed]=0.
        if not np.array_equal(public[:,69:759],self.history.reshape(len(ids),-1)):
            raise ValueError('Public history cannot be losslessly reconstructed from recorded controls')
        packed=np.concatenate([public[:,:69],public[:,759:]],1)
        self.history[:,1:]=self.history[:,:-1].copy();self.history[:,0]=public[:,:69]
        self.keys[:]=ids
        return packed,changed


class EpisodeTraceRecorderV4:
    def __init__(self,output,count,max_steps,body_names,settings,provenance):
        self.output=Path(output);self.settings=trace_settings(settings);self.provenance=dict(provenance)
        if self.settings is None or count<1 or max_steps<1:raise ValueError('Explicit positive trace dimensions required')
        if any(lane>=count for lane in self.settings['first_episode_lanes']):raise ValueError('Trace lane outside native batch')
        self.body_names=list(body_names);self.count=count;self.max_steps=max_steps
        shapes={'packed_public':(count,max_steps,104),'residual_actions':(count,max_steps,18),
            'base_actions':(count,max_steps,18),'native_actions':(count,max_steps,18),
            'link_poses':(count,max_steps,len(body_names),7),'ball_position_m':(count,max_steps,3)}
        required=sum(int(np.prod(shape))*4 for shape in shapes.values())+count*max_steps*4+count*10*69*4
        if required>self.settings['max_buffer_bytes']:raise ValueError('Declared CPU trajectory-buffer limit exceeded')
        if self.output.exists():raise ValueError('Never overwrite an existing trace producer')
        self.output.mkdir(parents=True)
        self.arrays={name:np.empty(shape,dtype=np.float32) for name,shape in shapes.items()}
        self.arrays['learner_iteration']=np.empty((count,max_steps),dtype=np.int32)
        self.buffer_bytes=required;self.packer=PublicHistoryPacker(count)
        self.lengths=np.zeros(count,dtype=np.int64);self.keys=np.full(count,-1,dtype=np.int64)
        self.last_clock=np.zeros(count,dtype=np.float32);self.public_hashes=[hashlib.sha256()for _ in range(count)]
        self.closed_per_lane=np.zeros(count,dtype=np.int64);self.contacts={};self.saved=[]
        self.success_selected=0;self.failure_selected=0;self.closed_observed=0;self.censored=0;self.verified_rows=0
        self.disabled_error=None
        self.flush()

    def statistics(self):
        return dict(schema=SCHEMA,enabled=self.disabled_error is None,disabled_error=self.disabled_error,
            allocated_host_buffer_bytes=self.buffer_bytes,verified_full_public_rows=self.verified_rows,
            closed_episodes_observed=self.closed_observed,censored_open_episodes=self.censored,
            saved_episode_count=len(self.saved),success_selected=self.success_selected,failure_selected=self.failure_selected,
            native_execution=bool(self.saved and self.provenance.get('native_source_confirmed',False)),
            frozen_policy_performance=False,formal_acceptance=False)

    def flush(self):
        value=dict(self.statistics(),selection_settings=self.settings,provenance=self.provenance,
            body_names=self.body_names,saved_episodes=self.saved,
            motion='Actual100Hz poses and actions; no interpolation. Select every second stored sample for50Hz video.',
            policy_scope='Changing weights, normalization and exploration; never a frozen-policy or formal score.',goal_complete=False)
        path=self.output/'manifest.json';tmp=path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

    def disable(self,error):
        self.disabled_error=repr(error)
        try:self.flush()
        except OSError:pass

    def reset(self,reason='canonical_ground_reset'):
        self.censored+=int((self.lengths>0).sum());self.lengths.fill(0);self.keys.fill(-1)
        self.packer.reset();self.contacts.clear();self.public_hashes=[hashlib.sha256()for _ in range(self.count)]
        if self.disabled_error is None:self.flush()

    def append(self,*,public,episode_ids,residual,base,actual,poses,ball,iteration):
        if self.disabled_error is not None:return
        for array,shape in [(residual,(self.count,18)),(base,(self.count,18)),(actual,(self.count,18)),
                (poses,(self.count,len(self.body_names),7)),(ball,(self.count,3))]:
            if array.shape!=shape or array.dtype!=np.float32 or not np.isfinite(array).all():
                raise ValueError('Actual finite float32 trace arrays with original shapes required')
        if np.max(np.abs(actual))>1.000001:raise ValueError('Native action outside original normalized limits')
        packed,changed=self.packer.pack(public,episode_ids)
        if np.any(changed & (self.lengths>0)):raise ValueError('An episode reset had no corresponding closed/aborted trace event')
        self.keys[changed]=np.asarray(episode_ids,dtype=np.int64)[changed]
        if np.any(self.lengths>=self.max_steps):raise ValueError('Recorded episode exceeded declared sample capacity')
        clock=public[:,775]
        if np.any(np.abs(clock[changed])>1e-6):raise ValueError('Every saved episode must start at the real zero clock')
        continuous=~changed
        if np.any(np.abs(clock[continuous]-self.last_clock[continuous]-.01)>2e-6):
            raise ValueError('Trace skipped or duplicated a native10ms control step')
        lanes=np.arange(self.count);index=self.lengths
        for name,value in [('packed_public',packed),('residual_actions',residual),('base_actions',base),
                ('native_actions',actual),('link_poses',poses),('ball_position_m',ball)]:
            self.arrays[name][lanes,index]=value
        self.arrays['learner_iteration'][lanes,index]=int(iteration)
        for lane in lanes:self.public_hashes[lane].update(public[lane].tobytes())
        self.lengths+=1;self.last_clock[:]=clock;self.verified_rows+=self.count

    def record_contacts(self,events):
        if self.disabled_error is not None:return
        for event in events:
            key=(event['env_id'],event['episode_index']);self.contacts.setdefault(key,[]).append(event)

    def finish(self,rows):
        if self.disabled_error is not None:return
        changed=False
        for row in rows:
            lane,episode=row['env_id'],row['episode_index'];key=(lane,episode)
            if self.keys[lane]!=episode or self.lengths[lane]<1:raise ValueError('Closed rally has no matching ground-start trace')
            reasons=[];success=bool(row['complete_smash'])
            if lane in self.settings['first_episode_lanes'] and self.closed_per_lane[lane]==0:reasons.append('first_episode_of_predeclared_lane')
            if success and self.success_selected<self.settings['max_success_traces']:
                reasons.append('first_complete_until_declared_limit');self.success_selected+=1
            if not success and self.failure_selected<self.settings['max_failure_traces']:
                reasons.append('first_failure_until_declared_limit');self.failure_selected+=1
            if reasons:
                length=int(self.lengths[lane]);data={name:array[lane,:length].copy() for name,array in self.arrays.items()}
                restored=unpack_public(data['packed_public']);digest=hashlib.sha256(restored.tobytes()).hexdigest()
                if digest!=self.public_hashes[lane].hexdigest():raise ValueError('Full public-observation SHA mismatch')
                data['episode_clock_s']=data['packed_public'][:,85].copy()
                path=self.output/f'lane{lane:04d}_episode{episode:06d}.npz';temp=path.with_suffix('.npz.tmp')
                if path.exists():raise ValueError('Preserve previous episode trace')
                with temp.open('xb')as stream:np.savez_compressed(stream,**data)
                temp.replace(path)
                metadata=dict(schema=SCHEMA,data_file=path.name,data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    env_id=lane,episode_index=episode,source_training_index=row.get('source_training_index',lane%128),
                    native_execution=bool(self.provenance.get('native_source_confirmed',False)),
                    source_kind=self.provenance.get('source_kind','CPU numerical fixture'),
                    samples=length,control_dt_s=.01,full_public794_sha256=digest,
                    full_public_reconstruction_bitwise_verified=True,selection_reasons=reasons,
                    actor_iterations_encountered=np.unique(data['learner_iteration']).tolist(),
                    policy_scope='Actual changing-policy training trace; no frozen-policy performance inherited.',
                    raw_episode=row,actual_contact_events=self.contacts.get(key,[]),
                    formal_cases_read=False,formal_acceptance=False,goal_complete=False)
                target=path.with_suffix('.json');temporary=target.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(metadata,indent=2)+'\n');temporary.replace(target)
                self.saved.append({name:metadata[name]for name in ['data_file','env_id','episode_index','source_training_index',
                    'samples','data_sha256','full_public794_sha256','selection_reasons','actor_iterations_encountered']})
                changed=True
            self.closed_per_lane[lane]+=1;self.closed_observed+=1
            self.lengths[lane]=0;self.keys[lane]=-1;self.packer.keys[lane]=-1;self.packer.history[lane]=0.
            self.public_hashes[lane]=hashlib.sha256();self.contacts.pop(key,None)
        if changed:self.flush()
