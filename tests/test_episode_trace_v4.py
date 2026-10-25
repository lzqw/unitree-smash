import hashlib
import json

import numpy as np
import pytest

from quadsmash.rl.episode_trace_v4 import PublicHistoryPacker,EpisodeTraceRecorderV4,unpack_public


def public(frame,history,time):
    tail=np.zeros(35,dtype=np.float32);tail[16]=time
    return np.concatenate([frame,history.reshape(-1),tail]).astype(np.float32)


def test_lossless_history_and_selective_reset_preserve_all794_values():
    rng=np.random.default_rng(17);packer=PublicHistoryPacker(2)
    history=np.zeros((2,10,69),dtype=np.float32);keys=np.array([3,7]);saved=[[],[]];packed=[[],[]]
    for step in range(6):
        if step==3:
            keys[0]+=1;history[0]=0.;saved[0]=[];packed[0]=[]
        frames=rng.normal(size=(2,69)).astype(np.float32)
        obs=np.stack([public(frames[lane],history[lane],.01*(step if lane else step%3))for lane in range(2)])
        value,changed=packer.pack(obs,keys)
        assert changed.tolist()==([True,True]if step==0 else[True,False]if step==3 else[False,False])
        for lane in range(2):saved[lane].append(obs[lane]);packed[lane].append(value[lane])
        history[:,1:]=history[:,:-1].copy();history[:,0]=frames
    for lane in range(2):assert np.array_equal(unpack_public(np.stack(packed[lane])),np.stack(saved[lane]))


def test_unaccounted_observation_history_is_rejected():
    packer=PublicHistoryPacker(1);obs=np.zeros((1,794),dtype=np.float32);obs[0,69]=1.
    with pytest.raises(ValueError,match='losslessly'):packer.pack(obs,np.array([0]))


def test_selected_complete_and_failure_traces_are_actual_inputs_and_not_frozen(tmp_path):
    settings=dict(max_success_traces=1,max_failure_traces=1,first_episode_lanes=[1],max_buffer_bytes=1000000)
    rec=EpisodeTraceRecorderV4(tmp_path/'trace',2,4,['base','Racket'],settings,
        dict(native_source_confirmed=False,source_kind='CPU unit fixture',training_indices=[0,1]))
    rng=np.random.default_rng(2);history=np.zeros((2,10,69),dtype=np.float32);original=[]
    for step in range(3):
        frames=rng.normal(size=(2,69)).astype(np.float32)
        obs=np.stack([public(frames[lane],history[lane],step*.01)for lane in range(2)])
        actions=np.full((2,18),step*.1,dtype=np.float32)
        poses=np.zeros((2,2,7),dtype=np.float32);poses[:,:,3]=1.;poses[:,:,2]=step
        ball=np.full((2,3),step,dtype=np.float32)
        rec.append(public=obs,episode_ids=np.array([5,9]),residual=actions,base=actions,actual=actions,
            poses=poses,ball=ball,iteration=4+step//2)
        original.append(obs.copy());history[:,1:]=history[:,:-1].copy();history[:,0]=frames
    rec.record_contacts([dict(env_id=0,episode_index=5,time_s=.02,string_contact=True)])
    rec.finish([dict(env_id=0,episode_index=5,complete_smash=True),dict(env_id=1,episode_index=9,complete_smash=False)])
    manifest=json.loads((tmp_path/'trace/manifest.json').read_text())
    assert len(manifest['saved_episodes'])==2 and not manifest['native_execution']
    for lane,episode in [(0,5),(1,9)]:
        path=tmp_path/f'trace/lane{lane:04d}_episode{episode:06d}.npz';meta=json.loads(path.with_suffix('.json').read_text())
        data=np.load(path);restored=unpack_public(data['packed_public'])
        assert np.array_equal(restored,np.stack(original)[:,lane])
        assert hashlib.sha256(restored.tobytes()).hexdigest()==meta['full_public794_sha256']
        assert meta['actor_iterations_encountered']==[4,5] and not meta['formal_acceptance']
        assert not meta['native_execution'] and meta['full_public_reconstruction_bitwise_verified']
        assert np.array_equal(data['link_poses'][:,0,2],np.arange(3,dtype=np.float32))
    assert rec.lengths.tolist()==[0,0]


def test_declared_storage_limit_and_canonical_reset_are_explicit(tmp_path):
    small=dict(max_success_traces=1,max_failure_traces=1,first_episode_lanes=[],max_buffer_bytes=1)
    with pytest.raises(ValueError,match='buffer limit'):
        EpisodeTraceRecorderV4(tmp_path/'too_large',2,800,['base']*28,small,{})
    settings=dict(small,max_buffer_bytes=1000000)
    rec=EpisodeTraceRecorderV4(tmp_path/'reset',1,4,['base'],settings,{})
    obs=np.zeros((1,794),dtype=np.float32);poses=np.zeros((1,1,7),dtype=np.float32);poses[:,:,3]=1.
    action=np.zeros((1,18),dtype=np.float32)
    rec.append(public=obs,episode_ids=np.array([2]),residual=action,base=action,actual=action,
        poses=poses,ball=np.zeros((1,3),dtype=np.float32),iteration=0)
    rec.reset();assert rec.censored==1 and rec.lengths[0]==0 and rec.packer.keys[0]==-1
    assert rec.statistics()['saved_episode_count']==0
    rec.disable(ValueError('Controlled test failure'));assert not rec.statistics()['enabled']
    assert 'Controlled test failure' in rec.statistics()['disabled_error']
