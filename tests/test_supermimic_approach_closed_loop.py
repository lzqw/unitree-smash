"""CPU integration contracts; no Isaac, policy training or physical evidence."""
import json
from types import SimpleNamespace
import pytest
import torch
from quadsmash.rl.supermimic_approach_data_v1 import active_gate,approach_window_masks,masked_lane_mean,gradient_parts,FIELDS
from quadsmash.rl.supermimic_frozen_refresh_v1 import FrozenRefresh
from quadsmash.rl.supermimic_physical_dataset_v1 import STATE_FIELDS
from quadsmash.rl.supermimic_physical_loop_replay_v1 import PhysicalLoopReplay

class Schema:
    def encode(self,record):
        n=len(record['paid_events']);return torch.zeros(n,2),torch.ones(n,2)

def row(t,elapsed,episode,done):
    n=len(elapsed);p={k:torch.zeros(n,w) for k,w in STATE_FIELDS}
    p.update(final_action=torch.zeros(n,18),foot_contact=torch.ones(n,4,dtype=torch.bool),
             ball_present=torch.zeros(n,dtype=torch.bool),elapsed_s=torch.tensor(elapsed)*.01)
    b=dict(physical=p,task={},recovery={},event_state=torch.zeros(n,8),episode_index=torch.tensor(episode),
           elapsed_tick=torch.tensor(elapsed),auxiliary=dict(reward_inputs=dict(applied_torque_all=torch.zeros(n,18))))
    return dict(before=b,after=b,executor_before={'tensors':{'controller/lanes':torch.arange(n)},'values':{'controller/count':n}},
                observation=torch.zeros(n,794),final_joint_targets=torch.zeros(n,18),reward=torch.full((n,),float(t+1)),
                terminated=torch.tensor(done),timeout=torch.zeros(n,dtype=torch.bool),reward_components={'complete5_terminal':torch.zeros(n)})

def test_gate_single_lane_and_effective_lane_normalization():
    assert active_gate(torch.ones(1));assert not active_gate(torch.zeros(1))
    assert not active_gate(torch.tensor([1.,0.,0.,0.]))
    x=torch.tensor([2.,50.],requires_grad=True);loss=masked_lane_mean(x,torch.tensor([True,False]))
    loss.backward();assert x.grad.tolist()==[1.,0.]

def test_failure_pool_distinguishes_unknown_unfinished_and_invalid_outgoing():
    t,n=20,2;meta={k:torch.full((t,n),float('nan')) for k in FIELDS+tuple('next_'+k for k in FIELDS[2:])}
    meta['approach_gate'][:]=1;meta['racket_distance'][:]=.1;meta['had_contact'][:]=0
    meta['completed_miss']=torch.zeros(t,n,dtype=torch.bool)
    valid=torch.ones(t-15,n,dtype=torch.bool)
    masks=approach_window_masks(meta,valid,16)
    assert masks['approach_active'].all() and masks['hit_prepare'].all()
    assert not masks['approach_failure'].any()
    meta['released'][:]=0;meta['next_released'][10,0]=1;meta['next_qualified'][10,0]=1;meta['next_attack'][10,0]=0
    masks=approach_window_masks(meta,valid,16)
    assert masks['approach_failure'][:,0].all() and not masks['approach_failure'][:,1].any()

def test_frozen_refresh_unequal_reset_boundaries_and_manifest(tmp_path):
    f=FrozenRefresh(2,3,.9,'policy_fixed',Schema(),stage_ticks=(0,1,2))
    # lane0 begins with an old suffix; lane1 begins at a real reset.
    sequence=[([2,0],[0,0],[True,False]),([0,1],[1,0],[False,False]),
              ([1,2],[1,0],[False,True]),([2,0],[1,1],[True,False])]
    for t,(elapsed,ep,done) in enumerate(sequence): f.append(row(t,elapsed,ep,done))
    rtg,report=f.result()
    assert report['complete_lanes']==2 and report['lengths']==[3,3]
    torch.testing.assert_close(rtg[0],torch.tensor([2+.9*3+.81*4,1+.9*2+.81*3],dtype=torch.float64))
    assert report['returns']==[9.,6.]
    path=tmp_path/'manifest.json';path.write_text(json.dumps(report));assert json.loads(path.read_text())['policy_id']=='policy_fixed'
    assert len(f.starts)<=3
    rows=list(f.calibration_rows())
    assert all(len(start['before']['physical']['joint_position'])==2 for _,start,_ in rows)
    assert report['stage_coverage']=={'0':2,'1':2,'2':2}
    assert {(stage,int(i)) for stage,_,mask in rows for i in torch.where(mask)[0]}=={(s,i) for s in range(3) for i in range(2)}

def test_incomplete_refresh_refuses_value_labels():
    f=FrozenRefresh(1,3,.9,'frozen',Schema());f.append(row(0,[0],[0],[False]))
    with pytest.raises(RuntimeError,match='Incomplete'):f.result()

def test_full_execution_snapshot_masks_all_lanes_not_fifo_last_eight():
    n=128;one=row(0,[0]*n,[0]*n,[False]*n)
    from quadsmash.rl.supermimic_physical_loop_replay_v1 import row_tensors
    package={'tensors':{k:v.unsqueeze(0) for k,v in row_tensors(one,Schema()).items()}}
    replay=PhysicalLoopReplay(package,Schema(),capacity_ticks=64);replay.new_stream()
    replay.append(one)
    start=replay.stage_starts['prepare'][0]
    assert start['_actor_lane_mask'].sum()==128
    assert start['executor_before']['values']['controller/count']==128
    assert start['observation'].shape==(128,794)
    metadata={k:torch.zeros(n) for k in FIELDS}
    metadata['approach_gate'][37]=1
    replay.append(one,approach_metadata=metadata)
    gated=replay.stage_starts['approach_active'][0]
    assert gated['_actor_lane_mask'].sum()==1 and gated['_actor_lane_mask'][37]
    assert gated['observation'].shape==(128,794)

def test_native_metadata_uses_real_gate_and_asset_kinematics():
    from quadsmash.rl.supermimic_approach_data_v1 import native_approach_metadata
    from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1,MODEL
    g=FullBodyGeometryV1(json.loads(MODEL.read_text())['asset_sha256'])
    r=row(0,[0,0],[0,0],[False,False]);r['before']['physical']['root_quaternion_w'][:,0]=1
    actor=SimpleNamespace(approach_gate=lambda obs: torch.tensor([[1.],[0.]]))
    metadata=native_approach_metadata(r,actor,g)
    assert metadata['approach_gate'].tolist()==[1.,0.]
    assert torch.isfinite(metadata['racket_distance']).all()
    assert metadata['next_attack'].isnan().all() # unavailable terminal evidence stays unknown

def test_gradient_components_identify_anchor_only():
    p=torch.nn.Parameter(torch.tensor([1.,2.]))
    stats=gradient_parts(dict(task=p.sum()*0,tracking=p.sum()*0,anchor=p.square().sum()),[p])
    assert stats['task_gradient_norm']==0 and stats['anchor_gradient_norm']>0
    assert p.grad is None


def test_compact_critic_snapshots_preserve_async_lane_time_identity():
    encoder=lambda start:torch.stack((start['before']['episode_index'],start['before']['elapsed_tick']),1).float()
    f=FrozenRefresh(2,3,.9,'frozen',Schema(),stage_ticks=(0,1,2),start_encoder=encoder)
    sequence=[([2,0],[0,0],[True,False]),([0,1],[1,0],[False,False]),
              ([1,2],[1,0],[False,True]),([2,0],[1,1],[True,False])]
    for t,(elapsed,ep,done) in enumerate(sequence):f.append(row(t,elapsed,ep,done))
    seen=set()
    for stage,start,mask in f.calibration_rows():
        assert 'executor_before' not in start
        for lane in torch.where(mask)[0].tolist():
            assert start['critic_input'][lane].tolist()==[1-lane,stage]
            seen.add((lane,stage))
    assert len(seen)==6


def test_initial_archive_survives_online_ring_wrap_and_holdout_exclusion():
    from quadsmash.rl.supermimic_physical_loop_replay_v1 import row_tensors
    initial=row(0,[0,0],[0,0],[False,False])
    replay=PhysicalLoopReplay({'tensors':{k:v.unsqueeze(0) for k,v in row_tensors(initial,Schema()).items()}},Schema(),capacity_ticks=64)
    replay.new_stream()
    for t in range(70):replay.append(row(t,[t,t],[1,1],[False,False]),retain_start=False,policy_version=2)
    d,_,_,sources,_,_=replay._ordered()
    assert len(d['state'])==65 and (sources==0).sum()==2
    window,lanes,report=replay.window(steps=16,batch=4,device='cpu',excluded_lanes=[1])
    assert lanes.tolist()==[0]*4
    assert not (window['episode']==0).any() # the singleton initial window cannot cross into online
