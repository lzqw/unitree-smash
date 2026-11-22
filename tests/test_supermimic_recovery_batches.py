import torch

from quadsmash.rl.supermimic_recovery_batches import recovery_batch, pooled_objective


def test_recovery_layout_preserves_recorded_start_identity_after_archive_restore():
    from quadsmash.rl.supermimic_physical_loop_replay_v1 import _current_start_view, current_start

    causal = dict(before=dict(elapsed_tick=torch.tensor([700]), episode_index=torch.tensor([0])),
                  executor_before={}, observation=torch.zeros(1, 3))
    origin = dict(native_tick=7098, stream_id=12, policy_version=133, actor_state_sha256='actor',
                  collection_mode='mixed_training')
    raw = dict(causal, **origin, reward=4., after={'event': True})
    live = current_start(raw)
    restored = current_start(_current_start_view(raw))
    for record in (live, restored):
        assert set(record) == set(causal) | set(origin)
        _, layout = recovery_batch({'touchdown': [(record, torch.tensor([True]))]}, 1, 1)
        assert layout[0]['native_origin'] == origin
        assert layout[0]['lane_episode_tick'] == [[0, 0, 700]]
    _, legacy = recovery_batch({'initial_archive': [(causal, torch.tensor([True]))]}, 1, 1)
    assert legacy[0]['native_origin'] is None


def test_sparse_starts_preserve_episode_context_and_do_not_repeat():
    def row(tick):
        return {'before':{'elapsed_tick':torch.full((4,),tick),'episode_index':torch.tensor([1,2,3,4])},
                'executor_before':{'tick':tick}}
    dense=row(400)
    rare=row(520)
    pools={'failure':[(rare,torch.tensor([0,1,0,0],dtype=torch.bool))],
           'archive':[(dense,torch.ones(4,dtype=torch.bool))]}
    batch,meta=recovery_batch(pools,4,1)
    assert [r['executor_before'] for r in batch]==[rare['executor_before'],dense['executor_before']]
    assert batch[0]['_actor_lane_mask'].tolist()==[False,True,False,False]
    keys=[(lane,ep) for record in meta for lane,ep,_ in record['lane_episode_tick']]
    assert len(set(keys))==len(keys)==4
    assert '_actor_lane_mask' not in dense


def test_pooled_gradient_uses_lane_and_reference_denominators_and_true_bootstrap():
    class Trainer:
        block_name='recovery'
        actor=torch.nn.Linear(1,1,bias=False)
        anchor_coeff=.5
        def tracking_coefficient(self,_):
            return 2.
        def _rollout_objective(self,state,effective_weights=None,tracking_parameters=()):
            n,h,w=state
            theta=self.actor.weight.squeeze()
            valid=torch.ones(h,n,dtype=torch.bool)
            return dict(gate_lanes=n,gate_steps=n*h,tracking_steps=h,tracking_active_lane_steps=n*h,
                        tracking_effective_weight_sum=w,tracking_nonzero_gradient_lanes=n,matched_references=n,
                        task=n*theta,tracking=h*theta,anchor=theta,reference_coverage=dict(selected_start_lanes=n,
                        gate_active_selected_start_lanes=n),effective_weights=valid[0],ret=theta.expand(n,1),
                        critic_data=(torch.full((h,n,2),1.),torch.ones(h,n,1),torch.zeros(h,n,1),torch.full((n,2),h),valid))
    trainer=Trainer()
    states=[(1,2,1.),(3,3,6.)]
    out=pooled_objective(trainer,states)
    out['loss'].backward()
    torch.testing.assert_close(trainer.actor.weight.grad.squeeze(),torch.tensor(2.5+2*(2+18)/7+.5))
    states0,_,_,final,valid=out['critic_data']
    assert states0.shape==(3,4,2) and final[:,0].tolist()==[2,3,3,3]
    assert valid.sum()==11 and not valid[0,0] and valid[1:,0].all()
    with torch.no_grad():
        torch.testing.assert_close(pooled_objective(trainer,states,out['effective_weights'])['loss'],out['loss'])


def test_loaded_history_retains_unmatched_native_substeps():
    from scripts.audit_supermimic_loaded_contact_history import compare
    def event(frame):
        return dict(frame=frame,friction_points=1,normal_points=1,anchors_local_m=[[.01,0.,0.]],
                    inferred_anchor_shape=[7],normal_w=[[1.,0.,0.]])
    audit=compare([event(2),event(4)],[event(3),event(4)])
    assert [(x['frame_minus'],x['frame_plus']) for x in audit]==[(2,None),(None,3),(4,4)]
    assert audit[-1]['inferred_same_shape']==[True]


def test_loaded_contacts_match_force_vectors_by_geometry_before_measuring_cancellation():
    from scripts.audit_supermimic_loaded_contact_history import compare
    common=dict(frame=4,friction_points=2,normal_points=1,inferred_anchor_shape=[7,7],normal_w=[[1.,0.,0.]])
    minus=dict(common,anchors_local_m=[[0.,0.,0.],[0.,.01,0.]],
               anchor_friction_dv_mps=[[0.,.4,0.],[0.,.6,0.]],tangent_dv_mps=[0.,1.,0.])
    plus=dict(common,anchors_local_m=[[0.,.01,0.],[0.,0.,0.]],
              anchor_friction_dv_mps=[[0.,.8,0.],[0.,.2,0.]],tangent_dv_mps=[0.,1.,0.])
    row,=compare([minus],[plus])
    assert row['matched_indices']==[[0,1],[1,0]]
    assert row['next_assignment_total_distance_margin_m']>.019
    assert row['net_delta_norm_mps']==0 and row['retained_net_error_mps']<1e-7
    assert abs(row['sum_anchor_delta_norm_mps']-.4)<1e-6


def test_cached_tracking_vjp_preserves_value_and_does_not_traverse_rollout_again():
    from quadsmash.rl.supermimic_recovery_batches import _gradient_value

    param = torch.tensor(2., requires_grad=True)
    unused = torch.tensor(7., requires_grad=True)
    middle = param.square()
    tracking = middle * 3.
    # Compute the lane diagnostic and the parameter derivative in the same traversal.
    state_grad, *derivatives = torch.autograd.grad(tracking, (middle, param, unused), allow_unused=True)
    assert state_grad == 3.
    reconstructed = _gradient_value(tracking, (param, unused), derivatives)
    assert reconstructed.item() == tracking.item()
    reconstructed.backward()
    assert param.grad == 12. and unused.grad is None


def test_recovery_prefix_is_causal_before_body_contact_and_reset_safe():
    from quadsmash.rl.supermimic_physical_loop_replay_v1 import PhysicalLoopReplay
    replay = PhysicalLoopReplay.__new__(PhysicalLoopReplay)
    replay.current_stream, replay.lanes, replay.stage_starts = 2, 4, {}
    def start(t):
        return dict(before=dict(elapsed_tick=torch.full((4,), t), episode_index=torch.zeros(4),
                                task=dict(had_contact=torch.ones(4,dtype=torch.bool), released=torch.ones(4,dtype=torch.bool)),
                                event_state=torch.zeros(4,8)), executor_before={'clock':t}, observation=t)
    past = start(100)
    past['before']['task']['released'][1] = False
    past['before']['event_state'][2,5] = 1
    current = start(119)
    current['before']['episode_index'][3] = 1
    replay.new_starts = []
    replay._retain_body_failure_prefix(past, torch.zeros(4,dtype=torch.bool), 400, 9)
    replay.new_starts = [past] + [start(t) for t in range(101,119)]
    event = torch.ones(4,dtype=torch.bool)
    replay._retain_body_failure_prefix(current,event,419,8)
    assert not replay.stage_starts
    replay._retain_body_failure_prefix(current,event,419,9)
    stored, = replay.stage_starts['pre_body_ground']
    assert stored['_actor_lane_mask'].tolist()==[True,False,False,False]
    assert stored['before'] is past['before'] and stored['executor_before']=={'clock':100}


def test_start_cache_roundtrip_keeps_only_current_state_and_native_identity(tmp_path, monkeypatch):
    import hashlib
    import pytest
    from quadsmash.rl import supermimic_physical_start_cache_v1 as cache

    for key in ('APPROACH_TICKS', 'RECOVERY_TICKS', 'CRITIC_TICKS'):
        monkeypatch.setattr(cache, key, (0, 1))
    archive = tmp_path/'archive'
    archive.mkdir()
    common = dict(before={}, executor_before={}, observation=torch.zeros(1, 3))
    new = dict(common, policy_version=17, native_tick=320, stream_id=3,
               collection_mode='mixed_training', actor_state_sha256='recorded')
    path = archive/'chunk.pt'
    torch.save(dict(rows=[dict(common, after={}), dict(new, reward=1.)]), path)
    manifest = dict(chunks=[dict(file=path.name, ticks=2, sha256=hashlib.sha256(path.read_bytes()).hexdigest())])
    for _ in range(2):
        starts = cache.load_start_cache(archive, manifest)
        assert starts[0].keys() == common.keys() and starts[1].keys() == new.keys()
        assert starts[1]['policy_version'] == 17

    saved_path = tmp_path/'physical_current_starts_v4.pt'
    saved = torch.load(saved_path, weights_only=True)
    saved['starts'][1]['after'] = {}
    torch.save(saved, saved_path)
    with pytest.raises(ValueError, match='unexpected fields'):
        cache.load_start_cache(archive, manifest)
