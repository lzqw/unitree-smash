import torch

from quadsmash.rl.supermimic_recovery_batches import recovery_batch, pooled_objective


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
        def _rollout_objective(self,state,effective_weights=None):
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
