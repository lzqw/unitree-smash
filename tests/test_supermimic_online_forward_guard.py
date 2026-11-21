import copy

import pytest
import torch

from scripts.supermimic_online_forward_guard import compare_forward
from scripts.replay_supermimic_recovery_wm_updates import fit_readout_rows


def forward_scores():
    return {h:{group:dict(windows=2,body_rmse=.2,root_angular_mae=.15,
                          ball_velocity_rmse=.3,ball_forward_sign_error=.1)
               for group in ('uniform','contact-1','contact+0','contact+1')}
            for h in (1,2,4,8,16,32,64)}


def test_equal_loaded_weights_pass_online_monitoring():
    initial=forward_scores()
    assert compare_forward(initial,copy.deepcopy(initial),list(initial))['passed']


@pytest.mark.parametrize('horizon,group,metric',[(2,'uniform','root_angular_mae'),
                         (64,'uniform','ball_velocity_rmse'),(1,'contact-1','ball_velocity_rmse')])
def test_physical_drift_blocks_candidate(horizon,group,metric):
    initial=forward_scores()
    candidate=copy.deepcopy(initial)
    candidate[horizon][group][metric]+=.02
    result=compare_forward(initial,candidate,list(initial))
    assert not result['passed']


def test_missing_event_never_accepts_online_model():
    initial=forward_scores()
    candidate=copy.deepcopy(initial)
    del candidate[4]['contact-1']
    assert not compare_forward(initial,candidate,list(initial))['passed']


def test_offline_readout_update_keeps_other_outputs_and_encoder():
    model=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Linear(5,4))
    before=copy.deepcopy(model.state_dict())
    fit_readout_rows(model,model[-1],slice(1,3))
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    model(torch.ones(2,3)).sum().backward()
    opt.step()
    assert torch.equal(model[0].weight,before['0.weight'])
    for k,v in model[-1].named_parameters():
        assert torch.equal(v[[0,3]],before['1.'+k][[0,3]])
        assert not torch.equal(v[1:3],before['1.'+k][1:3])


@pytest.mark.parametrize('valid',[True,False])
def test_strict_native_promotion_binds_online_world(tmp_path,valid):
    import json
    from quadsmash.rl.supermimic_long_run_contract_v1 import sha256,verify_native_promotion
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
    world={'weight':torch.ones(1)}
    token=tensor_digest(world.items())
    checkpoint=tmp_path/'weights.pt'
    check=dict(passed=True,spec_sha256='spec',world_sha256=token if valid else 'different')
    torch.save(dict(config=dict(wm_forward_guard_sha256='spec'),world=world,
                    metadata=dict(online_forward_validation=check)),checkpoint)
    scores=tmp_path/'metrics.json'
    scores.write_text(json.dumps(dict(policy_execution={},evaluated_checkpoint_sha256=sha256(checkpoint),
                     num_envs=128,control_ticks=800,numerical_failures=0,native_episode_return_mean=-10,
                     ball_side_complete_except_recovery=54,complete5=28)))
    if valid:
        assert verify_native_promotion(scores,checkpoint,expected_execution={})
    else:
        with pytest.raises(ValueError,match='deployed WM'):
            verify_native_promotion(scores,checkpoint,expected_execution={})


@pytest.mark.parametrize('direction',[[1.,0.,0.],[0.,0.,1.]])
def test_friction_tangent_pair_respects_joint_limit(direction):
    from scripts.audit_supermimic_friction_load_response import independent_tangent_rows
    zero=torch.zeros(1,3)
    result=independent_tangent_rows(zero,zero,torch.tensor([3.,4.,2.]),zero[0],
                 torch.tensor([[0.,0.,1.]]),torch.tensor([2.]),1.,torch.tensor([direction]),2)
    assert torch.allclose(result,torch.tensor([-1.2,-1.6,0.]),atol=1.e-6)
