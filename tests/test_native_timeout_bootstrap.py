import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
from quadsmash.rl.native_ppo import FinalObservationPPO

def test_uses_final_value_and_never_bootstraps_a_true_terminal():
    before=TensorDict({'policy':torch.tensor([[2.],[5.]])},batch_size=[2])
    policy=ActorCritic(before,{'policy':['policy'],'critic':['policy']},1,
        actor_hidden_dims=[1],critic_hidden_dims=[1],actor_obs_normalization=False,critic_obs_normalization=False)
    with torch.no_grad():
        for layer in policy.critic.modules():
            if isinstance(layer,torch.nn.Linear):
                layer.weight.fill_(1.)
                layer.bias.zero_()
    alg=FinalObservationPPO(policy,device='cpu',gamma=.9,num_learning_epochs=1,num_mini_batches=1)
    alg.init_storage('rl',2,1,before,[1])
    with torch.inference_mode():
        alg.act(before)
        reset_obs=TensorDict({'policy':torch.tensor([[-50.],[-90.]])},batch_size=[2])
        alg.process_env_step(reset_obs,torch.ones(2),torch.ones(2,dtype=torch.long),{
            'time_outs':torch.tensor([True,True]),'terminated':torch.tensor([False,True]),
            'final_observation_valid':torch.tensor([True,True]),
            'final_observation':{'policy':torch.tensor([[7.],[11.]])}})
    assert torch.allclose(alg.storage.rewards[0,:,0],torch.tensor([7.3,1.]))
    assert torch.allclose(alg.last_timeout_bonus,torch.tensor([6.3,0.]))
