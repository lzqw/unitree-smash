import copy
import unittest

import torch

from quadsmash.rl.success_retention_v1 import SuccessRetention


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(794,18)
        self.critic = torch.nn.Linear(794,1)
        self.log_std = torch.nn.Parameter(torch.zeros(18))
        self.register_buffer('normalization_count',torch.tensor(100))

    def act_inference(self,obs):
        return self.actor(obs['policy'])


class RetentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(771)
        self.policy=Policy()
        self.public=torch.randn(32,794)
        self.targets=torch.zeros(32,18)
        self.offsets=torch.tensor([0,16,32])
        self.configuration=dict(learning_rate=1e-4,steps_per_update=4,batch_size=16,
                                seed=891,anchor_weight=1.,manifest='test-only',manifest_sha256='test-only')

    def make(self,policy=None):
        return SuccessRetention(policy or self.policy,self.public,self.targets,
                                self.offsets,self.configuration)

    def test_actor_only_update_reduces_error(self):
        initial={k:v.clone() for k,v in self.policy.state_dict().items()}
        retention=self.make()
        stats=retention.step()
        self.assertLess(stats['mse_after'],stats['mse_before'])
        self.assertEqual(retention.completed_steps,4)
        for k,v in initial.items():
            if not k.startswith('actor.'):
                self.assertTrue(torch.equal(v,self.policy.state_dict()[k]),k)

    def test_critic_warmup_skips_optimizer_and_sampler(self):
        for p in self.policy.actor.parameters():p.requires_grad_(False)
        retention=self.make();rng=retention.generator.get_state().clone()
        self.assertTrue(retention.step()['skipped_actor_warmup'])
        self.assertFalse(retention.optimizer.state)
        self.assertTrue(torch.equal(rng,retention.generator.get_state()))

    def test_resume_restores_optimizer_and_episode_sampler(self):
        first=self.make();first.step()
        second_policy=copy.deepcopy(self.policy)
        second=self.make(second_policy)
        second.load_state_dict(copy.deepcopy(first.state_dict()))
        a,b=first.step(),second.step()
        self.assertEqual(a,b)
        for k,v in self.policy.state_dict().items():
            self.assertTrue(torch.equal(v,second_policy.state_dict()[k]),k)

    def test_rejects_changed_retention_contract(self):
        retention=self.make();saved=retention.state_dict()
        saved=copy.deepcopy(saved);saved['configuration']['manifest_sha256']='changed'
        with self.assertRaises(ValueError):retention.load_state_dict(saved)


if __name__=='__main__':unittest.main()
