import unittest
import torch
from quadsmash.control.observed_posthit_gate_v1 import ObservedPostHitGateV1


class GateTests(unittest.TestCase):
    def test_no_pre_hit_support_then_persistent_latch_and_reset(self):
        gate=ObservedPostHitGateV1(2,'cpu')
        obs=torch.zeros(2,794);obs[:,774]=1
        self.assertEqual(gate.update(obs).tolist(),[False,False])
        obs[:,775]=.01;obs[:,774]=0;obs[0,773]=1
        self.assertEqual(gate.update(obs).tolist(),[False,False])
        obs[:,775]=.02;obs[:,773]=0;obs[:,774]=1
        self.assertEqual(gate.update(obs).tolist(),[True,False])
        obs[:,775]=.03;obs[:,774]=0
        self.assertEqual(gate.update(obs).tolist(),[True,False])
        obs[:,775]=0;obs[:,774]=1
        self.assertEqual(gate.update(obs).tolist(),[False,False])

    def test_reject_mid_episode_start_and_duplicate_tick(self):
        gate=ObservedPostHitGateV1(1,'cpu');obs=torch.zeros(1,794);obs[:,775]=.5
        with self.assertRaises(ValueError):gate.update(obs)
        obs[:,775]=0;gate.update(obs)
        with self.assertRaises(ValueError):gate.update(obs)


if __name__=='__main__':unittest.main()
