import json
from pathlib import Path

import torch

from quadsmash.tasks.stand.recovery_baseline_diagnostics import strict_failures


def test_arm_only_excludes_concurrent_strict_failures():
    path = Path(__file__).resolve().parents[1] / 'configs/evaluation_protocol_v2.json'
    limits = json.loads(path.read_text())['recovery']
    zeros = torch.zeros(5)
    failed, arm_only = strict_failures(
        limits, support=torch.tensor([True, True, False, True, True]),
        tilt=zeros, speed=zeros, angular=zeros,
        leg_speed=torch.tensor([.8, 1.1, .7, 1., .2]),
        arm_speed=torch.tensor([1.1, .9, 1.2, 1., 1.1]),
        body_ground=torch.tensor([False, False, False, False, True]))
    assert failed[:, 4].tolist() == [True, True, True, False, True]
    assert arm_only.tolist() == [True, False, False, False, False]
