import numpy as np
import pytest
import torch
from quadsmash.control.learned_recovery_command_v1 import (
    INPUT_SIZE,SCHEMA,RecoveryCommandHead,NativeRecoveryCommandAdapter,compose_native_delta,
    admit_native_recovery_head)
from quadsmash.rl.recovery_command_dataset_v1 import compose_native_recovery_delta


def test_zero_initialization_and_torch_mapping_agree_with_audited_coordinate():
    head=RecoveryCommandHead(torch.zeros(INPUT_SIZE),torch.ones(INPUT_SIZE))
    assert torch.equal(head(torch.randn(5,INPUT_SIZE)),torch.zeros(5,18))
    generator=np.random.default_rng(17);base=generator.uniform(-1,1,(5,18));previous=generator.uniform(-1,1,(5,18))
    delta=generator.uniform(-1,1,(5,18));active=np.array([0,1,1,0,1],dtype=bool);limits=np.tile([-2.,3.],(18,1))
    expected=compose_native_recovery_delta(base,delta,previous,active,limits)
    actual=compose_native_delta(*[torch.tensor(x) for x in (base,delta,previous,active,limits)]).numpy()
    np.testing.assert_allclose(actual,expected,atol=1e-12)


def test_causal_gate_and_per_lane_reset_preserve_nonrecovery_commands():
    head=RecoveryCommandHead(torch.zeros(INPUT_SIZE),torch.ones(INPUT_SIZE))
    with torch.no_grad():head.network[-1].bias.fill_(.5)
    adapter=NativeRecoveryCommandAdapter(head,2,'cpu',np.tile([-2.,3.],(18,1)))
    base=torch.zeros(2,18)
    for tick in range(5):
        public=torch.zeros(2,794);public[:,775]=tick*.01
        if tick==1:public[:,774]=1.  # pre-hit support must not trigger
        if tick==2:public[0,773]=1.
        if tick==3:public[0,774]=1.
        action=adapter.action(public,base)
        assert torch.equal(action[1],base[1])
        if tick<3:assert torch.equal(action,base)
    assert adapter.active.tolist()==[True,False]
    public=torch.zeros(2,794);public[1,775]=.05
    action=adapter.action(public,base)
    assert not adapter.active.any() and torch.equal(action,base)


def test_missing_history_and_duplicate_tick_are_not_silently_accepted():
    head=RecoveryCommandHead(torch.zeros(INPUT_SIZE),torch.ones(INPUT_SIZE))
    adapter=NativeRecoveryCommandAdapter(head,1,'cpu',np.tile([-2.,3.],(18,1)))
    public=torch.zeros(1,794);public[:,775]=1.
    with pytest.raises(ValueError,match='ground reset'):adapter.action(public,torch.zeros(1,18))
    public[:,775]=0.;adapter.action(public,torch.zeros(1,18))
    with pytest.raises(ValueError,match='one call'):adapter.action(public,torch.zeros(1,18))


def test_checkpoint_admission_preserves_base_and_native_hardware_interface():
    head=RecoveryCommandHead(torch.zeros(INPUT_SIZE),torch.ones(INPUT_SIZE))
    limits=np.tile([-2.,3.],(18,1))
    ckpt=dict(schema=SCHEMA,head_state_dict=head.state_dict(),hidden=128,
        base_policy_sha256='base',protocol_sha256='protocol',joint_limits_rad=limits,
        control_dt_s=.01,arm_target_rate_rad_s=10.)
    restored=admit_native_recovery_head(ckpt,'base','protocol',limits,.01)
    assert torch.equal(restored(torch.zeros(1,INPUT_SIZE)),torch.zeros(1,18))
    with pytest.raises(ValueError,match='different frozen'):
        admit_native_recovery_head(ckpt,'other','protocol',limits,.01)
    with pytest.raises(ValueError,match='joint limits'):
        admit_native_recovery_head(ckpt,'base','protocol',limits*2,.01)
    with pytest.raises(ValueError,match='period'):
        admit_native_recovery_head(ckpt,'base','protocol',limits,.02)
