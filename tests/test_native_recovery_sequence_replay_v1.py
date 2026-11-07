"""Dispatch tests with archived states; these do not execute native physics."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scripts.native_recovery_sequence_replay_v1 import NativeRecoverySequenceReplay
from scripts.prepare_native_recovery_sequence_v1 import transform


def setup_replay(tmp_path):
    fixture=Path(__file__).parent/'fixtures/native_geometry_sequence_case15.npz'
    bundle=tmp_path/'bundle';bundle.mkdir();archive=bundle/'sequences.npz';archive.write_bytes(fixture.read_bytes())
    with np.load(archive,allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    manifest=dict(bundle_name=archive.name,bundle_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        policy_sha256='test-only-policy',native_execution=False,valid_expert_labels=False,
        ground_geometry_sha256='test-only-geometry',source_sha256={},control_dt_s=.01,
        configuration=dict(interface=dict(limits=arrays['joint_limits_rad'].tolist()),arm_target_rate_rad_s=10.),
        rows=[dict(case=15,query_tick=2)])
    (bundle/'manifest.json').write_text(json.dumps(manifest));output=tmp_path/'output';output.mkdir()
    c=NativeRecoverySequenceReplay(bundle,15,torch.tensor(arrays['joint_limits_rad']),
        'test-only-policy',output)
    c.query_public[775]=.02  # Synthetic test clock only, not the native bundle.
    public=torch.zeros(128,794);public[15]=torch.tensor(c.query_public)
    action=torch.full((128,18),.1);action[15]=torch.tensor(c.query_previous_action)
    finished=torch.zeros(128,dtype=torch.bool)
    for tick in range(2):
        public[15,775]=tick*.01
        torch.testing.assert_close(c.action(tick,public,action,finished),action,rtol=0,atol=0)
    def state(offset):
        q=c.expected_q[offset].copy();v=c.expected_v[offset].copy()
        return SimpleNamespace(q=q,v=v,contacts=c.expected_contacts[offset].copy(),
            joint_position=q[c.model.q_indices],measured_force_world=np.zeros((4,3)))
    return c,public,action,finished,state


def test_exact_prefix_and_untouched_lanes_then_bounded_handoff(tmp_path):
    c,public,action,finished,state=setup_replay(tmp_path);others=np.arange(128)!=15
    for offset in range(41):
        tick=offset+2;public[15,775]=tick*.01
        if offset==40:action[15,12:]=1.
        result=c.action(tick,public,action,finished,state(offset))
        torch.testing.assert_close(result[others],action[others],rtol=0,atol=0)
        if offset<40:np.testing.assert_allclose(c.decode(result[15]),c.commands[offset],atol=5e-7,rtol=0)
        else:assert max(abs(c.previous[12:]-c.commands[-1,12:]))<=.100001
    assert c.applied==40 and c.query_check['passed']
    saved=c.save();assert not saved['sustained_mpc_feedback_implemented']
    assert not saved['valid_expert_labels']


@pytest.mark.parametrize('kind',['configuration','velocity','contacts','previous'])
def test_query_mismatch_aborts_before_first_override(tmp_path,kind):
    c,public,action,finished,state=setup_replay(tmp_path);native=state(0);public[15,775]=.02
    if kind=='configuration':native.q[0]+=.01
    if kind=='velocity':native.v[0]+=.1
    if kind=='contacts':native.contacts[0]=not native.contacts[0]
    if kind=='previous':c.previous[12]+=.01
    with pytest.raises(ValueError,match='Actual query'):
        c.action(2,public,action,finished,native)
    assert c.applied==0 and not c.started


def test_entry_is_isolated_and_preserves_full_native_episode_loop():
    root=Path(__file__).resolve().parents[1]
    source=(root/'scripts/evaluate_shared_target_ground64_persistent_recovery_v1.py').read_text()
    result=transform(source,root)
    assert 'for tick in range(requested)' in result
    assert 'action=native_replay.action(tick,obs,action,finished,replay_state)' in result
    assert "sequence_result=native_replay.save()" in result
    assert 'len(first) != count or len(jumps) != count' in result
