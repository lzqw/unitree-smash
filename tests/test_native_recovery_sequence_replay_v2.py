"""Variable-length dispatch and the real timeout observation boundary."""
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.native_recovery_sequence_replay_v2 import NativeRecoverySequenceReplayV2
from test_native_recovery_sequence_replay_v1 import setup_replay


def test_eighty_commands_and_observed_endpoint_use_declared_length(tmp_path):
    c,public,action,finished,_=setup_replay(tmp_path)
    p=c.bundle/'manifest.json';m=json.loads(p.read_text());archive=c.bundle/m['bundle_name']
    with np.load(archive,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    # Synthetic dispatch fixture only: repeat the final reference/state.
    for suffix in ['commands_rad','predicted_q','predicted_v','predicted_contacts']:
        key='case_015_'+suffix;arrays[key]=np.concatenate([arrays[key],np.repeat(arrays[key][-1:],40,axis=0)])
    np.savez_compressed(archive,**arrays);m['bundle_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    m['rows'][0]['horizon_steps']=80;p.write_text(json.dumps(m))
    import torch
    v2=NativeRecoverySequenceReplayV2(c.bundle,15,torch.tensor(c.limits),'test-only-policy',c.output)
    v2.query_public[775]=.02
    for tick in range(83):
        public[15,775]=tick*.01;state=None
        if tick>=2:
            i=tick-2;q=v2.expected_q[i]
            state=SimpleNamespace(q=q,v=v2.expected_v[i],contacts=v2.expected_contacts[i],
                joint_position=q[v2.model.q_indices],measured_force_world=np.zeros((4,3)))
        output=v2.action(tick,public,action,finished,state)
        torch.testing.assert_close(output[:15],action[:15],rtol=0,atol=0)
        torch.testing.assert_close(output[16:],action[16:],rtol=0,atol=0)
    result=v2.save()
    assert result['applied_steps']==80 and result['prediction_endpoint_observed']
    assert result['records'][-1]['offset']==80


def test_plan_ending_at_timeout_does_not_fabricate_terminal_observation(tmp_path):
    c,public,action,finished,state=setup_replay(tmp_path)
    manifest_path=c.bundle/'manifest.json';m=json.loads(manifest_path.read_text())
    m['rows'][0].update(query_tick=760,horizon_steps=40);manifest_path.write_text(json.dumps(m))
    import torch
    v2=NativeRecoverySequenceReplayV2(c.bundle,15,torch.tensor(c.limits),'test-only-policy',c.output)
    v2.query_public[775]=7.6
    for tick in range(800):
        public[15,775]=tick*.01
        v2.action(tick,public,action,finished,state(tick-760) if tick>=760 else None)
    result=v2.save()
    assert result['applied_steps']==40 and result['query_match']['passed']
    assert not result['prediction_endpoint_observed']
    assert result['records'][-1]['offset']==39
    assert not result['sustained_mpc_feedback_implemented']


@pytest.mark.parametrize('steps',[0,41,True])
def test_invalid_or_post_timeout_length_is_rejected(tmp_path,steps):
    c,*_=setup_replay(tmp_path);p=c.bundle/'manifest.json';m=json.loads(p.read_text())
    m['rows'][0].update(query_tick=760,horizon_steps=steps);p.write_text(json.dumps(m))
    import torch
    with pytest.raises(ValueError):NativeRecoverySequenceReplayV2(c.bundle,15,torch.tensor(c.limits),'test-only-policy',c.output)
