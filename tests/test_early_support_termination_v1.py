"""A real episode ending must not apply a stale teacher to the reset episode."""
import json
import numpy as np
import pytest
import torch
from scripts.native_early_support_tail_v1 import NativeEarlySupportTailV1


def test_terminated_episode_passes_student_and_preserves_negative_evidence(tmp_path):
    adapter=object.__new__(NativeEarlySupportTailV1)
    adapter.case=3;adapter.start=280;adapter.steps=520;adapter.last_tick=281
    adapter.applied=2;adapter.early_termination_tick=None;adapter.output=tmp_path
    adapter.manifest={'bundle_sha256':'fixture-only'};adapter.query_check={'passed':True}
    shapes=dict(q=(25,),v=(24,),contacts=(4,),foot_force_world_n=(4,3),
        student_action=(18,),final_action=(18,),previous_target_rad=(18,),
        requested_target_rad=(18,),final_target_rad=(18,))
    adapter.records=[dict(control_tick=280+i,offset=i,mode='fixture',prediction_error=None,
        **{k:np.zeros(s) for k,s in shapes.items()}) for i in range(2)]
    adapter.feedback_records=[{'control_tick':280},{'control_tick':281}]
    student=torch.randn(128,18);finished=torch.zeros(128,dtype=torch.bool);finished[3]=True
    for tick in [282,283]:
        result=adapter.action(tick,torch.zeros(128,794),student,finished)
        assert torch.equal(result,student)
    assert adapter.applied==2 and adapter.early_termination_tick==282
    with pytest.raises(ValueError,match='clock'):
        adapter.action(285,torch.zeros(128,794),student,finished)
    result=adapter.save()
    assert not result['full_feedback_sequence_executed']
    assert not result['valid_expert_labels'] and result['applied_steps']==2
    assert result['early_termination_tick']==282
    with np.load(tmp_path/'sequence_native_trace.npz') as data:assert data['q'].shape==(2,25)
    assert json.loads((tmp_path/'sequence_native_result.json').read_text())['early_termination_tick']==282
