"""New tracking metadata must survive a genuine early native termination."""
import json
import numpy as np
import torch
from scripts.native_reference_tracking_v1 import NativeReferenceTrackingV1


def test_tracking_stops_after_termination_and_saves_negative_result(tmp_path):
    adapter=object.__new__(NativeReferenceTrackingV1)
    adapter.case=3;adapter.start=280;adapter.steps=520;adapter.last_tick=280
    adapter.applied=1;adapter.early_termination_tick=None;adapter.output=tmp_path
    adapter.manifest=dict(bundle_sha256='fixture',reference_tracking_definition={'test_only':True})
    adapter.query_check={'passed':True}
    shapes=dict(q=(25,),v=(24,),contacts=(4,),foot_force_world_n=(4,3),student_action=(18,),
        final_action=(18,),previous_target_rad=(18,),requested_target_rad=(18,),final_target_rad=(18,))
    adapter.records=[dict(control_tick=280,offset=0,mode='fixture',prediction_error=None,
        **{k:np.zeros(shape) for k,shape in shapes.items()})]
    adapter.feedback_records=[dict(control_tick=280,reference_tracking={'active':False})]
    student=torch.zeros((128,18));finished=torch.zeros(128,dtype=torch.bool);finished[3]=True
    for tick in (281,282):
        result=adapter.action(tick,torch.zeros((128,794)),student,finished)
        assert torch.equal(result,student)
    result=adapter.save()
    assert result['applied_steps']==1 and result['early_termination_tick']==281
    assert not result['full_feedback_sequence_executed'] and not result['valid_expert_labels']
    assert not result['online_mpc'] and len(result['feedback_records'])==1
    assert json.loads((tmp_path/'sequence_native_result.json').read_text())['reference_tracking_definition']=={'test_only':True}
