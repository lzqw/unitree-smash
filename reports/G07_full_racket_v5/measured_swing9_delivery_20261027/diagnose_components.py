"""One-frame counterfactual arithmetic probes; never a deployable policy."""
import copy
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('downloaded_component_support',HERE/'replay_downloaded.py')
support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
np,torch=support.np,support.torch
packet,runtime,args=support.packet,support.runtime,support.args


def main():
    manifest=support.read(packet/'MANIFEST.json')
    resolve=lambda n:packet/manifest['aliases'].get(n,n)
    factory_path=resolve('search/extra_execution_sources/swing_controller.py')
    spec=importlib.util.spec_from_file_location('component_swing_factory',factory_path)
    factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
    policy=torch.load(resolve('search/candidate_003.pt'),map_location='cpu',weights_only=False)
    inputs=support.read(resolve('search/INPUTS.json'))
    default,limits=[torch.tensor(inputs[n],dtype=torch.float32) for n in ['default_joint_position_rad','joint_limits_rad']]
    with np.load(resolve('search/public_and_actions_003.npz')) as data:
        public,native,valid=[data[n].copy()[:350] for n in ['observations','actions','valid_first_episode']]
        assert data['training_indices'].tolist()[4]==112
    results=[]
    for mode in ['unchanged','suppress_face','double_face','double_preparation_first','double_preparation_second','double_preparation_both']:
        controller=factory.MeasuredSwingController(policy['source_public_policy'],6,'cpu',policy['runtime_overrides'])
        teacher=controller.reference_controller.teacher
        kinematics64=copy.deepcopy(teacher.kinematics).double()
        face64=copy.deepcopy(teacher.face_feedback).double()
        preparation64=copy.deepcopy(teacher.joint_preparation).double()
        original_face=teacher.face_feedback.forward
        original_preparation=teacher.joint_preparation.forward
        context=dict(frame=-1,preparation_calls=0)
        probes=[]
        def face(actual,normal,limits,kinematics,offset):
            result=original_face(actual,normal,limits,kinematics,offset)
            if context['frame']==323:
                alternative=face64(actual.double(),normal.double(),limits.double(),kinematics64,offset.double()).float()
                probes.append(dict(component='face',original_delta=result[4].tolist(),double_delta=alternative[4].tolist()))
                if mode in ['suppress_face','double_face']:
                    result=result.clone();result[4]=torch.zeros_like(result[4]) if mode=='suppress_face' else alternative[4]
            return result
        def preparation(arm,obs,limits,kinematics,actual_joints=None,post_stroke_mask=None):
            call=context['preparation_calls'];context['preparation_calls']+=1
            result=original_preparation(arm,obs,limits,kinematics,actual_joints,post_stroke_mask)
            if context['frame']==323:
                alternate=preparation64(arm.double(),obs.double(),limits.double(),kinematics64,
                    None if actual_joints is None else actual_joints.double(),post_stroke_mask).float()
                probes.append(dict(component='preparation',call=call,original_target=result[4].tolist(),double_target=alternate[4].tolist()))
                selected=(mode=='double_preparation_both' or mode=='double_preparation_first' and call==0
                          or mode=='double_preparation_second' and call==1)
                if selected:result=result.clone();result[4]=alternate[4]
            return result
        teacher.face_feedback.forward=face
        teacher.joint_preparation.forward=preparation
        output=[];previous=None
        with torch.inference_mode():
            for frame,value in enumerate(public):
                context.update(frame=frame,preparation_calls=0)
                obs=torch.from_numpy(value)
                if previous is not None:
                    ids=torch.where(obs[:,775]<previous-1e-5)[0]
                    if len(ids):controller.reset(ids)
                output.append(controller.action({'policy':obs},default,limits).numpy().copy())
                previous=obs[:,775].clone()
        output=np.stack(output);delta=abs(output-native)
        result=dict(mode=mode,original_tolerance=5e-4,prefix_control_ticks=350,
            actual_valid_public_rows=int(valid.sum()),maximum_normalized_error=float(delta[valid].max()),
            rows_exceeding_original_tolerance=int(((delta.max(2)>5e-4)&valid).sum()),one_frame_probes=probes)
        results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='one_frame_probes'}),flush=True)
    assert not support.original_reads
    record=dict(checked_at=datetime.now(timezone.utc).isoformat(),candidate_id=3,
        results=results,counterfactual_frame=323,counterfactual_training_index=112,
        counterfactual_overrides_are_diagnostic_only=True,original_policy_not_changed=True,
        GPU_component_outputs_not_recorded=True,causal_GPU_branch_not_directly_observed=True,
        source_program_sha256=support.sha(Path(__file__)),observed_original_project_Python_reads=support.original_reads,
        native_execution=False,formal_acceptance=False)
    with args.output.open('x') as stream:stream.write(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
