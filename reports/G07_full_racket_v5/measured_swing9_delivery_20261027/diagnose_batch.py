"""Compare actual candidate3 GPU commands with CPU6 and padded CPU128 batches."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path

# Reuse the downloaded checker's argument parser and live-project read audit.
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('downloaded_replay_support',HERE/'replay_downloaded.py')
support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
np,torch=support.np,support.torch
packet,runtime,args=support.packet,support.runtime,support.args


def main():
    manifest=support.read(packet/'MANIFEST.json')
    resolve=lambda name:packet/manifest['aliases'].get(name,name)
    factory_path=resolve('search/extra_execution_sources/swing_controller.py')
    spec=importlib.util.spec_from_file_location('batch_sensitive_actual_swing_factory',factory_path)
    factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
    policy=torch.load(resolve('search/candidate_003.pt'),map_location='cpu',weights_only=False)
    inputs=support.read(resolve('search/INPUTS.json'))
    default,limits=[torch.tensor(inputs[n],dtype=torch.float32) for n in ['default_joint_position_rad','joint_limits_rad']]
    with np.load(resolve('search/public_and_actions_003.npz')) as data:
        public,native,valid=[data[n].copy() for n in ['observations','actions','valid_first_episode']]
        indices=data['training_indices'].tolist()
    all_results={};summaries=[]
    for count in [6,128]:
        controller=factory.MeasuredSwingController(policy['source_public_policy'],count,'cpu',policy['runtime_overrides'])
        previous=None;actions=[]
        with torch.inference_mode():
            for value in public:
                padded=np.tile(value,(int(np.ceil(count/6)),1))[:count].copy()
                obs=torch.from_numpy(padded)
                if previous is not None:
                    ids=torch.where(obs[:,775]<previous-1e-5)[0]
                    if len(ids):controller.reset(ids)
                actual=controller.action({'policy':obs},default,limits).numpy()[:6].copy()
                actions.append(actual);previous=obs[:,775].clone()
        actual=np.stack(actions);all_results['CPU'+str(count)]=actual
        delta=abs(actual-native);kept=delta[valid]
        bad=np.argwhere((delta.max(2)>5e-4)&valid)
        maximum=np.unravel_index(np.argmax(np.where(valid[:,:,None],delta,-1)),delta.shape)
        first=None
        if len(bad):
            frame,lane=map(int,bad[0]);j=int(delta[frame,lane].argmax())
            first=dict(frame=frame,training_index=indices[lane],clock_s=float(public[frame,lane,775]),joint_index=j,
                CPU_command=float(actual[frame,lane,j]),actual_GPU_command=float(native[frame,lane,j]),
                normalized_error=float(delta[frame,lane,j]))
        summary=dict(CPU_batch_size=count,actual_valid_rows=int(valid.sum()),
            maximum_normalized_error=float(kept.max()),maximum_leg_error=float(delta[:,:,:12][valid].max()),
            maximum_arm_error=float(delta[:,:,12:][valid].max()),rows_exceeding_original_tolerance=int(len(bad)),
            first_exceedance=first,maximum_error_frame=int(maximum[0]),
            maximum_error_training_index=indices[maximum[1]],maximum_error_joint_index=int(maximum[2]))
        summaries.append(summary);print(json.dumps(summary),flush=True)
    difference=abs(all_results['CPU6']-all_results['CPU128'])
    arrays=args.output.with_suffix('.npz')
    with arrays.open('xb') as stream:np.savez_compressed(stream,**all_results,native=native,valid=valid,clock_s=public[:,:,775])
    record=dict(checked_at=datetime.now(timezone.utc).isoformat(),candidate_id=3,summaries=summaries,
        CPU6_vs_CPU128_max_valid_difference=float(difference[valid].max()),
        original_tolerance=5e-4,tolerance_changed=False,
        actual_other122_GPU_public_inputs_not_recorded=True,padding_is_batch_diagnostic_only=True,
        exact_original_GPU_batch_not_reconstructed=True,observed_original_project_Python_reads=support.original_reads,
        source_program_sha256=support.sha(Path(__file__)),arrays_sha256=support.sha(arrays),
        controller_source_sha256=support.sha(factory_path),native_execution=False,formal_acceptance=False)
    assert not support.original_reads
    with args.output.open('x') as stream:stream.write(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
