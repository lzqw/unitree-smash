#!/usr/bin/env python3
"""Read-only native archive -> replay -> per-head WM forward/RTG audit.

R/A/R here denotes sampling and forward execution, NOT optimizer updates.
Historical model exposure is unknown; these are not certified holdout scores.
"""
import argparse,hashlib,json,time,traceback
from pathlib import Path
import torch
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop,tensor_digest
from quadsmash.rl.supermimic_physical_loop_replay_v1 import PhysicalLoopReplay
from quadsmash.rl.supermimic_approach_data_v1 import native_approach_metadata,APPROACH_QUOTA
from quadsmash.rl.supermimic_evidence_binding_v1 import complete_native_rtg
from quadsmash.rl.supermimic_wm_validation_v1 import validate_windows,command_sensitivity,event_aligned_windows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('archive','initial-packed-data','fresh-packed-data','physical-checkpoint','ball-checkpoint','task-checkpoint','safety-mapping','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(20261117)
    stage='LOAD'
    try:
        actor,adapter,_,launch,manifest=load_physical_loop(**{k:v for k,v in vars(a).items() if k not in ('output','initial_packed_data','fresh_packed_data')},device='cpu')
        actor.eval();adapter.eval()
        before=tensor_digest(adapter.state_dict().items());actor_before=tensor_digest(actor.state_dict().items())
        initial=torch.load(a.initial_packed_data,map_location='cpu',weights_only=True)
        from functools import partial
        from quadsmash.rl.supermimic_approach_data_v1 import recorded_approach_metadata
        labeler = partial(recorded_approach_metadata, schema=adapter.task_model.memory_schema,
                          actor=actor, geometry=adapter.physical_model.geometry)
        replay=PhysicalLoopReplay(initial,adapter.task_model.memory_schema,capacity_ticks=1600,approach_labeler=labeler)
        replay.new_stream(source='online',policy_version=0)
        stage='NATIVE_TO_REPLAY';ticks=0;started=time.perf_counter()
        for entry in manifest['chunks']:
            path=a.archive/entry['file']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Native chunk hash mismatch')
            rows=torch.load(path,map_location='cpu',weights_only=True)['rows']
            for row in rows:
                meta=native_approach_metadata(row,actor,adapter.physical_model.geometry)
                replay.append(row,retain_start=ticks in (270,460),policy_version=0,native_tick=ticks,approach_metadata=meta)
                ticks+=1
            del rows
            print(json.dumps(dict(replayed_ticks=ticks)),flush=True)
        appended_s=time.perf_counter()-started
        stage='REPLAY_WM_FORWARD';results=[]
        recovery_quota=dict(uniform=32,pre_touchdown=24,touchdown=24,post_touchdown=24,recovery_failure=16,stable_recovery=8)
        for attempt,head in enumerate(('recovery','approach','recovery')):
            start=time.perf_counter()
            window,lanes,sampling=replay.window(16,128,attempt,device='cpu',head=head,
                source_quota=dict(recent_online=64,older_online=32,initial_archive=32),
                stage_quota=APPROACH_QUOTA if head=='approach' else recovery_quota)
            sample_s=time.perf_counter()-start;start=time.perf_counter()
            errors=validate_windows(adapter,window,launch[lanes],horizons=(1,16))
            result=dict(head=head,sampling=sampling,sampling_s=sample_s,model_forward_s=time.perf_counter()-start,errors=errors)
            results.append(result);print(json.dumps(result),flush=True)
            (a.output/'progress.json').write_text(json.dumps(results,indent=2))
        stage='CONTACT_DIAGNOSTIC'
        fresh=torch.load(a.fresh_packed_data,map_location='cpu',weights_only=True)['tensors']
        aligned=event_aligned_windows(fresh,range(128),offsets=(-1,),maximum=16)['-1']
        pairs=aligned['starts'];sensitivity=None
        if len(pairs):
            t,lanes=pairs.unbind(1);at=t[None]+torch.arange(64)[:,None]
            window={k:v[at,lanes[None]] for k,v in fresh.items() if v.ndim>=2 and v.shape[:2]==fresh['episode'].shape}
            sensitivity=command_sensitivity(adapter,window,launch[lanes])
            errors=validate_windows(adapter,window,launch[lanes],horizons=(1,4,16,32,64))
        else:errors=None
        gamma=float(json.loads((a.archive.parent/'metrics.json').read_text())['native_return_gamma'])
        rtg,complete=complete_native_rtg(fresh,gamma)
        assert before==tensor_digest(adapter.state_dict().items()) and actor_before==tensor_digest(actor.state_dict().items())
        output=dict(replayed_ticks=ticks,new_transitions=ticks*128,replay_append_s=appended_s,head_forward_passes=results,
            contact_window_starts=pairs.tolist(),contact_errors=errors,command_sensitivity=sensitivity,
            native_gamma=gamma,completed_rtg_states=int(complete.sum()),excluded_incomplete_states=int((~complete).sum()),
            model_sha256=before,actor_sha256=actor_before,weights_unchanged=True,optimizer_updates=dict(actor=0,world=0,critic=0),
            limitations=['R/A/R sampling and WM forward only, no learning cycle',
                'Same development cases; historical WM training exposure unknown',
                'Command sensitivity is model-only; no paired native perturbation',
                'No updated-policy or gradient-caused native improvement claim'])
        (a.output/'result.json').write_text(json.dumps(output,indent=2))
    except BaseException as e:
        (a.output/'failure_traceback.txt').write_text(traceback.format_exc())
        (a.output/'failure.json').write_text(json.dumps(dict(stage=stage,type=type(e).__name__,message=str(e),optimizer_updates=0),indent=2))
        raise

if __name__=='__main__':main()
