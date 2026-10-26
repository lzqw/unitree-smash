"""Replay closed real training traces through the frozen body and outer mixer.

The selected traces are diagnostics from a still-running native job. They are
not a random sample or a new performance evaluation. No physics is executed.
"""
from datetime import datetime,timezone
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
MAIN=ROOT.parent/'unitree_smash'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(MAIN))
from quadsmash.rl.episode_trace_v4 import unpack_public
from quadsmash.control.full_racket_fixed_actor_body_v4 import make_rally_controller
from quadsmash.control.full_racket_residual_v4 import ResidualActionMixerV4

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not(HERE/'PLAN.json').exists()
    run=MAIN/'reports/G07_full_racket_v4/body_residual_leg15_fresh_n2048_updates48_01'
    owner=MAIN/'reports/G07_full_racket_v4/body_residual48_after_grid16_queue01'
    assert Path('/proc/1818808/cmdline').read_bytes() and 'train_full_racket_body_residual_ppo_v4.py' in Path('/proc/1818808/cmdline').read_bytes().decode().replace('\0',' ')
    admission=read(owner/'ADMISSION_DECISION.json')
    assert all(sha(MAIN/name)==digest for name,digest in admission['source_hashes'].items())
    manifest=read(run/'trajectory_traces/manifest.json');entries=manifest['saved_episodes']
    assert entries and manifest['disabled_error'] is None
    (HERE/'ACTUAL_MANIFEST_SNAPSHOT.json').write_text(json.dumps(manifest,indent=2)+'\n')
    base_path=Path(admission['source']['policy']);base=torch.load(base_path,map_location='cpu',weights_only=False)
    assert sha(base_path)==read(run/'INPUTS.json')['contract']['base_policy_sha256']
    streams=[];dependencies=[Path(__file__),base_path,run/'INPUTS.json',run/'source_hashes.json',HERE/'ACTUAL_MANIFEST_SNAPSHOT.json']
    for entry in entries:
        path=run/'trajectory_traces'/entry['data_file'];meta=read(path.with_suffix('.json'))
        assert sha(path)==meta['data_sha256']==entry['data_sha256']
        with np.load(path) as data:arrays={key:data[key].copy() for key in data.files}
        public=unpack_public(arrays.pop('packed_public'))
        assert hashlib.sha256(public.tobytes()).hexdigest()==meta['full_public794_sha256']
        assert abs(public[0,775])<1e-6 and len(public)==meta['samples']
        streams.append(dict(public=public,meta=meta,**arrays));dependencies.extend([path,path.with_suffix('.json')])
    paths=[MAIN/'quadsmash/control/full_racket_fixed_actor_body_v4.py',MAIN/'quadsmash/control/full_racket_residual_v4.py',MAIN/'quadsmash/control/full_racket_arm_target_rate_v4.py']
    dependencies.extend(paths)
    plan=dict(created_at=datetime.now(timezone.utc).isoformat(),source_sha256={str(p):sha(p) for p in dependencies},
        actual_selected_episodes=len(streams),replay_tolerance=5e-4,
        scope='Selected closed TRAIN trajectories from active native job. Replay actual public inputs and sampled residuals; no new robot rollout or performance score.',
        declared_before_replay=True,not_random_population_estimate=True,formal_acceptance=False)
    (HERE/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    count=len(streams);controller=make_rally_controller(base,count,'cpu')
    mixer=ResidualActionMixerV4(count,'cpu',[.15]*18)
    mixer_only=ResidualActionMixerV4(count,'cpu',[.15]*18)
    provenance=manifest['provenance'];default=torch.tensor(provenance['default_joint_position_rad'],dtype=torch.float32)
    limits=torch.tensor(provenance['joint_limits_rad'],dtype=torch.float32)
    joint_range=(limits[:,1]-limits[:,0]).numpy()
    maximum_base=maximum_total=maximum_mixer=0.;requested=[];realized=[];clock_groups=[]
    for tick in range(max(len(s['public']) for s in streams)):
        keep=np.array([tick<len(s['public']) for s in streams])
        # Finished lanes retain their last observation only as padding; they
        # are excluded from all comparisons and statistics.
        index=[min(tick,len(s['public'])-1) for s in streams]
        public=torch.from_numpy(np.stack([s['public'][i] for s,i in zip(streams,index)]))
        residual=torch.from_numpy(np.stack([s['residual_actions'][i] for s,i in zip(streams,index)]))
        recorded_base=np.stack([s['base_actions'][i] for s,i in zip(streams,index)])
        actual=np.stack([s['native_actions'][i] for s,i in zip(streams,index)])
        command=controller.action({'policy':public},default,limits)
        predicted=mixer.action(command,residual,public,default,limits).numpy()
        only=mixer_only.action(torch.from_numpy(recorded_base),residual,public,default,limits).numpy()
        maximum_base=max(maximum_base,float(abs(command.numpy()[keep]-recorded_base[keep]).max()))
        maximum_total=max(maximum_total,float(abs(predicted[keep]-actual[keep]).max()))
        maximum_mixer=max(maximum_mixer,float(abs(only[keep]-actual[keep]).max()))
        requested.append(np.clip(residual.numpy()[keep],-1,1)*.15)
        realized.append((actual[keep]-recorded_base[keep])*.5*joint_range)
        clock_groups.append(public.numpy()[keep,775])
    assert max(maximum_base,maximum_total,maximum_mixer)<=plan['replay_tolerance']
    request,real=np.concatenate(requested),np.concatenate(realized)
    groups=[]
    for label,slice_ in [('legs',slice(0,12)),('arm',slice(12,18))]:
        requested_values=request[:,slice_];realized_values=real[:,slice_]
        groups.append(dict(joints=label,requested_joint_delta_RMS_rad=float(np.sqrt(np.mean(requested_values**2))),
            realized_native_minus_base_joint_target_delta_RMS_rad=float(np.sqrt(np.mean(realized_values**2))),
            requested_vs_realized_RMS_rad=float(np.sqrt(np.mean((requested_values-realized_values)**2))),
            maximum_absolute_realized_delta_rad=float(abs(realized_values).max())))
    assert all(sha(Path(name))==digest for name,digest in plan['source_sha256'].items())
    assert all(sha(MAIN/name)==digest for name,digest in admission['source_hashes'].items())
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_selected_episodes=len(streams),
        actual_public_rows=len(request),source_base_public_policy_sha256=sha(base_path),
        frozen_two_actor_base_max_normalized_action_error=maximum_base,
        complete_base_plus_recorded_residual_max_normalized_action_error=maximum_total,
        mixer_only_max_normalized_action_error=maximum_mixer,tolerance=plan['replay_tolerance'],
        all_actual_frozen_base_and_outer_commands_reproduced=True,selected_joint_delta_statistics=groups,
        selection='Existing recorder first failures, fixed lanes and complete traces only; not all TRAIN lanes.',
        scope=plan['scope'],new_native_execution=False,formal_acceptance=False)
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
