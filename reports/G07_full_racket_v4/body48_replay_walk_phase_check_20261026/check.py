"""Test the recorded50Hz global walk hold phase of a later native episode."""
from datetime import datetime,timezone
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3];MAIN=ROOT.parent/'unitree_smash';HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(MAIN))
from quadsmash.rl.episode_trace_v4 import unpack_public
from quadsmash.control.full_racket_fixed_actor_body_v4 import make_rally_controller
from quadsmash.control.full_racket_residual_v4 import ResidualActionMixerV4

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    assert not(HERE/'PLAN.json').exists()
    run=MAIN/'reports/G07_full_racket_v4/body_residual_leg15_fresh_n2048_updates48_01'
    path=run/'trajectory_traces/lane1824_episode000003.npz';meta_path=path.with_suffix('.json')
    admission=read(MAIN/'reports/G07_full_racket_v4/body_residual48_after_grid16_queue01/ADMISSION_DECISION.json')
    source_path=Path(admission['source']['policy']);source=torch.load(source_path,map_location='cpu',weights_only=False)
    assert sha(path)==read(meta_path)['data_sha256']
    provenance=read(run/'trajectory_traces/manifest.json')['provenance']
    (HERE/'ACTUAL_JOINT_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    paths=[Path(__file__),path,meta_path,source_path,HERE/'ACTUAL_JOINT_PROVENANCE.json',
        MAIN/'quadsmash/control/go2_walk_transfer.py',MAIN/'quadsmash/control/go2_standing_transfer.py']
    plan=dict(created_at=datetime.now(timezone.utc).isoformat(),source_sha256={str(p):sha(p) for p in paths},
        hypothesis='Asynchronous lane resets clear lane history but preserve the walking actor global two-step hold clock. A later episode may start on an odd control tick.',
        paired_replay_phases=['fresh_global_zero','actual_native_global_tick'],
        original_replay_tolerance=5e-4,new_native_execution=False,formal_acceptance=False)
    (HERE/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    with np.load(path) as arrays:data={k:arrays[k].copy() for k in arrays.files}
    public=unpack_public(data['packed_public']);iterations=data['learner_iteration']
    boundary=int(np.flatnonzero(iterations!=iterations[0])[0])
    start=int(iterations[boundary])*128-boundary
    assert np.array_equal((start+np.arange(len(public)))//128,iterations)
    default,limits=[torch.tensor(provenance[k],dtype=torch.float32)for k in ['default_joint_position_rad','joint_limits_rad']]
    receipts=[]
    for label,tick in [('fresh_global_zero',0),('actual_native_global_tick',start)]:
        controller=make_rally_controller(source,1,'cpu');mixer=ResidualActionMixerV4(1,'cpu',[.15]*18)
        controller.reference_controller.teacher.walk.tick=tick
        base_error=full_error=0.;first_difference=None
        with torch.inference_mode():
            for i,(obs,residual,actual_base,actual) in enumerate(zip(public,data['residual_actions'],data['base_actions'],data['native_actions'])):
                tensor=torch.from_numpy(obs[None]);residual=torch.from_numpy(residual[None])
                base=controller.action({'policy':tensor},default,limits)
                command=mixer.action(base,residual,tensor,default,limits)
                error=float(abs(base.numpy()[0]-actual_base).max())
                base_error=max(base_error,error);full_error=max(full_error,float(abs(command.numpy()[0]-actual).max()))
                if first_difference is None and error>plan['original_replay_tolerance']:
                    first_difference=dict(row=i,episode_clock_s=float(obs[775]),error=error)
        receipts.append(dict(phase=label,initial_global_walk_tick=tick,
            max_base_action_error=base_error,max_full_action_error=full_error,
            first_difference_over_original_tolerance=first_difference))
    assert receipts[0]['max_base_action_error']>plan['original_replay_tolerance']
    assert max(receipts[1]['max_base_action_error'],receipts[1]['max_full_action_error'])<=plan['original_replay_tolerance']
    assert all(sha(Path(p))==s for p,s in plan['source_sha256'].items())
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_native_start_global_control_tick=start,
        actual_first_walk_hold_phase=start%2,actual_recorded_rows=len(public),paired_replays=receipts,
        interpretation='The mismatch came from assuming global walking phase0 when extracting a later episode. Restoring the independently recovered native phase reproduces commands within the unchanged tolerance. No native/controller code, hardware, physics, outcome or motion was changed.',
        original_tolerance_retained=True,new_native_execution=False,formal_acceptance=False)
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
