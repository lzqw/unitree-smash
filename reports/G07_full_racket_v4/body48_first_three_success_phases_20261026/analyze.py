"""Identify the actual actor collection phase of three closed TRAIN successes."""
from datetime import datetime,timezone
import hashlib,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
MAIN=ROOT.parent/'unitree_smash'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(MAIN))
from quadsmash.rl.episode_trace_v4 import unpack_public
from quadsmash.evaluation_v4 import independent_episode


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not(HERE/'PLAN.json').exists()
    run=MAIN/'reports/G07_full_racket_v4/body_residual_leg15_fresh_n2048_updates48_01'
    owner=MAIN/'reports/G07_full_racket_v4/body_residual48_after_grid16_queue01'
    stems=['lane0650_episode000000','lane1305_episode000001','lane1358_episode000002']
    progress=read(run/'progress.json')
    (HERE/'ACTUAL_PROGRESS_SNAPSHOT.json').write_text(json.dumps(progress,indent=2)+'\n')
    protocol_path=MAIN/'configs/g07_full_racket_eval_v4_r3.json'
    protocol=read(protocol_path);protocol['_root']=str(MAIN)
    warmup=read(owner/'ACTUAL_UPDATE7_CHECK.json')
    learning=read(owner/'ACTUAL_UPDATE8_CHECK.json')
    assert warmup['actor_weights_and_std_unchanged'] and not learning['actor_weights_and_std_unchanged']
    dependencies=[Path(__file__),protocol_path,owner/'ACTUAL_UPDATE7_CHECK.json',
        owner/'ACTUAL_UPDATE8_CHECK.json',HERE/'ACTUAL_PROGRESS_SNAPSHOT.json',
        MAIN/'scripts/train_full_racket_body_residual_ppo_v4.py',
        MAIN/'quadsmash/rl/full_racket_body_residual_ppo_v4.py',
        MAIN/'quadsmash/rl/episode_trace_v4.py']
    dependencies += [run/'trajectory_traces'/(stem+extension) for stem in stems for extension in ['.json','.npz']]
    plan=dict(created_at=datetime.now(timezone.utc).isoformat(),source_sha256={str(p):sha(p) for p in dependencies},
        scope='First three saved actual complete episodes only; no new simulation or frozen score.',
        collection_iteration_semantics='Zero-based runner.current_learning_iteration recorded before each collection; first actor optimization finishes update8, so changed actor weights first execute in collection iteration8.',
        contact_action='Last recorded public control clock at or before exact first contact substep.',
        no_causal_credit_assigned=True,new_native_execution=False,formal_acceptance=False)
    (HERE/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    cases=[]
    for stem in stems:
        meta=read(run/'trajectory_traces'/(stem+'.json'))
        data_path=run/'trajectory_traces'/(stem+'.npz')
        assert sha(data_path)==meta['data_sha256']
        row=meta['raw_episode'];result=independent_episode(row,protocol)
        assert result['complete']
        with np.load(data_path) as data:
            public=unpack_public(data['packed_public'])
            iterations=data['learner_iteration'].copy()
            residual=data['residual_actions'].copy()
        assert hashlib.sha256(public.tobytes()).hexdigest()==meta['full_public794_sha256']
        clocks=public[:,775]
        contact_index=int(np.searchsorted(clocks,row['contact_time_s'],side='right')-1)
        assert 0<=contact_index<len(clocks) and row['contact_time_s']-clocks[contact_index]<.010001
        contact_iteration=int(iterations[contact_index])
        cases.append(dict(env_id=meta['env_id'],episode_index=meta['episode_index'],
            source_training_index=meta['source_training_index'],complete=True,
            contact_time_s=row['contact_time_s'],preceding_control_clock_s=float(clocks[contact_index]),
            contact_collection_iteration_zero_based=contact_iteration,
            actor_mean_still_initial_zero_at_contact=contact_iteration<=7,
            episode_collection_iterations=np.unique(iterations).tolist(),
            final_collection_iteration_zero_based=int(iterations[-1]),
            episode_contains_changed_actor_execution=bool((iterations>=8).any()),
            actual_sampled_residual_at_contact=residual[contact_index].tolist(),
            outgoing_speed_mps=float(np.linalg.norm(row['outgoing_velocity_mps'])),
            first_landing_m=row['first_landing_m'],qualified_final_recovery_s=row['jump']['recovery_s']))
    assert all(c['actor_mean_still_initial_zero_at_contact'] for c in cases)
    assert all(sha(Path(p))==digest for p,digest in plan['source_sha256'].items())
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),cases=cases,
        snapshot_completed_update=progress['update'],snapshot_closed_complete=progress['complete'],
        all_three_contacts_before_changed_actor_execution=True,
        later_recovery_can_include_changed_actor=True,
        interpretation='These three attacks occurred under the initial zero-mean outer residual plus sampled exploration. Two episodes continue into learned-actor execution during recovery. They do not establish a learned frozen-policy improvement or causal effect of learning.',
        continue_registered48_course_and_four_fixed_comparisons=True,
        new_native_execution=False,formal_acceptance=False,goal_complete=False)
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(contact_collection_iterations=[c['contact_collection_iteration_zero_based'] for c in cases],
        final_collection_iterations=[c['final_collection_iteration_zero_based'] for c in cases],
        snapshot_update=progress['update'],snapshot_complete=progress['complete'],formal_acceptance=False)))


if __name__=='__main__':main()
