"""Recover frozen40 assessment after a post-training episode-ID verifier bug.

The successful native training and failed historical queue remain untouched.
All original training checks are retained; only the incorrect fresh-episode0
assertions are replaced by strict per-lane source-to-final identifier chains.
Never retrain or restart an existing native job from this recovery program.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
OLD=ROOT/'reports/G07_full_racket_v4/joint_placement_resume32_queue01/run.py'
spec=importlib.util.spec_from_file_location('historical_joint40_queue',OLD)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
core=old.core;probe=old.probe;TRAIN=old.TRAIN;EVALUATION=old.EVALUATION
STEPS=old.STEPS;UPDATES=old.UPDATES;CUMULATIVE_STEPS=old.CUMULATIVE_STEPS;NUM_ENVS=old.NUM_ENVS
PYTHON=old.PYTHON
STATE=HERE/'state.json';PLAN=HERE/'RECOVERY_PLAN.json'


def save_once(path,value):
    assert not path.exists(), 'Preserve existing evidence: '+str(path)
    core.atomic(path,value)


def write(state,**extra):
    value=dict(state=state,queue_pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),
        historical_queue_state_preserved=True,formal_cases_read=False,formal_acceptance=False,goal_complete=False,**extra)
    core.atomic(STATE,value);print(json.dumps(value),flush=True)


def verify_resumed_training():
    import numpy as np
    import torch
    from quadsmash.evaluation_v4 import independent_episode
    job = core.verify_job(TRAIN, STEPS)
    inputs, metrics = core.read(TRAIN / 'INPUTS.json'), core.read(TRAIN / 'metrics.json')
    assert inputs['resume_checkpoint_sha256'] == core.sha(probe.TRAIN / 'checkpoint.pt')
    assert inputs['initial_actor_checkpoint_sha256'] is None and not inputs['resize_ground_requested']
    assert inputs['contract'] == core.read(probe.TRAIN / 'INPUTS.json')['contract']
    assert metrics['completed_updates'] == len(metrics['history']) == UPDATES
    assert [r['update'] for r in metrics['history']] == list(range(9, 41))
    assert metrics['history'][-1]['actual_control_env_steps'] == STEPS
    assert core.finite(metrics['history']) and metrics['counts']['numerical_failures'] == 0
    native = torch.load(TRAIN / 'checkpoint.pt', map_location='cpu', weights_only=False)
    policy = torch.load(TRAIN / 'policy.pt', map_location='cpu', weights_only=False)
    assert native['iter'] == 40 and native['native']['canonical_restart']
    assert native['native']['total_timesteps'] == CUMULATIVE_STEPS
    assert native['native']['environment']['num_envs'] == NUM_ENVS
    assert not native['native']['environment']['tensors']['episode_length_buf'].any()
    assert all(torch.isfinite(v).all() for v in native['model_state_dict'].values() if torch.is_tensor(v))
    assert all(torch.isfinite(v).all() for state in native['optimizer_state_dict']['state'].values()
               for v in state.values() if torch.is_tensor(v))
    initialization = inputs['initialization']
    assert initialization == metrics['initialization'] == native['infos']['learner_initialization']
    assert initialization['mode'] == 'strict_canonical_ground_resume'
    assert initialization['initial_iteration'] == 8 and initialization['num_envs'] == NUM_ENVS
    assert native['infos']['residual_contract'] == inputs['contract']
    assert policy['schema'] == 'full-racket-residual-ppo-v4' and policy['native_execution']
    assert policy['asset_sha256'] == 'c2264fcece90b042c7c6a1cac7d516af83972729d0402947e3a519041e184cf2'
    assert core.sha(TRAIN / 'policy.pt') == metrics['policy_sha256']
    meta = policy['residual_training_metadata']
    assert meta['completed_updates'] == 40 and meta['actual_control_env_steps'] == STEPS
    assert meta['num_envs'] == NUM_ENVS and meta['base_policy_sha256'] == core.sha(core.BASE)
    assert meta['initialization'] == initialization
    assert core.finite(meta['export_actor_max_error']) and meta['export_actor_max_error'] <= 2e-5
    assert hashlib.sha256(policy['residual_actor_jit']).hexdigest() == meta['actor_jit_sha256']
    ground = core.read(TRAIN / 'initial_ground_states.json')
    protocol = core.read(ROOT / 'configs/g07_full_racket_eval_v4_r3.json')
    protocol['_root'] = str(ROOT)
    roots = np.asarray(ground['root_state_local'])
    assert ground['all_episode_clocks_zero'] and ground['source_training_indices'] == list(range(128)) * 16
    assert roots.shape == (NUM_ENVS, 13) and np.isfinite(roots).all()
    assert np.max(abs(roots[:, :3] - np.asarray(protocol['initialization']['root_position_m']))) <= 1e-5
    assert np.max(abs(roots[:, 3:7] - np.array([1., 0., 0., 0.]))) <= 1e-5
    assert np.max(abs(roots[:, 7:])) <= 1e-5
    assert np.max(abs(np.asarray(ground['joint_velocity_rad_s']))) <= 1e-5
    rows = core.read(TRAIN / 'episodes.json')
    assert core.sha(TRAIN / 'episodes.json') == core.sha(TRAIN / 'episodes.partial.json')
    # A strict resume preserves actual per-lane episode identifiers. Derive
    # each first expected identifier from the real source canonical checkpoint.
    source = torch.load(probe.TRAIN / 'checkpoint.pt', map_location='cpu', weights_only=False)
    initial_ids = source['native']['environment']['tensors']['episode_index'].cpu().numpy()
    final_ids = native['native']['environment']['tensors']['episode_index'].cpu().numpy()
    assert initial_ids.shape == final_ids.shape == (NUM_ENVS,)
    seen = [[] for _ in range(NUM_ENVS)]
    for row in rows:
        assert type(row['env_id']) is int and 0 <= row['env_id'] < NUM_ENVS
        seen[row['env_id']].append(row['episode_index'])
    assert all(seen), 'Every actual native lane must have closed an episode'
    assert all(ids == list(range(int(initial_ids[lane]), int(final_ids[lane])))
               for lane, ids in enumerate(seen)), 'Missing, duplicated or reordered native episodes'
    assert all(r['ground_init'] and r['source_training_index'] == r['env_id'] % 128 for r in rows)
    outcomes = [independent_episode(row, protocol) for row in rows]
    counts = dict(completed_episodes=len(rows), air=sum(r['airborne_contact'] for r in outcomes),
        conditional_recovery_count=sum(r['conditional_recovery'] for r in outcomes),
        complete=sum(r['complete'] for r in outcomes), numerical_failures=sum(r['numerical_failures'] for r in rows))
    assert counts == metrics['counts']
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), iteration=40,
        source_checkpoint_sha256=inputs['resume_checkpoint_sha256'],
        checkpoint_sha256=core.sha(TRAIN / 'checkpoint.pt'), policy_sha256=core.sha(TRAIN / 'policy.pt'),
        actual_control_env_steps=STEPS, cumulative_new_learner_control_env_steps=CUMULATIVE_STEPS,
        actual_gpu_hours=job['charged_gpu_hours'], num_envs=NUM_ENVS,
        initialization=initialization, independent_changing_policy_counts=counts,
        frozen_policy_performance=False, formal_acceptance=False, goal_complete=False)


def prepare():
    assert not PLAN.exists() and not STATE.exists() and not EVALUATION.exists()
    historical=core.read(old.STATE)
    assert historical['state']=='FAILED_VALIDATION_OR_JOB'
    for pid,needle in [(1638238,'joint_placement_resume32_queue01/run.py'),
        (1642834,TRAIN.name),(1642846,TRAIN.name),
        (1651319,'live_tangent_trial_20261025/queue.py'),
        (1653610,'joint_placement_iter40_video_queue01/run.py'),
        (1661032,'joint_placement_contact_confirmation_queue01/run.py'),
        (1661259,'joint_placement_iter40_targeted_video_queue01/run.py')]:
        assert not core.live(pid,needle), 'Existing source owner still live'
    expected=dict(core.read(old.PLAN)['source_hashes'])
    for path in [OLD,old.PLAN,Path(__file__).resolve(),HERE/'VERIFIER_DIFF.txt']:
        expected[str(path.relative_to(ROOT))]=core.sha(path)
    old.check_sources(expected)
    job=core.verify_job(TRAIN,STEPS)
    assert job['status']=='SUCCEEDED'
    receipt=verify_resumed_training()
    save_once(HERE/'TRAINING_RECEIPT.json',receipt)
    command=old.commands()[1]
    assert command==core.read(old.PLAN)['evaluation_command']
    save_once(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(),source_hashes=expected,
        reason='Historical verifier incorrectly required fresh episode0 after strict canonical resume; actual source identifiers1..3 are preserved.',
        correction='Retain every other original training check; require each lane records every episode ID exactly once in source-inclusive to final-exclusive range.',
        historical_error_file=str(old.HERE/'error.txt'),historical_error_sha256=core.sha(old.HERE/'error.txt'),
        old_source_owners_confirmed_exited=True,training_receipt_sha256=core.sha(HERE/'TRAINING_RECEIPT.json'),
        native_training_repeated=False,evaluation_command=command,
        unchanged_evaluation_seed=724201,unchanged_original_TRAIN_cases=128,
        unchanged_case_preselection=[19,23,107,0,8,58],targeted_cases_selected_from_training=[65,6],
        physical_parameters_rules_cases_thresholds_changed=False,
        previous_goal_turn_classification='progress: authoritative successful guard receipt and the exact postprocessing assertion failure were inspected; video delivery request was completed.',
        formal_cases_read=False,formal_acceptance=False,goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN),verified_sources=len(expected),native_training_repeated=False,training=receipt)),flush=True)


def main():
    assert PLAN.exists() and not STATE.exists() and not EVALUATION.exists()
    plan=core.read(PLAN);old.check_sources(plan['source_hashes'])
    receipt=verify_resumed_training()
    assert receipt['checkpoint_sha256']==core.read(HERE/'TRAINING_RECEIPT.json')['checkpoint_sha256']
    with (HERE/'assessment.guard.log').open('x') as log:
        child=subprocess.Popen(plan['evaluation_command'],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2'),
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write('RUNNING_FROZEN40_ORIGINAL_TRAIN128',guard_pid=child.pid,evaluation_output=str(EVALUATION))
        code=child.wait()
    assert code==0, 'Assessment guard failed; inspect evidence, never automatically restart'
    old.check_sources(plan['source_hashes'])
    summary=core.verify_evaluation(EVALUATION,TRAIN/'policy.pt')
    assert core.read(EVALUATION/'INPUTS.json')['reset_noise_seed']==724201
    save_once(HERE/'FIXED_TRAIN128_RESULT.json',summary)
    exports=[]
    for source,stem in [(TRAIN,'joint_iter40_training_episodes'),(EVALUATION,'joint_iter40_fixed_episodes')]:
        with (HERE/(stem+'.log')).open('x') as log:
            subprocess.run([PYTHON,'scripts/export_full_racket_train_csv_v4.py','--run',str(source),
                '--output-prefix',str(HERE/stem)],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'),
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        result=core.read(HERE/(stem+'.json'))
        assert result['original_runtime_metrics_match'] and not result['formal_acceptance']
        exports.append(result)
    old.check_sources(plan['source_hashes'])
    write('SUCCEEDED',training=receipt,independent_fixed_TRAIN128=summary,
        csv_exports=exports,next_action=old.decision(summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    with (HERE/'observer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepare_only:prepare()
        else:
            try:main()
            except BaseException as error:
                (HERE/'error.txt').write_text(traceback.format_exc());write('FAILED_RECOVERY_OR_ASSESSMENT',error=repr(error));raise
