"""One guarded body search, followed by a cold original TRAIN128 evaluation."""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT/'reports/G07_full_racket_v4'
PREVIOUS = REPORTS/'centered_leg06_full_cohort_warmup13_queue01'
SOURCE = REPORTS/'residual_ppo_centered_placement075_n2048_probe8_01/policy.pt'
SOURCE_EVAL = REPORTS/'residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01'
TRAIN = REPORTS/'fixed_actor_body_cem1280_gen2_01'
FROZEN = REPORTS/'fixed_actor_body_cem1280_gen2_fixed_train128_01'
CHECKS = REPORTS/'fixed_actor_body_candidate_20261026'
VIDEO = ROOT.parent/'unitree_smash_video_framing/reports/G07_full_racket_v4/orthographic_leg06_warmup13_after_native_queue01'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
PLAN, STATE = HERE/'PLAN.json', HERE/'state.json'
CAP = 2*1280*800
sys.path.insert(0,str(ROOT))
spec = importlib.util.spec_from_file_location('actual_completed_warmup13_owner',PREVIOUS/'run.py')
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
core = previous.core
stopping, child = False, None


def save(path,value):
    assert not path.exists(),str(path)
    core.atomic(path,value)


def write(state,**extra):
    core.atomic(STATE,dict(state=state,queue_pid=os.getpid(),observed_at=datetime.now(timezone.utc).isoformat(),
        formal_cases_read=False,formal_acceptance=False,goal_complete=False,**extra))


def check_hashes(expected):
    for name,digest in expected.items(): assert core.sha(ROOT/name) == digest,name


def prepare():
    assert not any(p.exists() for p in [PLAN,STATE,TRAIN,FROZEN])
    assert not core.live(1777456,'centered_leg06_full_cohort_warmup13_queue01/run.py')
    assert not core.live(1780693,'orthographic_leg06_warmup13_after_native_queue01/run.py')
    assert core.read(PREVIOUS/'state.json')['state'] == 'COMPLETED_FULL_COHORT_WARMUP13_AND_ORIGINAL_TRAIN128'
    assert core.read(VIDEO/'ACTUAL_NATIVE_INPUT_CHECK.json')['main302_inputs_verified_at_admission']
    assert not any(x['status'] == 'RUNNING' for x in core.read(ROOT/'status/GPU_LEDGER.json')['runs'])
    old = core.read(REPORTS/'centered_leg06_after075_intermediates_pilot8_queue01/SOURCE_SELECTION.json')
    candidates = old['all_candidates'] + [
        dict(candidate_id='leg06_pilot8',policy=str(previous.reference.TRAIN/'policy.pt'),evaluation=str(previous.reference.EVALUATION)),
        dict(candidate_id='leg06_full_cohort13',policy=str(previous.TRAIN/'policy.pt'),evaluation=str(previous.FROZEN))]
    measured = []
    for ordinal,candidate in enumerate(candidates):
        policy,evaluation = Path(candidate['policy']),Path(candidate['evaluation'])
        summary = core.verify_evaluation(evaluation,policy)
        assert not previous.reference.ready(summary), 'Review ready policy for formal validation before more search'
        measured.append(dict(candidate_id=candidate['candidate_id'],ordinal=ordinal,policy=str(policy),
            evaluation=str(evaluation),policy_sha256=core.sha(policy),summary=summary))
    chosen = max(measured,key=lambda c:(c['summary']['complete'],c['summary']['conditional_recovery_count'],
        c['summary']['airborne_contacts'],c['summary']['attack_outgoing'],-c['ordinal']))
    assert Path(chosen['policy']) == SOURCE and chosen['policy_sha256'] == '66c93b9f2bfd1f65812ca230312a45cda1d11f59e4a6f076ebdd98ae3c88a3b5'
    identity = core.read(CHECKS/'ACTUAL_SOURCE_IDENTITY_CHECK.json')
    contract = core.read(CHECKS/'CANDIDATE_CONTRACT_CHECK.json')
    assert identity['actual_source_policy_sha256'] == core.sha(SOURCE)
    assert identity['full800step_source_identity_max_action_error'] == 0.
    assert contract['cold_evaluator_only_change_is_explicit_controller_factory_import']
    for receipt in [identity,contract]:
        for name,digest in receipt['source_sha256'].items(): assert core.sha(Path(name)) == digest,name
    sources = sorted(ROOT.glob('quadsmash/**/*.py'))+sorted(ROOT.glob('scripts/*.py'))
    sources += sorted(ROOT.glob('configs/*.json'))+sorted(ROOT.glob('assets/*.usd'))
    sources += [Path(__file__),SOURCE,SOURCE_EVAL/'episodes.json',SOURCE_EVAL/'metrics.json',
        CHECKS/'ACTUAL_SOURCE_IDENTITY_CHECK.json',CHECKS/'CANDIDATE_CONTRACT_CHECK.json',
        VIDEO/'ACTUAL_NATIVE_INPUT_CHECK.json']
    expected = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):core.sha(p) for p in sources}
    search = [PYTHON,str(ROOT/'scripts/search_full_racket_fixed_actor_body_v4.py'),
        '--output',str(TRAIN),'--source',str(SOURCE),'--generations','2','--seed','750101','--headless']
    evaluate = [PYTHON,str(ROOT/'scripts/evaluate_full_racket_fixed_actor_body_train_v4.py'),
        '--output',str(FROZEN),'--checkpoint',str(TRAIN/'best_candidate.pt'),
        '--noise','none','--seed','724201','--headless']
    save(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(),source_hashes=expected,
        actual_twelve_source_selection=measured,selected=chosen,search_command=search,evaluation_command=evaluate,
        hypothesis='Earlier actual ball positions admit attack/target flight, but recorded fixed-root arm IK often cannot track them. Test forward stance plus earlier interception and coordinated jump timing, retaining both measured actors.',
        population='Two exact source references plus eight predeclared plane1.15/1.25m x waypoint.80/.95m x jump-phase advance0/.06s; retain original recovery-handover time and use existing normal bias.12rad.',
        subsequent_population='Second generation: unchanged source, actual best, exact updated mean and seven bounded Gaussian proposals; eight declared analytic coordinates only.',
        all128_cases_per_candidate=True,candidates=10,generations=2,seed=750101,
        search_env_steps_cap=CAP,search_wall_cap_s=10800,frozen_env_steps_cap=102400,frozen_wall_cap_s=2700,
        protocol_and_hardware_unchanged=True,source_actor_weights_normalizers_and_both_mixers_preserved=True,
        original_arm_target_rate_limit_rad_s=5.,automatic_retry=False,automatic_longer_training=False,
        source_custody='Hold all bound source/config/asset files and this PLAN through search, independent CPU verification and cold fixed128 parent exit.',
        formal_cases_read=False,formal_acceptance=False,goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN),bound_inputs=len(expected),source=chosen['candidate_id'])),flush=True)


def initial_check():
    import numpy as np
    path = TRAIN/'initial_ground_states_000.json'
    if not path.exists(): return None
    value = core.read(path); protocol = core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json')
    roots = np.asarray(value['root_state_local']); ids = np.asarray(value['episode_ids'])
    assert roots.shape == (1280,13) and ids.shape == (1280,) and np.isfinite(roots).all()
    assert value['all_clocks_zero'] and (ids >= 0).all()
    assert np.max(abs(roots[:,:3]-protocol['initialization']['root_position_m'])) < 1e-5
    assert np.max(abs(roots[:,3:7]-[1.,0.,0.,0.])) < 1e-5 and np.max(abs(roots[:,7:])) < 1e-5
    assert np.max(abs(np.asarray(value['joint_velocity_rad_s']))) < 1e-5
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_ground1280_checked=True,
        actual_initial_native_episode_ids=sorted(set(ids.tolist())),all_clocks_zero=True,source_sha256=core.sha(path),
        all_original128_cases_for_each_of10_candidates=True,formal_acceptance=False)


def run_job(output,command,seconds,cap,label,expected):
    global child
    assert not output.exists() and not stopping
    check_hashes(expected)
    assert not any(row['status'] == 'RUNNING' for row in core.read(ROOT/'status/GPU_LEDGER.json')['runs'])
    wrapped = [PYTHON,str(ROOT/'scripts/run_gpu_job.py'),'--goal','G07','--budget-window',
        'g07-full-racket-unlimited-20261024','--max-seconds',str(seconds),'--env-steps-cap',str(cap),
        '--output',str(output),'--require-artifact','completed.json','--',*command]
    checked = False
    with (HERE/(label+'.guard.log')).open('x') as log:
        child = subprocess.Popen(wrapped,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2'),
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(label,guard_pid=child.pid,output=str(output))
        try:
            while child.poll() is None:
                assert not stopping
                if output == TRAIN and not checked:
                    result = initial_check()
                    if result is not None: save(HERE/'ACTUAL_INITIAL_GROUND_CHECK.json',result); checked = True
                time.sleep(5)
        except BaseException:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM); child.wait(timeout=60)
            raise
    assert child.returncode == 0 and not stopping,'Preserve actual failure; no automatic retry'
    check_hashes(expected)
    job = core.read(output/'job.json')
    assert job['status'] == 'SUCCEEDED' and job['charged_env_steps'] == cap
    assert not core.live(job['pid'],output.name) and not core.live(job['child_pid'],output.name)
    assert 0 < core.read(output/'completed.json')['env_steps'] <= cap
    return job


def verify_search():
    import numpy as np
    import torch
    from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
    from quadsmash.control.full_racket_fixed_actor_body_v4 import FixedActorBodyControllerV4, SCHEMA
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases
    from scripts.initialize_centered_policy_v4 import same_tree
    protocol = core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json');protocol['_root'] = str(ROOT)
    cases,indices = training_cases(ROOT,protocol)
    inputs,metrics = core.read(TRAIN/'INPUTS.json'),core.read(TRAIN/'metrics.json')
    assert inputs['source_policy_sha256'] == core.sha(SOURCE) and inputs['training_indices'] == list(range(128))
    source = torch.load(SOURCE,map_location='cpu',weights_only=False)
    best = torch.load(TRAIN/'best_candidate.pt',map_location='cpu',weights_only=False)
    assert best['schema'] == SCHEMA and best['native_execution'] and not best['performance_inherited']
    assert same_tree(best['source_public_policy'],source)
    assert len(metrics['history']) == 2
    actual_steps = 0; best_actual = None; replays = []
    for generation,entry in enumerate(metrics['history']):
        assert entry['generation'] == generation
        rows = core.read(TRAIN/f'episodes_{generation:03d}.json')
        ground = core.read(TRAIN/f'initial_ground_states_{generation:03d}.json')
        pop = torch.load(TRAIN/f'population_{generation:03d}.pt',map_location='cpu',weights_only=False)
        assert len(rows) == 1280 and ground['all_clocks_zero']
        for i,row in enumerate(rows):
            assert row['env_id'] == row['jump']['env_id'] == i and row['candidate_id'] == i//128
            assert row['case_id'] == cases[i%128]['case_id'] and row['source_training_index'] == i%128
            assert row['episode_index'] == ground['episode_ids'][i] and row['ground_init']
            assert row['active_rally_launch_count'] <= 1
        summaries = [summarize_candidate(rows[i*128:(i+1)*128],protocol) for i in range(10)]
        assert same_tree(summaries,pop['summaries']) and same_tree(summaries,entry['summaries'])
        ranked = sorted(range(10),key=lambda i:tuple(summaries[i]['rank']),reverse=True)
        assert ranked == entry['ranked_candidate_indices'] == pop['ranked_candidate_indices']
        selected = ranked[0];rank = tuple(summaries[selected]['rank'])
        if best_actual is None or rank > best_actual[0]: best_actual = (rank,generation,selected,pop['parameters'][selected])
        actual_steps += entry['env_steps'];assert entry['env_steps'] == entry['control_ticks']*1280
        # Replay each candidate's two saved real public streams on CPU.
        with np.load(TRAIN/f'public_and_actions_{generation:03d}.npz') as arrays:
            public,actions,valid = (arrays[key].copy() for key in ['observations','actions','valid_first_episode'])
        assert public.shape[1:] == (20,794) and actions.shape[1:] == (20,18)
        replay = FixedActorBodyControllerV4(source,pop['parameters'].repeat_interleave(2,dim=0),20,'cpu')
        provenance = core.read(SOURCE.parent/'trajectory_traces/manifest.json')['provenance']
        default = torch.tensor(provenance['default_joint_position_rad'],dtype=torch.float32)
        limits = torch.tensor(provenance['joint_limits_rad'],dtype=torch.float32)
        previous_clock = None;maximum = 0.
        for obs,actual,keep in zip(public,actions,valid):
            tensor = torch.from_numpy(obs)
            if previous_clock is not None:
                reset_ids = torch.where(tensor[:,775] < previous_clock-1e-5)[0]
                if len(reset_ids): replay.reset(reset_ids)
            command = replay.action({'policy':tensor},default,limits).numpy()
            if keep.any(): maximum = max(maximum,float(np.max(abs(command[keep]-actual[keep]))))
            previous_clock = tensor[:,775].clone()
        replays.append(dict(generation=generation,actual_valid_public_rows=int(valid.sum()),
            CPU_vs_actual_GPU_max_normalized_action_error=maximum,tolerance=5e-4))
        assert maximum <= 5e-4,'Native action replay mismatch; preserve evidence for review'
    assert actual_steps == metrics['actual_control_env_steps'] == core.read(TRAIN/'completed.json')['env_steps']
    _,generation,selected,parameters = best_actual
    assert torch.equal(best['parameters'],parameters)
    assert best['actual_native_search']['generation'] == generation and best['actual_native_search']['candidate_id'] == selected
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_control_env_steps=actual_steps,
        selected_generation=generation,selected_candidate=selected,selected_rank=list(best_actual[0]),
        both_original_actor_exports_and_all_source_metadata_exactly_preserved=True,
        all2560_actual_selected_episode_rows_independently_checked=True,actual_public_replays=replays,
        public_policy_sha256=core.sha(TRAIN/'best_candidate.pt'),formal_cases_read=False,formal_acceptance=False)


def main():
    assert PLAN.is_file() and not STATE.exists()
    plan = core.read(PLAN); expected = dict(plan['source_hashes'],**{str(PLAN.relative_to(ROOT)):core.sha(PLAN)})
    check_hashes(expected)
    run_job(TRAIN,plan['search_command'],10800,CAP,'RUNNING_FIXED_ACTOR_BODY_SEARCH',expected)
    write('VERIFYING_COMPLETED_SEARCH_ON_CPU')
    save(HERE/'SEARCH_INDEPENDENT_CHECK.json',verify_search())
    run_job(FROZEN,plan['evaluation_command'],2700,102400,'RUNNING_COLD_ORIGINAL_TRAIN128',expected)
    result = core.verify_evaluation(FROZEN,TRAIN/'best_candidate.pt')
    save(HERE/'COLD_TRAIN128_RESULT.json',result)
    check_hashes(expected)
    write('COMPLETED_FIXED_ACTOR_BODY_SEARCH_AND_COLD128',summary=result,
        actual_steps=core.read(TRAIN/'completed.json')['env_steps'],source_custody_released_on_actual_parent_exit=True)


def stop(*_):
    global stopping
    stopping = True
    if child is not None and child.poll() is None: child.send_signal(signal.SIGTERM)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args = parser.parse_args()
    if args.prepare: prepare()
    else:
        with (HERE/'observer.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
            try: main()
            except BaseException:
                write('STOPPED' if stopping else 'FAILED',error=traceback.format_exc(),automatic_retry=False)
                raise
