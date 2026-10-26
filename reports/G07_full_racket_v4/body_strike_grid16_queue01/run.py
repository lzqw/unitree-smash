"""One declared strike grid, independent replay and cold original TRAIN128 evaluation."""
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
SOURCE = REPORTS/'body_candidate4_cold_queue01/candidate4_measured_public.pt'
COMPARISON = REPORTS/'fixed_actor_body_cem1280_gen2_01/best_candidate.pt'
ORIGINAL = REPORTS/'residual_ppo_centered_placement075_n2048_probe8_01/policy.pt'
SOURCE_EVAL = REPORTS/'fixed_actor_body_gen1_candidate4_fixed_train128_01'
TRAIN = REPORTS/'body_strike_grid16_n2048_01'
FROZEN = REPORTS/'body_strike_grid16_fixed_train128_01'
CHECKS = REPORTS/'fixed_actor_body_candidate_20261026'
VIDEO = ROOT.parent/'unitree_smash_video_framing/reports/G07_full_racket_v4/orthographic_leg06_warmup13_after_native_queue01'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
PLAN, STATE = HERE/'PLAN.json', HERE/'state.json'
CAP = 2048*800
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
    import torch
    from quadsmash.control.full_racket_body_strike_grid_v4 import population_and_labels
    from scripts.initialize_centered_policy_v4 import same_tree
    assert not any(p.exists() for p in [PLAN,STATE,TRAIN,FROZEN])
    assert core.read(REPORTS/'body_candidate4_cold_queue01/state.json')['state']=='COMPLETED_ALTERNATIVE_COLD128'
    assert not core.live(1803326,'body_candidate4_cold_queue01/run.py')
    assert not core.live(1794481,'fixed_actor_body_cem_gen2_queue01/run.py')
    assert not any(x['status']=='RUNNING' for x in core.read(ROOT/'status/GPU_LEDGER.json')['runs'])
    candidates=core.read(REPORTS/'fixed_actor_body_cem_gen2_queue01/PLAN.json')['actual_twelve_source_selection']
    candidates += [dict(candidate_id='body6',policy=str(COMPARISON),evaluation=str(REPORTS/'fixed_actor_body_cem1280_gen2_fixed_train128_01')),
        dict(candidate_id='body4',policy=str(SOURCE),evaluation=str(SOURCE_EVAL))]
    measured=[]
    for ordinal,c in enumerate(candidates):
        policy,evaluation=Path(c['policy']),Path(c['evaluation'])
        summary=core.verify_evaluation(evaluation,policy)
        assert not previous.reference.ready(summary)
        measured.append(dict(candidate_id=c['candidate_id'],ordinal=ordinal,policy=str(policy),evaluation=str(evaluation),
            policy_sha256=core.sha(policy),summary=summary))
    chosen=max(measured,key=lambda c:(c['summary']['complete'],c['summary']['conditional_recovery_count'],
        c['summary']['airborne_contacts'],c['summary']['attack_outgoing'],-c['ordinal']))
    assert Path(chosen['policy'])==SOURCE and chosen['summary']['complete']==1
    a,b=(torch.load(p,map_location='cpu',weights_only=False) for p in [SOURCE,COMPARISON])
    assert same_tree(a['source_public_policy'],b['source_public_policy'])
    population,labels=population_and_labels(a['parameters'],b['parameters'])
    proof=core.read(HERE/'CPU_REFERENCE_AND_GRID_CHECK.json')
    assert proof['passed'] and proof['candidate_count']==16
    for name,digest in proof['source_sha256'].items(): assert core.sha(Path(name))==digest,name
    sources=sorted(ROOT.glob('quadsmash/**/*.py'))+sorted(ROOT.glob('scripts/*.py'))
    sources+=sorted(ROOT.glob('configs/*.json'))+sorted(ROOT.glob('assets/*.usd'))
    sources += [Path(__file__),SOURCE,COMPARISON,ORIGINAL,HERE/'CPU_REFERENCE_AND_GRID_CHECK.json',
        SOURCE_EVAL/'episodes.json',SOURCE_EVAL/'metrics.json',SOURCE_EVAL/'INPUTS.json']
    expected={str(p.relative_to(ROOT)):core.sha(p) for p in sources}
    search=[PYTHON,str(ROOT/'scripts/search_full_racket_body_strike_grid_v4.py'),'--output',str(TRAIN),
        '--source',str(SOURCE),'--comparison',str(COMPARISON),'--generations','1','--seed','750201','--headless']
    evaluate=[PYTHON,str(ROOT/'scripts/evaluate_full_racket_fixed_actor_body_train_v4.py'),'--output',str(FROZEN),
        '--checkpoint',str(TRAIN/'best_candidate.pt'),'--noise','none','--seed','724201','--headless']
    save(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(),source_hashes=expected,selected=chosen,
        actual_fourteen_source_selection=measured,search_command=search,evaluation_command=evaluate,
        hypothesis='Measured body4 cold128 has59 air,42 physical jump successes but30 rule-qualified recoveries;17 air contacts net-blocked. Test interception position, desired normal and swing gain while preserving measured body timing and both learned actors.',
        population=population.tolist(),proposal_labels=labels,changed_coordinates_for_proposals=[7,24,37],
        candidate0_exact_body4=True,candidate1_exact_body6=True,
        candidates=16,generations=1,all128_cases_per_candidate=True,seed=750201,
        search_env_steps_cap=CAP,search_wall_cap_s=10800,frozen_env_steps_cap=102400,frozen_wall_cap_s=2700,
        original_arm_target_rate_limit_rad_s=5.,protocol_and_hardware_unchanged=True,
        automatic_retry=False,automatic_longer_training=False,
        source_custody='Hold bound inputs plus PLAN through actual parent exit, including independent replay and cold128.',
        formal_cases_read=False,formal_acceptance=False,goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN),bound_inputs=len(expected),selected=chosen['candidate_id'])),flush=True)


def initial_check():
    import numpy as np
    path = TRAIN/'initial_ground_states_000.json'
    if not path.exists(): return None
    value = core.read(path); protocol = core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json')
    roots = np.asarray(value['root_state_local']); ids = np.asarray(value['episode_ids'])
    assert roots.shape == (2048,13) and ids.shape == (2048,) and np.isfinite(roots).all()
    assert value['all_clocks_zero'] and (ids >= 0).all()
    assert np.max(abs(roots[:,:3]-protocol['initialization']['root_position_m'])) < 1e-5
    assert np.max(abs(roots[:,3:7]-[1.,0.,0.,0.])) < 1e-5 and np.max(abs(roots[:,7:])) < 1e-5
    assert np.max(abs(np.asarray(value['joint_velocity_rad_s']))) < 1e-5
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_ground2048_checked=True,
        actual_initial_native_episode_ids=sorted(set(ids.tolist())),all_clocks_zero=True,source_sha256=core.sha(path),
        all_original128_cases_for_each_of16_candidates=True,formal_acceptance=False)


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
    from quadsmash.control.full_racket_body_strike_grid_v4 import population_and_labels
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
    assert same_tree(best['source_public_policy'],source['source_public_policy'])
    comparison=torch.load(COMPARISON,map_location='cpu',weights_only=False)
    declared,_=population_and_labels(source['parameters'],comparison['parameters'])
    assert len(metrics['history']) == 1
    actual_steps = 0; best_actual = None; replays = []
    for generation,entry in enumerate(metrics['history']):
        assert entry['generation'] == generation
        rows = core.read(TRAIN/f'episodes_{generation:03d}.json')
        ground = core.read(TRAIN/f'initial_ground_states_{generation:03d}.json')
        pop = torch.load(TRAIN/f'population_{generation:03d}.pt',map_location='cpu',weights_only=False)
        assert len(rows) == 2048 and ground['all_clocks_zero']
        for i,row in enumerate(rows):
            assert row['env_id'] == row['jump']['env_id'] == i and row['candidate_id'] == i//128
            assert row['case_id'] == cases[i%128]['case_id'] and row['source_training_index'] == i%128
            assert row['episode_index'] == ground['episode_ids'][i] and row['ground_init']
            assert row['active_rally_launch_count'] <= 1
        summaries = [summarize_candidate(rows[i*128:(i+1)*128],protocol) for i in range(16)]
        assert same_tree(summaries,pop['summaries']) and same_tree(summaries,entry['summaries'])
        ranked = sorted(range(16),key=lambda i:tuple(summaries[i]['rank']),reverse=True)
        assert ranked == entry['ranked_candidate_indices'] == pop['ranked_candidate_indices']
        selected = ranked[0];rank = tuple(summaries[selected]['rank'])
        if best_actual is None or rank > best_actual[0]: best_actual = (rank,generation,selected,pop['parameters'][selected])
        actual_steps += entry['env_steps'];assert entry['env_steps'] == entry['control_ticks']*2048
        # Replay each candidate's two saved real public streams on CPU.
        with np.load(TRAIN/f'public_and_actions_{generation:03d}.npz') as arrays:
            public,actions,valid = (arrays[key].copy() for key in ['observations','actions','valid_first_episode'])
        assert public.shape[1:] == (32,794) and actions.shape[1:] == (32,18)
        assert torch.equal(pop['parameters'],declared)
        replay = FixedActorBodyControllerV4(source['source_public_policy'],pop['parameters'].repeat_interleave(2,dim=0),32,'cpu')
        provenance = core.read(ORIGINAL.parent/'trajectory_traces/manifest.json')['provenance']
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
        all2048_actual_selected_episode_rows_independently_checked=True,actual_public_replays=replays,
        public_policy_sha256=core.sha(TRAIN/'best_candidate.pt'),formal_cases_read=False,formal_acceptance=False)


def main():
    assert PLAN.is_file() and not STATE.exists()
    plan = core.read(PLAN); expected = dict(plan['source_hashes'],**{str(PLAN.relative_to(ROOT)):core.sha(PLAN)})
    check_hashes(expected)
    run_job(TRAIN,plan['search_command'],10800,CAP,'RUNNING_BODY_STRIKE_GRID16',expected)
    write('VERIFYING_COMPLETED_SEARCH_ON_CPU')
    save(HERE/'SEARCH_INDEPENDENT_CHECK.json',verify_search())
    run_job(FROZEN,plan['evaluation_command'],2700,102400,'RUNNING_COLD_ORIGINAL_TRAIN128',expected)
    result = core.verify_evaluation(FROZEN,TRAIN/'best_candidate.pt')
    save(HERE/'COLD_TRAIN128_RESULT.json',result)
    check_hashes(expected)
    write('COMPLETED_BODY_STRIKE_GRID16_AND_COLD128',summary=result,
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
