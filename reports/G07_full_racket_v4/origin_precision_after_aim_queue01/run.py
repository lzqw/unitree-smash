"""After the actual Aim owner exits, measure original and co-located origins."""
import argparse
from datetime import datetime,timezone
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

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
REPORTS=ROOT/'reports/G07_full_racket_v4';ISOLATED=ROOT.parent/'unitree_smash_landing_targets'
PREVIOUS=REPORTS/'contact_aim_after_body48_queue01';PREVIOUS_PID=1827656
PYTHON='/home/autodrive/Isaac/env_isaaclab/bin/python'
BASELINE=REPORTS/'origin16_native_precision_train29_n128_01'
COMPACT=REPORTS/'origin0_native_precision_train29_n128_01'
PROBE='scripts/probe_full_racket_origin_precision_v4.py'
PLAN=HERE/'PLAN.json';STATE=HERE/'state.json'
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('origin_precision_shared_guard',REPORTS/'residual_resume24_queue01/run.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
stopping=False;child=None


def check(expected):
    for name,digest in expected.items():assert core.sha(Path(name))==digest,name


def save(path,value):
    assert not path.exists(),'Preserve prior evidence: '+str(path)
    core.atomic(path,value)


def write(state,**extra):
    core.atomic(STATE,dict(state=state,queue_pid=os.getpid(),observed_at=datetime.now(timezone.utc).isoformat(),
        task_performance_measured=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False,**extra))


def prepare():
    assert not any(path.exists() for path in [PLAN,STATE,BASELINE,COMPACT])
    assert core.live(PREVIOUS_PID,'contact_aim_after_body48_queue01/run.py')
    assert not core.live(1873882,'prepose_branch_after_aim_queue01/run.py')
    assert core.read(REPORTS/'prepose_branch_after_aim_queue01/state.json')['state']=='STOPPED'
    expected=dict(core.read(PREVIOUS/'ADMISSION_DECISION.json')['source_sha256']);check(expected)
    assert not (ROOT/PROBE).exists()
    diagnostic=REPORTS/'aim006_contact_direction_diagnostic_20261027'
    rounding=core.read(diagnostic/'FLOAT32_ORIGIN_COMPARISON.json')
    assert rounding['count']==60
    assert rounding['summary']['ideal_float64_max_error_m']['maximum']>.03
    assert rounding['summary']['float32_world_Euler_max_error_m']['maximum']<1e-5
    for path in [Path(__file__),ISOLATED/PROBE,
                 diagnostic/'PLAN.json',diagnostic/'RESULT.json',diagnostic/'FLOAT32_ORIGIN_COMPARISON.json',
                 diagnostic/'analyze.py',diagnostic/'compare_origin_rounding.py',
                 ISOLATED/'configs/g07_full_racket_eval_v4_r3_origin0.json',
                 ISOLATED/'quadsmash/tasks/stand/full_racket_origin0_v4.py',
                 REPORTS/'residual_resume24_queue01/run.py']:
        expected[str(path)]=core.sha(path)
    save(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(),source_sha256=expected,
        wait_actual_parent_pid=PREVIOUS_PID,required_parent_terminal_state='COMPLETED_CONTACT_AIM_SEARCH_AND_COLD128',
        reason='Measured world-coordinate float32 position accumulation explains up to33.28mm precontact drift.',
        native_probe='Original TRAIN29 launch directly at diagnostic reset,128 replicas,50 control ticks, then stop; no task performance claim.',
        runtime_spacings_m=[16.,0.],max_seconds_each=1200,env_steps_cap_each=6400,seed=750601,
        compact_admission=dict(max_free_flight_position_error_m=.001,max_replica_ball_position_spread_m=.0001,
            max_replica_root_position_spread_m=.001,max_shuttle_contact_force_n=.001,unexpected_done_count=0,numerical_failures=0),
        geometry_hardware_task_cases_and_original_thresholds_retained=True,
        install_only_new_probe_after_actual_owner_exit=PROBE,automatic_retry=False,
        task_performance_measured=False,formal_cases_read=False,formal_acceptance=False))
    print(json.dumps(dict(prepared=str(PLAN),bound_inputs=len(expected),native_job_started=False)))


def run_job(output,spacing,expected):
    global child
    assert not output.exists() and not stopping
    check(expected)
    assert not any(row['status']=='RUNNING' for row in core.read(ROOT/'status/GPU_LEDGER.json')['runs'])
    command=[PYTHON,str(ROOT/'scripts/run_gpu_job.py'),'--goal','G07','--budget-window',
        'g07-full-racket-unlimited-20261024','--max-seconds','1200','--env-steps-cap','6400',
        '--output',str(output),'--require-artifact','completed.json','--',PYTHON,str(ROOT/PROBE),
        '--output',str(output),'--env-spacing',str(spacing),'--seed','750601','--headless']
    with (HERE/(output.name+'.guard.log')).open('x') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2'),
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write('RUNNING_NATIVE_PRECISION_PROBE',guard_pid=child.pid,spacing_m=spacing,output=str(output))
        while child.poll() is None:
            assert not stopping
            time.sleep(5)
    assert child.returncode==0 and not stopping,'Preserve actual failed probe; no auto retry'
    check(expected);core.verify_job(output,6400)


def verify(output,spacing):
    import numpy as np
    from scipy.integrate import solve_ivp
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases
    inputs,result=core.read(output/'INPUTS.json'),core.read(output/'RESULT.json')
    protocol=core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json');cases,indices=training_cases(ROOT,protocol)
    assert indices==list(range(128)) and inputs['case']==cases[29] and inputs['original_TRAIN_case_index']==29
    assert inputs['runtime_origin_spacing_m']==result['runtime_origin_spacing_m']==spacing
    assert inputs['asset_sha256']==protocol['asset']['sha256'] and inputs['protocol_sha256']==core.sha(ROOT/'configs/g07_full_racket_eval_v4_r3.json')
    assert inputs['filter_collisions'] and inputs['replicate_physics'] and result['actual_control_env_steps']==6400
    physics=core.read(ROOT/'configs/shuttle_physics_v2.json');wind=np.asarray(physics['wind_m_s']);drag=physics['quadratic_drag_k_per_m']
    def ode(_,state):
        relative=state[3:]-wind
        return np.r_[state[3:],np.array([0.,0.,-9.81])-drag*np.linalg.norm(relative)*relative]
    state=np.r_[cases[29]['launch']['position_m'],cases[29]['launch']['velocity_mps']]
    ideal=solve_ivp(ode,[0.,.5],state,rtol=1e-11,atol=1e-12,dense_output=True,max_step=.001,method='DOP853')
    with np.load(output/'native_probe.npz') as arrays:
        p,body,forces,clock,origins,done=[arrays[name].copy() for name in
            ['ball_position_m','root_position_m','shuttle_contact_force_n','clock_s','actual_environment_origins_m','done_counts']]
    assert p.shape==(51,128,3) and body.shape==p.shape and forces.shape==(50,128,3)
    assert np.array_equal(clock,np.arange(51)*.01) and np.isfinite(p).all() and np.isfinite(body).all() and np.isfinite(forces).all()
    assert origins.shape==(128,3) and inputs['actual_environment_origins_m']==origins.tolist()
    if spacing==0.:assert not origins.any()
    error=np.linalg.norm(p-ideal.sol(clock)[:3].T[:,None,:],axis=-1)
    calculated=dict(max_free_flight_position_error_m=float(error.max()),
        median_final_free_flight_position_error_m=float(np.median(error[-1])),
        max_replica_ball_position_spread_m=float(np.linalg.norm(p-p[:,:1],axis=-1).max()),
        max_replica_root_position_spread_m=float(np.linalg.norm(body-body[:,:1],axis=-1).max()),
        max_shuttle_contact_force_n=float(np.linalg.norm(forces,axis=-1).max()),unexpected_done_count=int(done.sum()))
    assert all(result[key]==value for key,value in calculated.items())
    assert result['numerical_failures']==0
    isolation=calculated['max_shuttle_contact_force_n']<=.001 and not calculated['unexpected_done_count'] and calculated['max_replica_root_position_spread_m']<=.001
    admitted=bool(isolation and calculated['max_free_flight_position_error_m']<=.001 and calculated['max_replica_ball_position_spread_m']<=.0001)
    assert result['collision_isolation_diagnostic_passed']==isolation and result['compact_precision_gate_passed']==admitted
    return result


def main():
    assert PLAN.exists() and not STATE.exists()
    plan=core.read(PLAN);expected=dict(plan['source_sha256']);expected[str(PLAN)]=core.sha(PLAN);check(expected)
    write('WAITING_FOR_ACTUAL_AIM_OWNER_EXIT',wait_parent_pid=PREVIOUS_PID,new_native_job_started=False)
    while core.live(PREVIOUS_PID,'contact_aim_after_body48_queue01/run.py'):
        assert not stopping
        time.sleep(5)
    assert core.read(PREVIOUS/'state.json')['state']==plan['required_parent_terminal_state'];check(expected)
    assert not (ROOT/PROBE).exists()
    with (ROOT/PROBE).open('xb') as stream:stream.write((ISOLATED/PROBE).read_bytes())
    expected[str(ROOT/PROBE)]=core.sha(ROOT/PROBE)
    save(HERE/'ADMISSION_DECISION.json',dict(checked_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=expected,actual_previous_owner_exited=True,task_performance_measured=False,formal_acceptance=False))
    run_job(BASELINE,16.,expected);baseline=verify(BASELINE,16.)
    save(HERE/'BASELINE_CHECK.json',baseline)
    run_job(COMPACT,0.,expected);compact=verify(COMPACT,0.)
    save(HERE/'COMPACT_CHECK.json',compact)
    evidence=[output/name for output in [BASELINE,COMPACT] for name in
        ['INPUTS.json','RESULT.json','native_probe.npz','job.json','completed.json','source_hashes.json']]
    receipt=dict(checked_at=datetime.now(timezone.utc).isoformat(),baseline_run=str(BASELINE),compact_run=str(COMPACT),
        baseline=baseline,compact=compact,source_sha256={str(path):core.sha(path) for path in evidence},
        admitted=bool(compact['compact_precision_gate_passed']),
        scope='Native precision and overlap/isolation engineering probe only. All original task and formal criteria retained; no policy performance inherited.',
        new_native_execution=True,task_performance_measured=False,formal_cases_read=False,formal_acceptance=False)
    save(HERE/'PRECISION_ADMISSION.json',receipt);check(expected)
    write('COMPLETED_PRECISION_PROBES_ADMITTED' if receipt['admitted'] else 'COMPLETED_PRECISION_PROBES_NOT_ADMITTED',
          admission=str(HERE/'PRECISION_ADMISSION.json'),source_custody_released_on_actual_parent_exit=True)


def stop(*_):
    global stopping
    stopping=True
    if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:
        with (HERE/'observer.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
            try:main()
            except BaseException:
                if child is not None and child.poll() is None:
                    child.send_signal(signal.SIGTERM);child.wait(timeout=60)
                write('STOPPED' if stopping else 'FAILED',error=traceback.format_exc(),automatic_retry=False);raise
