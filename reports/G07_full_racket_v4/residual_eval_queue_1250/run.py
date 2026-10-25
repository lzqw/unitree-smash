"""One guarded fixed-TRAIN evaluation after the live residual PPO exits."""
from datetime import datetime,timezone
import hashlib,json,math,os,signal,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
REPORTS=ROOT/'reports/G07_full_racket_v4'
PRIOR=REPORTS/'residual_ppo_pilot_train128_1230'
OUTPUT=REPORTS/'residual_ppo_fixed_train128_1250'
PYTHON='/home/autodrive/Isaac/env_isaaclab/bin/python'
STATE=HERE/'state.json';stopping=False;guard=None


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(state,**extra):
    data=dict(state=state,queue_pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),
        prior_output=str(PRIOR),next_output=str(OUTPUT),formal_acceptance=False,**extra)
    temp=STATE.with_suffix('.json.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(STATE)
    print(json.dumps({k:v for k,v in data.items() if k not in ['source_hashes','prior_metrics']}),flush=True)


def live(pid,needle):
    proc=Path(f'/proc/{pid}')
    try:
        status=(proc/'stat').read_text().split(') ',1)[1].split()[0]
        command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError:return False
    if status in ('Z','X'):return False
    if needle not in command:raise RuntimeError('PID identity changed; do not reuse it')
    return True


def stop(*_):
    global stopping
    stopping=True
    if guard is not None and guard.poll() is None:guard.terminate()


def finite_numbers(value):
    if isinstance(value,dict):return all(finite_numbers(v) for v in value.values())
    if isinstance(value,list):return all(finite_numbers(v) for v in value)
    return math.isfinite(value) if isinstance(value,(float,int)) else True


def main():
    global guard
    if STATE.exists() or OUTPUT.exists():raise RuntimeError('Existing queue/output; inspect the actual process instead')
    if not live(1559746,PRIOR.name) or not live(1559760,PRIOR.name):
        raise RuntimeError('The authorized predecessor must be verified live before queuing')
    files=sorted(ROOT.glob('quadsmash/**/*.py'))+[ROOT/'scripts/evaluate_full_racket_train_v4.py',
        ROOT/'scripts/run_gpu_job.py',ROOT/'configs/g07_full_racket_eval_v4_r3.json',
        ROOT/'configs/g07_full_racket_residual_ppo_v4.json']
    hashes={str(p):sha(p) for p in files};previous=None
    while not stopping:
        running=[pid for pid in [1559746,1559760] if live(pid,PRIOR.name)]
        if not running:break
        try:update=json.loads((PRIOR/'progress.json').read_text()).get('update')
        except (FileNotFoundError,json.JSONDecodeError):update=None
        if update!=previous or not STATE.exists():
            write('WAITING_FOR_VERIFIED_PRIOR_EXIT',verified_live_pids=running,
                  completed_update_at_check=update,source_hashes=hashes)
            previous=update
        time.sleep(10)
    if stopping:write('CANCELLED_BEFORE_LAUNCH');return
    job=json.loads((PRIOR/'job.json').read_text())
    if job['pid']!=1559746 or job['child_pid']!=1559760 or job['status']!='SUCCEEDED':
        write('SKIPPED_PRIOR_NOT_SUCCEEDED',prior_job=job);return
    receipt=json.loads((PRIOR/'completed.json').read_text())
    metrics=json.loads((PRIOR/'metrics.json').read_text())
    assert receipt['env_steps']==job['charged_env_steps']==131072
    assert metrics['completed_updates']==8 and len(metrics['history'])==8
    assert metrics['counts']['numerical_failures']==0 and finite_numbers(metrics['history'])
    if any(sha(Path(p))!=expected for p,expected in hashes.items()):
        write('SKIPPED_SOURCE_CHANGED');return
    import torch
    policy_path=PRIOR/'policy.pt';policy=torch.load(policy_path,map_location='cpu',weights_only=False)
    native=torch.load(PRIOR/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert native['iter']==8 and native['native']['canonical_restart']
    assert policy['schema']=='full-racket-residual-ppo-v4' and policy['native_execution']
    assert sha(policy_path)==metrics['policy_sha256']
    assert policy['asset_sha256']=='c2264fcece90b042c7c6a1cac7d516af83972729d0402947e3a519041e184cf2'
    meta=policy['residual_training_metadata']
    assert meta['completed_updates']==8 and meta['actual_control_env_steps']==131072
    assert meta['base_policy_sha256']=='51a8b19b17e19e60bd7482c7bcb7ed1d4e8b7f50b9317cac3c49007089e8b62d'
    assert math.isfinite(meta['export_actor_max_error']) and meta['export_actor_max_error']<=2e-5
    assert sha_bytes(policy['residual_actor_jit'])==meta['actor_jit_sha256']
    command=[PYTHON,str(ROOT/'scripts/run_gpu_job.py'),'--goal','G07','--budget-window',
        'g07-full-racket-unlimited-20261024','--max-seconds','2700','--env-steps-cap','102400',
        '--output',str(OUTPUT),'--require-artifact','completed.json','--',PYTHON,
        'scripts/evaluate_full_racket_train_v4.py','--output',str(OUTPUT),
        '--checkpoint',str(policy_path),'--noise','none','--headless']
    write('READY_TO_LAUNCH',policy_sha256=sha(policy_path),command=command,prior_metrics=metrics)
    with (HERE/'guard.stdout.log').open('w') as log:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
        guard=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        write('RUNNING_FIXED_TRAIN_EVALUATION',guard_pid=guard.pid,command=command,
              policy_sha256=sha(policy_path),source_hashes=hashes)
        code=guard.wait()
    result=json.loads((OUTPUT/'job.json').read_text()) if (OUTPUT/'job.json').exists() else None
    write('SUCCEEDED' if code==0 and result and result['status']=='SUCCEEDED' else 'EVALUATION_FAILED',
          exit_code=code,job=result)


def sha_bytes(data):return hashlib.sha256(data).hexdigest()


if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:main()
    except BaseException as error:
        write('FAILED_QUEUE_VALIDATION',error=repr(error));raise
