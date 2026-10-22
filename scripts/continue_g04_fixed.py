#!/usr/bin/env python3
"""Continue the same G04 PPO config with guarded independent evaluations.

No reward/config/asset search: each chunk resumes the complete optimizer state.
The ledger includes every failed attempt and reserves final evaluation capacity.
"""
import argparse,atexit,json,os,signal,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quadsmash.process_identity import write_identity
STOP=False
CHILD=None

def request_stop(_signal,_frame):
    global STOP
    STOP=True
    if CHILD is not None and CHILD.poll() is None:
        # Only the process group created by this driver. The GPU guard forwards
        # SIGTERM to its own separate simulator group for checkpoint-and-stop.
        os.killpg(CHILD.pid,signal.SIGTERM)

def run(command):
    global CHILD
    if STOP:return 130
    CHILD=subprocess.Popen(command,cwd=ROOT,start_new_session=True)
    try:code=CHILD.wait()
    finally:CHILD=None
    return 130 if STOP else code

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/jump_ppo_v2c.json'))
    p.add_argument('--num-envs',type=int,default=512)
    p.add_argument('--iterations-per-chunk',type=int,default=200)
    p.add_argument('--chunks',type=int,required=True)
    p.add_argument('--prefix',required=True)
    args=p.parse_args()
    directory=ROOT/f'runs/G04/{args.prefix}_driver'
    directory.mkdir(exist_ok=False)
    (directory/'run.pid').write_text(str(os.getpid())+'\n')
    write_identity(directory,os.getpid())
    atexit.register(lambda:(directory/'run.pid').unlink(missing_ok=True))
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    config=json.loads(args.config.read_text());rollout=config['runner']['num_steps_per_env']
    checkpoint=args.checkpoint.resolve()
    for chunk in range(1,args.chunks+1):
        if STOP:return 130
        ledger=json.loads((ROOT/'status/GPU_LEDGER.json').read_text())
        records=[r for r in ledger['runs'] if r['goal']=='G04']
        if any(r['status']=='RUNNING' for r in records):raise RuntimeError('an active job already exists')
        hours=8.-sum(r['charged_gpu_hours'] for r in records)
        steps=200_000_000-sum(r['charged_env_steps'] for r in records)
        # Reserve 240 s and 160k interactions for this chunk's final evaluation.
        iterations=min(args.iterations_per_chunk,(steps-160_000-args.num_envs*128)//(args.num_envs*rollout))
        max_hours=min(.5,hours-240/3600)
        iterations=min(iterations,max(0,int((max_hours*3600-60)/8)))
        if iterations<1 or max_hours<.03:
            print(json.dumps({'status':'BUDGET_EXHAUSTED','remaining_gpu_hours':hours,'remaining_steps':steps}),flush=True)
            return 2
        train=ROOT/f'runs/G04/{args.prefix}_{chunk:02d}'
        evaluate=ROOT/f'reports/G04/{args.prefix}_{chunk:02d}_eval'
        train_cmd=[sys.executable,'scripts/train.py','--stage','jump_recover','--seed','0',
            '--num-envs',str(args.num_envs),'--max-env-steps',str((iterations*rollout+128)*args.num_envs),
            '--max-gpu-hours',str(max_hours),'--run-dir',str(train),'--config',str(args.config.resolve()),
            '--resume-checkpoint',str(checkpoint)]
        print(json.dumps({'stage':'training','chunk':chunk,'command':train_cmd}),flush=True)
        code=run(train_cmd)
        if code:return code
        checkpoint=train/'native/checkpoint.pt'
        eval_cmd=[sys.executable,'scripts/evaluate.py','--stage','jump_recover','--checkpoint',str(checkpoint),
            '--config',str(args.config.resolve()),'--suite','dev_jump_v2','--episodes','400','--num-envs','200',
            '--reset-mode','ground','--max-seconds','240','--output',str(evaluate)]
        print(json.dumps({'stage':'evaluation','chunk':chunk,'command':eval_cmd}),flush=True)
        code=run(eval_cmd)
        if code:return code
        m=json.loads((evaluate/'metrics.json').read_text())
        passed=m['jump_recovery_success_rate']>=.9 and m['numerical_failures']==0 and m['ground_init_fraction']==1.
        state='PASSED' if passed else 'IN_PROGRESS'
        code=run([sys.executable,'scripts/finalize_g04_native.py','--eval-dir',str(evaluate),
            '--training-dir',str(train),'--state',state])
        if code:return code
        print(json.dumps({'stage':'evaluated','chunk':chunk,'checkpoint':str(checkpoint),'metrics':m}),flush=True)
        if passed:return 0
    print(json.dumps({'status':'CHUNKS_COMPLETED_BELOW_THRESHOLD','checkpoint':str(checkpoint)}),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
