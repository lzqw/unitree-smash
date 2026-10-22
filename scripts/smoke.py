#!/usr/bin/env python3
"""Public G03 native smoke, independent reload and controlled interruption."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PYTHON='/home/autodrive/Isaac/env_isaaclab/bin/python'

def run_job(output,script,cap,seconds,*extra,interrupt_after=None):
    command=[sys.executable,'scripts/run_gpu_job.py','--goal','G04','--max-seconds',str(seconds),
        '--env-steps-cap',str(cap),'--output',str(output),'--require-artifact',
        'interface.json' if 'interface' in script else 'resume.json','--',PYTHON,'-u',script,
        '--output',str(output),*extra,'--headless']
    process=subprocess.Popen(command,cwd=ROOT)
    requested=False
    if interrupt_after is not None:
        while process.poll() is None:
            progress=output/'progress.json'
            if progress.exists() and json.loads(progress.read_text())['completed_updates']>=interrupt_after:
                process.send_signal(signal.SIGTERM);requested=True;break
            time.sleep(.1)
    code=process.wait()
    if code:raise subprocess.CalledProcessError(code,command)
    if interrupt_after is not None:
        result=json.loads((output/'resume.json').read_text())
        if not requested or not result['stop_requested'] or result['updates']<interrupt_after:
            raise RuntimeError('controlled interruption was not observed by the native checkpoint handler')
        (output/'interruption.json').write_text(json.dumps(dict(requested_after_updates=interrupt_after,
            signal='SIGTERM through owned GPU guard',observed_updates=result['updates'],checkpoint_saved=True),indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs',type=int,default=32)
    parser.add_argument('--policy-steps',type=int,default=2000)
    parser.add_argument('--output',type=Path,default=Path('reports/G03'))
    args=parser.parse_args()
    if args.num_envs<32 or args.policy_steps<2000:parser.error('G03 requires at least 32 environments and 2000 policy steps')
    subprocess.run([sys.executable,'tools/verify_goal.py','--root','.','--goal','G02'],cwd=ROOT,check=True)
    root=args.output/('native_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f'))
    root.mkdir(parents=True)
    cpu=root/'cpu_tests.xml'
    subprocess.run([PYTHON,'-m','pytest','tests/test_native_observations.py','tests/test_native_timeout_bootstrap.py',
        'tests/test_process_identity.py',f'--junitxml={cpu}','-q'],cwd=ROOT,check=True)
    for stage,steps in [('jump',args.policy_steps),('virtual',30),('ball',30)]:
        run_job(root/stage,'scripts/isaac_interface_probe.py',(steps+4)*args.num_envs,240,
            '--steps',str(steps),'--num-envs',str(args.num_envs),'--stage',stage)
    run_job(root/'train','scripts/isaac_resume_probe.py',21000,180,'--iterations','10')
    run_job(root/'reload','scripts/isaac_resume_probe.py',4500,120,'--iterations','2',
        '--resume',str(root/'train/checkpoint.pt'),'--reference',str(root/'train/continuation_reference.pt'))
    run_job(root/'interrupt','scripts/isaac_resume_probe.py',21000,180,'--iterations','10',interrupt_after=2)
    run_job(root/'resume_interrupt','scripts/isaac_resume_probe.py',4500,120,'--iterations','2',
        '--resume',str(root/'interrupt/checkpoint.pt'),'--reference',str(root/'interrupt/continuation_reference.pt'))
    subprocess.run([PYTHON,'scripts/finalize_g03_reaudit.py','--run-dir',str(root),'--output',str(args.output)],cwd=ROOT,check=True)
    if args.output.resolve()==(ROOT/'reports/G03').resolve():
        subprocess.run([sys.executable,'tools/verify_goal.py','--root','.','--goal','G03'],cwd=ROOT,check=True)

if __name__=='__main__':main()
