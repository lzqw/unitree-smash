"""Evaluate the completed G1 arm, train the matched G0 arm and compare native outcomes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import psutil
import subprocess
import sys

from run_supermimic_research_scale import digest


def read(path):
    return json.loads(path.read_text())


def wait_for_arm(directory, pid):
    exit_file=directory/'exit.json'
    if not exit_file.exists():
        owner=psutil.Process(pid)
        argv=owner.cmdline()
        if str(directory) not in argv or 'scripts/run_supermimic_research_scale.py' not in argv:
            raise ValueError('Supplied PID does not own this experiment')
        owner.wait()
    if read(exit_file)['returncode']!=0:
        raise RuntimeError(f'Training failed; see {directory}/train.log')
    summary=read(directory/'train/training_completed.json')
    if (summary['recovery_accepted'],summary['approach_accepted'])!=(128,128):
        raise RuntimeError('Training endpoint lacks the requested retained update counts')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--main-dir',type=Path,required=True)
    ap.add_argument('--main-pid',type=int,required=True)
    ap.add_argument('--source-registration',type=Path,required=True)
    a=ap.parse_args()
    primary=a.main_dir.resolve()
    root=primary.parent
    reg=read(primary/'registration.json')
    source_registration=a.source_registration.resolve()
    if reg['arm']!='G1' or digest(source_registration)!=reg['previous_registration_sha256']:
        raise ValueError('Expected a registered main arm with the same source')
    def run(script,*args):
        subprocess.run([sys.executable,'scripts/'+script,*map(str,args)],cwd=Path(__file__).resolve().parents[1],
                       env=dict(os.environ,PYTHONPATH='.:scripts',OMP_NUM_THREADS='2'),check=True)
    with (root/'comparison.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        wait_for_arm(primary,a.main_pid)
        run('evaluate_supermimic_research_arm.py','--arm-dir',primary,'--tensorboard',root/'tensorboard/G1_evaluation')
        cmd=reg['command']
        control=root/'control_G0'
        if not control.exists():
            run('run_supermimic_research_scale.py','--arm','G0','--source-registration',source_registration,
                '--composition',cmd[cmd.index('--resume-training')+1],'--contact',cmd[cmd.index('--action-ball-checkpoint')+1],
                '--output',control,'--tensorboard',root/'tensorboard')
        if not (control/'exit.json').is_file():
            raise RuntimeError('Existing control arm lacks a terminal receipt')
        wait_for_arm(control,None)
        control_reg=read(control/'registration.json')
        differences={'--research-contact-gradient','--output','--tensorboard-dir'}
        def shared(arguments):
            ignored={arguments.index(key)+1 for key in differences}
            return [v for i,v in enumerate(arguments) if i not in ignored]
        if control_reg['initial']!=reg['initial'] or shared(control_reg['command'])!=shared(cmd):
            raise ValueError('Control changed a matched component or training setting')
        run('evaluate_supermimic_research_arm.py','--arm-dir',control,'--tensorboard',root/'tensorboard/G0_evaluation')
        for step in (128,256):
            before=control/f'fixed128_{step:04d}'
            after=primary/f'fixed128_{step:04d}'
            output=root/f'G1_vs_G0_{step:04d}.json'
            if not output.is_file():
                run('compare_supermimic_fixed_stages.py','--before',before,'--after',after,'--stage',f'G1_vs_G0_{step}',
                    '--expected-before',read(before/'metrics.json')['actor_state_sha256'],
                    '--expected-after',read(after/'metrics.json')['actor_state_sha256'],'--output',output)


if __name__=='__main__':
    main()
