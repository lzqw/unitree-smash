"""Bounded read-only follow-up after the guarded contact-detail execution."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['preparation','model-inputs','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text())
    if a.output.exists():raise ValueError('Fresh evidence directory required')
    plan=read(a.preparation/'plan.json');run=Path(plan['output']);a.output.mkdir(parents=True)
    pins={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [ROOT/'scripts/audit_native_landing_friction_v1.py',ROOT/'scripts/audit_pure_control_one_step_prediction_v1.py',
         a.preparation/'plan.json',a.model_inputs]}
    def status(state,**extra):
        path=a.output/'status.json';temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(state=state,time=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**extra),indent=2)+'\n')
        temp.replace(path)
    def check():
        for path,digest in pins.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Changed follow-up input '+path)
    commands=[['bash','scripts/python.sh','scripts/audit_native_landing_friction_v1.py',
        '--run',str(run),'--output',str((a.output/'native_friction.json').resolve())],
        ['bash','scripts/python.sh','scripts/audit_pure_control_one_step_prediction_v1.py',
        '--run',str(run),'--model-inputs',str(a.model_inputs.resolve()),'--sample-event','first_ground',
        '--output',str((a.output/'one_step_ground.json').resolve())]]
    (a.output/'inputs.json').write_text(json.dumps(dict(source_sha256=pins,commands=commands,
        training=False,physics_execution=False,waiting_for=str(a.preparation.resolve())),indent=2)+'\n')
    try:
        status('WAITING_NATIVE_COMPLETION');deadline=time.monotonic()+4500.
        while True:
            check();state=read(a.preparation/'status.json')['state']
            if state=='COMPLETED_DIAGNOSTIC':break
            if state=='FAILED':raise ValueError('Native run failed; no audit success inferred')
            if time.monotonic()>deadline:raise TimeoutError('Native follow-up wait budget')
            time.sleep(5)
        for i,command in enumerate(commands):
            check();status('AUDITING',step=i)
            with (a.output/('audit_%d.log'%i)).open('x') as log:
                subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',
                    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,
                    timeout=900,check=True)
        check();friction=read(a.output/'native_friction.json');prediction=read(a.output/'one_step_ground.json')
        status('COMPLETED',clean_cases=friction['clean_cases'],momentum_residual_ns=friction['momentum_residual_ns'],
            model_responses=prediction['model_responses'],prediction_errors=prediction['errors'],
            landing_success_not_inferred=True)
    except BaseException as e:status('FAILED',error=repr(e));raise


if __name__=='__main__':main()
