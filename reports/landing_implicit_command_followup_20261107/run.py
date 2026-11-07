"""Run the matched implicit-command candidate after the unchanged26 trial exits."""
import hashlib,json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
plan=json.loads((HERE/'plan.json').read_text())
def save(name,value):
    p=HERE/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)
def status(state,**kw):
    save('status.json',dict(state=state,pid=os.getpid(),time=datetime.now(timezone.utc).isoformat(),**kw))
try:
    worker=Path(plan['parent_worker']);parent=Path(plan['parent_output']);target=Path(plan['output'])
    deadline=time.monotonic()+5500;status('WAITING_FOR_COHORT26')
    while True:
        previous=json.loads((worker/'status.json').read_text())
        if previous['state']=='FAILED':raise ValueError('Parent trial failed; candidate not launched')
        if previous['state']=='COMPLETED':break
        if time.monotonic()>deadline:raise ValueError('Parent wait deadline; no parallel GPU launch')
        time.sleep(5)
    result=json.loads((worker/'result.json').read_text());audit=json.loads((worker/'native_command_audit.json').read_text())
    if (result['numerical_failures'] or result['complete5']<27 or
            result['stage_transitions']['complete5']['lost'] or not audit['passed']):
        raise ValueError('Parent numerical/mapping or old-success regression requires review')
    while Path('/proc',str(plan['parent_guard_pid'])).exists():
        if time.monotonic()>deadline:raise ValueError('Previous guard still live')
        time.sleep(2)
    for path,digest in plan['required_source_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise ValueError('Changed source: '+path)
    if target.exists():raise ValueError('Preserve previous candidate output')
    # Only now edit the shared entry. The running parent saw its pinned version.
    status('INSTALLING_DECLARED_ENTRY')
    patch='scripts/patches/implicit_command_native_entry_v1.patch'
    subprocess.run(['git','apply','--check',patch],cwd=ROOT,check=True)
    subprocess.run(['git','apply',patch],cwd=ROOT,check=True)
    if hashlib.sha256((ROOT/plan['entry']['path']).read_bytes()).hexdigest()!=plan['entry']['after_sha256']:
        raise ValueError('Patched evaluator does not match reviewed source')
    with (HERE/'entry_check.log').open('x') as log:
        subprocess.run(['bash','scripts/python.sh','scripts/check_implicit_command_native_entry_v1.py'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    status('RUNNING_IMPLICIT_COMMAND26',output=str(target))
    with (HERE/'guard_stdout.log').open('x') as log:
        env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
        subprocess.run(plan['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    status('CHECKING_IMPLICIT_COMMAND26')
    with (HERE/'audit_stdout.log').open('x') as log:
        subprocess.run(['bash','scripts/python.sh','scripts/audit_native_command_history_v1.py',
            '--run',str(target),'--arm-target-rate','10','--output',str(HERE/'native_command_audit.json')],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run(['bash','scripts/python.sh','scripts/compare_frozen_complete5_v2.py',
            '--before',str(parent),'--after',str(target),'--output',str(HERE/'paired_result.json')],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    m=json.loads((target/'metrics.json').read_text());paired=json.loads((HERE/'paired_result.json').read_text())
    save('result.json',dict(complete5=m['complete5'],air=m['airborne_contacts'],recovery=m['conditional_recovery_count'],
        numerical_failures=m['numerical_failures'],stage_transitions=paired['stage_transitions'],scope=plan['scope'],
        landing_mpc=m['landing_mpc']))
    status('COMPLETED',complete5=m['complete5'])
except BaseException as error:
    status('FAILED',error=repr(error));raise
