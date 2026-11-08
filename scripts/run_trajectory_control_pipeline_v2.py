"""Freeze completed model references, then run one serial native full128 trial."""
import argparse,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['model-after','primary-search','bank','native-after','parent-plan','fixture-run','output','native-output']:
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--model-pid',type=int,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    if a.output.exists() or a.native_output.exists():raise ValueError('Fresh pipeline and native outputs required')
    process=Path('/proc')/str(a.model_pid)/'cmdline';state=read(a.model_after/'status.json')['state']
    if state!='COMPLETED_REMAINING_CASES' and (not process.exists() or b'complete_geometry_bank_search_v1.py' not in process.read_bytes()):
        raise ValueError('No verified model completion process')
    a.output.mkdir(parents=True)
    def write(state,**kw):
        path=a.output/'pipeline_status.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(state=state,pid=os.getpid(),**kw),indent=2)+'\n');tmp.replace(path)
    def run(command,name,timeout):
        (a.output/(name+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
        with (a.output/(name+'.log')).open('x') as log:
            subprocess.run(command,cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    try:
        write('WAITING_VERIFIED_MODEL_COMPLETION',model_pid=a.model_pid);deadline=time.monotonic()+3900
        while state!='COMPLETED_REMAINING_CASES':
            if state=='FAILED':raise ValueError('Model completion failed; no automatic native launch')
            if not process.exists():raise ValueError('Model completion process disappeared')
            if time.monotonic()>deadline:raise TimeoutError('Wait budget ended; no new job launched')
            time.sleep(5);state=read(a.model_after/'status.json')['state']
        library=a.output/'reference_library.json';prepared=a.output/'native_preparation'
        write('BUILDING_IMMUTABLE_LIBRARY')
        command=['bash','scripts/python.sh','scripts/build_recovery_reference_library_v1.py',
            '--bank',str(a.bank.resolve()),'--search',str(a.primary_search.resolve()),'--output',str(library.resolve())]
        if (a.model_after/'search').exists():command+=['--additional-search',str((a.model_after/'search').resolve())]
        run(command,'library',60)
        run(['bash','scripts/python.sh','scripts/prepare_native_trajectory_control_v8.py',
            '--parent-plan',str(a.parent_plan.resolve()),'--output',str(prepared.resolve()),
            '--native-output',str(a.native_output.resolve()),'--fixture-run',str(a.fixture_run.resolve()),
            '--library',str(library.resolve())],'prepare',60)
        # The predecessor may itself be serially queued. Verify its queue
        # owner and wait for its real native-runner status before chaining.
        if not (a.native_after/'status.json').exists():
            owner=read(a.native_after/'serial_owner.json');queue_pid=int(owner['pid'])
            queue_cmd=Path('/proc')/str(queue_pid)/'cmdline'
            if not queue_cmd.exists() or b'run_serial_pure_control_v1.py' not in queue_cmd.read_bytes():
                raise ValueError('No verified native predecessor queue')
            write('WAITING_PREDECESSOR_NATIVE_START',queue_pid=queue_pid)
            wait_until=time.monotonic()+4500
            while not (a.native_after/'status.json').exists():
                if read(a.native_after/'serial_status.json')['state']=='FAILED':raise ValueError('Predecessor queue failed')
                if not queue_cmd.exists():raise ValueError('Predecessor queue disappeared')
                if time.monotonic()>wait_until:raise TimeoutError('No native restart after queue wait timeout')
                time.sleep(5)
        write('WAITING_SERIAL_NATIVE_SLOT',references=len(read(library)['references']))
        run(['bash','scripts/python.sh','-u','scripts/run_serial_pure_control_v1.py',
            '--after',str(a.native_after.resolve()),'--plan',str((prepared/'plan.json').resolve())],
            'native_serial',9600)
        write('COMPLETED_NATIVE_TRAJECTORY_TRIAL',native_status=read(prepared/'status.json'))
    except BaseException as e:write('FAILED',error=repr(e));raise


if __name__=='__main__':main()
