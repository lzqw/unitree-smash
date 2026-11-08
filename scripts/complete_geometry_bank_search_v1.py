"""Finish only model-search cases not fully processed by a bounded predecessor."""
import argparse,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['after','bank','run','model-inputs','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--predecessor-pid',type=int,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    if a.output.exists():raise ValueError('Fresh continuation required')
    cmd=Path('/proc')/str(a.predecessor_pid)/'cmdline'
    status=read(a.after/'status.json')['state']
    if status.startswith('RUNNING') and (not cmd.exists() or b'search_release_bank_geometry_v3.py' not in cmd.read_bytes()):
        raise ValueError('No verified live predecessor')
    a.output.mkdir(parents=True)
    def write(name,value):
        path=a.output/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
    deadline=time.monotonic()+2000.;write('status.json',dict(state='WAITING_VERIFIED_MODEL_SEARCH',pid=os.getpid()))
    try:
        while status.startswith('RUNNING'):
            if not cmd.exists():raise ValueError('Predecessor vanished without terminal status')
            if time.monotonic()>deadline:raise TimeoutError('No restart after wait timeout')
            time.sleep(5);status=read(a.after/'status.json')['state']
        if status!='COMPLETED_MODEL_SEARCH':raise ValueError('Predecessor did not complete normally')
        bank=read(a.bank);todo=[]
        for query in bank['queries']:
            path=a.after/('case_%03d.json'%query['case'])
            r=read(path) if path.exists() else {};fine=r.get('fine')
            if (len(r.get('coarse_candidates',[]))<19 or fine is None
                or (fine.get('failure') or {}).get('reason')=='compute_budget'):
                todo.append(query)
        bank['queries']=todo;bank['continuation_of']=str(a.bank.resolve())
        write('remaining_bank.json',bank)
        if todo:
            write('status.json',dict(state='RUNNING_REMAINING_CASES',pid=os.getpid(),cases=[r['case'] for r in todo]))
            command=['bash','scripts/python.sh','scripts/search_release_bank_geometry_v4.py',
                '--bank',str((a.output/'remaining_bank.json').resolve()),'--run',str(a.run.resolve()),
                '--model-inputs',str(a.model_inputs.resolve()),'--output',str((a.output/'search').resolve()),'--max-seconds','1800','--workers','8']
            write('command.json',command)
            with (a.output/'search.log').open('x') as log:subprocess.run(command,cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1950)
        write('status.json',dict(state='COMPLETED_REMAINING_CASES',cases=[r['case'] for r in todo],pid=os.getpid()))
    except BaseException as e:write('status.json',dict(state='FAILED',error=repr(e),pid=os.getpid()));raise


if __name__=='__main__':main()
