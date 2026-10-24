#!/usr/bin/env python3
"""Wait for one verified native job, then run reset and real-ball TRAIN smokes."""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PYTHON='/home/autodrive/Isaac/env_isaaclab/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--predecessor',type=Path,required=True);p.add_argument('--guard-pid',type=int,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sources=sorted(ROOT.glob('quadsmash/**/*.py'))+[ROOT/name for name in [
        'scripts/smoke_full_racket_eval_reset_v4.py','scripts/search_full_racket_ball_v4.py','scripts/run_gpu_job.py',
        'configs/g07_full_racket_eval_v4_r3.json','configs/court_net_v4.json','configs/full_racket_v4_r3.json']]
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    state=dict(pid=os.getpid(),predecessor=str(a.predecessor),guard_pid=a.guard_pid,
        state='WAITING_VERIFIED_PREDECESSOR',source_hashes=hashes,steps=[],formal_acceptance=False)
    def save():
        temp=a.output/'state.json.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(a.output/'state.json')
    def check_sources():
        for name,expected in hashes.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('Queued source changed: '+name)
    save();deadline=time.monotonic()+3000
    try:
        while not (a.predecessor/'job.json').is_file():
            if time.monotonic()>=deadline:raise TimeoutError('Predecessor observation bound reached; no automatic restart')
            proc=Path('/proc')/str(a.guard_pid)
            if not proc.exists():raise RuntimeError('Predecessor guard disappeared without terminal job receipt')
            command=(proc/'cmdline').read_bytes().replace(b'\x00',b' ').decode()
            if proc.stat().st_uid!=os.getuid() or 'scripts/run_gpu_job.py' not in command or str(a.predecessor) not in command:
                raise RuntimeError('Predecessor PID no longer matches expected project guard')
            time.sleep(3)
        receipt=json.loads((a.predecessor/'job.json').read_text())
        if receipt['status']!='SUCCEEDED' or not receipt['receipt_present']:
            raise RuntimeError('Predecessor did not complete successfully; inspect it before dependent work')
        jobs=[('reset_eval_native01',240,48,[PYTHON,'scripts/smoke_full_racket_eval_reset_v4.py']),
              ('ball_cem_smoke01',2100,6400,[PYTHON,'scripts/search_full_racket_ball_v4.py',
                '--warm-start',str(a.predecessor/'best_candidate.pt'),'--candidates','4','--generations','1',
                '--sigma-fraction','.06','--training-indices','1','65',
                '--ik-seeds','reports/G07_full_racket_v4/ik_flat_face_fast01/seeds.json'])]
        for name,seconds,cap,command in jobs:
            check_sources();output=ROOT/'reports/G07_full_racket_v4'/name
            if output.exists():raise ValueError('Refuse to overwrite existing native evidence: '+str(output))
            guard=[PYTHON,'scripts/run_gpu_job.py','--goal','G07','--budget-window','g07-full-racket-unlimited-20261024',
                '--max-seconds',str(seconds),'--env-steps-cap',str(cap),'--output',str(output),
                '--require-artifact','completed.json','--',*command,'--output',str(output),'--headless']
            with (a.output/(name+'.log')).open('w') as log:
                child=subprocess.Popen(guard,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                record=dict(name=name,guard_pid=child.pid,output=str(output),command=guard,status='RUNNING')
                state['steps'].append(record);state['state']='RUNNING_'+name;save()
                code=child.wait()
            record['exit_code']=code;record['status']='SUCCEEDED' if code==0 else 'FAILED';save()
            result=json.loads((output/'job.json').read_text())
            if code or result['status']!='SUCCEEDED':raise RuntimeError('Native stage failed: '+name)
        state['state']='COMPLETED_DEVELOPMENT_SMOKES';save()
    except BaseException:
        state['state']='STOPPED_FOR_INSPECTION';state['error']=traceback.format_exc();save();raise


if __name__=='__main__':main()
