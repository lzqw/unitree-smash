"""Continue stable-horizon candidates with corrected-clock control to episode end."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_float32_recovery_clock_v1 import evaluate
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['configuration','fixture-run','bank','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();read=lambda p:json.loads(Path(p).read_text())
    if args.output.exists():raise ValueError('Fresh full-model output required')
    cfg=read(args.configuration);library=read(cfg['model_reference_library'])
    nominal=read(args.fixture_run/'native_command_inputs.json')['default_joint_position_rad']
    bank={r['case']:r for r in read(args.bank)['queries']}
    changed=set(library['replaced_source_cases'])
    indices=[i for i,r in enumerate(library['references']) if r['source_case'] in changed]
    if not indices or len(indices)!=len(changed):raise ValueError('Missing changed candidate')
    args.output.mkdir(parents=True)
    def write(name,value):
        p=args.output/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');t.replace(p)
    files=[Path(__file__),args.configuration,args.bank,Path(cfg['model_reference_library']),args.fixture_run/'native_command_inputs.json']
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    pins={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write('inputs.json',dict(source_sha256=pins,selected_cases=sorted(changed),max_seconds=1800,
        workers=min(6,len(indices)),model_substeps=160,clock='float32, nearest control tick',training=False))
    started=time.monotonic();results=[]
    write('status.json',dict(state='RUNNING_FULL_MODEL_RECOVERY',pid=os.getpid(),completed=0,total=len(indices)))
    with ProcessPoolExecutor(max_workers=min(6,len(indices)),mp_context=mp.get_context('spawn')) as pool:
        jobs=[pool.submit(evaluate,cfg,nominal,i,int(bank[library['references'][i]['source_case']]['control_tick']),
            'nearest_tick',started+1800.) for i in indices]
        for future in as_completed(jobs):
            r=future.result();write('case_%03d.json'%r['source_case'],r)
            results.append({k:r[k] for k in ['source_case','steps','required_steps','failure','final_model_stable_s']})
            write('status.json',dict(state='RUNNING_FULL_MODEL_RECOVERY',pid=os.getpid(),completed=len(results),total=len(indices)))
    for name,expected in pins.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise ValueError('Changed input '+name)
    summary=dict(state='COMPLETED_FULL_MODEL_RECOVERY',results=results,wall_seconds=time.monotonic()-started,
        model_recovery_cases=[r['source_case'] for r in results if r['failure'] is None and r['steps']==r['required_steps'] and r['final_model_stable_s']>=.8],
        native_success=False,valid_expert_labels=False)
    write('result.json',summary);write('status.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
