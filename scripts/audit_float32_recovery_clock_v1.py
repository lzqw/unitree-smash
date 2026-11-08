"""Paired fine-model runs using the native float32 physical clock.

Only the trajectory indexing implementation differs between variants. The
plant, source state, physical limits, phase rule and full episode are equal.
This does not substitute for independent PhysX evaluation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.audit_recovery_long_reference_fine_v8 as audit
from scripts.pure_control_reference_clock_v12 import PureControlReferenceClockV12
from scripts.audit_joint_sequence_proposal_v1 import plain


def evaluate(configuration,nominal,index,start_tick,variant,deadline):
    original=audit.PureControlLongTrajectoryRecoveryV11
    base=original if variant=='floor' else PureControlReferenceClockV12

    class PhysicalClock(base):
        calls=0
        def command(self,state,phase,time_s):
            physical_time=float(np.float32(start_tick+self.calls)*np.float32(.01))
            self.calls+=1
            return super().command(state,phase,physical_time)

    audit.PureControlLongTrajectoryRecoveryV11=PhysicalClock
    try:result=audit.evaluate(configuration,nominal,index,'predictive_selection',deadline,800-start_tick)
    finally:audit.PureControlLongTrajectoryRecoveryV11=original
    result.update(clock_variant=variant,clock='float32(control_tick) * float32(0.01)',
        initial_control_tick=start_tick)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['configuration','fixture-run','bank','native-run','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cases',nargs='+',type=int,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh paired diagnostic required')
    read=lambda p:json.loads(Path(p).read_text())
    cfg=read(args.configuration)
    nominal=read(args.fixture_run/'native_command_inputs.json')['default_joint_position_rad']
    lib=read(cfg['model_reference_library'])['references']
    bank={r['case']:r for r in read(args.bank)['queries']}
    indices=[i for i,r in enumerate(lib) if r['source_case'] in args.cases]
    if len(indices)!=len(set(args.cases)):raise ValueError('Missing source')
    with np.load(args.native_run/'pure_control_commands.npz') as z:
        reconstructed=z['tick'].astype(np.float32)*np.float32(.01)
        if not np.array_equal(reconstructed,z['time_s']):raise ValueError('Clock model does not match recorded native samples')
        checked=len(reconstructed)
    args.output.mkdir(parents=True)
    def write(name,value):
        p=args.output/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');t.replace(p)
    files=[Path(__file__),args.configuration,args.bank,Path(cfg['model_reference_library']),
        args.fixture_run/'native_command_inputs.json',args.native_run/'pure_control_commands.npz']
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    pins={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write('inputs.json',dict(source_sha256=pins,clock_samples_exactly_reconstructed=checked,
        selected_cases=args.cases,workers=6,max_seconds=1800,plant_substeps=160,
        training=False,native_success=False))
    started=time.monotonic();results=[]
    write('status.json',dict(state='RUNNING_PAIRED_CLOCK_MODEL',pid=os.getpid(),completed=0,total=2*len(indices)))
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn')) as pool:
        jobs=[pool.submit(evaluate,cfg,nominal,i,int(bank[lib[i]['source_case']]['control_tick']),variant,started+1800.)
            for i in indices for variant in ['floor','nearest_tick']]
        for future in as_completed(jobs):
            r=future.result();write('case_%03d_%s.json'%(r['source_case'],r['clock_variant']),r)
            results.append({k:r[k] for k in ['source_case','clock_variant','steps','required_steps','failure','final_model_stable_s']})
            write('status.json',dict(state='RUNNING_PAIRED_CLOCK_MODEL',pid=os.getpid(),completed=len(results),total=len(jobs)))
    for name,expected in pins.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise ValueError('Paired input changed '+name)
    summary=dict(state='COMPLETED_PAIRED_CLOCK_MODEL',wall_seconds=time.monotonic()-started,
        results=results,native_success=False,valid_expert_labels=False)
    write('result.json',summary);write('status.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
