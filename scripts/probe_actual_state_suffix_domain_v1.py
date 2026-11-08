"""Finite model-state probes around the recorded-state recovery suffix.

The fixed plan and actual command history stay unchanged. These probes do not
widen native query admission or certify a recoverable region.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_native_replanned_suffix_v1 import validate
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.current_state_recovery_planner_v1 import predict_plan
from scripts.perturb_early_support_domain_v1 import perturb_state
from scripts.audit_joint_sequence_proposal_v1 import plain


def probe(spec,query,plan,tick,audit,deadline):
    state=perturb_state(state_from_row(query),spec)
    changed=dict(q=state.q,v=state.v,contacts=state.contacts,target=state.previous_command_target)
    row=predict_plan(plan,changed,tick,0,audit,160,deadline)
    return dict(probe=spec,query=changed,result=row['result'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--max-seconds',type=float,default=1200.)
    a=p.parse_args()
    if a.output.exists() or not 1<=a.workers<=4 or not 300<=a.max_seconds<=1800:raise ValueError('Fresh bounded probe required')
    s=json.loads(a.source.read_text());tick,expected=validate(s)
    _,knots,_,audit,context=source_context(s['configuration']['source'])
    plan=dict(original_knots_rad=knots,corrections=[dict(control_tick=tick,
        previous_command=s['query']['target'],knots_rad=s['best']['corrections_rad'])])
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    pins=dict(s['source_sha256']);pins[str(a.source.resolve())]=sha(a.source)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            path=Path(module.__file__).resolve();pins[str(path)]=sha(path)
    pins[str(Path(__file__).resolve())]=sha(__file__)
    for path,h in pins.items():
        if sha(path)!=h:raise ValueError('Changed probe input '+path)
    specs=[dict(name='nominal',kind='nominal')]
    for kind,size,unit in [('orientation',np.deg2rad(.5),'rad'),('linear_velocity',.02,'m/s'),('angular_velocity',.02,'rad/s')]:
        for axis in [0,1]:
            for sign in [-1,1]:specs.append(dict(name=f'{kind}_{axis}_{sign:+d}',kind=kind,world_axis=axis,amount=sign*size,unit=unit))
    began=time.monotonic();deadline=began+a.max_seconds;rows=[];a.output.parent.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=context) as pool:
        futures=[pool.submit(probe,spec,s['query'],plan,tick,audit,deadline) for spec in specs]
        for future in as_completed(futures):
            row=future.result();rows.append(row)
            print(json.dumps(dict(probe=row['probe']['name'],accepted=row['result']['accepted'],
                completed_steps=row['result']['completed_model_steps'],failure=row['result']['failure'])),flush=True)
            tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(plain(rows),allow_nan=False)+'\n')
            tmp.replace(a.output.with_suffix('.partial.json'))
    nominal=next(r['result'] for r in rows if r['probe']['kind']=='nominal')
    exact=nominal['accepted'] and all(np.array_equal(np.asarray([r[k] for r in nominal['trajectory']]),expected[k]) for k in expected)
    if not exact:raise ValueError('Nominal full suffix failed to reproduce')
    for path,h in pins.items():
        if sha(path)!=h:raise ValueError('Input changed during probes '+path)
    result=dict(scope=__doc__,source_sha256=pins,control_tick=tick,rows=rows,probe_count=len(rows),
        accepted_model_probes=sum(r['result']['accepted'] for r in rows),
        compute_expired_probes=sum((r['result']['failure'] or {}).get('reason')=='declared_compute_budget' for r in rows),
        nominal_exact=True,wall_seconds=time.monotonic()-began,substeps=160,
        native_execution=False,query_admission_widened=False,valid_expert_labels=False,
        limitations=['Finite single-axis model probes exclude combined, joint, actuator and sensor uncertainties.',
            'A different initial epoch from previous tick280 probes prevents direct causal attribution.',
            'Fixed-plan results do not measure the full replanning controller.'])
    a.output.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
