#!/usr/bin/env python3
"""Run bounded offline iterative recovery search at recorded risk states."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import zipfile
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.landing_sequence_linearization_v1 import make_nominal_profile
from scripts.landing_sequential_recovery_search_v1 import sequential_recovery_search, restore_feasibility
from scripts.audit_joint_sequence_proposal_v1 import plain


def query(item):
    cfg,source,row,iterations,seconds,starting=item
    if row['nominal_commands'] is None:
        return dict(case=row['case'],skipped='No full initial constrained rollout',accepted=False)
    case,tick=row['case'],row['query_tick']
    with np.load(Path(source)/'fixed_policy_traces'/f'case_{case:03d}.npz',allow_pickle=False) as z:
        obs=unpack_public(z['packed_public'])[tick];actions=z['dispatched_actions'][tick-1:tick+1]
    c=build_controller(cfg);limits=np.asarray(cfg['interface']['limits'])
    state=c.qp.model.state_from_public(obs,np.asarray(cfg['posture_target']))
    state.previous_command_target=limits[:,0]+.5*(actions[0]+1.)*np.diff(limits).ravel()
    reference=limits[:,0]+.5*(actions[1]+1.)*np.diff(limits).ravel()
    seed=c.solve(state,reference,lane=case,control_tick=tick,force=True)
    profile=make_nominal_profile(c,state,seed)
    if profile is None:raise ValueError('Stored nominal cannot be reproduced')
    commands=np.stack([n['command'] for n in profile]);error=float(np.max(abs(commands-row['nominal_commands'])))
    if error>1e-9:raise ValueError('Initial controller commands changed')
    if starting is None:
        result=sequential_recovery_search(c,state,reference,commands,iterations=iterations,time_limit_s=seconds)
    else:
        started=time.perf_counter()
        result=restore_feasibility(c,state,reference,starting['final_search_commands'],iterations=iterations)
        result.update(initial_verification_accepted=bool(seed['accepted']),
            newly_model_feasible=bool(result['accepted'] and not seed['accepted']),
            wall_seconds=time.perf_counter()-started)
    return plain(dict(case=case,query_tick=tick,complete5=row['complete5'],
        ball5_except_recovery=row['ball5_except_recovery'],nominal_command_error_rad=error,**result))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-report',type=Path,required=True)
    p.add_argument('--seed-code-revision',default='912dd52')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--iterations',type=int,default=6)
    p.add_argument('--query-seconds',type=float,default=180.)
    p.add_argument('--cases',type=int,nargs='+')
    p.add_argument('--start-search-result',type=Path)
    p.add_argument('--historical-source-archive',type=Path)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous evidence')
    if not 1<=a.workers<=4:raise ValueError('1..4 CPU workers required')
    seed=json.loads(a.seed_report.read_text());historical=[]
    # These two offline proposal programs may evolve. They did not generate
    # the nominal controller rollout. Validate their OLD source against its
    # explicit Git version; every controller/asset/input must still match.
    allowed={'scripts/landing_joint_sequence_proposal_v1.py','scripts/audit_joint_sequence_proposal_v1.py'}
    for name,digest in seed['source_sha256'].items():
        path=Path(name)
        if hashlib.sha256(path.read_bytes()).hexdigest()==digest:continue
        relative=str(path.relative_to(ROOT))
        if relative not in allowed:raise ValueError('Changed seed dependency '+name)
        old=subprocess.run(['git','show',a.seed_code_revision+':'+relative],cwd=ROOT,
            check=True,capture_output=True).stdout
        if hashlib.sha256(old).hexdigest()!=digest:raise ValueError('Historical source hash mismatch')
        historical.append(dict(path=relative,revision=a.seed_code_revision,sha256=digest))
    starting={}
    if a.start_search_result is not None:
        previous=json.loads(a.start_search_result.read_text())
        if Path(previous['seed_report']).resolve()!=a.seed_report.resolve():
            raise ValueError('Search iterates do not use the same seed report')
        for name,digest in previous['source_sha256'].items():
            path=Path(name)
            if hashlib.sha256(path.read_bytes()).hexdigest()==digest:continue
            relative=str(path.relative_to(ROOT))
            if relative not in allowed|{'scripts/landing_sequential_recovery_search_v1.py','scripts/audit_sequential_recovery_search_v1.py'}:
                raise ValueError('Changed restoration dependency '+name)
            if a.historical_source_archive is None:raise ValueError('Archived source needed')
            with zipfile.ZipFile(a.historical_source_archive) as archive:old=archive.read(relative)
            if hashlib.sha256(old).hexdigest()!=digest:raise ValueError('Archived source mismatch')
        starting={r['case']:r for r in previous['records'] if 'skipped' not in r}
    selected=[r for r in seed['records'] if a.cases is None or r['case'] in a.cases]
    if a.cases is not None and sorted(r['case'] for r in selected)!=sorted(a.cases):
        raise ValueError('Missing requested recorded state')
    if a.start_search_result is not None and any(r['nominal_commands'] is not None and r['case'] not in starting for r in selected):
        raise ValueError('Missing corresponding search iterate')
    args=[(seed['configuration'],seed['baseline'],row,a.iterations,a.query_seconds,starting.get(row['case'])) for row in selected]
    rows=[];a.output.parent.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(query,args):
            rows.append(row)
            print(json.dumps({k:row[k] for k in ['case','accepted','newly_model_feasible','wall_seconds'] if k in row}),flush=True)
            a.output.with_suffix('.partial.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    summary=dict(queries=len(rows),attempted=sum('skipped' not in r for r in rows),
        accepted_model_candidates=sum(r['accepted'] for r in rows),
        newly_model_feasible=sum(r.get('newly_model_feasible',False) for r in rows),
        accepted_search_steps=sum(sum(i['accepted_search_step'] for i in r.get('iterations',[])) for r in rows))
    files=[Path(__file__),a.seed_report,ROOT/'configs/go2_piper_racket_landing_model_v1.json']
    files += [p for p in [a.start_search_result,a.historical_source_archive] if p is not None]
    files += [Path(m.__file__) for n,m in sys.modules.items() if n.startswith('scripts.') and getattr(m,'__file__',None)]
    result=dict(scope=__doc__,native_execution=False,summary=summary,records=rows,
        seed_report=str(a.seed_report),historical_seed_sources=historical,
        stage='feasibility_restoration' if a.start_search_result else 'task_merit_search',
        limitations=['Fixed/causally predicted contact schedule, no free foothold optimization.',
            'No independent native stability or self-collision certificate.',
            'Infeasible intermediate iterates are search-only, never teacher labels.'],
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
