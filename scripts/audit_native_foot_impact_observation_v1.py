"""Check physics-record alignment and unchanged complete original-student replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_native_recovery_query_bank_v1 import audit as audit_bank


def audit(run, baseline):
    passive=audit_bank(run,baseline)
    if not passive['passed']:
        raise ValueError('Passive observer changed original control or trajectories')
    read=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    meta=read(run/'foot_impact_observation.json')
    if (not meta['complete_window'] or meta['lanes']!=[85,87] or
            meta['data_sha256']!=sha(run/'foot_impact_observation.npz')):
        raise ValueError('Unexpected physical observation coverage')
    with np.load(run/'foot_impact_observation.npz',allow_pickle=False) as z:
        physics={k:z[k].copy() for k in z.files}
    with np.load(run/'recovery_query_bank.npz',allow_pickle=False) as z:
        bank={k:z[k].copy() for k in z.files}
    expected=np.arange(297*160,306*160+1)
    if not np.array_equal(physics['physics_frame'],expected):
        raise ValueError('Physical timestep lost or duplicated')
    if any(not np.isfinite(v).all() for v in physics.values()):
        raise ValueError('Nonfinite observation')
    if np.any(physics['episode_index']):
        raise ValueError('Observation window crossed a first-episode reset')
    comparisons=[]
    for tick in range(297,307):
        pi=int(np.flatnonzero(physics['physics_frame']==tick*160)[0])
        bi=int(np.flatnonzero(bank['control_ticks']==tick)[0])
        for li,lane in enumerate(meta['lanes']):
            fields={k:bool(np.array_equal(v[pi,li],bank[k][bi,lane]))
                for k,v in physics.items() if k!='physics_frame'}
            comparisons.append(dict(tick=tick,case=lane,fields_exact=fields))
    aligned=all(all(r['fields_exact'].values()) for r in comparisons)
    pins=dict(passive['source_sha256'])
    for p in [run/'foot_impact_observation.json',run/'foot_impact_observation.npz',
              Path(__file__).resolve(),ROOT/'scripts/native_foot_impact_observer_v1.py']:
        pins[str(p.resolve())]=sha(p)
    return dict(passed=aligned,passive_original_replay=passive['passed'],
        complete_physics_window=True,control_boundary_alignment=comparisons,
        source_sha256=pins,valid_expert_labels=False,
        native_reset_capable=False,control_performance_not_inferred=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['run','baseline','output']:
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh result required')
    result=audit(args.run.resolve(),args.baseline.resolve())
    tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(args.output)
    print(json.dumps({k:result[k] for k in ['passed','passive_original_replay','complete_physics_window']}))
    if not result['passed']:raise SystemExit(2)
