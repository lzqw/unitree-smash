"""Replan a complete remaining recovery from an actual recorded native state.

No simulator state is reset. This is an offline query to the same full-tail
planner intended for subsequent receding-horizon native execution. A returned
model plan is not an expert demonstration or evidence of physical recovery.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.recovery_family_queries_v1 import transfer_reference
from scripts.search_preserved_early_support_v1 import initialize
from scripts.current_state_recovery_planner_v3 import CurrentStateRecoveryPlannerV3
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','search','native-run','bundle','output']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--tick',type=int,default=400)
    p.add_argument('--max-seconds',type=float,default=600.)
    a=p.parse_args()
    if a.output.exists() or not 281<=a.tick<800 or not 120<=a.max_seconds<=1800:
        raise ValueError('Fresh output and bounded actual-time query required')
    read=lambda path:json.loads(Path(path).read_text())
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    setup=read(a.plan);search=read(a.search);manifest=read(a.bundle/'manifest.json')
    native=read(a.native_run/'sequence_native_result.json');job=read(a.native_run/'job.json')
    if (not search['verified'] or not native['native_execution'] or not native['query_match']['passed']
            or native['case']!=search['case'] or job['status']!='SUCCEEDED'
            or native['bundle_sha256']!=manifest['bundle_sha256']
            or sha(a.bundle/manifest['bundle_name'])!=manifest['bundle_sha256']):
        raise ValueError('Matching completed native source and verified model bundle required')
    cfg=manifest['foot_space_definition']
    if (not np.array_equal(cfg['original_knots'],search['original_knots']) or
            not np.array_equal(cfg['transition_knots'],search['verified']['replay']['transition_knots']) or
            cfg['braking']!=search['braking']):
        raise ValueError('Native controller differs from the warm plan')
    ticks=np.asarray([r['control_tick'] for r in native['records']]);matches=np.flatnonzero(ticks==a.tick)
    if len(matches)!=1:raise ValueError('No unique actual first-episode state at query time')
    index=int(matches[0])
    with np.load(a.native_run/'sequence_native_trace.npz',allow_pickle=False) as z:
        if (index==0 or ticks[index-1]!=a.tick-1 or
                not np.array_equal(z['previous_target_rad'][index],z['final_target_rad'][index-1])):
            raise ValueError('Actual command history is discontinuous')
        query=dict(q=z['q'][index].copy(),v=z['v'][index].copy(),contacts=z['contacts'][index].copy(),
                   target=z['previous_target_rad'][index].copy())
    source=read(setup['source']);_,_,_,geometry,context=source_context(source['configuration']['source'])
    initialize(*context)
    # Keep the original tick-280 reference origin, rather than rebuilding it
    # around the later observation and silently changing the warm controller.
    context,_=transfer_reference(context,state_from_row(search['query']))
    initialize(*context);state=state_from_row(query)
    warm=dict(contract='foot_space_current_state_plan_v1',original_knots_rad=search['original_knots'],
        transition_knots_rad=search['verified']['replay']['transition_knots'],braking=search['braking'],corrections=[])
    pins=dict(search['source_sha256'])
    for path in [a.plan,a.search,a.bundle/'manifest.json',a.bundle/manifest['bundle_name'],
                 a.native_run/'sequence_native_result.json',a.native_run/'sequence_native_trace.npz',a.native_run/'job.json']:
        pins[str(path.resolve())]=sha(path)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            path=str(Path(module.__file__).resolve());pins[path]=sha(path)
    pins[str(Path(__file__).resolve())]=sha(__file__)
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Changed input '+path)
    planner=CurrentStateRecoveryPlannerV3(context,geometry,workers=2,max_seconds=a.max_seconds)
    try:result=planner.solve(state,a.tick,warm,preceding_stable_ticks=0)
    finally:planner.close()
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Input changed during solve '+path)
    result.update(source_sha256=pins,case=native['case'],query_is_native=True,
        query_run=str(a.native_run.resolve()),native_execution=False,valid_expert_labels=False,
        note='No native replay of the revised commands; all original isolation/physics gates remain required.')
    temporary=a.output.with_suffix('.tmp');temporary.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')
    temporary.replace(a.output)
    print(json.dumps(dict(accepted=result['accepted'],tick=a.tick,wall_seconds=result['wall_seconds'],
        stages=[row['stage'] for row in result['records']])))


if __name__=='__main__':main()
