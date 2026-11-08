"""Exercise recovery search from a recorded native state with no suffix solution.

The successful offline correction is deliberately not supplied to the planner.
The current state and previous command remain the recorded native query. This
is a model search experiment, not a native reset or a task success result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_native_replanned_suffix_v1 import validate
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.current_state_recovery_service_v1 import CurrentStateRecoveryServiceV1
from scripts.search_preserved_early_support_v1 import initialize
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=600.)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior evidence; use a fresh output')
    source=json.loads(a.source.read_text());tick,_=validate(source)
    _,knots,_,audit,initargs=source_context(source['configuration']['source'])
    initialize(*initargs)
    state=state_from_row(source['query'])
    # No successful correction knots enter the search, including as an elite.
    plan=dict(original_knots_rad=knots,corrections=[])
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    pins=dict(source['source_sha256']);pins[str(a.source.resolve())]=sha(a.source)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            path=Path(module.__file__).resolve();pins[str(path)]=sha(path)
    pins[str(Path(__file__).resolve())]=sha(__file__)
    service=CurrentStateRecoveryServiceV1(initargs,audit,pins,workers=4,max_seconds=a.max_seconds)
    try:result=service.solve(state,tick,plan)
    finally:service.close()
    for path,h in pins.items():
        if sha(path)!=h:raise ValueError('Input changed during search: '+path)
    stages=[r['stage'] for r in result['records']]
    warm=result['records'][0]['row']['result']
    result.update(scope=__doc__,source_sha256=pins,
        successful_suffix_supplied=False,warm_plan_accepted=warm['accepted'],
        search_branch_exercised='coarse_candidates' in stages,
        current_state_source=str(a.source),native_execution=False,valid_expert_labels=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')
    tmp.replace(a.output)
    print(json.dumps({k:result[k] for k in ['accepted','warm_plan_accepted','search_branch_exercised',
        'wall_seconds','successful_suffix_supplied']}),flush=True)


if __name__=='__main__':main()
