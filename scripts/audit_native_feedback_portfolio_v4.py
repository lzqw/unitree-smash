"""Portfolio provenance plus unchanged full native dispatch/trajectory audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import audit_native_frictional_recovery_v1 as original


def audit_portfolio(result):
    if result['planner_version']!='verified_feedback_portfolio_v4':
        raise ValueError('Wrong executed planner')
    expected=['original_arm_margin','arm_speed_reserve']
    if result['feedback_candidate_order']!=expected or result['per_query_soft_budget_s']!=60.:
        raise ValueError('Undeclared feedback candidates or budget')
    counts={}
    for row in result['planning_records']:
        decision=row['decision']
        if decision['observed_control_tick']!=row['tick'] or decision['planner_version']!='verified_feedback_portfolio_v4':
            raise ValueError('Planner used wrong native phase')
        data=decision['feedback_portfolio'];attempts=data['attempts'];selected=data['selected']
        if [a['name'] for a in attempts]!=expected[:len(attempts)]:
            raise ValueError('Unexpected candidate order')
        admitted=[a['name'] for a in attempts if a['generation_failure'] is None and
                  a['verification'] is not None and a['verification']['accepted']]
        if selected!=(admitted[0] if admitted else None) or len(admitted)>1:
            raise ValueError('Unverified feedback seed selection')
        if data['original_shifted_seed_used']!=(selected is None):
            raise ValueError('Incorrect original-plan fallback label')
        counts[selected or 'shifted_plan']=counts.get(selected or 'shifted_plan',0)+1
    if counts!=result['feedback_seed_selection_counts']:raise ValueError('Wrong selection totals')
    return dict(verified=True,selection_counts=counts,query_count=len(result['planning_records']),
                phase_and_selection_provenance_checked=True,does_not_replace_native_execution_audit=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['run','bundle','output','sphere-audit']:parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise ValueError('Preserve prior audit')
    result=json.loads((a.run/'sequence_native_result.json').read_text())
    inputs=json.loads((a.run/'INPUTS.json').read_text())
    validation=Path(result['feedback_validation'])
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if (str(validation)!=inputs['feedback_validation'] or sha(validation)!=result['feedback_validation_sha256']
            or inputs['feedback_validation_sha256']!=result['feedback_validation_sha256']
            or inputs['recovery_planner_version']!='verified_feedback_portfolio_v4'):
        raise ValueError('Changed feedback reference input')
    extra=audit_portfolio(result)
    # Original routine still checks all 128 trajectories, current observation
    # and command history, every accepted nonlinear plan, and original limits.
    # Its nonzero exit must remain nonzero even if this provenance check passes.
    try: original.main()
    finally:
        if a.output.exists():
            report=json.loads(a.output.read_text());report['feedback_portfolio_audit']=extra
            report['source_sha256'][str(Path(__file__).resolve())]=sha(__file__)
            report['source_sha256'][str(validation)]=sha(validation)
            temporary=a.output.with_suffix('.tmp')
            temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temporary.replace(a.output)


if __name__=='__main__':main()
