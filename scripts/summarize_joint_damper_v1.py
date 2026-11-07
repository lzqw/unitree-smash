#!/usr/bin/env python3
"""Paired recorded-state assessment; never promote from CPU feasibility alone."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--later-failures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Keep prior evidence')
    old,new,fail=[json.loads(x.read_text()) for x in (a.baseline,a.candidate,a.later_failures)]
    baseline={(r['case'],r['tick']):r for r in old['records']}
    candidate={(r['case'],r['tick']):r for r in new['records']}
    assert len(baseline)==len(old['records']) and len(candidate)==len(new['records'])
    assert baseline.keys()==candidate.keys()
    # Current candidate and recorded native inputs must be reproducible. The
    # old audit may have used another controller revision; do not rewrite it.
    for path,digest in new['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    def compare(keys,field):
        before={k for k in keys if baseline[k][field]}
        after={k for k in keys if candidate[k][field]}
        return dict(before=len(before),after=len(after),retained_count=len(before&after),
                    lost=sorted(before-after),gained=sorted(after-before))
    rescued={3,8,58,61,67,85,95,115}
    groups=dict(all_recorded=baseline.keys(),
        previously_applied=[k for k,r in baseline.items() if r['native_recorded_first_step_applied']],
        rescued_eight_cases=[k for k in baseline if k[0] in rescued],
        later_joint_limit_failures=[(r['case'],r['tick']) for r in fail['records']
                                   if r['category']=='joint_position_constraint'])
    metrics={name:dict(samples=len(keys),first_step=compare(keys,'candidate_first_step_accepted'),
                      full_horizon=compare(keys,'candidate_horizon_accepted')) for name,keys in groups.items()}
    source=[a.baseline,a.candidate,a.later_failures,Path(__file__)]
    result=dict(scope=__doc__,native_execution=False,source=str(a.candidate),
        candidate_contract=new['contract'],groups=metrics,
        later_failure_changes=[dict(case=k[0],tick=k[1],
            before_stage=baseline[k]['candidate_failed_stage'],after_stage=candidate[k]['candidate_failed_stage'],
            before_rejection=baseline[k]['rejection'],after_rejection=candidate[k]['rejection'])
            for k in groups['later_joint_limit_failures']],
        rejection_counts=dict(Counter(str(r['rejection']) for r in candidate.values() if not r['candidate_horizon_accepted'])),
        wall_seconds=new['summary']['wall_seconds'],
        source_sha256={str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in source})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{m:{j:v[m][j] for j in ('before','after')} for m in ('first_step','full_horizon')}
                      for k,v in metrics.items()}))


if __name__=='__main__':main()
