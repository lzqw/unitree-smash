#!/usr/bin/env python3
"""Keep compact full-horizon search evidence separate from native outcomes."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior evidence')
    d=json.loads(a.audit.read_text())
    for path,digest in d['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    records=[]
    for r in d['records']:
        search=r['reference_shooting'];ledger=search['candidates'];baseline=ledger[0]
        best=min(ledger,key=lambda c:tuple(c['rank']))
        assert baseline['generation']==-1
        assert len(ledger)==search['candidate_count']
        assert (not search['selected_new_plan']) or (r['candidate_horizon_accepted'] and best['accepted'])
        records.append(dict(case=r['case'],tick=r['tick'],
            baseline_accepted=baseline['accepted'],baseline_valid_steps=baseline['valid_steps'],
            baseline_rejection=baseline['rejection'],baseline_terminal=baseline['terminal'],
            selected_new_plan=search['selected_new_plan'],accepted=bool(r['candidate_horizon_accepted']),
            candidate_count=len(ledger),accepted_candidate_count=sum(x['accepted'] for x in ledger),
            selected_knots_world_mps2=search['selected_knots_world_mps2'],
            best_rank=search['best_rank'],best_valid_steps=best['valid_steps'],best_rejection=best['rejection'],
            best_terminal=best['terminal'],wall_seconds=search['wall_seconds']))
    result=dict(scope=__doc__,native_execution=False,contract=d['contract'],samples=len(records),
        baseline_complete_predictions=sum(r['baseline_accepted'] for r in records),
        selected_complete_predictions=sum(r['accepted'] for r in records),
        candidates=sum(r['candidate_count'] for r in records),
        wall_seconds=sum(r['wall_seconds'] for r in records),records=records,
        source_sha256={str(a.audit.resolve()):hashlib.sha256(a.audit.read_bytes()).hexdigest(),
                       str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','source_sha256','contract')}))


if __name__=='__main__':main()
