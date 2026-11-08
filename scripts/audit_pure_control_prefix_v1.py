"""Compare actual frozen-prefix traces strictly before each causal takeover."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','baseline','failure-chain','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();cases=json.loads(a.failure_chain.read_text())['cases'];results=[];pins={}
    for case in cases:
        lane=case['case'];file='fixed_policy_traces/case_%03d.npz'%lane
        paths=[a.run/file,a.baseline/file]
        with np.load(paths[0]) as x,np.load(paths[1]) as y:
            n=min(case['takeover_tick'],len(y['packed_public']),len(x['packed_public']))
            if n!=case['takeover_tick']:raise ValueError('Incomplete matching prefix')
            item=dict(case=lane,takeover_tick=n)
            for key in ['packed_public','dispatched_actions']:
                error=np.abs(x[key][:n]-y[key][:n]);different=np.flatnonzero(error.max(1)>1e-6)
                item[key]=dict(first_different_tick=int(different[0]) if len(different) else None,
                    max_abs=float(error.max()) if error.size else 0.)
            results.append(item)
        for path in paths:pins[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    result=dict(comparison_scope='strictly before own takeover; 1e-6 per-field diagnostic tolerance',cases=results,
        observation_diff_cases=[r['case'] for r in results if r['packed_public']['first_different_tick'] is not None],
        action_diff_cases=[r['case'] for r in results if r['dispatched_actions']['first_different_tick'] is not None],
        source_sha256=pins,physical_bitwise_reproducibility_not_assumed=True)
    tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(a.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','source_sha256']}))


if __name__=='__main__':main()
