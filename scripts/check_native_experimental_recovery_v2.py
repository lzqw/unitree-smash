"""Verify the native adapter in a fresh process without offline worker state."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.native_experimental_recovery_v2 import NativeExperimentalRecoveryV2
from scripts.search_landing_support_tail_v3 import _WORKER


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or _WORKER:raise ValueError('Fresh process/output required')
    a.output.mkdir(parents=True);m=json.loads((a.bundle/'manifest.json').read_text());case=m['case']
    with np.load(a.bundle/m['bundle_name'],allow_pickle=False) as z:query=z['query_public'].copy()
    with np.load(Path(m['baseline'])/f'fixed_policy_traces/case_{case:03d}.npz',allow_pickle=False) as z:actions=z['dispatched_actions'].copy()
    limits=torch.tensor(m['configuration']['interface']['limits'])
    outcomes={}
    for name,offset in [('recorded_query',0.),('wrong_query',.001)]:
        out=a.output/name;out.mkdir();adapter=NativeExperimentalRecoveryV2(a.bundle,case,limits,m['policy_sha256'],out)
        public=torch.zeros((128,794));student=torch.zeros((128,18));finished=torch.zeros(128,dtype=torch.bool)
        for tick in range(280):
            student[case]=torch.tensor(actions[tick]);adapter.action(tick,public,student,finished)
        public[case]=torch.tensor(query);student[case]=torch.tensor(actions[280])
        state=copy.deepcopy(adapter.initial);state.q[0]+=offset;state.root_position_world[0]+=offset
        state.measured_force_world=np.zeros((4,3))
        try:
            final=adapter.action(280,public,student,finished,state)
            if offset:raise AssertionError('Wrong native query was accepted')
            others=np.arange(128)!=case
            if not torch.equal(final[others],student[others]) or not adapter.query_check['passed']:
                raise AssertionError('Invalid selected-lane dispatch')
            outcomes[name]=dict(passed=True,query=adapter.query_check,first_target_rad=adapter.previous.tolist())
        except ValueError as error:
            if not offset or 'query differs' not in str(error):raise
            outcomes[name]=dict(rejected=True,error=str(error))
    if _WORKER:raise AssertionError('Native adapter initialized offline global worker')
    report=dict(passed=True,offline_worker_remained_empty=True,native_execution=False,checks=outcomes)
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
