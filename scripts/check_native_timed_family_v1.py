"""Check family dispatch on recorded queries; no simulated outcome is claimed."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.native_timed_recovery_family_v1 import NativeTimedRecoveryFamilyV1,CASES
from scripts.native_recovery_query_bank_v1 import state_from_sample
from scripts.diagnose_frozen_trace_divergence_v1 import load_trace
from scripts.search_landing_support_tail_v3 import _WORKER


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','bank','bank-audit','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or _WORKER:raise ValueError('Fresh independent process/output required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    audit=read(a.bank_audit)
    if not audit['passed']:raise ValueError('Verified passive state bank required')
    for path,digest in audit['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Bank source changed '+path)
    m=read(a.bundle/'manifest.json');meta=read(a.bank/'recovery_query_bank.json')
    with np.load(a.bank/'recovery_query_bank.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    public=[];actions=[];pins=dict(audit['source_sha256'])
    for case in range(128):
        obs,action,files=load_trace(Path(m['baseline']),case);pins.update(files)
        public.append(obs[:286]);actions.append(action[:286])
    public=torch.as_tensor(np.stack(public,axis=1));actions=torch.as_tensor(np.stack(actions,axis=1))
    a.output.mkdir(parents=True);adapter=NativeTimedRecoveryFamilyV1(a.bundle,85,
        torch.tensor(m['configuration']['interface']['limits']),m['policy_sha256'],a.output)
    states={};environment=SimpleNamespace(num_envs=128)
    adapter.set_environment(environment)
    # Dependency injection supplies recorded measurements, not physics transitions.
    adapter.model.state_from_native=lambda env,lane:copy.deepcopy(states[lane])
    finished=torch.zeros(128,dtype=torch.bool);outside=~np.isin(np.arange(128),CASES)
    for tick in range(286):
        if tick>=280:
            index=int(np.flatnonzero(data['control_ticks']==tick)[0])
            for case in CASES:
                sample={k:v[index,case] for k,v in data.items() if k!='control_ticks'}
                states[case]=state_from_sample(adapter.model,sample,meta['origins_world'][case])
        final=adapter.action(tick,public[tick],actions[tick],finished,states.get(85))
        if not torch.equal(final[outside],actions[tick,outside]):raise ValueError('Undeclared lane changed')
        if tick<280 and not torch.equal(final,actions[tick]):raise ValueError('Prefix changed')
    if _WORKER:raise ValueError('Unexpected offline optimizer global dependency')
    if len(adapter.alignments)!=25 or len(adapter.family_records)!=150:raise ValueError('Missing cases')
    first=[r for r in adapter.family_records if r['tick']==280]
    continuity=max(float(np.max(abs(r['requested_target_rad']-r['previous_target_rad']))) for r in first)
    if continuity>1e-12:raise ValueError('Discontinuous first transferred command')
    for path in [Path(__file__),a.bundle/'manifest.json',ROOT/'scripts/native_timed_recovery_family_v1.py',
                 a.bank/'recovery_query_bank.json',a.bank/'recovery_query_bank.npz']:
        pins[str(path.resolve())]=sha(path)
    result=dict(passed=True,scope=__doc__,source_sha256=pins,cases=list(CASES),
        recorded_query_ticks=list(range(280,286)),prefix_unchanged=True,unselected_actions_unchanged=True,
        first_transferred_command_difference_rad=continuity,offline_worker_empty=True,
        native_execution=False,valid_expert_labels=False)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))


if __name__=='__main__':main()
