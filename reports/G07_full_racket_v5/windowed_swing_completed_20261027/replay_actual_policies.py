"""Audit all7 actual own-input action streams with the original tolerance."""
import copy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
MAIN=HERE.parents[2]
ROOT=MAIN.parent/'unitree_smash_contact_recovery'
RUN=HERE.parent/'windowed_swing7_n128_01'
sys.path.insert(0,str(ROOT))
from scripts.initialize_centered_policy_v4 import same_tree


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def wait_candidate(candidate_id, started):
    while True:
        history=read(RUN/'metrics.json')['history']
        matches=[e for e in history if e['candidate_id']==candidate_id]
        if matches:
            assert len(matches)==1
            return matches[0]
        receipt=read(RUN/'job.json') if (RUN/'job.json').exists() else None
        live=[]
        for pid,ticks in [(1994961,'71584847'),(1994976,'71584859')]:
            path=Path('/proc',str(pid))
            if path.exists() and (path/'stat').read_text().rsplit(')',1)[1].split()[19]==ticks:
                live.append(pid)
        assert live and receipt is None, 'Missing candidate after original job exit; reconcile raw job without restarting it'
        if time.monotonic()-started>1200:
            raise TimeoutError('Read-only observation expired; the actual native job is not stopped or restarted')
        time.sleep(5)


def main():
    started=time.monotonic()
    factory_path=HERE.parent/'recovery_prior_native_20261027/prior_controller.py'
    spec=importlib.util.spec_from_file_location('completed_actual_window_dispatch',factory_path)
    factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
    inputs=read(RUN/'INPUTS.json')
    defaults,limits=[torch.tensor(inputs[k],dtype=torch.float32) for k in ['default_joint_position_rad','joint_limits_rad']]
    files=[Path(__file__),factory_path,HERE/'ACTUAL_COMPLETED_SEARCH_CHECK.json',RUN/'INPUTS.json']
    results=[]
    for candidate_id in range(7):
        entry=wait_candidate(candidate_id,started)
        i=entry['candidate_id'];policy_path=RUN/f'candidate_{i:03d}.pt';data_path=RUN/f'public_and_actions_{i:03d}.npz'
        source=torch.load(policy_path,map_location='cpu',weights_only=False);before=copy.deepcopy(source)
        controller=factory.make_base(source,6,'cpu')
        with np.load(data_path) as data:
            public,native,valid=[data[k].copy() for k in ['observations','actions','valid_first_episode']]
            assert data['training_indices'].tolist()==[0,29,69,100,112,123]
        assert public.shape==(800,6,794)
        maximum=maximum_rate=0.;previous=previous_target=None;violating_rows=[];digest=hashlib.sha256()
        for frame,(obs,expected,keep) in enumerate(zip(public,native,valid)):
            values=torch.from_numpy(obs);unchanged=values.clone();reset=torch.zeros(6,dtype=torch.bool)
            if previous is not None:
                reset=values[:,775]<previous-1e-5
                ids=torch.where(reset)[0]
                if len(ids):controller.reset(ids)
            actual=controller.action({'policy':values},defaults,limits)
            assert torch.equal(values,unchanged) and torch.isfinite(actual).all() and (abs(actual)<=1).all()
            error=abs(actual.numpy()-expected).max(1)
            if keep.any():maximum=max(maximum,float(error[keep].max()))
            for lane in np.flatnonzero(keep & (error>5e-4)):
                violating_rows.append(dict(frame=frame,training_index=[0,29,69,100,112,123][lane],maximum_error=float(error[lane])))
            target=limits[:,0]+.5*(actual+1)*(limits[:,1]-limits[:,0])
            if previous_target is not None and (~reset).any():
                maximum_rate=max(maximum_rate,float(abs(target[~reset,12:]-previous_target[~reset,12:]).max())/.01)
            previous=values[:,775].clone();previous_target=target.clone();digest.update(actual.numpy().tobytes())
        assert same_tree(source,before) and maximum_rate<=5.0005
        result=dict(candidate_id=i,actual_valid_public_rows=int(valid.sum()),all_public_rows=4800,
            CPU_vs_actual_GPU_max_action_error=maximum,original_tolerance=5e-4,
            within_original_tolerance=maximum<=5e-4,maximum_arm_target_rate_rad_s=maximum_rate,
            violating_valid_rows=violating_rows,CPU_command_stream_sha256=digest.hexdigest(),
            source_actors_normalizers_and_public_inputs_unchanged=True)
        results.append(result);files.extend([policy_path,data_path])
        print(json.dumps({k:v for k,v in result.items() if k not in ['violating_valid_rows','CPU_command_stream_sha256']}),flush=True)
    while not (HERE/'ACTUAL_COMPLETED_SEARCH_CHECK.json').exists():
        if time.monotonic()-started>1200:
            raise TimeoutError('Completed-search raw verification is still pending; do not restart any native job')
        time.sleep(5)
    completed=read(HERE/'ACTUAL_COMPLETED_SEARCH_CHECK.json')
    assert all(sha(Path(p))==h for p,h in completed['source_sha256'].items())
    assert [x['candidate_id'] for x in completed['results']]==list(range(7))
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),results=results,
        total_valid_actual_public_rows=sum(x['actual_valid_public_rows'] for x in results),
        all7_within_original_tolerance=all(x['within_original_tolerance'] for x in results),
        failed_candidate_ids=[x['candidate_id'] for x in results if not x['within_original_tolerance']],
        source_sha256={str(p):sha(p) for p in files},new_native_execution=False,
        audit_completion_not_equated_to_all_tolerance_checks_passing=True,formal_acceptance=False)
    with (HERE/'ACTUAL_POLICY_REPLAY_AUDIT.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
