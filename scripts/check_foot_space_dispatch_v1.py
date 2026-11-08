"""Check complete measured-state dispatch and immutable nominal evidence."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import pinocchio as pin
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.native_foot_space_recovery_v1 import NativeFootSpaceRecoveryV1
from scripts.check_early_support_dispatch_v1 import state_from_arrays
from quadsmash.rl.episode_trace_v4 import unpack_public


def check(bundle,run=None):
    bundle=Path(bundle);manifest=json.loads((bundle/'manifest.json').read_text());case=manifest['rows'][0]['case']
    path=(Path(run) if run else Path(manifest['baseline']))/f'fixed_policy_traces/case_{case:03d}.npz'
    with np.load(path,allow_pickle=False) as z:
        public=unpack_public(z['packed_public']);original_student=z['dispatched_actions'].copy()
    if len(public)!=800:raise ValueError('Complete native first episode required')
    if run:
        with np.load(Path(run)/'sequence_dispatch.npz',allow_pickle=False) as z:
            student=z['student'].copy();actual=z['final'].copy();valid=z['first_episode'].copy()
        with np.load(Path(run)/'sequence_native_trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
        if student.shape!=(800,128,18) or trace['q'].shape!=(520,25):raise ValueError('Complete native dispatch required')
    else:
        student=np.zeros((800,128,18),np.float32);student[:,case]=original_student
        valid=np.ones((800,128),bool);actual=None
    maximum=0.;perturbation=None
    with tempfile.TemporaryDirectory(prefix='foot-space-dispatch-') as directory:
        adapter=NativeFootSpaceRecoveryV1(bundle,case,torch.tensor(manifest['configuration']['interface']['limits']),
            manifest['policy_sha256'],directory)
        nominal=adapter.commands.copy()
        for tick in range(800):
            obs=torch.zeros((128,794));obs[case]=torch.as_tensor(public[tick]);state=None
            if tick>=280:
                i=tick-280
                q,v,c,f=(trace['q'][i],trace['v'][i],trace['contacts'][i],trace['foot_force_world_n'][i]) if run else (
                    adapter.expected_q[i],adapter.expected_v[i],adapter.expected_contacts[i],np.zeros((4,3)))
                state=state_from_arrays(adapter.model,q,v,c,f)
                if not run and i==400:
                    a,_=adapter.feedback.action(80+i,state,adapter.previous.copy())
                    changed=copy.deepcopy(state);delta=np.zeros(24);delta[3]=.03
                    changed.q=pin.integrate(adapter.model.model,state.q,delta)
                    changed.root_rotation_world=pin.Quaternion(changed.q[3:7]).matrix()
                    b,_=adapter.feedback.action(80+i,changed,adapter.previous.copy())
                    perturbation=float(np.max(abs(a-b)))
            result=adapter.action(tick,obs,torch.as_tensor(student[tick]),torch.as_tensor(~valid[tick]),state)
            others=np.arange(128)!=case
            if not np.array_equal(result.numpy()[others],student[tick,others]):raise ValueError('Other lanes changed')
            if run:maximum=max(maximum,float(np.max(abs(result.numpy()-actual[tick]))))
            elif tick>=280:maximum=max(maximum,float(np.max(abs(adapter.decode(result[case].numpy())-nominal[tick-280]))))
        if maximum>1e-6 or adapter.applied!=520 or not np.array_equal(adapter.commands,nominal):
            raise ValueError('Feedback mapping or nominal immutability check failed')
        if not run and (perturbation is None or perturbation<=1e-6):raise ValueError('Feedback did not respond to state')
        error=maximum
        if run:
            requested=np.asarray([r['requested_target_rad'] for r in adapter.records])
            error=float(np.max(abs(requested-trace['requested_target_rad'])))
            if error>1e-6:raise ValueError('Native requested target differs')
    return dict(maximum_normalized_action_error=maximum if run else None,maximum_target_error_rad=error,
        perturbation_response_rad=perturbation,commands=520,query_match=adapter.query_check,
        native_execution=bool(run),native_physics_verified=False,valid_expert_labels=False)


def verify_feedback_dispatch(run,bundle):return check(bundle,run)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh output required')
    r=check(a.bundle);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
