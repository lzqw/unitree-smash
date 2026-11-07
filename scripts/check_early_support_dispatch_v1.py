"""Check measured-state feedback dispatch without mistaking CPU replay for physics."""
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
from scripts.native_early_support_tail_v1 import NativeEarlySupportTailV1
from scripts.loaded_pinocchio_model_v1 import PublicState
from quadsmash.rl.episode_trace_v4 import unpack_public


def state_from_arrays(model,q,v,contacts,force):
    q=np.asarray(q).copy();v=np.asarray(v).copy()
    state=PublicState(q,v,pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),
        q[model.q_indices].copy(),v[model.v_indices].copy(),np.asarray(contacts,bool).copy(),0.)
    state.measured_force_world=np.asarray(force).copy()
    return state


def verify_feedback_dispatch(run,bundle):
    run,bundle=Path(run),Path(bundle)
    manifest=json.loads((bundle/'manifest.json').read_text())
    with np.load(run/'sequence_dispatch.npz',allow_pickle=False) as z:
        student=z['student'].copy();final=z['final'].copy();valid=z['first_episode'].copy()
    with np.load(run/'sequence_native_trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    with np.load(run/'fixed_policy_traces/case_003.npz',allow_pickle=False) as z:public=unpack_public(z['packed_public'])
    if student.shape!=(800,128,18) or len(public)!=800 or trace['q'].shape!=(520,25):
        raise ValueError('Complete native first episode required')
    maximum=0.
    with tempfile.TemporaryDirectory(prefix='early-feedback-audit-') as directory:
        adapter=NativeEarlySupportTailV1(bundle,3,torch.tensor(manifest['configuration']['interface']['limits']),
                                        manifest['policy_sha256'],directory)
        nominal=adapter.commands.copy()
        for tick in range(800):
            obs=torch.zeros((128,794));obs[3]=torch.as_tensor(public[tick])
            state=None
            if tick>=280:
                i=tick-280
                state=state_from_arrays(adapter.model,trace['q'][i],trace['v'][i],trace['contacts'][i],trace['foot_force_world_n'][i])
            actual=adapter.action(tick,obs,torch.as_tensor(student[tick]),torch.as_tensor(~valid[tick]),state)
            maximum=max(maximum,float(np.max(abs(actual.numpy()-final[tick]))))
        if not np.array_equal(adapter.commands,nominal) or maximum>1e-6:
            raise ValueError('Native feedback or immutable nominal trajectory differs')
        requested=np.asarray([r['requested_target_rad'] for r in adapter.records])
        error=float(np.max(abs(requested-trace['requested_target_rad'])))
        if error>1e-6:raise ValueError('Actual request differs from measured-state feedback')
    return dict(maximum_normalized_action_error=maximum,maximum_target_error_rad=error,
                model_reference_not_substituted_for_measured_state=True)


def check_model(bundle):
    bundle=Path(bundle);m=json.loads((bundle/'manifest.json').read_text())
    with np.load(Path(m['baseline'])/'fixed_policy_traces/case_003.npz',allow_pickle=False) as z:
        public=unpack_public(z['packed_public']);student=z['dispatched_actions'].copy()
    with tempfile.TemporaryDirectory(prefix='early-interface-check-') as directory:
        adapter=NativeEarlySupportTailV1(bundle,3,torch.tensor(m['configuration']['interface']['limits']),m['policy_sha256'],directory)
        original=adapter.commands.copy();maximum=0.;perturbation=None
        for tick in range(800):
            obs=torch.zeros((128,794));obs[3]=torch.as_tensor(public[tick])
            actions=torch.zeros((128,18));actions[3]=torch.as_tensor(student[tick])
            state=None
            if tick>=280:
                offset=tick-280
                state=state_from_arrays(adapter.model,adapter.expected_q[offset],adapter.expected_v[offset],
                                        adapter.expected_contacts[offset],np.zeros((4,3)))
                if offset==400:
                    prior=adapter.previous.copy()
                    a,_=adapter.feedback.action(80+offset,state,prior)
                    altered=copy.deepcopy(state);delta=np.zeros(24);delta[3]=.03
                    altered.q=pin.integrate(adapter.model.model,state.q,delta)
                    altered.root_rotation_world=pin.Quaternion(altered.q[3:7]).matrix()
                    b,_=adapter.feedback.action(80+offset,altered,prior)
                    perturbation=float(np.max(abs(a-b)))
            result=adapter.action(tick,obs,actions,torch.zeros(128,dtype=torch.bool),state)
            if not torch.equal(result[np.arange(128)!=3],actions[np.arange(128)!=3]):raise ValueError('Other lanes changed')
            if tick>=280:
                target=adapter.decode(result[3].numpy())
                maximum=max(maximum,float(np.max(abs(target-original[tick-280]))))
        if (maximum>1e-6 or perturbation is None or perturbation<=1e-6 or
                adapter.applied!=520 or not np.array_equal(adapter.commands,original)):
            raise ValueError('Model-input feedback interface check failed')
        return dict(scope='CPU interface replay on model states; zero force placeholders are not native telemetry.',
            control_ticks=800,feedback_commands=520,maximum_nominal_target_error_rad=maximum,
            perturbed_attitude_changes_target_rad=perturbation,query_match=adapter.query_check,
            nominal_commands_immutable=True,other_lanes_unchanged=True,native_execution=False,valid_expert_labels=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh result required')
    result=check_model(a.bundle);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
