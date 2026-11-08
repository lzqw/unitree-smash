"""Check full-prefix and measured-state suffix command mapping on CPU.

Native validation later uses the actual recorded trajectory. Neither check
turns CPU command replay into a new physical rollout.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.native_replanned_suffix_v1 import NativeReplannedSuffixV1
from scripts.check_early_support_dispatch_v1 import state_from_arrays
from quadsmash.rl.episode_trace_v4 import unpack_public


def verify_feedback_dispatch(run,bundle):
    run,bundle=Path(run),Path(bundle);m=json.loads((bundle/'manifest.json').read_text())
    with np.load(run/'sequence_dispatch.npz',allow_pickle=False) as z:
        student=z['student'].copy();final=z['final'].copy();valid=z['first_episode'].copy()
    with np.load(run/'sequence_native_trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    with np.load(run/'fixed_policy_traces/case_003.npz',allow_pickle=False) as z:public=unpack_public(z['packed_public'])
    if student.shape!=(800,128,18) or len(public)!=800 or trace['q'].shape!=(520,25):
        raise ValueError('Complete actual first episode required')
    maximum=0.
    with tempfile.TemporaryDirectory(prefix='suffix-dispatch-audit-') as directory:
        adapter=NativeReplannedSuffixV1(bundle,3,torch.tensor(m['configuration']['interface']['limits']),m['policy_sha256'],directory)
        nominal=adapter.commands.copy()
        for tick in range(800):
            obs=torch.zeros((128,794));obs[3]=torch.as_tensor(public[tick]);state=None
            if tick>=280:
                i=tick-280
                state=state_from_arrays(adapter.model,trace['q'][i],trace['v'][i],trace['contacts'][i],trace['foot_force_world_n'][i])
            result=adapter.action(tick,obs,torch.as_tensor(student[tick]),torch.as_tensor(~valid[tick]),state)
            maximum=max(maximum,float(np.max(abs(result.numpy()-final[tick]))))
        requests=np.asarray([r['requested_target_rad'] for r in adapter.records])
        error=float(np.max(abs(requests-trace['requested_target_rad'])))
        if (maximum>1e-6 or error>1e-6 or not np.array_equal(adapter.commands,nominal)
                or not adapter.suffix_query or not adapter.suffix_query['passed']):
            raise ValueError('Actual suffix dispatch or query differs')
    return dict(maximum_normalized_action_error=maximum,maximum_target_error_rad=error,
        model_reference_not_substituted_for_measured_state=True,suffix_query_match=adapter.suffix_query)


def check_model(bundle):
    bundle=Path(bundle);m=json.loads((bundle/'manifest.json').read_text())
    source=Path(m['replanned_suffix_definition']['native_prefix_run'])
    with np.load(source/'fixed_policy_traces/case_003.npz',allow_pickle=False) as z:public=unpack_public(z['packed_public'])
    with np.load(source/'sequence_dispatch.npz',allow_pickle=False) as z:
        student=z['student'].copy();old_final=z['final'].copy()
    with np.load(source/'sequence_native_trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    with tempfile.TemporaryDirectory(prefix='suffix-interface-') as directory:
        adapter=NativeReplannedSuffixV1(bundle,3,torch.tensor(m['configuration']['interface']['limits']),m['policy_sha256'],directory)
        nominal=adapter.commands.copy();maximum=0.;prefix_error=0.
        for tick in range(800):
            obs=torch.zeros((128,794));obs[3]=torch.as_tensor(public[tick]);state=None
            if tick>=280:
                i=tick-280
                if tick<adapter.suffix_start:
                    state=state_from_arrays(adapter.model,trace['q'][i],trace['v'][i],trace['contacts'][i],trace['foot_force_world_n'][i])
                else:
                    state=state_from_arrays(adapter.model,adapter.expected_q[i],adapter.expected_v[i],adapter.expected_contacts[i],np.zeros((4,3)))
            result=adapter.action(tick,obs,torch.as_tensor(student[tick]),torch.zeros(128,dtype=torch.bool),state)
            if not np.array_equal(result.numpy()[np.arange(128)!=3],student[tick,np.arange(128)!=3]):
                raise ValueError('Other lane output changed')
            if tick<adapter.suffix_start:
                prefix_error=max(prefix_error,float(np.max(abs(result.numpy()-old_final[tick]))))
            else:
                target=adapter.decode(result[3].numpy())
                maximum=max(maximum,float(np.max(abs(target-nominal[tick-280]))))
        if (maximum>1e-6 or prefix_error>1e-6 or adapter.applied!=520
                or not np.array_equal(adapter.commands,nominal) or not adapter.suffix_query['passed']):
            raise ValueError('Preserved prefix / candidate suffix interface mismatch')
        negative_dir=Path(directory)/'changed-query';negative_dir.mkdir()
        negative=NativeReplannedSuffixV1(bundle,3,torch.tensor(m['configuration']['interface']['limits']),m['policy_sha256'],negative_dir)
        for tick in range(negative.suffix_start):
            obs=torch.zeros((128,794));obs[3]=torch.as_tensor(public[tick]);state=None
            if tick>=280:
                i=tick-280
                state=state_from_arrays(negative.model,trace['q'][i],trace['v'][i],trace['contacts'][i],trace['foot_force_world_n'][i])
            negative.action(tick,obs,torch.as_tensor(student[tick]),torch.zeros(128,dtype=torch.bool),state)
        tick=negative.suffix_start;i=tick-280
        state=state_from_arrays(negative.model,negative.expected_q[i],negative.expected_v[i],negative.expected_contacts[i],np.zeros((4,3)))
        state.v[0]+=.01
        obs=torch.zeros((128,794));obs[3]=torch.as_tensor(public[tick]);rejected=False
        try:negative.action(tick,obs,torch.as_tensor(student[tick]),torch.zeros(128,dtype=torch.bool),state)
        except ValueError as error:
            if str(error)!='Native suffix query differs; new candidate not dispatched':raise
            rejected=True
        if not rejected or negative.applied!=i or negative.last_tick!=tick-1:
            raise ValueError('Mismatched query reached suffix dispatch')
        return dict(scope='CPU replay: actual prefix inputs and predicted suffix states; not new native physics.',
            control_ticks=800,feedback_commands=520,suffix_commands=800-adapter.suffix_start,
            maximum_prefix_normalized_action_error=prefix_error,maximum_nominal_target_error_rad=maximum,
            original_query_match=adapter.query_check,suffix_query_match=adapter.suffix_query,
            nominal_commands_immutable=True,other_lanes_unchanged=True,mismatched_query_rejected_before_dispatch=True,
            native_execution=False,valid_expert_labels=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh output required')
    result=check_model(a.bundle);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
