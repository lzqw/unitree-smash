"""CPU command-interface check with an actual isolated planning-service call.

Uses recorded prefix inputs and model suffix states. This does not advance
PhysX and must not be reported as a native recovery success.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.native_current_state_recovery_v1 import NativeCurrentStateRecoveryV1
from scripts.check_early_support_dispatch_v1 import state_from_arrays
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.audit_joint_sequence_proposal_v1 import plain
from scripts.audit_native_current_state_dispatch_v1 import verify_feedback_dispatch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh output required')
    manifest=json.loads((a.bundle/'manifest.json').read_text())
    source=Path(manifest['replanned_suffix_definition']['native_prefix_run'])
    with np.load(source/'fixed_policy_traces/case_003.npz',allow_pickle=False) as z:public=unpack_public(z['packed_public'])
    with np.load(source/'sequence_dispatch.npz',allow_pickle=False) as z:
        student=z['student'].copy();old_final=z['final'].copy()
    with np.load(source/'sequence_native_trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    pins={}
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            path=Path(module.__file__).resolve();pins[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    pins[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='current-state-native-interface-') as directory:
        temp=Path(directory);bundle=temp/'bundle';bundle.mkdir();out=temp/'output';out.mkdir()
        shutil.copyfile(a.bundle/manifest['bundle_name'],bundle/manifest['bundle_name'])
        manifest['current_state_planning']=dict(control_ticks=[700],max_seconds_per_call=180.)
        (bundle/'manifest.json').write_text(json.dumps(manifest)+'\n')
        adapter=NativeCurrentStateRecoveryV1(bundle,3,torch.tensor(manifest['configuration']['interface']['limits']),
            manifest['policy_sha256'],out)
        nominal=adapter.commands.copy();prefix_error=0.;first_command_error=None;dispatched=[]
        try:
            for tick in range(800):
                obs=torch.zeros((128,794));obs[3]=torch.as_tensor(public[tick]);state=None
                if tick>=280:
                    i=tick-280
                    if tick<adapter.suffix_start:
                        state=state_from_arrays(adapter.model,trace['q'][i],trace['v'][i],trace['contacts'][i],trace['foot_force_world_n'][i])
                    else:
                        state=state_from_arrays(adapter.model,adapter.expected_q[i],adapter.expected_v[i],adapter.expected_contacts[i],np.zeros((4,3)))
                result=adapter.action(tick,obs,torch.as_tensor(student[tick]),torch.zeros(128,dtype=torch.bool),state)
                dispatched.append(result.numpy().copy())
                if not np.array_equal(result.numpy()[np.arange(128)!=3],student[tick,np.arange(128)!=3]):
                    raise ValueError('Other lane command changed')
                if tick<350:prefix_error=max(prefix_error,float(np.max(abs(result.numpy()-old_final[tick]))))
                if tick==700:
                    first_command_error=float(np.max(abs(adapter.decode(result[3].numpy())-
                        adapter.planning_records[0]['first_command'])))
            if (prefix_error>1e-6 or first_command_error is None or first_command_error>1e-6
                    or adapter.applied!=520 or not np.array_equal(adapter.commands,nominal)
                    or len(adapter.planning_records)!=1 or not adapter.planning_records[0]['accepted']):
                raise ValueError('Current-state planning interface mismatch')
            saved=adapter.save()
            np.savez_compressed(out/'sequence_dispatch.npz',student=student,final=np.stack(dispatched),
                first_episode=np.ones((800,128),dtype=bool))
            (out/'fixed_policy_traces').mkdir()
            shutil.copyfile(source/'fixed_policy_traces/case_003.npz',out/'fixed_policy_traces/case_003.npz')
            dispatch_audit=verify_feedback_dispatch(out,bundle)
            # An altered query must be detected without re-running the solver.
            plan_path=out/'current_state_planning.json';recorded=json.loads(plan_path.read_text())
            recorded['calls'][0]['query']['v'][0]+=.01
            plan_path.write_text(json.dumps(recorded)+'\n');negative=False
            try:verify_feedback_dispatch(out,bundle)
            except ValueError as error:
                if str(error)!='Planning query differs from actual trace v':raise
                negative=True
            if not negative:raise ValueError('Altered planning query was not detected')
            check=dict(scope=__doc__,control_ticks=800,feedback_commands=520,prefix_error=prefix_error,
                planned_first_command_error_rad=first_command_error,nominal_commands_immutable=True,
                other_lane_commands_unchanged=True,planning_calls=adapter.planning_records,
                saved_online_flag=saved['online_mpc'],service_closed=adapter.service is None,
                source_sha256=pins,native_execution=False,valid_expert_labels=False)
            check.update(dispatch_audit=dispatch_audit,altered_query_rejected=True)
        finally:adapter.close()
    for path,h in pins.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h:raise ValueError('Source changed '+path)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(plain(check),indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:check[k] for k in ['control_ticks','feedback_commands','prefix_error',
        'planned_first_command_error_rad','service_closed','native_execution']}),flush=True)


if __name__=='__main__':main()
