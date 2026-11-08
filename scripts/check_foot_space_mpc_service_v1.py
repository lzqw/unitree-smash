"""Exercise the real CPU service at the recorded native replanning boundary.

Replays measured inputs up to tick 400, performs a real solve and verifies the
new command mapping. It does not execute that new command in native physics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.native_foot_space_mpc_v1 import NativeFootSpaceMPCV1
from scripts.check_early_support_dispatch_v1 import state_from_arrays
from scripts.audit_joint_sequence_proposal_v1 import plain
from quadsmash.rl.episode_trace_v4 import unpack_public


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['bundle','run','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh boundary check required')
    manifest=json.loads((a.bundle/'manifest.json').read_text());case=manifest['rows'][0]['case']
    with np.load(a.run/f'fixed_policy_traces/case_{case:03d}.npz',allow_pickle=False) as z:
        public=unpack_public(z['packed_public'])
    with np.load(a.run/'sequence_dispatch.npz',allow_pickle=False) as z:
        student=z['student'].copy();actual=z['final'].copy();valid=z['first_episode'].copy()
    with np.load(a.run/'sequence_native_trace.npz',allow_pickle=False) as z:
        trace={k:z[k].copy() for k in z.files}
    if student.shape!=(800,128,18) or trace['q'].shape!=(520,25):raise ValueError('Full original native record required')
    with tempfile.TemporaryDirectory(prefix='foot-space-mpc-service-') as directory:
        adapter=NativeFootSpaceMPCV1(a.bundle,case,torch.tensor(manifest['configuration']['interface']['limits']),
            manifest['policy_sha256'],directory)
        maximum=0.;pid=None
        try:
            for tick in range(401):
                obs=torch.zeros((128,794));obs[case]=torch.as_tensor(public[tick]);state=None
                if tick>=280:
                    i=tick-280;state=state_from_arrays(adapter.model,trace['q'][i],trace['v'][i],
                        trace['contacts'][i],trace['foot_force_world_n'][i])
                command=adapter.action(tick,obs,torch.as_tensor(student[tick]),torch.as_tensor(~valid[tick]),state)
                if tick<400:maximum=max(maximum,float(np.max(abs(command.numpy()-actual[tick]))))
                if not np.array_equal(command.numpy()[np.arange(128)!=case],student[tick,np.arange(128)!=case]):
                    raise ValueError('Non-selected command changed')
            if maximum!=0. or len(adapter.planning_records)!=1:raise ValueError('Preserved real prefix differs')
            answer=adapter.planning_records[0];pid=adapter.service.process.pid;ready=adapter.service.ready
            error=float(np.max(abs(adapter.decode(command[case].numpy())-answer['first_command'])))
            if error>1e-6 or not answer['accepted'] or not answer['independent_replay_exact']:
                raise ValueError('Service plan and native command mapping differ')
            output=dict(passed=True,case=case,query_tick=400,preserved_native_prefix_steps=400,
                preserved_prefix_action_error=maximum,first_command_mapping_error_rad=error,
                cpu_service_ready=ready,cpu_service_result=answer['cpu_service'],
                planner_seconds=answer['wall_seconds'],independent_replay_exact=True,
                planned_remaining_steps=len(answer['prediction']['trajectory']),
                stages=[r['stage'] for r in answer['records']],
                new_native_execution=False,revised_commands_physically_executed=False,valid_expert_labels=False)
        finally:adapter.close()
        if pid is None or Path('/proc',str(pid)).exists():raise ValueError('CPU service did not terminate cleanly')
        output['service_cleanup_verified']=True
    files=[a.bundle/'manifest.json',a.run/'sequence_native_trace.npz',a.run/'sequence_dispatch.npz',Path(__file__).resolve()]
    files += [ROOT/'scripts'/name for name in ['native_foot_space_mpc_v1.py',
        'current_state_recovery_service_v3.py','current_state_recovery_planner_v3.py']]
    output['source_sha256']={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    a.output.write_text(json.dumps(plain(output),indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='source_sha256'}))


if __name__=='__main__':main()
