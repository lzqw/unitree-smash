"""Recompute recorded MPC dispatch without claiming a new physical execution."""
import copy
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from scripts.native_foot_space_mpc_v1 import NativeFootSpaceMPCV1
from scripts.check_early_support_dispatch_v1 import state_from_arrays
from quadsmash.rl.episode_trace_v4 import unpack_public


class RecordedPlanningResponses:
    def __init__(self,calls):self.calls=copy.deepcopy(calls);self.index=0

    def solve(self,state,tick,plan,*,preceding_stable_ticks=0):
        if self.index>=len(self.calls):raise ValueError('Missing recorded planning call')
        result=self.calls[self.index];self.index+=1
        if (result['control_tick']!=tick or result['preceding_stable_ticks']!=preceding_stable_ticks
                or not result['accepted'] or not result['independent_replay_exact']):
            raise ValueError('Recorded planning time, history or acceptance differs')
        for key,value in [('q',state.q),('v',state.v),('contacts',state.contacts),('target',state.previous_command_target)]:
            if not np.array_equal(result['query'][key],value):raise ValueError('Recorded query differs: '+key)
        if len(result['prediction']['trajectory'])!=800-tick:raise ValueError('Incomplete full-tail evidence')
        return copy.deepcopy(result)

    def close(self):pass


def verify_feedback_dispatch(run,bundle):
    run=Path(run);bundle=Path(bundle);manifest=json.loads((bundle/'manifest.json').read_text())
    case=manifest['rows'][0]['case'];planning=json.loads((run/'foot_space_current_state_planning.json').read_text())
    service=RecordedPlanningResponses(planning['calls'])
    with np.load(run/f'fixed_policy_traces/case_{case:03d}.npz',allow_pickle=False) as z:
        public=unpack_public(z['packed_public'])
    with np.load(run/'sequence_dispatch.npz',allow_pickle=False) as z:
        student=z['student'].copy();actual=z['final'].copy();valid=z['first_episode'].copy()
    with np.load(run/'sequence_native_trace.npz',allow_pickle=False) as z:
        trace={k:z[k].copy() for k in z.files}
    if (len(public)!=800 or student.shape!=(800,128,18) or actual.shape!=student.shape
            or valid.shape!=(800,128) or trace['q'].shape!=(520,25)):
        raise ValueError('Complete recorded native execution required')
    maximum=0.
    with tempfile.TemporaryDirectory(prefix='foot-space-mpc-recompute-') as directory:
        adapter=NativeFootSpaceMPCV1(bundle,case,torch.tensor(manifest['configuration']['interface']['limits']),
            manifest['policy_sha256'],directory)
        adapter.service=service;nominal=adapter.commands.copy()
        try:
            for tick in range(800):
                obs=torch.zeros((128,794));obs[case]=torch.as_tensor(public[tick]);state=None
                if tick>=280:
                    i=tick-280
                    state=state_from_arrays(adapter.model,trace['q'][i],trace['v'][i],
                        trace['contacts'][i],trace['foot_force_world_n'][i])
                command=adapter.action(tick,obs,torch.as_tensor(student[tick]),torch.as_tensor(~valid[tick]),state)
                maximum=max(maximum,float(np.max(abs(command.numpy()-actual[tick]))))
            if (maximum>1e-6 or adapter.applied!=520 or service.index!=len(service.calls)
                    or not np.array_equal(nominal,adapter.commands)):
                raise ValueError('Recorded MPC dispatch or nominal evidence differs')
            requested=np.asarray([r['requested_target_rad'] for r in adapter.records])
            error=float(np.max(abs(requested-trace['requested_target_rad'])))
            if error>1e-6:raise ValueError('Actual target request differs from reconstructed feedback')
        finally:adapter.close()
    return dict(maximum_normalized_action_error=maximum,maximum_target_error_rad=error,
        commands=520,query_match=adapter.query_check,planning_calls=service.index,
        planning_resolved_again=False,native_execution=True,native_physics_verified=False,valid_expert_labels=False)
