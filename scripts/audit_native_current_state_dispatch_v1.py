"""Recompute dispatched controls from recorded actual states and accepted plans.

This never reruns optimization or substitutes predicted states for observations.
It validates action arbitration, not the physical quality of recovery.
"""
import copy
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from scripts.native_replanned_suffix_v1 import NativeReplannedSuffixV1
from scripts.check_early_support_dispatch_v1 import state_from_arrays
from scripts.solve_recorded_recovery_suffix_v1 import SuffixCommandCorrectionV1
from quadsmash.rl.episode_trace_v4 import unpack_public


def verify_feedback_dispatch(run,bundle):
    run,bundle=Path(run),Path(bundle);manifest=json.loads((bundle/'manifest.json').read_text())
    calls=json.loads((run/'current_state_planning.json').read_text())['calls']
    if not calls or any(not c['accepted'] or not c['independent_replay_exact'] for c in calls):
        raise ValueError('Accepted independently replayed planning records required')
    expected_ticks=manifest['current_state_planning']['control_ticks']
    if [c['control_tick'] for c in calls]!=expected_ticks:raise ValueError('Missing or reordered planning calls')
    by_tick={c['control_tick']:c for c in calls}
    with np.load(run/'sequence_dispatch.npz',allow_pickle=False) as z:
        student=z['student'].copy();final=z['final'].copy();valid=z['first_episode'].copy()
    with np.load(run/'sequence_native_trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    with np.load(run/'fixed_policy_traces/case_003.npz',allow_pickle=False) as z:public=unpack_public(z['packed_public'])
    if student.shape!=(800,128,18) or len(public)!=800 or trace['q'].shape!=(520,25):
        raise ValueError('Complete original episode trace required')
    errors=[];maximum=0.
    with tempfile.TemporaryDirectory(prefix='current-state-dispatch-audit-') as directory:
        adapter=NativeReplannedSuffixV1(bundle,3,torch.tensor(manifest['configuration']['interface']['limits']),
            manifest['policy_sha256'],directory)
        base=adapter.suffix_feedback.base;nominal=adapter.commands.copy();prior=None
        for tick in range(800):
            obs=torch.zeros((128,794));obs[3]=torch.as_tensor(public[tick]);state=None
            if tick>=280:
                i=tick-280
                state=state_from_arrays(adapter.model,trace['q'][i],trace['v'][i],trace['contacts'][i],trace['foot_force_world_n'][i])
            if tick in by_tick:
                call=by_tick[tick];query=call['query'];plan=call['plan']
                for key,x in [('q',state.q),('v',state.v),('contacts',state.contacts),('target',adapter.previous)]:
                    if not np.array_equal(query[key],x):raise ValueError('Planning query differs from actual trace '+key)
                if not np.array_equal(plan['original_knots_rad'],manifest['replanned_suffix_definition']['original_knots_rad']):
                    raise ValueError('Frozen reference changed')
                if prior is not None:
                    executed=[p for p in plan['corrections'] if p['control_tick']<tick]
                    if len(executed)!=len(prior['corrections']):raise ValueError('Executed patch history changed')
                    for a,b in zip(executed,prior['corrections']):
                        if a['control_tick']!=b['control_tick'] or any(not np.array_equal(a[k],b[k]) for k in ['previous_command','knots_rad']):
                            raise ValueError('Executed correction rewritten')
                feedback=base;last=279
                for patch in plan['corrections']:
                    if not last<patch['control_tick']<=tick:raise ValueError('Invalid correction epoch')
                    last=patch['control_tick']
                    feedback=SuffixCommandCorrectionV1(feedback,last,patch['previous_command'],patch['knots_rad'])
                state.previous_command_target=adapter.previous.copy()
                requested,_=feedback.action(80+tick-280,state,adapter.previous.copy())
                if not np.array_equal(requested,call['first_command']):raise ValueError('Planned first command differs')
                adapter.suffix_feedback=feedback;prior=copy.deepcopy(plan)
                errors.append(dict(control_tick=tick,query_exact=True,first_command_exact=True))
            result=adapter.action(tick,obs,torch.as_tensor(student[tick]),torch.as_tensor(~valid[tick]),state)
            if tick==adapter.suffix_start:
                definition=manifest['replanned_suffix_definition']
                prior=dict(corrections=[dict(control_tick=tick,
                    previous_command=adapter.suffix_feedback.previous_at_start.copy(),
                    knots_rad=definition['corrections_rad'])])
            maximum=max(maximum,float(np.max(abs(result.numpy()-final[tick]))))
        requests=np.asarray([r['requested_target_rad'] for r in adapter.records])
        error=float(np.max(abs(requests-trace['requested_target_rad'])))
        if maximum>1e-6 or error>1e-6 or not np.array_equal(adapter.commands,nominal):
            raise ValueError('Recorded current-state plan did not reproduce actual dispatch')
    return dict(maximum_normalized_action_error=maximum,maximum_target_error_rad=error,
        model_reference_not_substituted_for_measured_state=True,planning_queries=errors,
        planning_not_rerun=True,nominal_commands_immutable=True)
