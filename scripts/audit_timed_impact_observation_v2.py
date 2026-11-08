"""Validate passive high-rate observation against the executed timed parent.

This validates recorder transparency, not isolation from the original student
or admission of the recorded controller as an expert.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_frozen_trace_divergence_v1 import compare
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.native_recovery_query_bank_v1 import state_from_sample


def audit(run,parent):
    read=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if read(run/'job.json')['status']!='SUCCEEDED' or read(parent/'job.json')['status']!='SUCCEEDED':
        raise ValueError('Both original ground runs must complete')
    rows=[];pins={}
    manifests=[read(path/'fixed_policy_traces/manifest.json') for path in [parent,run]]
    samples=[{r['source_training_index']:r['samples'] for r in m['episodes']} for m in manifests]
    if any(set(counts)!=set(range(128)) for counts in samples):raise ValueError('Full original cohort required')
    for case in range(128):
        row,more=compare(parent,run,case);pins.update(more)
        # Match the complete first episode, including unchanged early termination.
        if samples[0][case]!=samples[1][case] or row['compared_samples']!=samples[0][case]:
            raise ValueError('First-episode length changed or partial comparison')
        rows.append(row)
    unchanged=all(r['maximum_public_error']==0 and r['maximum_action_error']==0 for r in rows)
    meta=read(run/'foot_impact_observation.json')
    if (not meta['complete_window'] or meta['lanes']!=[85,87]
            or abs(meta['physics_dt_s']-.0000625)>1e-12
            or meta['data_sha256']!=sha(run/'foot_impact_observation.npz')):
        raise ValueError('Wrong observation coverage or clock')
    with np.load(run/'foot_impact_observation.npz',allow_pickle=False) as z:physics={k:z[k].copy() for k in z.files}
    with np.load(run/'sequence_native_trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    if not np.array_equal(physics['physics_frame'],np.arange(288*160,300*160+1)):
        raise ValueError('Missing physics frames')
    if any(not np.isfinite(v).all() for v in physics.values()) or physics['episode_index'].any():
        raise ValueError('Nonfinite or reset observation')
    model=LoadedPinocchioModelV1();alignment=[]
    for tick in range(288,301):
        i=(tick-288)*160;j=tick-280
        sample={k:v[i,0] for k,v in physics.items() if k!='physics_frame'}
        state=state_from_sample(model,sample,meta['origins_world'][0])
        errors=dict(q=float(np.max(abs(state.q-trace['q'][j]))),
                    v=float(np.max(abs(state.v-trace['v'][j]))),
                    previous_target=float(np.max(abs(state.previous_command_target-trace['previous_target_rad'][j]))),
                    force=float(np.max(abs(state.measured_force_world-trace['foot_force_world_n'][j]))))
        passed=(errors['q']<1e-12 and errors['v']<1e-12 and errors['force']==0
                and errors['previous_target']<5e-6 and np.array_equal(state.contacts,trace['contacts'][j]))
        alignment.append(dict(tick=tick,passed=passed,errors=errors))
    for p in [Path(__file__),run/'foot_impact_observation.json',run/'foot_impact_observation.npz',
              run/'sequence_native_trace.npz',run/'sequence_native_result.json',run/'job.json',
              parent/'job.json',ROOT/'scripts/native_timed_impact_observer_v1.py',
              ROOT/'scripts/native_foot_impact_observer_v1.py',ROOT/'scripts/native_recovery_query_bank_v1.py']:
        pins[str(p.resolve())]=sha(p)
    return dict(scope=__doc__,passed=unchanged and all(r['passed'] for r in alignment),
                passive_parent_replay=unchanged,controller_unchanged=True,
                strict_student_isolation_not_claimed=True,case_comparisons=rows,
                complete_physics_window=True,control_boundary_alignment=alignment,
                source_sha256=pins,valid_expert_labels=False,native_reset_capable=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','parent','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh audit output required')
    result=audit(a.run.resolve(),a.parent.resolve())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['passed','passive_parent_replay','complete_physics_window']}))
    if not result['passed']:raise SystemExit(2)
