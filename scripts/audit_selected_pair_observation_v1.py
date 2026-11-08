"""Check selected pair measurements against an unchanged native parent run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_frozen_trace_divergence_v1 import compare
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.native_recovery_query_bank_v1 import state_from_sample


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['run','parent','bundle','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if a.output.exists():raise ValueError('Fresh audit required')
    for run in [a.run,a.parent]:
        j=read(run/'job.json')
        if j['status']!='SUCCEEDED' or j['exit_code']!=0:raise ValueError('Completed native parents required')
    dispatch_path=a.output.with_name('dispatch_audit.json')
    subprocess.run([sys.executable,str(ROOT/'scripts/audit_native_experimental_recovery_v1.py'),
        '--run',str(a.run),'--bundle',str(a.bundle),'--output',str(dispatch_path)],check=True)
    dispatch=read(dispatch_path);meta=read(a.run/'filtered_contact_observation.json');rawmeta=read(a.run/'foot_impact_observation.json')
    case=dispatch['case'];expected=[case,125]
    if (meta['lanes']!=expected or rawmeta['lanes']!=expected or not meta['high_rate_complete']
            or not meta['panel_complete'] or not rawmeta['complete_window']
            or meta['declared_first_physics_frame']!=318*160 or meta['declared_last_physics_frame']!=328*160
            or meta['selected_observer_source_sha256']!=sha(ROOT/'scripts/native_selected_pair_observer_v1.py')
            or rawmeta['data_sha256']!=sha(a.run/'foot_impact_observation.npz')):
        raise ValueError('Wrong selected observer/source/window')
    arrays={}
    for stem in ['filtered_foot_pairs','filtered_ground_panel','foot_impact_observation','sequence_native_trace']:
        path=a.run/(stem+'.npz')
        if stem.startswith('filtered') and sha(path)!=meta['data_sha256'][path.name]:raise ValueError('Changed pair data')
        with np.load(path,allow_pickle=False) as z:arrays[stem]={k:z[k].copy() for k in z.files}
        if any(not np.isfinite(v).all() for v in arrays[stem].values()):raise ValueError('Nonfinite telemetry')
    pairs=arrays['filtered_foot_pairs'];raw=arrays['foot_impact_observation'];panel=arrays['filtered_ground_panel'];trace=arrays['sequence_native_trace']
    frames=np.arange(318*160,328*160+1)
    if (not np.array_equal(raw['physics_frame'],frames) or not np.array_equal(pairs['physics_frame'],frames)
            or not np.array_equal(panel['physics_frame'],np.arange(160,128001,160))
            or pairs['pair_buffer_invalid'].any() or raw['episode_index'].any()):
        raise ValueError('Incomplete/reset/invalid selected window')
    normal=pairs['pair_normal_force_world_n'];friction=pairs['pair_friction_force_world_n']
    model=LoadedPinocchioModelV1();alignment=[]
    for tick in range(318,329):
        i=(tick-318)*160;j=tick-280
        state=state_from_sample(model,{k:v[i,0] for k,v in raw.items() if k!='physics_frame'},rawmeta['origins_world'][0])
        errors=dict(q=float(np.max(abs(state.q-trace['q'][j]))),v=float(np.max(abs(state.v-trace['v'][j]))),
            previous_command=float(np.max(abs(state.previous_command_target-trace['previous_target_rad'][j]))),
            controller_ground_force=float(np.max(abs(normal[i,0,:,0]-trace['foot_force_world_n'][j]))),
            panel_ground_force=float(np.max(abs(normal[i,:,:,0]-panel['ground_normal_force_world_n'][tick-1,expected]))))
        passed=errors['q']<1e-12 and errors['v']<1e-12 and errors['previous_command']<5e-6 and max(errors[k] for k in ['controller_ground_force','panel_ground_force'])<1e-3
        alignment.append(dict(tick=tick,passed=passed,errors=errors))
    manifests=[read(p/'fixed_policy_traces/manifest.json') for p in [a.parent,a.run]]
    counts=[{r['source_training_index']:r['samples'] for r in m['episodes']} for m in manifests]
    if any(set(c)!=set(range(128)) for c in counts):raise ValueError('Full TRAIN128 required')
    comparisons=[];pins={}
    for case_i in range(128):
        row,source=compare(a.parent,a.run,case_i);pins.update(source)
        row['complete_episode_length_equal']=counts[0][case_i]==counts[1][case_i]
        comparisons.append(row)
    unchanged=all(r['complete_episode_length_equal'] and r['compared_samples']==counts[0][r['case']]
        and r['maximum_public_error']==0 and r['maximum_action_error']==0 for r in comparisons)
    summaries=[];selected_events=[]
    for i in range(4):
        for j in range(i+1,4):
            ia=1+[k for k in range(4) if k!=i].index(j);ib=1+[k for k in range(4) if k!=j].index(i)
            magnitude=np.linalg.norm(normal[:,0,i,ia],axis=1);hit=np.flatnonzero(magnitude>1.)
            summaries.append(dict(feet=[meta['feet'][i],meta['feet'][j]],peak_normal_n=float(magnitude.max()),
                first_force_over_1n_s=float(frames[hit[0]]*.0000625) if len(hit) else None,
                sampled_exceedance_s=float(len(hit)*.0000625),
                normal_antisymmetry_max_n=float(np.max(abs(normal[:,:,i,ia]+normal[:,:,j,ib])))))
            if len(hit):
                k=int(np.argmax(magnitude));selected_events.append(dict(physics_frame=int(frames[k]),
                    feet=[meta['feet'][i],meta['feet'][j]],normal_force_world_n=normal[k,0,i,ia].tolist(),
                    friction_force_world_n=friction[k,0,i,ia].tolist(),
                    ground_normal_force_world_n=normal[k,0,:,0].tolist()))
    paths=[Path(__file__),ROOT/'scripts/native_selected_pair_observer_v1.py',dispatch_path,
        a.run/'filtered_contact_observation.json',a.run/'foot_impact_observation.json',
        a.run/'job.json',a.parent/'job.json']+[a.run/(k+'.npz') for k in arrays]
    pins.update({str(p.resolve()):sha(p) for p in paths})
    passed=unchanged and all(r['passed'] for r in alignment)
    result=dict(dispatch,passed_dispatch_integrity=dispatch['passed_dispatch_integrity'] and passed,
        filtered_observation_passed=passed,passive_parent_replay=unchanged,control_boundary_alignment=alignment,
        case_comparisons=comparisons,pair_summaries=summaries,selected_peak_events=selected_events,
        original_student_isolation_not_inferred=True,valid_expert_labels=False,source_sha256=pins)
    temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(a.output)
    print(json.dumps(dict(passed=passed,complete5=result['complete5'],pair_summaries=summaries)))
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
