"""Compare actual grid references with prior cold runs; no causal claim."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
MAIN=ROOT.parent/'unitree_smash'
HERE=Path(__file__).resolve().parent
R=MAIN/'reports/G07_full_racket_v4'
GRID=R/'body_strike_grid16_n2048_01'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not(HERE/'RESULT.json').exists()
    assert read(GRID/'job.json')['status']=='SUCCEEDED'
    population=torch.load(GRID/'population_000.pt',map_location='cpu',weights_only=False)['parameters']
    a=torch.load(R/'body_candidate4_cold_queue01/candidate4_measured_public.pt',map_location='cpu',weights_only=False)
    b=torch.load(R/'fixed_actor_body_cem1280_gen2_01/best_candidate.pt',map_location='cpu',weights_only=False)
    assert torch.equal(population[0],a['parameters']) and torch.equal(population[1],b['parameters'])
    ground=read(GRID/'initial_ground_states_000.json');rows=read(GRID/'episodes_000.json')
    metrics=read(GRID/'metrics.json')['history'][0]['summaries']
    groups=[];source_paths=[Path(__file__),GRID/'population_000.pt',GRID/'initial_ground_states_000.json',
        GRID/'episodes_000.json',GRID/'metrics.json',GRID/'all_first_episodes_000.npz',GRID/'public_and_actions_000.npz']
    with np.load(GRID/'all_first_episodes_000.npz') as data:
        ids=np.array([128*c+i for c in [0,1,2,4,6,8] for i in [0,1]])
        allposes=data['link_poses'];grid_roots=allposes[:,ids,0].copy();del allposes
        grid_clocks=data['episode_clock_s'][:,ids].copy();grid_valid=data['valid_first_episode'][:,ids].copy()
    np.savez_compressed(HERE/'actual_selected_roots.npz',roots=grid_roots,clocks=grid_clocks,valid=grid_valid,env_ids=ids)
    for c,run in enumerate(['fixed_actor_body_gen1_candidate4_fixed_train128_01','fixed_actor_body_cem1280_gen2_fixed_train128_01']):
        cold=R/run;initial=read(cold/'initial_states.json');source_paths.extend([cold/'initial_states.json',cold/'all_first_episodes.npz',cold/'public_and_actions_first2.npz'])
        # Initial cold evaluator state uses the same root/joint state layout.
        initial_key=next(k for k in ['root_state_local','robot_root_state_local'] if k in initial)
        old_root=np.asarray(initial[initial_key]);new_root=np.asarray(ground['root_state_local'])[c*128:(c+1)*128]
        root_error=float(abs(new_root-old_root).max())
        with np.load(cold/'all_first_episodes.npz') as data:
            old_roots=data['link_poses'][:,:2,0].copy();old_clocks=data['episode_clock_s'][:,:2].copy();old_valid=data['valid_first_episode'][:,:2].copy()
        np.savez_compressed(HERE/f'actual_cold_reference{c}_roots.npz',roots=old_roots,clocks=old_clocks,valid=old_valid)
        with np.load(GRID/'public_and_actions_000.npz') as data:grid_initial_public=data['observations'][0,c*2:c*2+2].copy()
        with np.load(cold/'public_and_actions_first2.npz') as data:cold_initial_public=data['observations'][0].copy()
        group=rows[c*128:(c+1)*128]
        groups.append(dict(reference_candidate=c,exact_measured_parameters_preserved=True,
            initial_all128_root_state_max_difference=root_error,
            initial_first2_public_max_difference=float(abs(grid_initial_public-cold_initial_public).max()),
            grid_summary=metrics[c],cold_summary=read(cold/'metrics.json'),
            grid_body_landing_count=sum(r['jump']['body_landing'] for r in group),
            grid_physical_jump_success=sum(r['jump']['success'] for r in group),
            grid_terminated_count=sum(r['jump']['terminated'] for r in group)))
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),exact_references=groups,
        all_gain1p3_proposals_airborne_zero=all(metrics[i]['airborne_contacts']==0 for i in [3,5,7,9,11,13,15]),
        actual_reference_root_samples_saved=True,reference_first2_TRAIN_indices=[0,1],
        causes_not_established=True,new_native_execution=False,formal_acceptance=False,
        limitations=['Different batch size, seed and native instances; no isolated causal attribution.',
            'Public inputs and initial state closeness do not prove subsequent physical trajectories identical.',
            'All high-gain proposals failing in this population does not prove a general impossibility.'],
        source_sha256={str(p):sha(p) for p in source_paths})
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(references=[{k:v for k,v in g.items() if k in ['reference_candidate','initial_all128_root_state_max_difference','initial_first2_public_max_difference','grid_body_landing_count','grid_physical_jump_success','grid_terminated_count']} for g in groups])))

if __name__=='__main__':main()
