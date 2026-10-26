"""Check actual measured references and all declared proposals on real inputs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_body_strike_grid_v4 import population_and_labels
from quadsmash.control.full_racket_fixed_actor_body_v4 import FixedActorBodyControllerV4
from scripts.initialize_centered_policy_v4 import same_tree

reports=ROOT/'reports/G07_full_racket_v4'
paths=[reports/'body_candidate4_cold_queue01/candidate4_measured_public.pt',
    reports/'fixed_actor_body_cem1280_gen2_01/best_candidate.pt']
a,b=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
assert same_tree(a['source_public_policy'],b['source_public_policy'])
population,labels=population_and_labels(a['parameters'],b['parameters'])
assert torch.equal(population[0],a['parameters']) and torch.equal(population[1],b['parameters'])
streams=[]
for run in ['fixed_actor_body_gen1_candidate4_fixed_train128_01', 'fixed_actor_body_cem1280_gen2_fixed_train128_01']:
    path=reports/run/'public_and_actions_first2.npz';paths.append(path)
    with np.load(path) as data: streams.append({k:data[k].copy() for k in data.files})
provenance=json.loads((reports/'residual_ppo_centered_placement075_n2048_probe8_01/trajectory_traces/manifest.json').read_text())['provenance']
default=torch.tensor(provenance['default_joint_position_rad'],dtype=torch.float32)
limits=torch.tensor(provenance['joint_limits_rad'],dtype=torch.float32)
controller=FixedActorBodyControllerV4(a['source_public_policy'],population.repeat_interleave(2,dim=0),32,'cpu')
maximum=[0.,0.];count=[0,0];previous=None
for tick in range(len(streams[0]['observations'])):
    public=np.tile(streams[0]['observations'][tick],(16,1));public[2:4]=streams[1]['observations'][tick]
    obs=torch.from_numpy(public)
    if previous is not None:
        reset=torch.where(obs[:,775]<previous-1e-5)[0]
        if len(reset):controller.reset(reset)
    action=controller.action({'policy':obs},default,limits).numpy()
    assert action.shape==(32,18) and np.isfinite(action).all() and (abs(action)<=1).all()
    for i,data in enumerate(streams):
        keep=data['valid_first_episode'][tick]
        if keep.any():
            maximum[i]=max(maximum[i],float(abs(action[i*2:i*2+2][keep]-data['actions'][tick][keep]).max()))
            count[i]+=int(keep.sum())
    previous=obs[:,775].clone()
assert max(maximum)<=5e-4,maximum
paths += [Path(__file__),HERE/'run.py',ROOT/'scripts/search_full_racket_body_strike_grid_v4.py',
    ROOT/'quadsmash/control/full_racket_body_strike_grid_v4.py',ROOT/'quadsmash/control/full_racket_fixed_actor_body_v4.py']
result=dict(checked_at=datetime.now(timezone.utc).isoformat(),candidate_count=16,
    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    exact_reference_max_normalized_action_errors=maximum,reference_valid_public_input_rows=count,
    tolerance=5e-4,all16_proposal_actions_finite_bounded_on_actual_public_inputs=True,
    both_original_residual_actors_exactly_preserved=True,proposal_labels=labels,
    new_native_execution=False,new_performance_measured=False,formal_acceptance=False,passed=True)
target=HERE/'CPU_REFERENCE_AND_GRID_CHECK.json';assert not target.exists()
target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
