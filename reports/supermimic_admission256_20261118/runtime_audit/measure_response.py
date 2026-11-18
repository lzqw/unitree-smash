"""Read-only same-prefix physical response measurement; creates no passing receipt."""
import hashlib,json
from pathlib import Path
import torch
from quadsmash.rl.supermimic_native_pair_capture_v1 import BRANCHES,tree_digest
from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
from quadsmash.rl.supermimic_action_ball_v1 import ActionConditionedBallV1
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_probe_region_v1 import contact_mode_groups

def main():
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parent
    runs={k:torch.load(root/k/'native_probe.pt',map_location='cpu',weights_only=True) for k in BRANCHES}
    ref=runs['nominal_a'];lane=ref['spec']['lane'];sl=slice(lane,lane+1)
    for k,r in runs.items():
        for field in ('spec','seed','task_contract','policy_execution','physics','prefix_sha256','executor_before_sha256','execution_code_sha256'):
            if r[field]!=ref[field]:raise ValueError('Native repeatability mismatch: '+k+':'+field)
        if tree_digest(r['actor_state'])!=tree_digest(ref['actor_state']):raise ValueError('Actor changed')
        if tree_digest(r['raw']['start'])!=tree_digest(ref['raw']['start']):raise ValueError('Full start state changed')
        if any(r.get(k)!=0 for k in ('actor_updates','world_updates','critic_updates')):raise ValueError('Unexpected updates')
    physical=ref['raw']['start']['before']['physical'];state=encode_physical(physical)[sl]
    suite=Path(ref['task_contract']['suite']['path'])
    if not suite.is_absolute():suite=Path.cwd()/suite
    if hashlib.sha256(suite.read_bytes()).hexdigest()!=ref['task_contract']['suite']['sha256']:raise ValueError('Suite changed')
    case=json.loads(suite.read_text())['cases'][lane]
    launch=torch.tensor([case['launch']['position_m']+case['launch']['velocity_mps']])
    base_path=Path('/tmp/supermimic_racket_dynamics_20261114_YRFhKS/run/world.pt')
    ball_path=Path('/tmp/supermimic_ball_sequence_20261114_axsonnit/run/ball.pt')
    base=PhysicalWorldV2(ref['task_contract']['asset']['sha256']).eval()
    base.load_state_dict(torch.load(base_path,map_location='cpu',weights_only=True)['model'],strict=True)
    ball=BallResidualV2().eval();ball.load_state_dict(torch.load(ball_path,map_location='cpu',weights_only=True)['model'],strict=True)
    candidate_path=root.parent/'action_ball_neighborhood/action_ball_candidate.pt'
    candidate=ActionConditionedBallV1().eval();candidate.load_state_dict(torch.load(candidate_path,map_location='cpu',weights_only=True)['model'],strict=True)
    base.requires_grad_(False);ball.requires_grad_(False);candidate.requires_grad_(False)
    modes={}
    for name,r in runs.items():
        modes[name]={k:v[lane].tolist() for k,v in r['raw']['task_after'].items() if k in ('had_contact','released','frame_hit','string_hit','legal_initial_contact','airborne_string_hit','rally_fault')}
        modes[name]['foot_contact']=r['raw']['next_feet'][lane].bool().tolist()
    native_delta=(runs['plus']['raw']['next_state']-runs['minus']['raw']['next_state'])[sl,52:55]
    minus=runs['minus']['raw']['final_joint_targets'][sl];plus=runs['plus']['raw']['final_joint_targets'][sl]
    delta=plus-minus
    result=dict(scope='single calibration episode; no validity bounds fitted; no training admission',
        optimizer_updates=0,diagnostic_device='cpu',native_device=ref['physics'],spec=ref['spec'],
        source_sha256={k:hashlib.sha256((root/k/'native_probe.pt').read_bytes()).hexdigest() for k in BRANCHES},
        models_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in dict(physical=base_path,legacy_ball=ball_path,candidate=candidate_path).items()},
        nominal_repeat_max=float((runs['nominal_a']['raw']['next_state']-runs['nominal_b']['raw']['next_state']).abs().max()),
        final_target_delta_rad=delta.tolist(),final_target_delta_norm=float(delta.norm()),
        native_ball_velocity_delta=native_delta.tolist(),native_response_norm=float(native_delta.norm()),
        native_contact_modes=modes,contact_mode_group=contact_mode_groups(modes,1,'approach'),models={})
    kwargs=dict(previous_action=physical['final_action'][sl],feet=physical['foot_contact'][sl].float(),present=physical['ball_present'][sl],elapsed=physical['elapsed_s'][sl],launch=launch)
    def predict(target,action_model):return ball_transition(base,ball,state,target,**kwargs,action_model=action_model)[0]
    for model,action_model in (('legacy',None),('action_conditioned',candidate)):
        with torch.no_grad():
            pred={k:predict(r['raw']['final_joint_targets'][sl],action_model) for k,r in runs.items()}
            pd=pred['plus'][:,52:55]-pred['minus'][:,52:55]
        target=((minus+plus)/2).detach().requires_grad_(True)
        velocity=predict(target,action_model)[:,52:55]
        grads=[torch.autograd.grad(velocity[:,j].sum(),target,retain_graph=True,allow_unused=True)[0] for j in range(3)]
        jac=torch.stack([torch.zeros_like(target) if g is None else g for g in grads],1)
        projected=(jac*delta[:,None]).sum(-1).detach()
        def cosine(x,y):
            denom=float(x.norm()*y.norm())
            return None if denom==0 else float((x*y).sum()/denom)
        result['models'][model]=dict(predicted_velocity_delta=pd.tolist(),projected_jacobian=projected.tolist(),
            response_cosine=cosine(pd,native_delta),jacobian_cosine=cosine(projected,native_delta),
            response_error_norm=float((pd-native_delta).norm()),jacobian_finite=bool(torch.isfinite(jac).all()),
            branch_velocity={k:dict(native=r['raw']['next_state'][sl,52:55].tolist(),predicted=pred[k][:,52:55].tolist(),
                rmse=float((r['raw']['next_state'][sl,52:55]-pred[k][:,52:55]).square().mean().sqrt())) for k,r in runs.items()})
    with (root/'measured_response.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
