import torch
import pytest
from quadsmash.rl.supermimic_native_response_data_v1 import normalized_response_loss


def test_response_supervision_uses_final_command_and_has_model_gradient():
    p=torch.tensor([[.02,0.],[.99,.5]],requires_grad=True)
    target_minus=torch.zeros(2,18);target_plus=target_minus.clone();target_plus[0,12]=.02
    truth=torch.tensor([[.04,0.],[0.,0.]])
    loss,counts=normalized_response_loss(torch.zeros_like(p),p,torch.zeros_like(truth),truth,target_minus,target_plus,torch.ones(2))
    assert float(loss.detach())==pytest.approx(.5) and counts=={'supported':1,'saturated':1}
    loss.backward();assert p.grad[0,0]<0 and torch.equal(p.grad[1],torch.zeros(2))


def test_exact_response_zero_even_with_shared_prediction_bias():
    minus=torch.tensor([[10.,20.]])
    plus=minus+torch.tensor([[.5,1.]])
    t=torch.zeros(1,18);u=t.clone();u[0,12]=.02
    loss,_=normalized_response_loss(minus,plus,torch.zeros(1,2),torch.tensor([[.5,1.]]),t,u,torch.ones(2))
    assert loss==0


def test_no_supported_probes_not_claimed_as_valid():
    p=torch.ones(2,3,requires_grad=True);t=torch.zeros(2,18)
    loss,counts=normalized_response_loss(p,p,p.detach(),p.detach(),t,t,torch.ones(3))
    assert counts['supported']==0 and counts['saturated']==2
    loss.backward();assert torch.equal(p.grad,torch.zeros_like(p))


def test_robot_point_supervision_respects_quaternion_double_cover():
    from scripts.fit_supermimic_native_response import robot_point_loss
    truth=torch.zeros(2,55);truth[:,3]=1
    prediction=truth.clone();prediction[:,3]=-1;prediction.requires_grad_(True)
    loss=robot_point_loss(prediction,truth,torch.ones(55))
    assert loss==0
    loss.backward();assert torch.isfinite(prediction.grad).all()


def test_native_response_loader_requires_real_complete_hashed_prefixes(tmp_path):
    import json,hashlib
    from quadsmash.rl.supermimic_native_response_data_v1 import load_response_data
    from quadsmash.rl.supermimic_native_pair_capture_v1 import BRANCHES
    from quadsmash.rl.supermimic_physical_dataset_v1 import STATE_FIELDS
    (tmp_path/'status.json').write_text(json.dumps({'status':'RUNNING'}))
    split=tmp_path/'split.json';split.write_text(json.dumps({'train_lanes':list(range(96)),'validation_lanes':list(range(96,128))}))
    with pytest.raises(ValueError,match='not complete'):load_response_data(tmp_path,split)
    suite=tmp_path/'suite.json';suite.write_text(json.dumps({'cases':[{'launch':{'position_m':[0,0,1],'velocity_mps':[1,2,3]}} for _ in range(128)]}))
    contract={'suite':{'path':str(suite),'sha256':hashlib.sha256(suite.read_bytes()).hexdigest()}}
    spec=dict(schema='native-paired-prefix-batch-spec-v2',head='approach',tick=272,lanes=list(range(128)),epsilon_request=.02,request_directions=[[0.]*12+[1.]+[0.]*5 for _ in range(128)])
    physical={name:torch.zeros(128,width) for name,width in STATE_FIELDS}
    physical.update(final_action=torch.zeros(128,18),foot_contact=torch.zeros(128,4),ball_present=torch.ones(128,dtype=torch.bool),elapsed_s=torch.ones(128))
    start={'before':{'physical':physical,'episode_index':torch.zeros(128,dtype=torch.long)}}
    records=[]
    for b in BRANCHES:
        raw=dict(start=start,next_state=torch.zeros(128,55),final_joint_targets=torch.zeros(128,18),command_valid_lanes=torch.ones(128,dtype=torch.bool))
        value=dict(schema='native-paired-prefix-run-v1',branch=b,spec=spec,seed=1,task_contract=contract,policy_execution={'gate':'causal'},physics={'device':'cuda:0'},prefix_sha256='same',executor_before_sha256='same',execution_code_sha256={'capture':'same'},restore_mode='from-ground-identical-prefix',actor_state={'p':torch.ones(1)},actor_updates=0,world_updates=0,critic_updates=0,raw=raw)
        folder=tmp_path/b;folder.mkdir();path=folder/'native_probe.pt';torch.save(value,path)
        records.append({'branch':b,'exit_code':0,'probe_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    status={'status':'COLLECTION_FINISHED','actor_updates':0,'completed':records}
    (tmp_path/'status.json').write_text(json.dumps(status))
    data,meta=load_response_data(tmp_path,split)
    assert data['state'].shape==(128,55) and set(meta['train_lanes']).isdisjoint(meta['validation_lanes'])
    target=tmp_path/'plus/native_probe.pt';value=torch.load(target,weights_only=True)
    value['raw']['next_state'][0,0]=float('nan');torch.save(value,target)
    with pytest.raises(ValueError,match='hash changed'):load_response_data(tmp_path,split)
    status['completed'][-1]['probe_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    (tmp_path/'status.json').write_text(json.dumps(status))
    with pytest.raises(ValueError,match='Nonfinite'):load_response_data(tmp_path,split)
