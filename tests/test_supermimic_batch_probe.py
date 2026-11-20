import pytest
import torch
from quadsmash.rl.supermimic_native_pair_capture_v1 import (
    validate_spec,probe_lanes,probe_valid_lanes,perturb_request,
)


def spec():
    return dict(schema='native-paired-prefix-batch-spec-v2',tick=272,head='approach',
                lanes=[17,30],request_directions=[[0.]*12+[1.]+[0.]*5,[0.]*13+[-1.]+[0.]*4],epsilon_request=.02)


def test_batch_intervention_only_selected_lanes_and_coordinates():
    s=spec();before=torch.zeros(128,18)
    plus=perturb_request(before,s,'plus');minus=perturb_request(before,s,'minus')
    expected=torch.zeros_like(before);expected[17,12]=.02;expected[30,13]=-.02
    assert torch.equal(plus,expected) and torch.equal(minus,-expected)
    assert before.count_nonzero()==0
    assert torch.equal(perturb_request(before,s,'nominal_a'),before)
    assert probe_lanes(s)==[17,30]


def test_single_lane_legacy_and_batch_are_identical():
    s=spec();s['lanes']=[17];s['request_directions']=s['request_directions'][:1]
    old=dict(schema='native-paired-prefix-spec-v1',tick=272,head='approach',lane=17,
             request_direction=s['request_directions'][0],epsilon_request=.02)
    request=torch.randn(128,18).clamp(-1,1)
    for branch in ('nominal_a','nominal_b','minus','plus'):
        assert torch.equal(perturb_request(request,s,branch),perturb_request(request,old,branch))


@pytest.mark.parametrize('change',[
    {'lanes':[17,17]}, {'lanes':[]}, {'lanes':[True,30]}, {'lane':17},
    {'request_directions':[[0.]*18,[0.]*18]}, {'request_directions':[[1.]*18]},
    {'request_directions':[[float('nan')]*18,[1.]*18]},
])
def test_invalid_batch_specs_rejected(change):
    with pytest.raises(ValueError):validate_spec(dict(spec(),**change))


def test_target_reset_is_rejected_unrelated_reset_is_explicit():
    done=torch.zeros(128,dtype=torch.bool);done[7]=True
    valid=probe_valid_lanes(done,[17,30]);assert not valid[7] and valid[17] and valid[30]
    done[30]=True
    with pytest.raises(ValueError,match='Target probe lane'):probe_valid_lanes(done,[17,30])


def test_saturation_not_faked_as_executable_request_delta():
    request=torch.ones(128,18);result=perturb_request(request,spec(),'plus')
    assert result[17,12]==1 and result[30,13]==pytest.approx(.98)
    assert torch.equal(request,torch.ones_like(request))


def test_action_hold_window_is_validated():
    s=dict(spec(), action_tick=268, action_hold_ticks=4)
    assert validate_spec(s)==s
    with pytest.raises(ValueError, match='hold window'):
        validate_spec(dict(s, action_tick=270))
    with pytest.raises(ValueError, match='positive integer'):
        validate_spec(dict(s, action_hold_ticks=0))


def test_batch_bundle_preserves_lane_alignment_and_explicit_holdout_subset(tmp_path):
    import json,hashlib
    from scripts.build_supermimic_probe_region import build
    from quadsmash.rl.supermimic_native_pair_capture_v1 import BRANCHES,tree_digest
    from quadsmash.rl.supermimic_physical_dataset_v1 import STATE_FIELDS
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
    from quadsmash.rl.supermimic_evidence_binding_v1 import json_sha
    suite=tmp_path/'suite.json'
    suite.write_text(json.dumps({'cases':[{'launch':{'position_m':[i,0,1],'velocity_mps':[1,2,3]}} for i in range(128)]}))
    contract={'suite':{'path':str(suite),'sha256':hashlib.sha256(suite.read_bytes()).hexdigest()}}
    actor={'weight':torch.ones(1)};code={'capture':'fixed'};execution={'gate':'causal'}
    identity=dict(execution_code_sha256=code,actor_state_sha256=tensor_digest(actor.items()),
                  task_contract_sha256=json_sha(contract),policy_execution=execution)
    physical={name:torch.zeros(128,width) for name,width in STATE_FIELDS}
    physical.update(final_action=torch.zeros(128,18),foot_contact=torch.zeros(128,4),ball_present=torch.ones(128,dtype=torch.bool),elapsed_s=torch.full((128,),2.72))
    start={'before':{'physical':physical,'episode_index':torch.zeros(128,dtype=torch.long)}}
    paths={}
    for b in BRANCHES:
        raw=dict(start=start,next_state=torch.arange(128.)[:,None].expand(-1,55).clone(),
                 final_joint_targets=torch.arange(128.)[:,None].expand(-1,18).clone(),final_normalized=torch.zeros(128,18),
                 task_after={k:torch.ones(128,dtype=torch.bool) for k in ('had_contact','released','frame_hit','string_hit')},
                 next_feet=torch.zeros(128,4),recovery_after={'body_ground_seen':torch.zeros(128,dtype=torch.bool)},command_valid_lanes=torch.ones(128,dtype=torch.bool))
        run=dict(schema='native-paired-prefix-run-v1',branch=b,spec=spec(),seed=1,task_contract=contract,policy_execution=execution,
                 physics={'device':'cuda:0'},prefix_sha256='same',executor_before_sha256='same',execution_code_sha256=code,
                 actor_updates=0,world_updates=0,critic_updates=0,actor_state=actor,raw=raw,restore_mode='from-ground-identical-prefix')
        paths[b]=tmp_path/(b+'.pt');torch.save(run,paths[b])
    result=build(paths,identity,tmp_path/'bundle.pt',selected_lanes=[30])
    assert result['native_probe_count']==1 and result['passed'] is None
    bundle=torch.load(tmp_path/'bundle.pt',weights_only=True)
    assert bundle['lane_ids']==[30] and bundle['intervention_lanes']==[17,30]
    assert bundle['pairs']['native_plus'].shape==(1,55)
    assert bundle['pairs']['native_plus'][0,0]==30 and bundle['pairs']['launch'][0,0]==30
    assert bundle['sample_ids'][0].endswith(':30:0')
    with pytest.raises(ValueError,match='unique measured'):build(paths,identity,tmp_path/'wrong.pt',[7])
    build(paths,identity,tmp_path/'both.pt')
    both=torch.load(tmp_path/'both.pt',weights_only=True)
    assert len(both['pairs']['native_contact_modes']['plus'])==2
    changed=torch.load(paths['plus'],weights_only=True);changed['raw']['start']['before']['physical']['elapsed_s'][30]+=.01
    torch.save(changed,paths['plus'])
    with pytest.raises(ValueError,match='Different full native start'):build(paths,identity,tmp_path/'bad.pt')
