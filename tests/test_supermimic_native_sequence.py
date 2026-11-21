import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from quadsmash.physics.native import ContactEpisodes
from quadsmash.physics.rally_rules_v4 import RallyRulesV4
from quadsmash.tasks.stand.contact_flight_recovery_v5 import ContactFlightRecoveryV5
from quadsmash.rl.supermimic_native_sequence_v1 import NativeTaskSequenceV1, replay_task_sequence
from quadsmash.rl.supermimic_native_metrics import NativeReturnAccumulator


def test_original_substep_task_functions_replay_and_inputs_do_not_alias():
    court=json.loads(Path('configs/court_net_v4.json').read_text())
    env=SimpleNamespace(cfg=SimpleNamespace(decimation=160),
        rally=RallyRulesV4(2,'cpu',court),
        contact_recovery=ContactFlightRecoveryV5(2,'cpu',{'recovery':{'continuous_stable_window_s':.8}}),
        contact_events=ContactEpisodes(2,'cpu'))
    old=copy.deepcopy(env)
    recorder=NativeTaskSequenceV1(env)
    jump=SimpleNamespace(flight_candidate=torch.ones(2,dtype=torch.bool),
        takeoff_time=torch.ones(2),flight_apex_rise=torch.full((2,),.1),
        flight_clear_time=torch.full((2,),.2),valid_jump_landed=torch.zeros(2,dtype=torch.bool),
        flight_qualified=torch.ones(2,dtype=torch.bool),landing_time=torch.zeros(2),
        body_ground_seen=torch.zeros(2,dtype=torch.bool))
    recorder.begin()
    env.rally.launch(torch.arange(2),torch.ones(2))
    force=torch.ones(2,3)
    env.contact_events.update(force,torch.ones(2))
    env.contact_recovery.metric(jump,torch.zeros(2,dtype=torch.bool),.001)
    env.contact_recovery.contact(jump,legal_air_strings_touching=torch.ones(2,dtype=torch.bool),
        time=torch.full((2,),1.2),rally_contact_time=torch.full((2,),1.2))
    jump.flight_candidate.zero_();jump.valid_jump_landed.fill_(True);jump.landing_time.fill_(1.4)
    for stable,dt in (([True,True],.799),([False,True],.001),([True,True],.001)):
        env.contact_recovery.metric(jump,torch.tensor(stable),dt)
    force.zero_()
    env.contact_events.update(force,torch.ones(2))
    env.contact_events.update(force,torch.ones(2))
    trace=recorder.finish()
    assert env.contact_recovery.stable_seen.tolist()==[False,True]
    replay_task_sequence(trace,old.rally,old.contact_recovery,old.contact_events)
    for left,right in ((env.rally,old.rally),(env.contact_recovery,old.contact_recovery),(env.contact_events,old.contact_events)):
        for key,value in vars(left).items():
            if isinstance(value,torch.Tensor):
                torch.testing.assert_close(value,getattr(right,key),atol=0,rtol=0,equal_nan=True)
    assert trace['streams']['contact.update']['tensors']['filtered_force'][0].eq(1).all()
    recorder.close()


def test_discounted_native_return_is_separate_and_resets_per_lane():
    values=NativeReturnAccumulator(2,'cpu',gamma=.5)
    values.add(torch.tensor([10.,2.]));values.add(torch.tensor([20.,4.]))
    row=values.records([0])[0]
    assert row['native_episode_return']==30
    assert row['native_episode_discounted_return']==20
    values.reset(torch.tensor([0]));values.add(torch.tensor([3.,8.]))
    assert values.records([0])[0]['native_episode_discounted_return']==3
    assert values.records([1])[1]['native_episode_discounted_return']==6


def test_sparse_racket_label_uses_same_physics_frame_and_world_origin():
    rec = NativeTaskSequenceV1.__new__(NativeTaskSequenceV1)
    xyz = torch.arange(6, dtype=torch.float).reshape(2,1,3)
    data = SimpleNamespace(body_link_pos_w=xyz, body_link_lin_vel_w=2*xyz, body_link_ang_vel_w=3*xyz)
    rec.env = SimpleNamespace(robot=SimpleNamespace(data=data), racket_body_id=0, cfg=SimpleNamespace(decimation=160),
        shuttle=SimpleNamespace(data=SimpleNamespace(root_pos_w=xyz[:,0]-1)), episode_index=torch.tensor([5,6]),
        first_tool_normal=torch.tensor([[1.,0,0],[1.,0,0]]), physics_dt=.0000625, _last_physics_frame=118)
    rec.start_frame = 100
    rec.reference_targets = []
    rec.record_racket_target(dict(legal_air_strings_touching=torch.tensor([True,True]),
        time=torch.tensor([1.,2.]), rally_contact_time=torch.tensor([.9,2.])))
    label, = rec.reference_targets
    assert label['lanes'].tolist() == [1] and label['episode'].tolist() == [6]
    assert label['substep']==18 and label['offset_s']==18*.0000625
    torch.testing.assert_close(label['features'][0],torch.cat((torch.ones(3),2*xyz[1,0],
                                                              torch.tensor([1.,0,0]),3*xyz[1,0])))
