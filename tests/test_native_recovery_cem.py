import torch
from scripts.native_recovery_cem import correction, panel_layout


def test_feedback_phase_clipping_and_native_joint_order():
    obs = torch.zeros(4,794)
    obs[:,8] = -1
    coeff = torch.zeros(4,24)
    coeff[:,0] = 1.5
    coeff[:,20] = .4
    delta = correction(obs, torch.full((4,),.14), torch.tensor([True,True,False,False]),
                       torch.tensor([.04,.12,.12,.4]), coeff)
    torch.testing.assert_close(delta[:,10], torch.tensor([0.,1.5,.4,0.]))
    assert torch.count_nonzero(delta)==2
    assert ((-1.22 + delta[1,10]).clamp(-1,1) > 0).item()


def test_native_cem_panel_rotation_and_companions():
    bank=[dict(case_id=f'{p}{i}',panel=p) for p in 'AB' for i in range(128)]
    seen=set()
    for g in range(16):
        layout=panel_layout(bank,g)
        assert len(layout)==128 and layout[:16]*8==layout
        assert all(layout[:16:2]==layout[i:i+16:2] for i in range(0,128,16))
        seen.update(c['case_id'] for c in layout)
    assert len(seen)==256


def test_resident_search_keeps_strict_rescue_and_nominal_anchor(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.native_recovery_cem import search
    from quadsmash.rl import supermimic_terminal_scoring as scoring
    import json

    monkeypatch.setattr(scoring, 'score_terminal_episode', lambda row, protocol: row)
    cases = [dict(case_id=f'{p}{i}', panel=p) for p in 'AB' for i in range(128)]
    clock = torch.zeros(128,794)
    clock[:,8] = -1
    clock[:,775] = .15
    release = torch.zeros(128)
    on = torch.ones(128,dtype=torch.bool)
    xyz = torch.zeros(128,3)
    env = SimpleNamespace(device='cpu', episode_index=torch.zeros(128,dtype=torch.long),
        rally=SimpleNamespace(had_contact=on, released=on, release_time=release, robot_net_fault=~on),
        robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=xyz)), scene=SimpleNamespace(env_origins=xyz),
        body_ground_seen=~on, landed_once=on, rally_records=[])
    def step(action):
        for i,c in enumerate(panel_layout(cases,0)):
            fall = i%16==0 and i<16
            env.rally_records.append(dict(case_id=c['case_id'],env_id=i,episode_index=0,numerical_failures=0,
                jump=dict(terminated=False),body_ground_seen=fall,robot_net_fault=False,
                qualified_airborne_string_contact=True,physical_recovery_success=not fall,
                ball_side_complete_except_recovery=True,contact_flight_recovery=dict(success=not fall,recovery_s=.8)))
        return {'policy':clock},None,on,None
    class Policy:
        def __init__(self):
            self.release_arm_reference=torch.nn.Linear(1,1)
            self.actor=SimpleNamespace(recovery=torch.nn.Linear(1,1))
        def install_release_arm_reference(self):
            pass
        def _gated_distribution(self, x, gate, release_time):
            return SimpleNamespace(loc=torch.zeros(len(x),18))
    ckpt=tmp_path/'seed.pt'
    ckpt.write_bytes(b'test')
    args=SimpleNamespace(output=tmp_path,checkpoint=ckpt,seed=1,recovery_cem_generations=1)
    search(args,env,SimpleNamespace(step=step),{'policy':clock},Policy(),Policy(),cases,{},dict(signal=False))
    folder=tmp_path/'generation_01'
    report=json.loads((folder/'results.json').read_text())
    assert report['teachers']==['A0'] and len(report['scores'])==8
    examples=torch.load(folder/'examples.pt',weights_only=True)
    assert sum(e['anchor'] for e in examples)==15
    assert examples[0]['delta_raw'].abs().max()>0
