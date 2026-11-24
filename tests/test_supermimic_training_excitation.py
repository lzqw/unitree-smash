import torch
import pytest

from scripts.build_supermimic_training_excitation_specs import make_spec, training_lanes
from quadsmash.rl.supermimic_native_pair_capture_v1 import TrainingWindowRecorder, validate_spec


def test_excitation_uses_train_lanes_and_distinct_reproducible_arm_commands():
    lanes = training_lanes([range(96,128), [0,2,17]])
    assert set(lanes).isdisjoint(set(range(96,128)) | {0,2,17})
    spec = make_spec(lanes, 267, 8, 8, .02, 123)
    assert spec == make_spec(lanes, 267, 8, 8, .02, 123)
    assert (spec['action_tick'], spec['tick']) == (259,274)
    d = torch.tensor(spec['request_directions'])
    assert not d[:,:12].any() and d.unique(dim=0).size(0)==len(lanes)
    torch.testing.assert_close(d.norm(dim=1),torch.ones(len(lanes)))
    with pytest.raises(ValueError,match='16-step'):
        validate_spec(dict(spec, tick=267))


def test_probe_window_rejects_resets_anywhere_in_rollout():
    recorder = TrainingWindowRecorder([4,77])
    recorder.rows=[dict(episode=torch.zeros(2),terminated=torch.zeros(2,dtype=torch.bool),
                        timeout=torch.zeros(2,dtype=torch.bool)) for _ in range(16)]
    recorder.rows[2]['terminated'][1]=True
    with pytest.raises(ValueError,match='boundary'):
        recorder.payload()


def test_held_action_response_uses_the_whole_prefix(monkeypatch):
    from types import SimpleNamespace
    from scripts import fit_supermimic_mixed_body_rollout as fit

    # A one-tick command produces a response that persists after reconvergence.
    def step(model, ball, state, targets, previous, feet, present, elapsed, launch):
        return state + model.gain*targets[:, :1], feet, present
    monkeypatch.setattr(fit,'ball_transition',step)
    monkeypatch.setattr(fit,'racket_features',lambda state,geom:state[:,:12])
    model=SimpleNamespace(gain=torch.tensor(.5,requires_grad=True),geometry=None)
    windows={}
    for sign,value in [('minus',-1.),('plus',1.)]:
        targets=torch.zeros(2,2,18)
        targets[0,0,0]=value
        truth=targets[:,:,:1].cumsum(0).expand(-1,-1,55)
        windows[sign]=dict(state=torch.zeros_like(truth),next_state=truth,
                           targets=targets,previous_action=torch.zeros_like(targets),
                           feet=torch.zeros(2,2,4),present=torch.ones(2,2,dtype=torch.bool),
                           elapsed=torch.zeros(2,2))
    point,jac,support=fit.sequence_terms(model,None,windows,torch.arange(2),torch.zeros(2,6),2)
    for tick in (1,2):
        assert support[f'tick_{tick}']['supported']==1
        assert support[f'tick_{tick}']['saturated']==1
        assert support[f'tick_{tick}']['jacobian_loss']>0
    assert torch.autograd.grad(jac+point,model.gain)[0]<0


def test_distinct_window_capacity_uses_only_signed_eligible_lanes():
    from quadsmash.rl.supermimic_training_audit_v1 import identification_window_capacity
    first = dict(measured_windows=2*62)
    assert identification_window_capacity([first],128)['unavoidable_shortfall']==4
    assert identification_window_capacity([first, dict(measured_windows=16)],128)['unavoidable_shortfall']==0
    assert identification_window_capacity([],0)['unavoidable_shortfall']==0


def test_signed_long_pair_masks_contacts_on_the_shared_clock_and_retains_gradients(monkeypatch):
    from quadsmash.rl import supermimic_precontact_body as body

    names = ('had_contact', 'legal_initial_contact', 'airborne_string_hit', 'string_hit',
             'released', 'robot_net_fault', 'crossed_net', 'landed_target', 'outgoing_velocity')
    columns = {'rally/' + name: slice(i, i + (3 if name == 'outgoing_velocity' else 1))
               for i, name in enumerate(names)}

    def pose(state, _):
        rotation = torch.eye(3).flatten().expand(len(state), -1)
        return torch.cat((state[:, :3], rotation, state[:, 7:10], state[:, 10:13]), -1)

    monkeypatch.setattr(body, 'racket_state', pose)

    class World:
        geometry = None
        dt = .01
        gain = torch.tensor(.5, requires_grad=True)

        def __call__(self, state, targets, *args, detach_contacts):
            output = state.clone()
            output[:, :3] += self.gain * targets[:, :3] * self.dt
            output[:, 7:10] = self.gain * targets[:, :3]
            return output, None, None

    def branch(sign):
        shape = (32, 3)
        state = torch.zeros(*shape, 55)
        state[:, :, 3] = 1.
        state[:, :, 49] = .2
        targets = torch.zeros(*shape, 18)
        targets[..., :3] = sign
        state[..., :3] = .01 * sign * torch.arange(32)[:, None, None]
        truth = state.clone()
        truth[..., :3] += .01 * sign
        truth[..., 7:10] = sign
        events = torch.zeros(*shape, 11)
        if sign == 1:
            events[22:, 0, 0] = 1.
            events[23:, 0, 1:5] = 1.
            events[23:, 0, 8] = 4.3
            events[30:, 1, :5] = 1.
            events[30:, 1, 6:8] = 1.
            events[30:, 1, 8] = 4.6
        return dict(state=state, next_state=truth, targets=targets, previous_action=targets,
                    values=events.clone(), next_values=events, next_valid=torch.ones_like(events),
                    feet=torch.zeros(*shape, 4), present=torch.ones(shape, dtype=torch.bool),
                    elapsed=torch.arange(32)[:, None].expand(shape) * .01)

    model = World()
    loss, support, report = body.body_pair_loss(model, branch(-1), branch(1), torch.zeros(3, 6), columns)
    assert support == {4: 3, 8: 3, 16: 3, 20: 3, 32: 1}
    assert set(report[20]) == {'low_speed', 'marginal_success', 'unclassified'}
    assert set(report[32]) == {'unclassified'}
    assert report[20]['low_speed']['normal_response_mae_mps'] > 0
    assert torch.autograd.grad(loss, model.gain)[0] < 0


@pytest.mark.parametrize('steps', [20, 32])
def test_extended_probe_spec_declares_actual_recorded_boundaries(steps):
    spec = make_spec([4, 77], 267, steps + 2, 4, .01, 123, steps)
    assert spec['tick'] - spec['action_tick'] + 1 == steps
    recorder = TrainingWindowRecorder([4, 77], steps)
    assert recorder.steps == steps


def test_delayed_probe_labels_require_exact_branch_actor_prefix_and_episode():
    from quadsmash.rl.supermimic_native_pair_capture_v1 import tree_digest
    from quadsmash.rl.supermimic_training_probe_replay import terminal_response_group, verified_terminal_records

    spec = dict(lanes=[1])
    actor = {'w': torch.ones(1)}
    root = dict(spec=spec, actor_state=actor, prefix_sha256='prefix', branch='minus', seed=9,
                raw={'start': {'before': {'episode_index': torch.tensor([0, 4])}}})
    probe = tree_digest(spec)
    label = dict(env_id=1, episode_index=4, prefix='prefix', branch='minus', probe=probe,
                 actor=tree_digest(actor), stream=tree_digest((probe, 'prefix', 9, 'minus')),
                 scoring_schema='approach_ball_only_v1', numerical_failures=0,
                 approach_ball_terminal=dict(ball_terminal=True, numerical_abort=False),
                 qualified_airborne_string_contact=True, ball_side_complete_except_recovery=True,
                 rejection_reasons=[], outgoing_speed_mps=4.68)
    assert terminal_response_group(verified_terminal_records(dict(root, terminal_labels={1: label}))[1]) == 1
    assert terminal_response_group(None) == 3
    with pytest.raises(ValueError, match='different native probe episode'):
        verified_terminal_records(dict(root, terminal_labels={1: dict(label, episode_index=5)}))
