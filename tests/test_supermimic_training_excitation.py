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
