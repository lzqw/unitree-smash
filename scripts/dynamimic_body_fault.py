"""Native first-body-contact risk over the current and next seven control ticks."""
import torch
import torch.nn.functional as F

from quadsmash.rl.dynamimic_recovery_wm_v1 import local_robot_state, rotation


def features(q, v, feet):
    return torch.cat((q[:, 2:3], rotation(q[:, 3:7])[:, 2], q[:, 7:], v, feet.to(q)), -1)


def fit_native_body_fault(data):
    tick, lane = data['active'].nonzero().T
    train = lane % 16 >= 2
    q, v = local_robot_state(data['after'][tick, lane])
    x = features(q, v, data['feet'][tick, lane])
    center = x[train].mean(0)
    scale = x[train].std(0).clamp_min(.01)
    x_train = (x[train] - center) / scale
    fault = data['active'] & data['body']
    imminent = F.pad(fault, (0, 0, 0, 7)).unfold(0, 8, 1).any(-1)[tick, lane]
    y = imminent[train]
    if not y.any() or y.all():
        raise ValueError('Body-fault fit requires native onset and non-onset windows')
    weight = x.new_zeros(x.shape[1], requires_grad=True)
    bias = x.new_zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([weight, bias], max_iter=100, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        score = x_train @ weight + bias
        loss = .5 * (F.softplus(-score[y]).mean() + F.softplus(score[~y]).mean()) + .001 * weight.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    coef = weight.detach() / scale
    intercept = bias.detach() - center @ coef
    model = torch.cat((coef, intercept))
    predicted = (x @ coef + intercept >= 0)
    metrics = {}
    for name, selected in [('imminent', imminent), ('outside_window', ~imminent)]:
        mask = ~train & selected
        metrics[name] = dict(ticks=int(mask.sum()), flagged=int(predicted[mask].sum()))
    for name in ('body', 'strict', 'relaxed'):
        event = data['active'] & data[name]
        mask = ~train & event[tick, lane] & (tick == event.int().argmax(0)[lane])
        metrics['first_'+name] = dict(cases=int(mask.sum()), flagged=int(predicted[mask].sum()))
    return model, metrics


def body_fault_score(q, v, feet, coefficients):
    # Class-balanced score for smooth reward shaping, not a calibrated probability.
    return (features(q, v, feet) @ coefficients[:-1] + coefficients[-1]).sigmoid()
