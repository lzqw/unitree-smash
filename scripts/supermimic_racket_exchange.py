"""Offline attribution of nominal racket motion and signed perturbations in the world frame."""
import torch
from torch.nn.functional import normalize

from audit_supermimic_paired_body_transfer import response_metrics
from evaluate_supermimic_contact_rollout import tick
from fit_supermimic_unloaded_racket_motion import multiply_quaternions
from quadsmash.rl.supermimic_contact_pilot_v1 import _quat_rotate


def exchange_racket_path(base, nominal, signed):
    """Transport the signed world-frame pose and twist change onto the other nominal motion."""
    c, q, v, w = base
    c0, q0, v0, w0 = nominal
    c1, q1, v1, w1 = signed
    inverse = normalize(q0.double(), dim=-1) * q0.new_tensor([1., -1., -1., -1.])
    delta = multiply_quaternions(normalize(q1.double(), dim=-1), inverse)
    delta = torch.where(delta[..., :1] < 0, -delta, delta)
    rotation = normalize(multiply_quaternions(delta, normalize(q.double(), dim=-1)), dim=-1).to(q)
    # Differentiate R_signed R_nominal.T R_base using spatial angular velocities.
    omega = w1 + _quat_rotate(delta, (w-w0).double()).to(w)
    return c + (c1-c0), rotation, v + (v1-v0), omega


@torch.no_grad()
def crossed_rollouts(paths, first, prior, cfg, material, reference, comparison):
    nominal = paths['nominal_a']
    responses = {}
    for base in ('body', 'native'):
        for change in ('body', 'native'):
            velocities = {}
            for branch in ('minus', 'plus'):
                ball = torch.cat((first[:, :3], first[:, 7:10]), -1)
                spin = first[:, 10:13]
                for neutral, signed in zip(nominal, paths[branch], strict=True):
                    path = (signed[change] if base == change else
                            exchange_racket_path(neutral[base], neutral[change], signed[change]))
                    ball, _, spin = tick(prior, ball, None, None, cfg['fitted'], cfg['solver']['pair_contact_offset_m'],
                                         measured_path=path, initial_spin=spin, compound_racket=True,
                                         racket_material=material, return_spin=True)
                velocities[branch] = ball[:, 3:]
            responses[f'{base}_nominal_{change}_change'] = velocities['plus'] - velocities['minus']
    last_tick = str(reference['ticks'][-1])
    def observed(key):
        return torch.tensor([case['ticks'][last_tick][key] for case in reference['cases']])
    for key, original in [('native_nominal_native_change', 'measured_racket'), ('body_nominal_body_change', 'body_racket')]:
        torch.testing.assert_close(responses[key], observed(original), atol=2e-6, rtol=1e-5)
    true, repeat = observed('native'), observed('native_repeat')
    lanes = [case['lane'] for case in reference['cases']]
    groups = {}
    for name, group in comparison['groups'].items():
        indices = [lanes.index(lane) for lane in group['case_ids']]
        if indices:
            groups[name] = {key: response_metrics(true[indices], values[indices], repeat[indices])
                            for key, values in responses.items()}
    return dict(rotation_transport='world relative rotation with spatial angular velocity transport',
                groups=groups, lanes=lanes, signed_ball_velocity_differences={k:v.tolist() for k,v in responses.items()},
                native_differences=true.tolist(), native_repeat=repeat.tolist(),
                scope='Development trajectory substitutions. Unmodified controls reproduce the original signed ball velocities.')
