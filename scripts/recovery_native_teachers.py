"""Paired post-release feedback probes on a separate training panel."""
import json
import torch


def training_layout(path, protocol_sha, seed):
    from quadsmash.rl.supermimic_independent_panel import fixed128_cases, load_panel
    data = json.loads(path.read_text())
    cases = data['cases']
    if not path.name.startswith('train256_') or len(cases) != 256:
        raise ValueError('Expected an explicitly declared training panel')
    selected = [case for panel in ('A', 'B') for case in [c for c in cases if c['panel'] == panel][:16]]
    held = fixed128_cases(protocol_sha, seed)[0] + load_panel()['cases']
    val = path.with_name('val256_strict.spec.json')
    held += json.loads(val.read_text())['cases']
    signature = lambda row: json.dumps(row['launch'], sort_keys=True)
    if len(set(map(signature, selected))) != 32 or set(map(signature, selected)) & set(map(signature, held)):
        raise ValueError('Training panel overlaps evaluation or has duplicate launches')
    return selected * 4


class NativeTeachers:
    def __init__(self, device):
        self.branch = torch.arange(4, device=device).repeat_interleave(32)
        self.side = torch.tensor([1., -1., 1., -1.], device=device)
        self.front = torch.tensor([1., 1., -1., -1.], device=device)
        self.frames = []

    def feedback(self, obs):
        # Canonical policy order is four hips, four thighs, four calves, six arm joints.
        q, qd = obs[:, 9:27], obs[:, 27:45]
        correction = (-.12 * q - .018 * qd).clamp(-.12, .12)
        roll = (obs[:, 7] + .12 * obs[:, 3]).clamp(-.6, .6)
        pitch = (obs[:, 6] - .12 * obs[:, 4]).clamp(-.6, .6)
        correction[:, :4] += .12 * roll[:, None]
        correction[:, 4:8] += .13 * pitch[:, None] * self.front - .08 * roll[:, None] * self.side
        correction[:, 8:12] -= .15 * pitch[:, None] * self.front
        return correction.clamp(-.16, .16)

    def apply(self, tick, obs, action, rally, episode_index):
        gate = rally.had_contact & rally.released & (episode_index == 0)
        scale = obs.new_tensor([0., .5, 1., -.5])[self.branch]
        adjusted = (action + self.feedback(obs) * scale[:, None] * gate[:, None]).clamp(-1., 1.)
        if tick % 4 == 0:
            self.frames.append((obs.clone(), adjusted.clone(), (adjusted-action).clone(), gate.clone()))
        return adjusted

    def save(self, folder):
        values = [torch.stack(column).cpu() for column in zip(*self.frames)]
        torch.save(dict(zip(('obs', 'requested', 'delta', 'gate'), values)), folder/'teachers.pt')
