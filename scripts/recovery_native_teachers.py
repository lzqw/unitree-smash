"""Paired post-release feedback probes on a separate training panel."""
import json
import torch


def training_layout(path, protocol_sha, seed, shard=0):
    from quadsmash.rl.supermimic_independent_panel import fixed128_cases, load_panel
    data = json.loads(path.read_text())
    cases = data['cases']
    if not path.name.startswith('train256_') or len(cases) != 256:
        raise ValueError('Expected an explicitly declared training panel')
    if shard not in range(8):
        raise ValueError('Recovery shard outside training panel')
    selected = [case for panel in ('A', 'B') for case in
                [c for c in cases if c['panel'] == panel][16*shard:16*(shard+1)]]
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
        self.release_tick = torch.full_like(self.branch, -1)
        self.arm_masks = []

    def feedback(self, obs):
        # Canonical policy order is four hips, four thighs, four calves, six arm joints.
        q, qd = obs[:, 9:27], obs[:, 27:45]
        supported = (obs[:, 57:61] > .5).any(-1)
        roll = (obs[:, 7] - .12 * obs[:, 3]).clamp(-.6, .6)
        pitch = (obs[:, 6] + .12 * obs[:, 4]).clamp(-.6, .6)
        vx, vy = obs[:, 0:2].clamp(-1., 1.).unbind(-1)
        quiet = obs[:, 6:8].norm(dim=-1) < .21
        kd = torch.where(quiet, .047, .025)
        legs = -.12 * q[:, :12] - kd[:, None] * qd[:, :12]
        legs[:, :4] += (.12 * roll - .04 * vy)[:, None]
        legs[:, 4:8] += (.13 * pitch - .05 * vx)[:, None] * self.front - .08 * roll[:, None] * self.side
        legs[:, 8:12] -= .15 * pitch[:, None] * self.front
        airborne = torch.zeros_like(legs)
        airborne[:, 4:8] = .035
        airborne[:, 8:12] = -.065
        arm = -.16 * q[:, 12:] - .025 * qd[:, 12:]
        return torch.cat((torch.where(supported[:, None], legs, airborne), arm), -1).clamp(-.16, .16)

    def apply(self, tick, obs, action, rally, episode_index):
        gate = rally.had_contact & rally.released & (episode_index == 0)
        first = torch.where(self.release_tick < 0, tick, self.release_tick)
        self.release_tick = torch.where(gate, first, -1)
        arm_allowed = gate & (tick - self.release_tick >= 4)
        scale = obs.new_tensor([0., .5, 1., -.5])[self.branch]
        delta = self.feedback(obs)
        delta[:, 12:] *= arm_allowed[:, None]
        adjusted = (action + delta * scale[:, None] * gate[:, None]).clamp(-1., 1.)
        if tick % 4 == 0:
            self.frames.append((obs.clone(), adjusted.clone(), (adjusted-action).clone(), gate.clone()))
            self.arm_masks.append(arm_allowed.clone())
        return adjusted

    def save(self, folder):
        values = [torch.stack(column).cpu() for column in zip(*self.frames)]
        tape = dict(zip(('obs', 'requested', 'delta', 'gate'), values))
        tape.update(arm_allowed=torch.stack(self.arm_masks).cpu(), feedback_version=2)
        torch.save(tape, folder/'teachers.pt')
