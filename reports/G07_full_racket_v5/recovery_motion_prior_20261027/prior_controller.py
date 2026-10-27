"""A bounded shared residual trajectory, seeded by an actual recovered episode.

Runtime inputs are the public clock, causal source forecast and an observed
air/contact phase. A phase3 observation is latched through landing. The source
case identity is provenance only and is never a runtime input.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller
from quadsmash.control.full_racket_residual_v4 import ResidualActionMixerV4

HERE = Path(__file__).resolve().parent
SCHEMA = 'g07-shared-native-recovery-prior-v5'


def load_seed(directory=HERE):
    directory = Path(directory)
    metadata = json.loads((directory / 'SEED_RECEIPT.json').read_text())
    path = directory / metadata['table_file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['table_sha256']:
        raise ValueError('Recorded recovery prior changed')
    with np.load(path) as data:
        arrays = {key: data[key].copy() for key in data.files}
    return metadata, arrays


def candidates():
    definitions = [
        ('unchanged_source', 'full', 'raw', 'absolute', 0.),
        ('full_actual_trace_absolute', 'full', 'raw', 'absolute', 1.),
        ('legs_actual_trace_absolute', 'legs', 'raw', 'absolute', 1.),
        ('arms_actual_trace_absolute', 'arms', 'raw', 'absolute', 1.),
        ('legs_lowpass_absolute', 'legs', 'ema025', 'absolute', 1.),
        ('legs_actual_trace_half', 'legs', 'raw', 'absolute', .5),
        ('legs_actual_trace_planned_phase', 'legs', 'raw', 'planned', 1.),
        ('legs_actual_trace_after_observed_air_hit', 'legs', 'raw', 'observed_hit', 1.),
        ('legs_lowpass_after_observed_air_hit', 'legs', 'ema025', 'observed_hit', 1.),
    ]
    return [dict(candidate_id=i, name=name, group=group, table=table, alignment=alignment, gain=gain)
            for i, (name, group, table, alignment, gain) in enumerate(definitions)]


def make_base(source, count, device):
    if not source.get('native_execution'):
        raise ValueError('Require an actually measured source controller')
    if source.get('schema') in ['g07-measured-swing-contact-recovery-v5', 'g07-public-early-lateral-intercept-v5']:
        path = HERE.parent / 'early_tangent_intercept_20261027/early_lateral_controller.py'
        spec = importlib.util.spec_from_file_location('preserved_public_lateral_factory', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if source['schema'] == 'g07-public-early-lateral-intercept-v5':
            if hashlib.sha256(path.read_bytes()).hexdigest() != source['factory_sha256']:
                raise ValueError('Measured early-lateral factory changed')
            return module.EarlyLateralController(source['source_public_policy'], count, device,
                source['early_lateral']['gain'], source['early_lateral']['start_clock_s'])
        return module.source_controller(source, count, device)
    return make_rally_controller(source, count, device)


class RecoveryPriorController:
    def __init__(self, source, seed, arrays, count, device, *, group, table, alignment, gain):
        if group not in ['full', 'legs', 'arms'] or table not in ['raw', 'ema025']:
            raise ValueError('Require one declared prior group/table')
        if alignment not in ['absolute', 'planned', 'observed_hit'] or gain not in [0., .5, 1.]:
            raise ValueError('Require a declared causal alignment and bounded gain')
        if source['asset_sha256'] != seed['asset_sha256']:
            raise ValueError('Recovery prior belongs to a different physical asset')
        if source['protocol_sha256'] not in [seed['protocol_sha256'], seed['source_protocol_sha256']]:
            raise ValueError('Recovery prior requires the admitted unchanged origin0 task')
        self.source = make_base(source, count, device)
        self.reference = self.source.reference_controller
        self.clock = torch.as_tensor(arrays['clock_s'], dtype=torch.float32, device=device)
        self.table = torch.as_tensor(arrays['residual_' + table], dtype=torch.float32, device=device)
        if self.clock.shape != (800,) or self.table.shape != (800, 18):
            raise ValueError('Require the actual800-row18-joint seed')
        if not torch.isfinite(self.table).all() or (abs(self.table) > 1).any() or not (self.clock[1:] > self.clock[:-1]).all():
            raise ValueError('Require finite bounded requests and a strictly increasing public clock')
        self.group, self.alignment, self.gain = group, alignment, float(gain)
        self.reference_event = float(seed['actual_reference_planned_event_time_s'])
        self.reference_public_hit = float(seed['actual_reference_first_public_air_hit_phase_s'])
        self.mixer = ResidualActionMixerV4(count, device, seed['residual_joint_scale_rad'])
        self.air_hit_seen = torch.zeros(count, dtype=torch.bool, device=device)
        self.first_public_hit_clock = torch.zeros(count, device=device)
        self.last_request = None

    @property
    def reference_controller(self):
        return self.reference

    @property
    def logs(self):
        return self.source.logs

    @logs.setter
    def logs(self, value):
        self.source.logs = value

    def reset(self, ids=None):
        self.source.reset(ids)
        self.mixer.reset(ids)
        if ids is None:
            self.air_hit_seen.zero_()
            self.first_public_hit_clock.zero_()
            self.last_request = None
        else:
            self.air_hit_seen[ids] = False
            self.first_public_hit_clock[ids] = 0.

    def sample(self, time):
        upper = torch.searchsorted(self.clock, time.contiguous()).clamp(0, len(self.clock) - 1)
        lower = (upper - 1).clamp_min(0)
        denominator = (self.clock[upper] - self.clock[lower]).clamp_min(1e-8)
        fraction = ((time - self.clock[lower]) / denominator).clamp(0., 1.)
        value = torch.lerp(self.table[lower], self.table[upper], fraction[:, None])
        # Preserve actual stored requests bitwise at the real control clocks.
        value = torch.where((time == self.clock[upper])[:, None], self.table[upper], value)
        supported = (time >= self.clock[0]) & (time <= self.clock[-1])
        return torch.where(supported[:, None], value, torch.zeros_like(value))

    @torch.inference_mode()
    def action(self, observation, default, limits):
        public = observation['policy']
        base = self.source.action(observation, default, limits)
        if self.gain == 0.:
            return base
        time = public[:, 775]
        # This is an observed phase, not a persistent hit bit. Remember the
        # first actual air/contact phase for this episode and keep it through
        # the later grounded phase without consulting hidden rally state.
        first = (public[:, 773] > .5) & (~self.air_hit_seen)
        self.first_public_hit_clock[first] = time[first]
        self.air_hit_seen |= first
        if self.alignment == 'planned':
            offset = torch.where(self.reference.planned,
                self.reference_event - self.reference.event_time, torch.zeros_like(time))
            query = time + offset
        elif self.alignment == 'observed_hit':
            query = time + (self.reference_public_hit - self.first_public_hit_clock)
        else:
            query = time
        request = self.sample(query) * self.gain
        if self.group == 'legs':
            request[:, 12:] = 0.
        elif self.group == 'arms':
            request[:, :12] = 0.
        if self.alignment == 'observed_hit':
            request = torch.where(self.air_hit_seen[:, None], request, torch.zeros_like(request))
        self.last_request = request.clone()
        result = self.mixer.action(base, request, public, default, limits)
        # An ablation of one joint group leaves the other source commands
        # bitwise intact, including the source's original arm rate limiter.
        if self.group == 'legs':
            result[:, 12:] = base[:, 12:]
        elif self.group == 'arms':
            result[:, :12] = base[:, :12]
        return result
