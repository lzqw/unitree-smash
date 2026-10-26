"""Joint timing/recovery adaptation over intact actors and preparation guards."""
import torch

from .full_racket_residual_gain_v4 import ResidualGainControllerV4, validate_gains
from .full_racket_body_search_v4 import COORDINATES, LOW, HIGH

SCHEMA = 'full-racket-frozen-source-gain-body-adaptation-v4'


def body_candidates(source):
    phase_scale = float(source['parameters'][4])
    definitions = [
        ('unchanged_source', [[1., 1.], [1., 1.]], {}),
        ('both_residual_layers_off', [[0., 0.], [0., 0.]], {}),
        ('jump_40ms_earlier_keep_handover', [[1., 1.], [1., 1.]], {2: -.04, 3: .04*phase_scale}),
        ('jump_40ms_later_keep_handover', [[1., 1.], [1., 1.]], {2: .04, 3: -.04*phase_scale}),
        ('recovery_handover_phase_plus015', [[1., 1.], [1., 1.]], {3: .15}),
        ('recovery_handover_phase_minus015', [[1., 1.], [1., 1.]], {3: -.15}),
        ('walk_plus010_jump_earlier', [[1., 1.], [1., 1.]], {1: .10, 2: -.04, 3: .04*phase_scale}),
        ('walk_plus010_jump_later', [[1., 1.], [1., 1.]], {1: .10, 2: .04, 3: -.04*phase_scale}),
    ]
    return [dict(candidate_id=i, name=name, gains=validate_gains(gains),
                 body_deltas={str(key):value for key, value in delta.items()})
            for i, (name, gains, delta) in enumerate(definitions)]


class GainBodyAdaptationControllerV4(ResidualGainControllerV4):
    def __init__(self, source, gains, count, device, body_deltas=None):
        super().__init__(source, gains, count, device)
        self.body_deltas = {} if body_deltas is None else dict(body_deltas)
        if set(self.body_deltas)-{str(key) for key in COORDINATES}:
            raise ValueError('Body adaptation may change only original algorithm search coordinates')
        self.runtime_parameters = self.reference.parameters.detach().clone()
        for key, value in self.body_deltas.items():
            delta = torch.as_tensor(value, dtype=torch.float64)
            if delta.shape or not torch.isfinite(delta):
                raise ValueError('Require one finite shared body delta per coordinate')
            self.runtime_parameters[:, int(key)] += float(value)
        for index, low, high in zip(COORDINATES, LOW, HIGH):
            values = self.runtime_parameters[:, index]
            if ((values < low) | (values > high)).any():
                raise ValueError('Body adaptation outside original algorithm bounds')
        self.reference.parameters.copy_(self.runtime_parameters)

    def reset(self, ids=None):
        super().reset(ids)
        if ids is None:
            self.reference.parameters.copy_(self.runtime_parameters)
        else:
            self.reference.parameters[ids] = self.runtime_parameters[ids]


def make_rally_controller(checkpoint, count, device):
    if checkpoint.get('schema') != SCHEMA:
        from .full_racket_residual_gain_v4 import make_rally_controller as previous
        return previous(checkpoint, count, device)
    source = checkpoint['source_public_policy']
    if checkpoint['asset_sha256'] != source['asset_sha256'] or not torch.equal(checkpoint['parameters'], source['parameters']):
        raise ValueError('Gain/body policy changed the immutable embedded source')
    return GainBodyAdaptationControllerV4(source, checkpoint['residual_gains_outer_to_inner'], count, device,
                                         checkpoint['runtime_body_deltas'])
