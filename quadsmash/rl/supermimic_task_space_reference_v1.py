"""Success-guided task-space references for the physical SuperMimic loop.

The bank is an auxiliary training signal only.  It never supplies future
events to the actor and is never used by native scoring.  References are
selected from a causal initial physical state, then sampled by elapsed time;
the stored trajectories remain ordinary native-success evidence.
"""
from dataclasses import dataclass
import hashlib
import torch

from .supermimic_physical_dataset_v1 import decode_physical


APPROACH_FEATURES = ('racket_position', 'racket_velocity', 'racket_normal',
                     'center', 'center_velocity', 'root_angular_velocity')
RECOVERY_FEATURES = ('center', 'center_velocity', 'root_angular_velocity',
                     'feet')


def task_space_features(state, geometry, mode):
    """Compute differentiable task-space quantities from a 55D physical state."""
    p = decode_physical(state)
    fk = geometry(p['joint_position'], p['joint_velocity'], p['root_position_w'],
                  p['root_quaternion_w'], p['root_link_velocity_w'],
                  p['root_angular_velocity_w'])
    racket_id = geometry.ids['Racket']
    racket_position = fk['position'][:, racket_id]
    racket_velocity = fk['velocity'][:, racket_id]
    # The racket local z axis is the declared face-normal convention in the
    # pinned geometry asset.  Normalize defensively at the singular boundary.
    racket_normal = fk['rotation'][:, racket_id, :, 2]
    racket_normal = racket_normal / racket_normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    common = dict(center=fk['center'], center_velocity=fk['center_velocity'],
                  root_angular_velocity=p['root_angular_velocity_w'])
    if mode == 'approach':
        common.update(racket_position=racket_position, racket_velocity=racket_velocity,
                      racket_normal=racket_normal)
        names = APPROACH_FEATURES
    elif mode == 'recovery':
        common['feet'] = fk['feet'].reshape(len(state), -1)
        names = RECOVERY_FEATURES
    else:
        raise ValueError(f'Unknown task-space reference mode: {mode}')
    return torch.cat([common[name] for name in names], dim=-1)


@dataclass
class TaskSpaceReferenceBankV1:
    """A compact bank of event-validated native reference trajectories."""
    approach_sequences: torch.Tensor
    recovery_sequences: torch.Tensor
    approach_keys: torch.Tensor
    recovery_keys: torch.Tensor
    elapsed: torch.Tensor
    geometry_sha256: str
    source_sha256: str = ''
    match_distance_threshold: float = float('inf')
    schema: str = 'success-guided-task-space-reference-v1'

    def to(self, device):
        return type(self)(*(x.to(device) if torch.is_tensor(x) else x for x in (
            self.approach_sequences, self.recovery_sequences, self.approach_keys,
            self.recovery_keys, self.elapsed)), self.geometry_sha256, self.source_sha256,
            self.match_distance_threshold)

    @property
    def manifest(self):
        return dict(schema=self.schema, geometry_sha256=self.geometry_sha256,
                    source_sha256=self.source_sha256,
                    approach_count=int(self.approach_sequences.shape[0]),
                    recovery_count=int(self.recovery_sequences.shape[0]),
                    horizon_steps=int(self.approach_sequences.shape[1]),
                    approach_feature_dim=int(self.approach_sequences.shape[-1]),
                    recovery_feature_dim=int(self.recovery_sequences.shape[-1]))

    @staticmethod
    def _key(state):
        p = decode_physical(state)
        return torch.cat((p['root_position_w'], p['root_link_velocity_w'],
                          p['ball_position_w'], p['ball_velocity_w']), -1)

    @classmethod
    def from_packed(cls, tensors, geometry, *, source_sha256='', horizon=None):
        """Build references from packed native data using terminal event flags.

        The archive contains cumulative event bits, so success selection is
        performed only at the terminal sample.  Faulted or body-ground lanes
        are excluded.  This does not invent labels for incomplete trajectories.
        """
        states = tensors['state']
        events = tensors['event_state']
        if states.ndim != 3 or events.ndim != 3:
            raise ValueError('Packed reference data must be [time,lane,*]')
        t = states.shape[0] if horizon is None else min(int(horizon), states.shape[0])
        states = states[:t]
        events = events[:t]
        # event indices follow physical-world-v2: qualified contact, cross,
        # target, recovery, body-ground, frame and fault.
        final = events[-1]
        approach_ok = (final[:, 1] > .5) & (final[:, 2] > .5) & (final[:, 3] > .5)
        approach_ok &= ~(final[:, 5] > .5) & ~(final[:, 6] > .5) & ~(final[:, 7] > .5)
        recovery_ok = (final[:, 4] > .5) & ~(final[:, 5] > .5) & ~(final[:, 7] > .5)
        # The bank stores physical states; geometry is applied lazily by the
        # caller when features are computed.  We use root/ball keys now.
        def sequence(mask, event_index):
            ids = mask.nonzero(as_tuple=False).flatten()
            if len(ids) == 0:
                return states.new_empty((0, t, 0)), states.new_empty((0, 13))
            seq = states[:, ids].permute(1, 0, 2).contiguous()
            event = events[:, ids, event_index] > .5
            first = event.float().argmax(0)
            key = cls._key(seq[torch.arange(len(ids)), first])
            return seq, key
        approach, approach_keys = sequence(approach_ok, 1)
        recovery, recovery_keys = sequence(recovery_ok, 4)
        # Feature tensors are materialized by bind() after the geometry is
        # available.  Keep the physical sequence schema explicit in the bank.
        elapsed = tensors['elapsed'][:t, 0].detach().clone()
        geometry_sha = getattr(geometry, 'model_sha256', '')
        return cls(approach, recovery, approach_keys, recovery_keys, elapsed,
                   geometry_sha, source_sha256)

    def _select(self, initial_physical, mode):
        keys = self.approach_keys if mode == 'approach' else self.recovery_keys
        sequences = self.approach_sequences if mode == 'approach' else self.recovery_sequences
        if sequences.shape[0] == 0:
            return None
        distance = (self._key(initial_physical)[:, None] - keys[None]).square().mean(-1)
        return distance.argmin(-1)

    def bind(self, initial_physical, mode):
        """Return an immutable reference id for each imagination lane."""
        keys = self.approach_keys if mode == 'approach' else self.recovery_keys
        sequences = self.approach_sequences if mode == 'approach' else self.recovery_sequences
        if sequences.shape[0] == 0:
            return None
        distance=(self._key(initial_physical)[:, None] - keys[None]).square().mean(-1)
        best=distance.argmin(-1)
        threshold=float(self.match_distance_threshold)
        matched=torch.isfinite(distance.gather(1,best[:,None]).squeeze(1))
        if threshold < float('inf'):
            matched &= distance.gather(1,best[:,None]).squeeze(1) <= threshold
        return dict(indices=best.detach(), matched=matched.detach(), mode=mode)

    def loss(self, predicted_physical, elapsed, reference, mode, geometry=None, valid_mask=None):
        """Return task-space loss; geometry is required for live features."""
        if reference is None:
            return predicted_physical.new_zeros(())
        if geometry is None:
            raise ValueError('Task-space tracking requires the verified geometry module')
        sequences = self.approach_sequences if mode == 'approach' else self.recovery_sequences
        if sequences.shape[-1] != 55:
            # Sequences are physical states until lazily transformed by the
            # geometry-aware helper below.
            raise ValueError('Reference bank physical sequence schema is corrupted')
        indices = reference['indices'].to(predicted_physical.device)
        selected = sequences.to(predicted_physical.device)[indices]
        # Use one target time per lane, with linear interpolation between the
        # same native 100Hz samples.  Selection is detached; predicted feature
        # computation remains differentiable.
        time = elapsed.flatten().clamp(float(self.elapsed[0]), float(self.elapsed[-1]))
        position = time / max(float(self.elapsed[1] - self.elapsed[0]), 1e-6)
        lo = position.floor().long().clamp(0, selected.shape[1] - 1)
        hi = (lo + 1).clamp(0, selected.shape[1] - 1)
        alpha = (position - lo.float()).unsqueeze(-1)
        target_state = selected[torch.arange(len(selected), device=selected.device), lo] * (1-alpha) \
            + selected[torch.arange(len(selected), device=selected.device), hi] * alpha
        target = task_space_features(target_state, geometry, mode).detach()
        actual = task_space_features(predicted_physical, geometry, mode)
        mask=reference.get('matched',torch.ones(len(actual),dtype=torch.bool,device=actual.device))
        if valid_mask is not None: mask &= valid_mask.to(mask.device)
        if not mask.any(): return actual.new_zeros(())
        actual=actual[mask]; target=target[mask]
        scale = actual.new_tensor(1.).clamp_min(1e-6)
        return (actual-target).square().mean() / scale


def sha256_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
