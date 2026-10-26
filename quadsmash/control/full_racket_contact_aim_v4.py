"""Choose an interior string aiming point without changing physical geometry.

The measured source policy, its learned actors, joint commands, rate limiters,
and physical support guard remain intact. Only the analytic contact point and
its matching planning Jacobian are replaced in the instantiated controller.
The immutable source checkpoint remains embedded in every proposal.
"""
import numpy as np
import torch

from quadsmash.control.full_racket_body_residual_v4 import make_rally_controller as make_source
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4

SCHEMA = 'full-racket-frozen-source-contact-aim-v4'
SOURCE_SCHEMAS = ('full-racket-frozen-actors-body-search-v4', 'full-racket-body-residual-ppo-v4')


def aim_candidates():
    # The old upper-string reference is retained as candidate0. No physical
    # contact is relabelled by these algorithmic reference coordinates.
    return [(0., .10), (0., .08), (0., .06), (0., .04), (0., 0.),
            (-.015, .06), (.015, .06)]


class ContactAimControllerV4:
    def __init__(self, source, aim_uv_m, count, device):
        if source.get('schema') not in SOURCE_SCHEMAS or not source.get('native_execution'):
            raise ValueError('Require an actually executed measured body or body-residual source')
        model = FullRacketGeometryV4()
        uv = np.asarray(aim_uv_m, dtype=float)
        if uv.shape != (2,) or not np.isfinite(uv).all() or ((uv/model.half_extents[1:])**2).sum() > 1:
            raise ValueError('Choose a finite aiming point inside the unchanged strings')
        self.source = make_source(source, count, device)
        self.reference = self.source.reference_controller
        teacher = self.reference.teacher
        original_uv = teacher.contact_offset[1:].detach().cpu().numpy()
        self.aim_uv_m = uv.tolist()
        self.original_aim_uv_m = original_uv.tolist()
        if not np.array_equal(uv.astype(np.float32), original_uv):
            # x is the unchanged ball-center offset from the physical string
            # surface. Neither the racket asset nor the observed pose moves.
            teacher.contact_offset = teacher.contact_offset.clone()
            teacher.contact_offset[1:] = torch.as_tensor(uv, dtype=teacher.contact_offset.dtype, device=device)
            seed = self.reference.checkpoint['seed']
            self.reference.jacobian = model.contact_jacobian(np.asarray(seed['joint_position_rad']),
                uv=uv, face=seed.get('face_sign', 1), base_position=(0, 0, 0))
            j = self.reference.jacobian
            self.reference.inverse = j.T @ np.linalg.inv(j @ j.T + .04**2*np.eye(3))
        node = self.source
        while hasattr(node, 'actor') or hasattr(node, 'source'):
            if hasattr(node, 'actor'):
                for parameter in node.actor.parameters():
                    parameter.requires_grad_(False)
                node = node.base
            else:
                node = node.source

    @property
    def logs(self):
        return self.source.logs

    @logs.setter
    def logs(self, value):
        self.source.logs = value

    @property
    def reference_controller(self):
        return self.reference

    def reset(self, ids=None):
        self.source.reset(ids)

    @torch.inference_mode()
    def action(self, observation, default, limits):
        return self.source.action(observation, default, limits)


def make_rally_controller(checkpoint, count, device):
    if checkpoint.get('schema') != SCHEMA:
        return make_source(checkpoint, count, device)
    source = checkpoint['source_public_policy']
    if checkpoint['asset_sha256'] != source['asset_sha256']:
        raise ValueError('Contact aim proposal changed the source physical asset')
    if not torch.equal(checkpoint['parameters'], source['parameters']):
        raise ValueError('This family changes only the reference contact point')
    return ContactAimControllerV4(source, checkpoint['aim_uv_m'], count, device)
