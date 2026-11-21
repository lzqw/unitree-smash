"""Mass-normalized tangential impulse at the predicted racket time of impact."""
import torch


def calibrated_tangent_dv(row, present, friction, sticking):
    normal = row['normal']
    relative = row['relative_velocity']
    tangent = relative - (relative*normal).sum(-1,keepdim=True)*normal
    bound = friction*row['impulse_dv'].norm(dim=-1,keepdim=True)
    scale = torch.minimum(torch.ones_like(bound)*sticking, bound/tangent.norm(dim=-1,keepdim=True).clamp_min(1e-12))
    active = present.reshape(-1).bool() & row['active'].reshape(-1).bool() & (row['object_id']==0)
    return -tangent*scale*active[:,None], active


class TOISlip:
    """Compose the calibrated analytic and supervised residual impulses before post-impact flight."""
    def __init__(self, friction, sticking, residual=None):
        self.friction=friction
        self.sticking=sticking
        self.residual=residual

    def toi_response(self, row, present):
        dv,active=calibrated_tangent_dv(row,present,self.friction,self.sticking)
        if self.residual is not None:
            dv=dv+self.residual.toi_response(row,present)['impulse_residual_dv']*active[:,None]
        return dict(impulse_residual_active=active,impulse_residual_dv=dv)
