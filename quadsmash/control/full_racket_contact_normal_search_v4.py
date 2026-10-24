"""Optional bounded face orientation target, never an assigned shuttle velocity."""
import torch

NORMAL_PARAMETER_NAMES = ['contact_normal_downward_pitch_bias_rad']
NORMAL_LOW = [-.12]
NORMAL_HIGH = [.12]


def extend_contact_normal_parameters(parameters):
    if parameters.ndim != 1 or len(parameters) not in (37, 38) or not torch.isfinite(parameters).all():
        raise ValueError('Require a finite37/38-vector')
    if len(parameters) == 38:
        if not NORMAL_LOW[0] <= float(parameters[37]) <= NORMAL_HIGH[0]:
            raise ValueError('Contact normal bias exceeds declared algorithm bounds')
        return parameters.clone()
    return torch.cat((parameters, parameters.new_zeros(1)))


def contact_normal_body(parameters, reference_world, forward, left, up):
    """Positive bias rotates world+X toward world-Z; only public axes enter."""
    if parameters.shape[1] <= 37:
        # Preserve the original operation order for every legacy controller.
        return reference_world[0]*forward + reference_world[1]*left + reference_world[2]*up
    if parameters.shape[1] != 38:
        raise ValueError('Unknown contact normal parameter layout')
    angle = parameters[:, 37:38]
    cosine, sine = torch.cos(angle), torch.sin(angle)
    x = cosine*reference_world[0] + sine*reference_world[2]
    z = -sine*reference_world[0] + cosine*reference_world[2]
    return x*forward + reference_world[1]*left + z*up
