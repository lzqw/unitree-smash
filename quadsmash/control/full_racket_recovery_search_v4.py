"""Optional policy coordinates for the existing standing recovery controller."""
import torch

RECOVERY_PARAMETER_NAMES = ['recovery_thigh_bias_rad', 'recovery_front_back_thigh_bias_rad',
    'recovery_calf_bias_rad', 'recovery_roll_feedback_gain', 'recovery_actor_gain']
# These bound searched leg-target policy parameters, never native actuators or
# the qualified-jump/recovery-success criterion.
RECOVERY_LOW = [.05, -.15, -.45, -1., .20]
RECOVERY_HIGH = [.45, .15, .05, 0., 1.20]


def extend_recovery_parameters(parameters, legacy_recovery):
    if parameters.ndim != 1 or len(parameters) not in (32, 37, 38) or not torch.isfinite(parameters).all():
        raise ValueError('Require a finite32/37/38-vector')
    recovery = parameters[32:37] if len(parameters) >= 37 else torch.as_tensor(
        legacy_recovery, device=parameters.device, dtype=parameters.dtype)
    lower = parameters.new_tensor(RECOVERY_LOW)
    upper = parameters.new_tensor(RECOVERY_HIGH)
    if (recovery.shape != (5,) or not torch.isfinite(recovery).all()
            or (recovery < lower).any() or (recovery > upper).any()):
        raise ValueError('Recovery policy parameters exceed declared algorithm bounds')
    return parameters.clone() if len(parameters) >= 37 else torch.cat((parameters, recovery))


def recovery_parameters(parameters, legacy_recovery):
    if parameters.shape[1] in (37, 38):
        return parameters[:, 32:37]
    if parameters.shape[1] <= 32:
        return legacy_recovery[None].expand(len(parameters), -1)
    raise ValueError('Unknown recovery parameter layout')
