"""Preserve unmodified normalized commands through position-target admission.

Only joints actually changed by the existing target admission are re-encoded.
This avoids adding a floating-point decode/encode perturbation to a no-op.
It does not change limits, determine admission, or bypass an accepted teacher.
"""
import numpy as np
import torch


def exact_admitted_action(original, requested_target, admitted_target, encoded_admitted):
    if original.shape != (18,) or not torch.is_floating_point(original):
        raise ValueError('One original floating-point 18D action required')
    requested,admitted,encoded=[np.asarray(x,dtype=np.float64) for x in
                               [requested_target,admitted_target,encoded_admitted]]
    if any(x.shape!=(18,) or not np.isfinite(x).all() for x in [requested,admitted,encoded]):
        raise ValueError('Finite original requested and admitted targets required')
    if not torch.isfinite(original).all() or torch.any(abs(original)>1.) or np.any(abs(encoded)>1.+1e-12):
        raise ValueError('Bounded original and encoded actions required')
    changed=admitted!=requested
    result=original.clone()
    if changed.any():
        mask=torch.as_tensor(changed,device=original.device)
        replacement=torch.as_tensor(encoded,device=original.device,dtype=original.dtype)
        result[mask]=replacement[mask]
    return result
