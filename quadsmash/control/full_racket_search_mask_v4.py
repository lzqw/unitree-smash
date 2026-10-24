"""Optional block-coordinate sampling; task/controller execution is unchanged."""
import torch


def search_coordinate_mask(width, indices):
    if indices is None:
        return torch.ones(width, dtype=torch.bool)
    if (not indices or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < width for i in indices)):
        raise ValueError('Require distinct search coordinate indices within the parameter layout')
    mask = torch.zeros(width, dtype=torch.bool)
    mask[indices] = True
    return mask


def project_reference_coordinates(values, reference, mask):
    if (reference.ndim != 1 or mask.shape != reference.shape or mask.dtype != torch.bool
            or values.shape[-1] != len(reference)):
        raise ValueError('Search values/reference/mask layout mismatch')
    return torch.where(mask, values, reference)
