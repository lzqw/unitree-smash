"""Declared TRAIN-only strike grid around the measured body candidate4.

The two measured body references preserve their actual parameters. Proposals
change only arm velocity gain, interception plane and desired normal bias.
Both learned residual actors, their mixers and the physical task stay intact.
"""
import itertools
import torch

CANDIDATE_COUNT = 16
COORDINATES = (7, 24, 37)


def population_and_labels(reference, comparison):
    for value in (reference, comparison):
        if value.shape != (38,) or not torch.isfinite(value).all():
            raise ValueError('Require two finite measured body parameter vectors')
    population = reference[None].repeat(CANDIDATE_COUNT, 1)
    population[1] = comparison
    labels = [dict(kind='exact_measured_candidate4'), dict(kind='exact_measured_candidate6')]
    settings = list(itertools.product((1., 1.08, 1.16), (.085, .12),
        (float(reference[7]), 1.3)))
    settings += [(1.24, .085, float(reference[7])), (1.24, .085, 1.3)]
    for index, (plane, normal, gain) in enumerate(settings, 2):
        population[index, 24] = plane
        population[index, 37] = normal
        population[index, 7] = gain
        labels.append(dict(kind='candidate4_strike_proposal', intercept_plane_x_m=plane,
            normal_down_bias_rad=normal, arm_velocity_gain=gain))
    fixed = [index for index in range(38) if index not in COORDINATES]
    assert torch.equal(population[0], reference) and torch.equal(population[1], comparison)
    assert torch.equal(population[2:, fixed], reference[fixed][None].expand(14, -1))
    assert ((population[:, 7] >= .3) & (population[:, 7] <= 1.5)).all()
    assert ((population[:, 24] >= .8) & (population[:, 24] <= 1.45)).all()
    assert (population[:, 37].abs() <= .1200001).all()
    assert population.shape == (CANDIDATE_COUNT, 38) and len(labels) == CANDIDATE_COUNT
    return population, labels
