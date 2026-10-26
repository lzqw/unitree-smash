"""TRAIN search proposals and ranking; original native acceptance stays separate."""
import itertools
import numpy as np
import torch

from quadsmash.evaluation_v4 import independent_episode
from .full_racket_target_landing_curriculum_v4 import norm_outgoing_shaping, target_landing_shaping

COORDINATES = (0, 1, 2, 3, 4, 24, 25, 37)
LOW = (.35, .45, 1.10, 1.20, .85, .80, -.12, -.12)
HIGH = (1., 1.20, 1.60, 1.80, 1.15, 1.45, .12, .12)
INITIAL_SIGMA = (.06, .07, .035, .035, .035, .06, .006, .015)


def initial_population(reference):
    if reference.shape != (38,) or not torch.isfinite(reference).all():
        raise ValueError('Require the measured38-coordinate reference')
    population = reference[None].repeat(10, 1)
    for i, (plane, waypoint, phase_advance) in enumerate(
            itertools.product((1.15, 1.25), (.80, .95), (0., .06)), 2):
        population[i, 24] = plane
        population[i, 1] = waypoint
        population[i, 2] = reference[2] - phase_advance
        # Preserve the recovery-handover clock relative to predicted contact
        # while testing earlier execution of the unchanged jump backbone.
        population[i, 3] = reference[3] + phase_advance * reference[4]
        population[i, 37] = .12
    validate_population(population, reference)
    return population


def validate_population(population, reference):
    if population.shape != (10, 38) or not torch.isfinite(population).all():
        raise ValueError('Require10 finite full parameter vectors')
    unchanged = [i for i in range(38) if i not in COORDINATES]
    if not torch.equal(population[:, unchanged], reference[unchanged][None].expand(10, -1)):
        raise ValueError('Search changed an undeclared coordinate')
    low = torch.tensor(LOW, device=population.device)
    high = torch.tensor(HIGH, device=population.device)
    if ((population[:, COORDINATES] < low) | (population[:, COORDINATES] > high)).any():
        raise ValueError('Candidate outside declared algorithm bounds')


def next_population(reference, population, ranked_indices, mean, sigma, generator):
    elite = population[ranked_indices[:2]][:, COORDINATES]
    mean = .3 * mean + .7 * elite.mean(0)
    floor = torch.tensor(INITIAL_SIGMA, device=population.device) * .2
    sigma = torch.maximum(.3 * sigma + .7 * elite.std(0, unbiased=False), floor)
    low, high = (torch.tensor(x, device=population.device) for x in (LOW, HIGH))
    result = reference[None].repeat(10, 1)
    sampled = mean[None] + sigma[None] * torch.randn((8, len(COORDINATES)),
        generator=generator, device=population.device)
    sampled[0] = mean
    result[2:, COORDINATES] = sampled.clamp(low, high)
    result[1] = population[ranked_indices[0]]
    validate_population(result, reference)
    return result, mean, sigma


def summarize_candidate(rows, protocol):
    checked = [independent_episode(row, protocol) for row in rows]
    attacks, chains, partial_scores = [], [], []
    for row, result in zip(rows, checked):
        velocity = np.asarray(row['outgoing_velocity_mps'])
        attack = bool(result['airborne_contact'] and row['released'] and
            np.isfinite(velocity).all() and np.linalg.norm(velocity) >= 8. and
            velocity[0] > 0 and velocity[2] <= -1.)
        valid_cross = bool(result['airborne_contact'] and row['crossed_net'] and not result['fault'])
        chain = bool(attack and valid_cross and row['landed_target'])
        # These terms guide TRAIN optimization only. A partial chain never
        # substitutes for the independently recomputed complete criterion.
        score = (4 * result['airborne_contact'] + 4 * result['conditional_recovery'] +
            8 * attack + 8 * valid_cross + 12 * (valid_cross and row['landed_target']) +
            .3 * norm_outgoing_shaping(row) +
            target_landing_shaping(row, (2., 4.), 1.) -
            2 * min(row['closest_string_gap_m'], .3) / .3)
        attacks.append(attack); chains.append(chain); partial_scores.append(float(score))
    summary = dict(episodes=len(rows), complete=sum(x['complete'] for x in checked),
        airborne_contacts=sum(x['airborne_contact'] for x in checked),
        conditional_recovery_count=sum(x['conditional_recovery'] for x in checked),
        attack_outgoing=sum(attacks), actual_attack_cross_target_chains=sum(chains),
        numerical_failures=sum(row['numerical_failures'] for row in rows),
        mean_TRAIN_partial_score=float(np.mean(partial_scores)), formal_acceptance=False)
    summary['rank'] = [int(summary['numerical_failures'] == 0), summary['complete'],
        summary['actual_attack_cross_target_chains'], round(summary['mean_TRAIN_partial_score'], 8),
        summary['conditional_recovery_count'], summary['airborne_contacts']]
    return summary
