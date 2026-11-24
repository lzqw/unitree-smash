"""Six-parameter native Approach search and its measured ball-side fitness."""
import math

ARM_BIAS = 'actor.approach.4.bias'
ARM_BIAS_LIMIT = .025


def validate_offsets(offsets, *, minimum=2, maximum=7):
    if not isinstance(offsets, list) or not minimum <= len(offsets) <= maximum:
        raise ValueError('Invalid number of native Approach search branches')
    for row in offsets:
        if len(row) != 6 or any(not math.isfinite(value) or abs(value) > ARM_BIAS_LIMIT + 1e-8 for value in row):
            raise ValueError('Approach search must use six bounded final-layer arm biases')
    if len({tuple(row) for row in offsets}) != len(offsets):
        raise ValueError('Duplicate native Approach offsets')
    return offsets


def actor_with_bias(state, offset):
    result = dict(state)
    bias = result[ARM_BIAS].clone()
    if bias.shape != (18,):
        raise ValueError('Approach final bias must follow the declared 18-joint contract')
    bias[-6:] += bias.new_tensor(offset)
    result[ARM_BIAS] = bias
    return result


def candidate_layout(cases, source_indices, offsets):
    if len(cases) != 128 or len(source_indices) != 32 or len(set(source_indices)) != 32:
        raise ValueError('Native search needs 32 distinct training cases')
    if any(type(i) is not int or not 0 <= i < 128 or i % 2 == 0 for i in source_indices):
        raise ValueError('Reserve the even TRAIN64 cases for the final paired evaluation')
    return [cases[i] for _ in offsets for i in source_indices]


def native_fitness(rows, court, signed_string_margin):
    """Rank complete returns first, then reward measured margins on hard cases."""
    if not rows or any(row['numerical_failures'] for row in rows):
        raise ValueError('Native fitness requires complete finite PhysX episodes')

    def margin(value, scale, missing=-1.):
        return max(-1., min(1., value / scale)) if value is not None and math.isfinite(value) else missing

    details = []
    low, high = court['target_x_limits_m']
    width = court['target_y_half_width_m']
    for row in rows:
        motion = row.get('first_approach_contact_motion') or {}
        string = bool(row['qualified_airborne_string_contact'])
        speed = row['outgoing_velocity_mps']
        outgoing = math.sqrt(sum(x*x for x in speed)) if row['released'] and all(map(math.isfinite, speed)) else None
        landing = row['first_landing_m']
        inside = (min(landing[0] - low, high - landing[0], width - abs(landing[1]))
                  if all(map(math.isfinite, landing)) else None)
        contact = signed_string_margin(motion) if motion else None
        speed_margin = None if outgoing is None else outgoing - 4.5
        net_margin = row['net_margin_m'] if row['crossed_net'] else None
        aggregate = (.30 * margin(contact, .025) + .15 * margin(speed_margin, .45)
                     + .25 * margin(net_margin, .08) + .30 * margin(inside, .30))
        details.append(dict(case_id=row['case_id'], success=bool(row['ball_side_complete_except_recovery']),
                            string=string, robot_net=bool(row['robot_net_fault']), frame=bool(row['frame_hit'] and not row['string_hit']),
                            low_speed=bool(string and speed_margin is not None and speed_margin < 0),
                            landed_target=bool(row['landed_target']), ball_net=bool(row['shuttle_net_contact']),
                            string_margin_m=contact, speed_margin_mps=speed_margin,
                            net_margin_m=net_margin if net_margin is not None and math.isfinite(net_margin) else None,
                            landing_margin_m=inside, margin_score=aggregate))
    return dict(success=sum(row['success'] for row in details),
                robot_net=sum(row['robot_net'] for row in details),
                qualified=sum(row['string'] for row in details),
                margin_score=sum(row['margin_score'] for row in details) / len(details), rows=details)


def rank_key(result):
    return result['success'], -result['robot_net'], result['qualified'], result['margin_score']
