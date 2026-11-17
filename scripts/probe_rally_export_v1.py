"""Lossless first-episode join for small native contact-development panels.

The authoritative scorer requires the real JumpEnv.episode_records entry
in addition to FullRacketEnvV4.rally_records. Never fabricate a jump result,
infer it from a contact flag, or substitute a post-reset live state.
"""


def first_episode_probe_rows(rally_records, jump_records, cases, indices):
    count = len(cases)
    if not count or len(indices) != count or len(set(indices)) != count:
        raise ValueError('Distinct indexed first-episode cases required')

    first = {}
    for row in rally_records:
        lane = row.get('env_id')
        if row.get('episode_index') != 0:
            continue
        if not isinstance(lane, int) or not 0 <= lane < count or lane in first:
            raise ValueError('Invalid or repeated first-rally episode')
        first[lane] = row

    # JumpEnv stores rows in chronological completion order; unlike the rally
    # records, these rows do not have a native episode_index. As in the formal
    # evaluation entry, the first row for a lane is its first episode.
    jumps = {}
    for row in jump_records:
        lane = row.get('env_id')
        if not isinstance(lane, int) or not 0 <= lane < count:
            raise ValueError('Invalid jump episode lane')
        jumps.setdefault(lane, row)

    expected = set(range(count))
    if set(first) != expected or set(jumps) != expected:
        missing_rally = sorted(expected-set(first))
        missing_jump = sorted(expected-set(jumps))
        raise ValueError(
            f'Incomplete first-episode source: rally={missing_rally}, jump={missing_jump}')

    rows = []
    for lane in range(count):
        row = first[lane]
        if row.get('case_id') not in (None, cases[lane]['case_id']):
            raise ValueError('Rally case identity differs from frozen case')
        if row.get('ground_init') is not True:
            raise ValueError('Missing or false native ground initialization')
        rows.append(dict(
            row, jump=dict(jumps[lane]), case_id=cases[lane]['case_id'],
            source_training_index=indices[lane],
            source_case_index=indices[lane]))
    return rows
