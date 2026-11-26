"""Paired 32-case decision for moving the native feedback probe to larger training."""
import argparse
import json
import math
from pathlib import Path


def compare(rows):
    if len(rows) != 128 or len(set(r['case_id'] for r in rows[:32])) != 32:
        raise ValueError('Expected four branches over the same 32 training launches')
    scores = []
    nominal = rows[:32]
    for b, scale in enumerate((0., .5, 1., -.5)):
        part = rows[b*32:(b+1)*32]
        if any(x['case_id'] != y['case_id'] or y['numerical_failures'] for x,y in zip(nominal, part)):
            raise ValueError('Invalid probe case pairing or numerical failure')
        score = dict(scale=scale, qualified_body_ground=sum(
            r['body_ground_seen'] and r['qualified_airborne_string_contact'] for r in part))
        for k in ('body_ground_seen','physical_recovery_success','strict_physical_recovery',
                  'robot_net_fault','ball_side_complete_except_recovery','qualified_airborne_string_contact'):
            wins = sum(bool(y[k]) and not x[k] for x,y in zip(nominal,part))
            losses = sum(bool(x[k]) and not y[k] for x,y in zip(nominal,part))
            n = wins + losses
            p = sum(math.comb(n,j) for j in range(wins,n+1)) / 2**n
            score[k] = dict(count=sum(bool(r[k]) for r in part), gained=wins, lost=losses,
                            p_one_sided_increase=p)
        scores.append(score)
    passed = []
    for b,score in enumerate(scores[1:],1):
        strict = score['strict_physical_recovery']
        relaxed = score['physical_recovery_success']
        if (score['qualified_body_ground'] < scores[0]['qualified_body_ground']
                and score['body_ground_seen']['count'] <= scores[0]['body_ground_seen']['count']
                and strict['gained'] > strict['lost'] and relaxed['gained'] > relaxed['lost']
                and max(strict['p_one_sided_increase'],relaxed['p_one_sided_increase']) <= .05
                and score['robot_net_fault']['gained'] == 0
                and score['ball_side_complete_except_recovery']['lost'] == 0):
            passed.append(b)
    return dict(branches=scores, qualified_branches=passed, expand=bool(passed),
                paired_launches=32,
                criterion='Lower body-ground among qualified hits, no increase overall; strict and relaxed each improve with one-sided paired exact p <= 0.05; no new robot-net or lost ball-side success. Exploratory per-branch tests.')


def score_folder(folder):
    import torch
    metadata = json.loads((folder/'metrics.json').read_text())
    completion = json.loads((folder/'completed.json').read_text())
    if (completion['episodes'] != 128 or metadata['numerical_failures']
            or metadata['evaluation_scope'] != 'Recovery training teacher branches'):
        raise ValueError('Expected completed native training probes')
    tape = torch.load(folder/'teachers.pt', mmap=True, weights_only=True, map_location='cpu')
    if tape.get('feedback_version') != 2:
        raise ValueError('Pilot uses an incompatible feedback version')
    blocked = tape['gate'] & ~tape['arm_allowed']
    if not blocked.any() or tape['delta'][:, :, 12:][blocked].ne(0).any():
        raise ValueError('Recorded arm correction inside the four-tick release hold')
    result = compare(json.loads((folder/'episodes_summary.json').read_text()))
    result.update(control_ticks=metadata['control_ticks'], numerical_failures=metadata['numerical_failures'],
                  elapsed_s=metadata['elapsed_s'], recorded_held_lane_frames=int(blocked.sum()))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folder',type=Path)
    print(json.dumps(score_folder(p.parse_args().folder),indent=2))
