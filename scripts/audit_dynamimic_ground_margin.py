"""Compare the imagined ground penalty with native first faults and stability milestones."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1, MODEL
from scripts.fit_dynamimic_recovery_wm import read_data


def audit(native):
    data, manifest = read_data(native, ('active', 'strict', 'relaxed'))
    geometry = FullBodyGeometryV1(json.loads(MODEL.read_text())['asset_sha256'])
    geometry.enable_depth_batches()
    sites = (geometry.root_id, *geometry.heads,
             *(geometry.ids[name + '_hip'] for name in ('FL', 'FR', 'RL', 'RR')))
    report = {}
    for event in ('body', 'strict', 'relaxed'):
        eligible = data[event] & data['active'] & data['live']
        lanes = eligible.any(0).nonzero().flatten()
        ticks = eligible.int().argmax(0)[lanes]
        state = data['after'][ticks, lanes]
        positions = geometry(state[:, 13:31], state[:, 31:49], state[:, :3], state[:, 3:7],
                             state[:, 7:10], state[:, 10:13])['position']
        minimum = positions[:, sites, 2].amin(1)
        for split, mask in [('train', lanes % 16 >= 2), ('held', lanes % 16 < 2)]:
            heights = minimum[mask]
            if not len(heights):
                continue
            report[event + '_' + split] = dict(cases=len(heights), penalty_active=int((heights < .14).sum()),
                median_link_height_m=float(heights.median()), median_root_height_m=float(state[mask, 2].median()),
                mean_ground_penalty=float(20 * (.14 - heights).relu().square().mean()))
    return dict(behavior_sha256=manifest['checkpoint_sha256'], companion_sha256=manifest['companion_sha256'],
                sample='first active native tick per case and event; milestone and later fault may overlap',
                events=report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    args = parser.parse_args()
    with torch.no_grad():
        print(json.dumps(audit(args.native), indent=2))
