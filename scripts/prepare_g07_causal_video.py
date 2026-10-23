#!/usr/bin/env python3
"""Prepare a measured fixed-scenario video without rewriting raw evaluation data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scenario', type=int, choices=[0, 1], required=True)
    p.add_argument('--geometry-reference', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Use a new output directory')
    manifest = json.loads((a.input/'eval_manifest.json').read_text())
    rows = json.loads((a.input/'episodes.json').read_text())
    traces = json.loads((a.input/'traces_fixed_scenarios_0_1.json').read_text())
    poses = np.load(a.input/f'link_poses_fixed_scenario_{a.scenario}.npy')
    names = json.loads((a.input/'body_names.json').read_text())
    if names != json.loads((a.geometry_reference/'body_names.json').read_text()):
        raise ValueError('Geometry link order differs from measured evaluation')
    hashes = json.loads((a.geometry_reference/'source_hashes.json').read_text())
    if hashes.get('assets/quadsmash_standard_v2.usd') != manifest['asset_sha256']:
        raise ValueError('Visual geometry asset differs from evaluation asset')
    trace = traces[a.scenario]
    if len(poses) != len(trace) or not len(trace):
        raise ValueError('Measured trace/pose mismatch')
    selected = [r for r in rows if r['scenario'] == a.scenario]
    if len(selected) != 1:
        raise ValueError('Missing or duplicate source scenario')
    free = lambda r: not r['robot_net_contact'] and not r['numerical_failures']
    metrics = dict(controller='learned hybrid with causal model prediction',
                   net_geometry_version=manifest['net_geometry_version'], episodes=len(rows),
                   complete=sum(r['complete_smash'] and free(r) for r in rows),
                   recovery=sum(r['recovery_success'] and free(r) for r in rows),
                   selected_video_lane=a.scenario, best=selected[0], formal_acceptance=False)
    scope = dict(kind='Derived presentation only; source evaluation untouched', input=str(a.input),
                 checkpoint_sha256=manifest['checkpoint_sha256'], scenario=a.scenario,
                 selection='One of two fixed recorded indices, all source episodes retained',
                 stats_scope='Entire source panel, not only the displayed episode',
                 source_episodes_sha256=digest(a.input/'episodes.json'),
                 source_poses_sha256=digest(a.input/f'link_poses_fixed_scenario_{a.scenario}.npy'),
                 asset_sha256=manifest['asset_sha256'])
    a.output.mkdir(parents=True)
    for name in ['geometry.json', 'geometry.npz']:
        shutil.copyfile(a.geometry_reference/name, a.output/name)
    np.save(a.output/'link_poses.npy', poses)
    (a.output/'trajectory.json').write_text(json.dumps(trace)+'\n')
    (a.output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    (a.output/'video_scope.json').write_text(json.dumps(scope, indent=2)+'\n')
    print(json.dumps(scope), flush=True)


if __name__ == '__main__':
    main()
