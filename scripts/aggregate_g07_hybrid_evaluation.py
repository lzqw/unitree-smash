#!/usr/bin/env python3
"""Recompute both frozen-panel outcomes; never certify a nominal curriculum."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--policy-lock', type=Path, required=True)
    parser.add_argument('--panel-output', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol, lock = read(args.protocol), read(args.policy_lock)
    from quadsmash.physics.net_geometry import PROFILE_VERSION
    if protocol.get('net_geometry_version') != PROFILE_VERSION:
        raise ValueError('Flat-net curriculum cannot satisfy the original side-height contract')
    if lock['protocol_sha256'] != digest(args.protocol):
        raise ValueError('Protocol does not match immutable policy lock')
    if len(args.panel_output) != len(protocol['panels']):
        raise ValueError('Both frozen panels are required')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new aggregate directory')
    all_rows, summaries, jobs, seen = [], [], [], set()
    for directory in args.panel_output:
        manifest, supplied, job = read(directory/'eval_manifest.json'), read(directory/'metrics.json'), read(directory/'job.json')
        if not manifest.get('formal_evaluation_eligible') or job['status'] != 'SUCCEEDED':
            raise ValueError('A curriculum, failed or unlocked run cannot satisfy this evaluation')
        if (manifest['protocol_sha256'] != digest(args.protocol)
                or manifest['policy_lock_sha256'] != digest(args.policy_lock)
                or manifest['checkpoint_sha256'] != lock['policy_sha256']):
            raise ValueError('Panel policy/protocol/implementation lock mismatch')
        references = [p for p in protocol['panels'] if p['sha256'] == manifest['launch_panel_sha256']]
        if len(references) != 1 or references[0]['id'] in seen:
            raise ValueError('Unknown or duplicate panel')
        reference = references[0]
        seen.add(reference['id'])
        rows = read(directory/'episodes.json')
        receipt = read(directory/'completed.json')
        if (len(rows) != reference['episodes'] or receipt['episodes'] != len(rows)
                or set(r['scenario'] for r in rows) != set(range(len(rows)))):
            raise ValueError('Missing, repeated or discarded evaluation scenarios')
        count = len(rows)
        air = sum(r['airborne_contact'] for r in rows)
        rec = sum(r['airborne_contact'] and r['recovery_success'] and not r['robot_net_contact'] for r in rows)
        full = sum(r['complete_smash'] and not r['robot_net_contact'] and r['ground_init'] for r in rows)
        summary = dict(panel=reference['id'], sample_count=count,
                       ground_init_fraction=sum(r['ground_init'] for r in rows)/count,
                       airborne_contact_rate=air/count,
                       recovery_given_airborne_contact=rec/air if air else 0.,
                       complete_smash_success_rate=full/count,
                       numerical_failures=sum(r['numerical_failures'] for r in rows))
        for key in ['ground_init_fraction', 'airborne_contact_rate', 'recovery_given_airborne_contact',
                    'complete_smash_success_rate', 'numerical_failures']:
            if abs(summary[key]-supplied[key]) > 1e-12:
                raise ValueError('Reported metric differs from raw episodes: '+key)
        summaries.append(summary)
        jobs.append(job)
        all_rows.extend(dict(r, frozen_panel_id=reference['id']) for r in rows)
    metrics = dict(sample_count=sum(p['sample_count'] for p in summaries),
                   numerical_failures=sum(p['numerical_failures'] for p in summaries),
                   checkpoint_sha256=lock['policy_sha256'], protocol=protocol['version'],
                   panel_metrics=summaries, aggregation='minimum panel fraction; sum failures')
    for key in ['ground_init_fraction', 'airborne_contact_rate', 'recovery_given_airborne_contact',
                'complete_smash_success_rate']:
        metrics[key] = min(p[key] for p in summaries)
    contract = read(ROOT/'configs/acceptance.json')
    if digest(ROOT/'configs/acceptance.json') != protocol['acceptance_contract_sha256']:
        raise ValueError('Original acceptance contract changed')
    checks = contract['goals']['G07']['checks']
    passed = all(metrics[key] >= value['value'] if value['op'] == 'ge' else
                 metrics[key] == value['value'] for key, value in checks.items())
    metrics['thresholds_met'] = passed
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    (args.output/'episodes.json').write_text(json.dumps(all_rows, indent=2)+'\n')
    with (args.output/'episodes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    provenance = dict(protocol_path=str(args.protocol), protocol_sha256=digest(args.protocol),
                      policy_lock_sha256=digest(args.policy_lock), panel_outputs=[str(p) for p in args.panel_output],
                      checkpoint_sha256=lock['policy_sha256'], source_lock=lock,
                      run=dict(started_at=min(j['started_at'] for j in jobs), finished_at=max(j['finished_at'] for j in jobs),
                               gpu_hours=sum(j['charged_gpu_hours'] for j in jobs),
                               env_steps=sum(j['charged_env_steps'] for j in jobs)), jobs=jobs)
    (args.output/'eval_manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(metrics), flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
