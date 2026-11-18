#!/usr/bin/env python3
"""Verify an indexed native archive and build holdout-free dual-head references.

SQLite is an evidence index, never a substitute for raw bytes or terminal
scores. Other archives are not imported implicitly. No Isaac or optimizer.
"""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
import torch
from quadsmash.rl.supermimic_data_split_v1 import explicit_lane_split
from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1, MODEL
from quadsmash.rl.supermimic_policy_execution_v1 import contract
from quadsmash.rl.supermimic_task_space_reference_v1 import TaskSpaceReferenceBankV1


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def indexed_terminal_records(connection, archive, records):
    indexed = set()
    query = ('SELECT e.record_json FROM episodes e JOIN occurrences o ON e.id=o.episode '
             'WHERE o.archive=? AND o.identity_known=1 AND o.raw_available=1 '
             "AND e.scoring_schema='historical-independent-complete5-v1'")
    canonical = lambda record: json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)
    for (raw,) in connection.execute(query, (str(archive),)):
        indexed.add(canonical(json.loads(raw)))
    identities = set()
    for record in records:
        identity = (record['env_id'], record['episode_index'])
        if identity in identities:
            raise ValueError('Duplicate scored episode identity')
        identities.add(identity)
        if record.get('scoring_schema') != 'historical-independent-complete5-v1' or canonical(record) not in indexed:
            raise ValueError('Terminal score not independently indexed for this archive')
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('database', 'archive', 'packed-data', 'episodes-summary', 'split-manifest', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--contract-root', type=Path,
                   help='Root containing the exact asset/config files named by the archive contract. '
                        'Defaults to this checkout and never searches other worktrees implicitly.')
    p.add_argument('--match-distance-threshold', type=float, default=1.0)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    torch.set_num_threads(2)
    archive = a.archive.resolve()
    manifest_path = archive / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    with sqlite3.connect(a.database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        registered = connection.execute('SELECT manifest_sha256 FROM archives WHERE path=?', (str(archive),)).fetchone()
        if not registered or registered[0] != sha(manifest_path):
            raise ValueError('Raw archive differs from indexed manifest')
        records = indexed_terminal_records(connection, archive, json.loads(a.episodes_summary.read_text()))
    if manifest.get('policy_execution') != contract():
        raise ValueError('Unknown or incompatible policy execution contract')
    if manifest.get('num_envs') != 128 or manifest.get('control_ticks') != 800 or manifest.get('pending_transition'):
        raise ValueError('Expected complete original128 frozen native archive')
    task = manifest['task_contract']
    if task.get('control_dt_s') != .01 or task.get('episode_s') != 8.:
        raise ValueError('Task timebase differs')
    repo = (a.contract_root or Path(__file__).resolve().parents[1]).resolve()
    def verify_file(path, expected):
        path = Path(path)
        if not path.is_absolute():
            path = repo / path
        if sha(path) != expected:
            raise ValueError('Contract file hash mismatch: ' + str(path))
    for key in ('asset', 'suite'):
        verify_file(task[key]['path'], task[key]['sha256'])
    verify_file(task['protocol_path'], task['protocol_sha256'])
    chunks = []
    for item in manifest['chunks']:
        path = (archive / item['file']).resolve()
        if path.parent != archive or sha(path) != item['sha256']:
            raise ValueError('Raw chunk path/hash mismatch: ' + str(path))
        chunks.append(dict(file=item['file'], sha256=item['sha256'], bytes=path.stat().st_size))
    packed_sha = sha(a.packed_data)
    receipt = json.loads((a.packed_data.parent / 'completed.json').read_text())
    if receipt['sha256'] != packed_sha:
        raise ValueError('Packed data receipt hash mismatch')
    package = torch.load(a.packed_data, map_location='cpu', weights_only=True)
    if package.get('source_manifest') != manifest or not bool(package['first_done'].all()):
        raise ValueError('Packed data identity/completeness differs from raw archive')
    _, heldout = explicit_lane_split(json.loads(a.split_manifest.read_text()), 128)
    geometry = FullBodyGeometryV1(json.loads(MODEL.read_text())['asset_sha256'])
    with torch.no_grad():
        bank = TaskSpaceReferenceBankV1.from_packed(package['tensors'], geometry, source_sha256=packed_sha,
            policy_id=manifest['policy_id'], terminal_records=records, excluded_lanes=heldout.tolist(),
            match_distance_threshold=a.match_distance_threshold)
    if not bank.manifest['protocol_verified'] or not all(bank.manifest[h + '_count'] for h in ('recovery', 'approach')):
        raise ValueError('Insufficient verified references for both heads; do not enable guided training')
    destination = a.output / 'reference_bank.pt'
    torch.save(bank.payload(), destination)
    audit = dict(bank.manifest, database=str(a.database), archive=str(archive), packed_sha256=packed_sha,
                 manifest_sha256=sha(manifest_path), episodes_summary_sha256=sha(a.episodes_summary),
                 split_sha256=sha(a.split_manifest), bank_sha256=sha(destination), verified_chunks=chunks,
                 policy_execution=manifest['policy_execution'], task_contract=task,
                 entries=[{k: e[k] for k in ('mode', 'lane_id', 'episode_id', 'policy_id', 'digest')} for e in bank.entries],
                 actor_updates=0, native_steps=0, tracking_gradient_validated=False,
                 scope='One source contract verified; other indexed archives remain excluded until independently verified')
    (a.output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(dict(bank.manifest, output=str(destination), bank_sha256=audit['bank_sha256'])), flush=True)


if __name__ == '__main__':
    main()
