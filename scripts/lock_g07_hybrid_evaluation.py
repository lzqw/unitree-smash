#!/usr/bin/env python3
"""Freeze one policy and its implementation before either 200-case panel."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Evaluation locks are immutable; use a new version')
    protocol = json.loads(args.protocol.read_text())
    for panel in protocol['panels']:
        if digest(ROOT/panel['path']) != panel['sha256']:
            raise ValueError('Frozen panel hash mismatch')
    sources = sorted(ROOT.glob('quadsmash/**/*.py'))
    sources += [ROOT/'scripts/evaluate_g07_causal_hybrid.py', ROOT/'configs/g07_go2_walk_transfer.json',
                ROOT/'configs/shuttle_physics_v2.json', ROOT/'configs/evaluation_protocol_v2.json']
    lock = dict(created_at=datetime.now(timezone.utc).isoformat(),
                policy_path=str(args.checkpoint.resolve().relative_to(ROOT)), policy_sha256=digest(args.checkpoint),
                protocol_path=str(args.protocol.resolve().relative_to(ROOT)), protocol_sha256=digest(args.protocol),
                git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2)+'\n')
    print(json.dumps(dict(path=str(args.output), policy_sha256=lock['policy_sha256'],
                         protocol_sha256=lock['protocol_sha256'])), flush=True)


if __name__ == '__main__':
    main()
