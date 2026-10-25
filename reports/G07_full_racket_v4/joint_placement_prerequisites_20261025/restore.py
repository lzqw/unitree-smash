"""Restore absent historical artifacts byte-for-byte; never alter old evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    assert source != ROOT and not (HERE / 'RESTORATION.json').exists()
    spec = importlib.util.spec_from_file_location('historical_verifier', ROOT / 'tools/verify_goal.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    before = verifier.verify(ROOT, 'G06')
    plan = {}
    evidence_hashes = {}
    for index in range(7):
        relative = f'reports/G{index:02d}/evidence.json'
        evidence_path = ROOT / relative
        assert sha(evidence_path) == sha(source / relative), 'Historical evidence differs'
        evidence_hashes[relative] = sha(evidence_path)
        for artifact in json.loads(evidence_path.read_bytes())['artifacts']:
            name = artifact['path']
            destination = verifier.inside(ROOT, name)
            if destination.exists():
                assert destination.is_file() and sha(destination) == artifact['sha256']
                continue
            assert not destination.is_symlink(), 'Do not replace a dangling user symlink'
            original = verifier.inside(source, name)
            assert original.is_file() and sha(original) == artifact['sha256']
            plan[name] = dict(path=name, source=str(original), sha256=artifact['sha256'],
                              bytes=original.stat().st_size)
    assert before['status'] == 'FAILED' and plan, 'Only restore actually missing prerequisites'
    for name, record in plan.items():
        destination = ROOT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f'.restore-{os.getpid()}.tmp')
        assert not destination.exists() and not temporary.exists()
        try:
            with temporary.open('xb') as stream, Path(record['source']).open('rb') as original:
                shutil.copyfileobj(original, stream)
            assert sha(temporary) == record['sha256']
            # An exclusive hard link makes publication atomic without replacing
            # any file another workspace owner may have created meanwhile.
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        assert not destination.is_symlink() and sha(destination) == record['sha256']
    after = verifier.verify(ROOT, 'G06')
    assert after['status'] == 'PASSED' and all(v['passed'] for v in after['checks'].values())
    assert all(sha(ROOT / name) == checksum for name, checksum in evidence_hashes.items())
    assert all(sha(Path(record['source'])) == record['sha256'] for record in plan.values())
    g07 = verifier.verify(ROOT, 'G07')
    assert not g07['checks']['G07']['passed'] and not (ROOT / 'reports/G07/evidence.json').exists()
    receipt = dict(restored_at=datetime.now(timezone.utc).isoformat(),
        scope='Local restoration of exact historical prerequisite bytes in an isolated candidate worktree. Old short-racket evidence is unchanged; no new policy performance or full-racket acceptance is inherited.',
        source_root=str(source), destination_root=str(ROOT), restored=list(plan.values()),
        total_restored_bytes=sum(record['bytes'] for record in plan.values()),
        original_evidence_sha256=evidence_hashes, script_sha256=sha(Path(__file__).resolve()),
        prior_historical_prerequisite_verification=before,
        restored_historical_prerequisite_verification=after,
        full_goal_structural_verification=g07,
        source_files_unchanged=True, historical_evidence_unchanged=True,
        original_main_workflow_edited=False, GPU_jobs_started=False,
        candidate_native_training_run=False, candidate_native_asset_composition_run=False,
        candidate_frozen_performance=None, formal_acceptance=False, goal_complete=False,
        activation='Review the actual original iter96 fixed assessment and deliver its videos/raw data first. Respect all live source owners and the single GPU0 guard; this restoration starts no successor job.')
    temporary = HERE / 'RESTORATION.json.tmp'
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    temporary.replace(HERE / 'RESTORATION.json')
    print(json.dumps(dict(restored_files=len(plan), restored_bytes=receipt['total_restored_bytes'],
        historical_G00_through_G06='PASSED', candidate_native_training_run=False,
        goal_complete=False, receipt=str(HERE / 'RESTORATION.json'))))


if __name__ == '__main__':
    main()
