"""Package real intermediate16/24 inference evidence without touching live owners."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT/'reports/G07_full_racket_v4'
QUEUE = REPORTS/'centered_placement075_intermediate_selection_queue01'
RUN = REPORTS/'residual_ppo_centered_placement075_n2048_resume32_02'
RSL = Path('/home/autodrive/ZCY/UNITREE/rsl_rl')
ARCHIVE = ROOT/'reports/G07_placement075_native_intermediate16_24_CPU_export_evidence_20261026.zip'


def read(path):
    return json.loads(path.read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    assert not ARCHIVE.exists() and not (HERE/'ARCHIVE_CHECK.json').exists()
    for pid, needle in [(1734680, RUN.name), (1744903, QUEUE.name+'/run.py')]:
        assert needle.encode() in Path('/proc', str(pid), 'cmdline').read_bytes()
    plan = read(QUEUE/'PREDECLARED_PLAN.json')
    assert all(digest((ROOT/n).read_bytes()) == h for n, h in plan['source_hashes'].items())
    assert digest((QUEUE/'PREDECLARED_PLAN.json').read_bytes()) == read(QUEUE/'DETACHED_LAUNCH_RECEIPT.json')['plan_sha256']
    paths = set()
    for iteration in [16, 24]:
        receipt = read(QUEUE/f'exports/iter{iteration}/EXPORT_CHECK.json')
        assert receipt['actor_export_max_error'] == 0. and receipt['actual_public_samples'] == 1366
        for name, expected in receipt['source_sha256'].items():
            path = ROOT/name
            assert digest(path.read_bytes()) == expected
            paths.add(path)
        paths.update([QUEUE/f'exports/iter{iteration}/policy.pt', QUEUE/f'exports/iter{iteration}/EXPORT_CHECK.json'])
    paths.update([Path(__file__), HERE/'check_restored.py', RUN/'source_hashes.json', RUN/'source_snapshot.zip',
        QUEUE/'exports/iter24/CPU_EXPORT_DURING_VERIFIED_TRAINING.json'])
    tracked = subprocess.check_output(['git', 'ls-files', 'quadsmash'], cwd=ROOT, text=True).splitlines()
    paths.update(ROOT/n for n in tracked if n.endswith('.py'))
    payloads = {str(p.relative_to(ROOT)): p.read_bytes() for p in sorted(paths)}
    # Immutable copy of a live publication, not a final training receipt.
    progress = (RUN/'progress.json').read_bytes()
    assert read(RUN/'progress.json')['update'] > 24
    payloads[str((RUN/'progress.json').relative_to(ROOT))] = progress
    for path in [*sorted((RSL/'rsl_rl').rglob('*.py')), RSL/'LICENSE', RSL/'setup.py', RSL/'pyproject.toml']:
        payloads['vendor/rsl_rl/'+str(path.relative_to(RSL))] = path.read_bytes()
    payloads['vendor/rsl_rl/LOCAL_DIFF.patch'] = subprocess.check_output(['git', 'diff', '--binary', 'HEAD', '--', 'rsl_rl', 'setup.py'], cwd=RSL)
    payloads['README_CPU_EXPORT_EVIDENCE.txt'] = (
        'Actual placement075 native16/24 intermediate inference policies and CPU fidelity evidence.\n'
        'Contains diagnostic states, source canonical8/public8, fixed base846, 1366 actual saved public inputs,\n'
        'current export code, project Python sources and exact locally used RSL sources plus BSD license.\n'
        '16/24 diagnostic model_i files are NOT canonical resume checkpoints. No frozen16/24 score exists yet.\n'
        'The native training job remains active. Captured progress is not a final receipt or population score.\n'
        'This archive is self-contained DATA for the CPU source/weight check; Python/Torch/numpy/scipy/tensordict remain external software.\n'
        'It does not include all USD/STL assets or prove a fresh native physics replay. The previously released293MB stage8 bundle supplies native model assets.\n'
        'After extraction into a fresh directory, from that directory run:\n'
        "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python reports/G07_full_racket_v4/placement075_intermediate_CPU_delivery_20261026/check_restored.py --root . --output CPU_RESTORE_CHECK.json\n"
        'Full objective remains original A/B200 each with worse-panel50/70/85 and NF0, currently unproven.\n'
    ).encode()
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), archive=ARCHIVE.name,
        project_git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        original_native_launch_git_sha='6f69aaf9eedfe0e94d4e98abe56097b4e0b77114',
        RSL_base_git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=RSL, text=True).strip(),
        actual_RSL_source_and_local_diff_included=True,
        software_versions={n: importlib.metadata.version(n) for n in ['torch', 'numpy', 'scipy', 'tensordict']},
        actual_captured_progress=json.loads(progress), complete_native_job=False,
        scope='Standalone CPU data/export evidence, not complete native training or simulation assets or frozen performance.',
        payloads={n: dict(bytes=len(b), sha256=digest(b)) for n, b in payloads.items()},
        formal_cases_read=False, new_native_execution=False, formal_acceptance=False, goal_complete=False)
    with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        archive.writestr('CPU_EXPORT_MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(payloads)+1 and len(set(archive.namelist())) == len(archive.namelist())
        for name, data in payloads.items():
            assert digest(archive.read(name)) == digest(data)
    checked = dict(checked_at=datetime.now(timezone.utc).isoformat(), archive=str(ARCHIVE),
        bytes=ARCHIVE.stat().st_size, sha256=digest(ARCHIVE.read_bytes()), payloads_checked=len(payloads),
        all_members_CRC_and_payload_SHA_passed=True, actual_native_still_running_at_capture=True,
        no_claim_of_complete_native_job_or_formal_performance=True, source_archive_manifest=manifest,
        formal_acceptance=False, goal_complete=False)
    (HERE/'ARCHIVE_CHECK.json').write_text(json.dumps(checked, indent=2)+'\n')
    print(json.dumps({k: checked[k] for k in ['archive', 'bytes', 'sha256', 'payloads_checked']}))


if __name__ == '__main__':
    main()
