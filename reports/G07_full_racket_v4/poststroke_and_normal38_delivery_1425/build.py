"""Package immutable CEM20 results and separately identified CEM21 initialization."""
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
REPORTS = ROOT / 'reports/G07_full_racket_v4'
OLD = REPORTS / 'ball_post_stroke_retreat37_cem20'
NEW = REPORTS / 'ball_attack_recovered_normal38_cem21'
NAME = 'G07_attacking_recovery_two_generations_and_normal38_native_20261024_1425'
DESTINATION = REPORTS / f'{NAME}.zip'
assert not DESTINATION.exists(), 'Never replace a published bundle'
assert (REPORTS / 'normal38_initialization_1421/REPORT.json').exists()
entries = {}

def add(path):
    assert path.is_file() and not path.is_symlink(), path
    entries[str(path.relative_to(ROOT))] = path.read_bytes()

def folder(name):
    for path in sorted((REPORTS / name).rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            add(path)

for path in sorted(OLD.iterdir()):
    if (re.search(r'_(000|001|002)\.(json|npz|pt)$', path.name)
        or path.name in ['job.json', 'interrupted.json', 'geometry.json', 'geometry.npz',
            'asset_dependencies.json', 'source_hashes.json', 'source_snapshot.zip', 'dirty.patch',
            'stdout.log', 'best_candidate.pt', 'resume.pt', 'generation_start.pt', 'metrics.json',
            'GENERATION0_SUMMARY.json', 'GENERATION0_SUMMARY.csv',
            'GENERATION1_SUMMARY.json', 'GENERATION1_SUMMARY.csv']):
        add(path)
for name in ['attack_recovery_gen0_env101', 'attack_recovery_repeat_gen1_env3',
             'poststroke_retreat37_terminal_1420', 'attack_recovered_normal38_ready03_1415',
             'attack_recovered_normal38_bank_cpu04_1415', 'normal38_initialization_1421',
             'recovered_attack_landing_forecast_1358']:
    folder(name)
for name in ['source_snapshot.zip', 'source_hashes.json', 'dirty.patch', 'geometry.json',
             'geometry.npz', 'asset_dependencies.json']:
    add(NEW / name)
for name in ['poststroke_retreat37_stop_1417.json',
             'attack_recovered_normal38_ready02_1358/CPU_TESTS.txt',
             'attack_recovered_normal38_ready02_1358/ACTUAL_ROW_SCORE_AUDIT.json',
             'attack_recovered_normal38_legacy_commands01_1355/REPORT.json',
             'MODEL_RULES.json', 'video/Current_effect_attack_recovery_20261024_1400_verification.json',
             'video/Current_effect_attack_recovery_20261024_1400_upload_receipt.json']:
    add(REPORTS / name)
# Extract the exact new launch sources at repository-relative paths for replay.
# The prior37 execution's sources remain in its own immutable source_snapshot.zip.
with zipfile.ZipFile(NEW / 'source_snapshot.zip') as source:
    for name in source.namelist():
        assert not name.startswith('/') and '..' not in Path(name).parts
        entries[name] = source.read(name)
for path in ROOT.glob('tests/test_full_racket*.py'):
    add(path)
add(ROOT / 'README.md'); add(ROOT / 'status/NEXT.md'); add(Path(__file__))
live = [r for r in json.loads((ROOT / 'status/GPU_LEDGER.json').read_text())['runs']
        if r.get('output') == str(NEW)]
assert len(live) == 1
assert live[0]['status'] == 'RUNNING' and Path(f'/proc/{live[0]["child_pid"]}').exists()
launch = dict(record=live[0], captured_at=datetime.now(timezone.utc).isoformat(),
    scope='Live launch reservation only; charged_gpu_hours here is the2hour reservation, not terminal actual use.')
entries[str((NEW / 'LIVE_LAUNCH_CAPTURE_1425.json').relative_to(ROOT))] = (json.dumps(launch, indent=2)+'\n').encode()
terminal = json.loads((REPORTS / 'poststroke_retreat37_terminal_1420/REPORT.json').read_text())
manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
    scope='Actual CEM20 two complete TRAIN generations, separately preserved339tick generation2 interruption, repeated attacking+qualified-recovery cases with target misses, and CEM21 exact source/input/124CPU tests/actual100tick initialization. CEM21 has no completed performance in this archive.',
    prior_native_source_commit=terminal['source_commit'], new_native_source_commit=live[0]['git_sha'],
    prior_completed_generation_counts=terminal['completed_generation_counts'],
    prior_actual_control_env_steps=terminal['actual_control_env_steps'],
    prior_actual_gpu_hours=terminal['actual_gpu_hours'], prior_guard_terminal_status='FAILED',
    current_run=NEW.name, current_native_ticks_included=100,
    distinction='256candidate executions cover2distinct TRAIN cases, not256independent distribution cases. No formal panel score. Old and new source namespaces and results are separate.',
    dependencies=['G07_runtime_dependencies_20261024_0608.zip',
        'Current_effect_attack_recovery_20261024_1400.mp4',
        'Current_effect_attack_recovery_20261024_1400_evidence.zip'],
    no_acceptance_gates_or_physical_assets_changed=True, formal_acceptance=False,
    files=[dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
           for name, data in sorted(entries.items())])
manifest_name = str((OUT / 'MANIFEST.json').relative_to(ROOT))
manifest_bytes = (json.dumps(manifest, indent=2)+'\n').encode()
(OUT / 'MANIFEST.json').write_bytes(manifest_bytes)
with zipfile.ZipFile(DESTINATION, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
    for name, data in sorted(entries.items()):
        archive.writestr(name, data, compress_type=zipfile.ZIP_STORED if Path(name).suffix in ('.npz', '.zip') else zipfile.ZIP_DEFLATED)
    archive.writestr(manifest_name, manifest_bytes)
with zipfile.ZipFile(DESTINATION) as archive:
    assert len(archive.namelist()) == len(entries)+1
    for record in manifest['files']:
        data = archive.read(record['path'])
        assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
    assert archive.read(manifest_name) == manifest_bytes
record = dict(name=DESTINATION.name, bytes=DESTINATION.stat().st_size,
    sha256=hashlib.sha256(DESTINATION.read_bytes()).hexdigest(),
    archive_members=len(entries)+1, all_data_members_sha256_verified=True,
    scope=manifest['scope'], prior_native_source_commit=manifest['prior_native_source_commit'],
    new_native_source_commit=manifest['new_native_source_commit'], formal_acceptance=False)
(REPORTS / f'{NAME}.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
