"""Package completed slew50trial and a separately identified live positive-family prefix."""
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
REPORTS=ROOT/'reports/G07_full_racket_v4'
CURRENT=REPORTS/'ball_slew50_near_attack_positive38_cem24'
OUT=Path(__file__).parent
NAME='G07_slew50_two_generations_and_positive_search_20261024_1605'
DEST=REPORTS/(NAME+'.zip')
assert not DEST.exists()
entries={}
def add(path):
    assert path.is_file() and not path.is_symlink(),path
    entries[str(path.relative_to(ROOT))]=path.read_bytes()
def folder(name):
    for path in sorted((REPORTS/name).rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:add(path)
static=['job.json','interrupted.json','source_hashes.json','source_snapshot.zip','dirty.patch',
    'geometry.json','geometry.npz','asset_dependencies.json','metrics.json','best_candidate.pt',
    'resume.pt','generation_start.pt','stdout.log','GENERATION0_SUMMARY.json','GENERATION0_SUMMARY.csv',
    'GENERATION1_SUMMARY.json','GENERATION1_SUMMARY.csv']
for run in ['ball_arm_target_slew50_normal38_cem23']:
    for path in sorted((REPORTS/run).iterdir()):
        if path.name in static or re.search(r'_(000|001|002)\.(json|npz|pt)$',path.name):add(path)
for name in ['slew50_terminal_1556','positive38_initialization_1600',
    'slew50_near_attack_gen0_c54_1540','slew50_near_attack_positive38_ready01_1545',
    'slew50_near_attack_positive38_bank_cpu02_1550','servo_contact38_ready01_1600',
    'servo_contact38_bank_cpu01_1600','protocol_frozen01']:
    folder(name)
for name in ['slew50_two_gen_stop_1555.json','MODEL_RULES.json']:
    add(REPORTS/name)
for name in ['source_snapshot.zip','source_hashes.json','dirty.patch','geometry.json','geometry.npz','asset_dependencies.json']:
    add(CURRENT/name)
with zipfile.ZipFile(CURRENT/'source_snapshot.zip') as source:
    for name in source.namelist():
        assert not name.startswith('/') and '..' not in Path(name).parts
        entries[name]=source.read(name)
for path in ROOT.glob('tests/test_full_racket*.py'):add(path)
for name in ['README.md','AGENTS.md','PLAN.md','TRAINING_SPEC.md','ACCEPTANCE.md','status/NEXT.md','status/STATUS.json']:
    add(ROOT/name)
add(Path(__file__))
ledger=json.loads((ROOT/'status/GPU_LEDGER.json').read_text())
live=[record for record in ledger['runs'] if record.get('output')==str(CURRENT)]
assert len(live)==1 and live[0]['status']=='RUNNING' and Path(f'/proc/{live[0]["child_pid"]}').exists()
prior=json.loads((REPORTS/'ball_arm_target_slew50_normal38_cem23/source_hashes.json').read_text())
assert prior==json.loads((CURRENT/'source_hashes.json').read_text())
manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),
    scope='Actual complete CEM23two TRAIN generations/256candidate executions over2distinctcases, full100Hz target/state telemetry, plus separately saved73tick generation2interruption.0attacks/full. CurrentCEM24fresh positive family shares identical control/physics sources and has only100initial native ticks in this archive; later contact/landing/recovery not inferred. Existing Cartesian-gain grid is CPU-only and not queued.',
    actual_source_commits=dict(CEM23='4ef7961a79ffb50be48fd9175f197138d62dad88',CEM24=live[0]['git_sha']),
    current_and_prior_slew_control_and_physics_sources_identical=True,
    distinct_TRAIN_cases_per_completed_generation=2,
    current_live_job_reservation=live[0],
    resource_note='Current live2GPUhours is reservation only. CEM23terminal actual.614570775GPUh/214144steps, FAILED/exit0/no completed receipt, declaredcap409600retained.',
    completed_trial_reports=['reports/G07_full_racket_v4/slew50_terminal_1556/REPORT.json'],
    current_prefix_report='reports/G07_full_racket_v4/positive38_initialization_1600/REPORT.json',
    assumptions='Original actuator/model assumptions and all task gates unchanged.100Hz telemetry does not certify physics-substep velocity maxima. Near-net free-flight/localIK forecasts are not native performance or infeasibility proofs.',
    reproducibility='Repository-relative current launch source and original frozen case files included. Historical launches keep their exact source_snapshot.zip namespace. Extract runtime dependency archive too; follow its absolute USD-layer mapping. Do not overlap or restore a live resource ledger blindly.',
    dependencies=['G07_runtime_dependencies_20261024_0608.zip'],formal_acceptance=False,
    files=[dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in sorted(entries.items())])
manifest_bytes=(json.dumps(manifest,indent=2)+'\n').encode()
(OUT/'MANIFEST.json').write_bytes(manifest_bytes)
with zipfile.ZipFile(DEST,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
    for name,data in sorted(entries.items()):
        archive.writestr(name,data,compress_type=zipfile.ZIP_STORED if Path(name).suffix in ['.npz','.zip'] else zipfile.ZIP_DEFLATED)
    archive.writestr('SLEW50_POSITIVE_UPDATE_MANIFEST.json',manifest_bytes)
with zipfile.ZipFile(DEST) as archive:
    assert len(archive.namelist())==len(entries)+1
    for record in manifest['files']:
        data=archive.read(record['path'])
        assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
record=dict(name=DEST.name,bytes=DEST.stat().st_size,sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),
    archive_members=len(entries)+1,all_data_members_sha256_verified=True,scope=manifest['scope'],formal_acceptance=False)
(REPORTS/(NAME+'.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
