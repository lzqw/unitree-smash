"""Package completed native trials and a separately identified live5prefix."""
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
REPORTS=ROOT/'reports/G07_full_racket_v4'
CURRENT=REPORTS/'ball_arm_target_slew50_normal38_cem23'
OUT=Path(__file__).parent
NAME='G07_normal_slew_results_and_current_program_20261024_1525'
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
for run in ['ball_attack_recovered_normal38_cem21','ball_arm_target_slew45_normal38_cem22']:
    for path in sorted((REPORTS/run).iterdir()):
        if path.name in static or re.search(r'_(000|001|002)\.(json|npz|pt)$',path.name):add(path)
for name in ['normal38_terminal_1516','normal38_gen1_tracking_1510','slew45_terminal_1544',
    'slew45_both_case_recovery_gen0_c15_1540','slew45_initialization_1520','slew45_contact_prefix_1535',
    'slew50_initialization_1550','near_net_planning_1505','arm_target_rate_telemetry_ready_1500',
    'arm_target_rate_cpu02_1500','arm_target_rate45_normal38_ready02_1515',
    'arm_target_rate45_normal38_bank_cpu03_1515','arm_target_rate50_normal38_ready01_1540',
    'arm_target_rate50_normal38_bank_cpu01_1540','attack_recovered_normal38_ready03_1415',
    'protocol_frozen01']:
    folder(name)
for name in ['normal38_two_gen_stop_1512.json','slew45_one_gen_stop_1540.json','MODEL_RULES.json']:
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
prior=json.loads((REPORTS/'ball_arm_target_slew45_normal38_cem22/source_hashes.json').read_text())
assert prior==json.loads((CURRENT/'source_hashes.json').read_text())
manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),
    scope='Completed CEM21two TRAIN generations/256executions, separately saved50tick interruption; completed CEM22one4.5slew generation/128executions plus127tick interruption and full arm target/state telemetry; currentCEM23exact same code with5software-target rate, input and first100native ticks only. All completed full success counts0. Current5profile task outcomes are unmeasured.',
    actual_source_commits=dict(CEM21='d1e7a53559cfd5c752e753c3a71a7c3c8a96cdff',CEM22=live[0]['git_sha'],CEM23=live[0]['git_sha']),
    current_and_prior_slew_control_and_physics_sources_identical=True,
    distinct_TRAIN_cases_per_completed_generation=2,
    current_live_job_reservation=live[0],
    resource_note='The live job charged_gpu_hours2is its reservation, not terminal actual usage. Stopped jobs retain their true FAILED/exit0/no-completed-receipt status, elapsed hours and declared interaction caps.',
    completed_trial_reports=['reports/G07_full_racket_v4/normal38_terminal_1516/REPORT.json',
        'reports/G07_full_racket_v4/slew45_terminal_1544/REPORT.json'],
    current_prefix_report='reports/G07_full_racket_v4/slew50_initialization_1550/REPORT.json',
    assumptions='Original actuator/model assumptions and all task gates unchanged.100Hz telemetry does not certify physics-substep velocity maxima. Near-net free-flight/localIK forecasts are not native performance or infeasibility proofs.',
    reproducibility='Repository-relative current launch source and original frozen case files included. Historical launches keep their exact source_snapshot.zip namespace. Extract runtime dependency archive too; follow its absolute USD-layer mapping. Do not overlap or restore a live resource ledger blindly.',
    dependencies=['G07_runtime_dependencies_20261024_0608.zip'],formal_acceptance=False,
    files=[dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in sorted(entries.items())])
manifest_bytes=(json.dumps(manifest,indent=2)+'\n').encode()
(OUT/'MANIFEST.json').write_bytes(manifest_bytes)
with zipfile.ZipFile(DEST,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
    for name,data in sorted(entries.items()):
        archive.writestr(name,data,compress_type=zipfile.ZIP_STORED if Path(name).suffix in ['.npz','.zip'] else zipfile.ZIP_DEFLATED)
    archive.writestr('NORMAL_SLEW_UPDATE_MANIFEST.json',manifest_bytes)
with zipfile.ZipFile(DEST) as archive:
    assert len(archive.namelist())==len(entries)+1
    for record in manifest['files']:
        data=archive.read(record['path'])
        assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
record=dict(name=DEST.name,bytes=DEST.stat().st_size,sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),
    archive_members=len(entries)+1,all_data_members_sha256_verified=True,scope=manifest['scope'],formal_acceptance=False)
(REPORTS/(NAME+'.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
