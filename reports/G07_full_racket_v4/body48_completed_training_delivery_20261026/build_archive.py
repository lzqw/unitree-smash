"""Stream the actual completed native48 training course into a portable archive.

No GPU execution or process control. Immutable training data may be sealed while
the existing owner evaluates frozen policies. All source versions are retained.
"""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from check_data import TRAIN, HERE_REL
REPORTS = ROOT/'reports/G07_full_racket_v4'
PARENT = REPORTS/'body_residual48_after_grid16_queue01'
ARCHIVE = REPORTS/'G07_body_residual48_completed_native_training_and_all_weights_20261026.zip'
COMPANION_NAME = 'G07_placement075_stage8_to40_completed_native_and_frozen_TRAIN128_20261026.zip'
COMPANION_SHA = '44528da938ad422ed6a5515a32c65df3b763421288d2b4ff6f9236ab36c517bb'


def read(path):
    return json.loads(path.read_text())


def digest_stream(stream):
    digest, size = hashlib.sha256(), 0
    for block in iter(lambda:stream.read(8*1024*1024),b''):
        digest.update(block)
        size += len(block)
    return size, digest.hexdigest()


def sha(path):
    with path.open('rb') as stream:
        return digest_stream(stream)[1]


def atomic(path,value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def verify_completed():
    run = ROOT/TRAIN
    job = read(run/'job.json')
    assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0 and job['receipt_present']
    assert job['charged_env_steps'] == read(run/'completed.json')['env_steps'] == 12582912
    receipt = read(PARENT/'TRAINING_RECEIPT.json')
    assert receipt['actual_control_env_steps'] == 12582912 and receipt['source_teacher_exact']
    assert receipt['checkpoint_sha256'] == sha(run/'checkpoint.pt')
    assert receipt['policy_sha256'] == sha(run/'policy.pt')
    audit = read(PARENT/'TRAINING_TRACE_AUDIT.json')
    assert audit['all_selected_trajectories_checked'] and audit['selected_episodes_checked'] > 0
    checked = read(HERE/'ACTUAL_COMPLETED_DATA_CHECK.json')
    assert checked['canonical_checkpoint_sha256'] == receipt['checkpoint_sha256']
    assert checked['public_policy_sha256'] == receipt['policy_sha256']
    assert checked['independently_recomputed_training_counts'] == receipt['independent_changing_policy_counts']
    plan = read(HERE/'PLAN.json')
    for name,digest in {**plan['static_sources'],**plan['main_inputs']}.items():
        assert sha(Path(name)) == digest,name
    return receipt, checked


def build():
    receipt, checked = verify_completed()
    assert not ARCHIVE.exists() and not ARCHIVE.with_suffix('.zip.partial').exists()
    companion = REPORTS/COMPANION_NAME
    assert sha(companion) == COMPANION_SHA
    members, collisions, omissions = {}, [], []

    def add(name,kind,source,digest,size):
        path = PurePosixPath(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert name != 'MANIFEST.json'
        item = dict(kind=kind,source=source,sha256=digest,bytes=size)
        previous = members.get(name)
        if previous and previous['sha256'] != digest:
            historical = 'historical_source_versions/'+previous['sha256']+'/'+name
            assert historical not in members or members[historical]['sha256'] == previous['sha256']
            members[historical] = previous
            collisions.append(dict(original_name=name,preserved_previous_member=historical))
        members[name] = item

    def file(path):
        assert path.is_file() and path.resolve().is_relative_to(ROOT)
        add(path.relative_to(ROOT).as_posix(),'file',path,sha(path),path.stat().st_size)

    def data(name,value):
        add(name,'bytes',value,hashlib.sha256(value).hexdigest(),len(value))

    def folder(directory):
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.lock':
                continue
            if '.partial.' in path.name:
                final = path.with_name(path.name.replace('.partial.','.'))
                if final.exists() and sha(final) == sha(path):
                    omissions.append(dict(omitted=str(path.relative_to(ROOT)),
                        identical_to=str(final.relative_to(ROOT)),sha256=sha(path)))
                    continue
            file(path)

    config = read(HERE/'CHECK_INPUTS.json')
    source_policy = ROOT/config['source_policy_relative_path']
    source_evaluation = ROOT/config['source_evaluation_relative_path']
    # The completed training directory is immutable once its native child exits.
    # The owner's mutable state, frozen exports and evaluation logs stay outside.
    folder(ROOT/TRAIN)
    folder(source_evaluation)
    file(source_policy)
    for name in ['INPUTS.json','metrics.json','job.json','completed.json','source_hashes.json','source_snapshot.zip']:
        file(source_policy.parent/name)
    parent_names = [
        'PLAN.json','run.py','check_source_zero.py','export_intermediate.py',
        'ADMISSION_DECISION.json','SOURCE_SELECTION.json','CPU_PREPARATION_CHECK.json',
        'CPU_NONZERO_EXPORT_CHECK.json','ACTUAL_SELECTED_SOURCE_ZERO_CHECK.json',
        'ACTUAL_UPDATE1_CHECK.json','ACTUAL_UPDATE7_CHECK.json','ACTUAL_UPDATE8_CHECK.json',
        'ROOT_ACTUAL_INITIAL_STATE_CHECK.json','ROOT_ACTUAL_UPDATE36_CHECK.json',
        'TRAINING_RECEIPT.json','TRAINING_TRACE_AUDIT.json','RUNNING_FRESH_BODY_RESIDUAL48.guard.log',
    ]
    for name in parent_names:
        file(PARENT/name)
    # Snapshots are authoritative. If snapshots differ, retain each old version;
    # the current training snapshot is flattened last at its original root paths.
    for run in [source_evaluation, ROOT/TRAIN]:
        with zipfile.ZipFile(run/'source_snapshot.zip') as snapshot:
            for name,digest in read(run/'source_hashes.json').items():
                value = snapshot.read(name)
                assert hashlib.sha256(value).hexdigest() == digest
                data(name,value)
    suite = read(ROOT/'configs/g07_full_racket_eval_v4_r3.json')['training_suite']
    assert sha(ROOT/suite['path']) == suite['sha256']
    assert len(read(ROOT/suite['path'])['cases']) == 128
    file(ROOT/suite['path'])
    with zipfile.ZipFile(companion) as old:
        manifest = json.loads(old.read('MANIFEST.json'))
        for name,item in manifest['source_files'].items():
            if name.startswith('vendor/rsl_rl/'):
                value = old.read(name)
                assert hashlib.sha256(value).hexdigest() == item['sha256']
                data(name,value)
        layers = manifest['external_runtime_layers']
    assert sum(name.startswith('vendor/rsl_rl/') for name in members) == 30
    for name in ['check_data.py','build_archive.py','run.py','REPORT.md','PLAN.json','CHECK_INPUTS.json',
                 'PREFLIGHT_CHECK.json','DETACHED_LAUNCH_RECEIPT.json','NATIVE48_EPISODES.csv',
                 'NATIVE48_EPISODES.json','ACTUAL_COMPLETED_DATA_CHECK.json']:
        file(HERE/name)
    reproduction = (
        'Completed native48 training only:2048 ground environments,48 updates,12582912 control environment steps.\n'
        'Fresh zero-mean outer residual over two fixed source residual actor tiers;7 critic-only updates and41 actor updates.\n'
        'All48 diagnostic boundaries, canonical48/public48, initial model state, all closed episode/contact/planning rows and selected100Hz traces are preserved.\n'
        'The included source teacher has original fixed TRAIN128 measurements. It is not the newly learned48 policy score.\n'
        'Changing-policy training totals are not fixed-policy performance. New13/24/36/48 frozen assessments are delivered separately.\n'
        'Exact native source snapshots are flattened; collisions are retained under historical_source_versions.\n'
        'Pinned RSL Python/license is bundled. Isaac/Kit/Torch and other runtime packages remain external.\n'
        'The companion listed in MANIFEST provides full USD/STL/external Go2 layers and the walking actor. Restore its runtime_external paths as recorded.\n'
        'Canonical checkpoint may restart at its recorded ground boundary; model_i diagnostics are inference/update evidence and not canonical physical restart states.\n'
        'CPU check:CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python '+HERE_REL+'/check_data.py --root . --output fresh_check.json --require-bundled\n'
        'Trace audit:CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python scripts/check_full_racket_training_traces_v4.py --run '+TRAIN+' --output fresh_traces.json\n'
        'Original formal A/B cases are excluded. No formal acceptance or full-goal completion is claimed.\n'
    ).encode()
    data('REPRODUCE.txt',reproduction)
    payloads = {name:{k:item[k] for k in ['bytes','sha256']} for name,item in members.items()}
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),payloads=payloads,
        scope='Completed changing-policy native48 course, all diagnostic/canonical/public weights and measured source128 evidence; new frozen comparisons delivered separately.',
        actual_training_receipt=receipt,actual_CPU_data_check=checked,
        source_version_collisions=collisions,duplicate_omissions=omissions,
        required_native_runtime_companion=dict(name=COMPANION_NAME,sha256=COMPANION_SHA,
            bytes=companion.stat().st_size,url='https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/'+COMPANION_NAME),
        external_runtime_layers=layers,native_source_bytes_preserved=True,
        source_custody='After sealing, validate archive or fresh extraction bytes only; no extended hold on main worktree.',
        new_native_execution=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    temporary = ARCHIVE.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as output:
        for name,item in members.items():
            stream = item['source'].open('rb') if item['kind'] == 'file' else io.BytesIO(item['source'])
            with stream,output.open(name,'w',force_zip64=True) as target:
                shutil.copyfileobj(stream,target,8*1024*1024)
        output.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist())) == len(payloads)+1
        for name,item in payloads.items():
            with archive.open(name) as stream:
                size,digest = digest_stream(stream)
            assert size == item['bytes'] and digest == item['sha256'],name
    temporary.replace(ARCHIVE)
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),archive=str(ARCHIVE),
        bytes=ARCHIVE.stat().st_size,sha256=sha(ARCHIVE),payloads=len(payloads),
        all_CRC_and_payload_SHA_verified=True,all48_diagnostics_included=True,
        new_native_execution=False,automatic_upload=False,formal_acceptance=False,goal_complete=False)
    atomic(HERE/'ARCHIVE_CHECK.json',result)
    return result


if __name__ == '__main__':
    print(json.dumps(build()),flush=True)
