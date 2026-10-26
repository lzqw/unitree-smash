"""Seal completed fresh leg06 TRAIN8/frozen128 data; no GPU or automatic upload."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from check_data import validate, TRAIN_NAME, FROZEN_NAME, SOURCE_NAME, BASE_NAME
REPORTS = ROOT/'reports/G07_full_racket_v4'
PARENT = REPORTS/'centered_leg06_after075_intermediates_pilot8_queue01'
ARCHIVE = REPORTS/'G07_leg06_pilot8_native_training_and_frozen_TRAIN128_20261026.zip'
COMPANION_NAME = 'G07_placement075_stage8_to40_completed_native_and_frozen_TRAIN128_20261026.zip'
COMPANION_SHA = '44528da938ad422ed6a5515a32c65df3b763421288d2b4ff6f9236ab36c517bb'


def read(path): return json.loads(path.read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def build():
    assert not Path('/proc/1768388').exists(), 'Wait for actual native/frozen/video parent exit'
    assert read(PARENT/'state.json')['state'] == 'READY_FOR_VISUAL_REVIEW'
    assert not ARCHIVE.exists() and not ARCHIVE.with_suffix('.zip.partial').exists()
    before = validate(ROOT)
    companion = REPORTS/COMPANION_NAME
    assert sha(companion) == COMPANION_SHA
    members, collisions, omissions = {}, [], []
    def add(name, data):
        if name in members and members[name] != data:
            historic = 'historical_source_versions/'+hashlib.sha256(members[name]).hexdigest()+'/'+name
            members[historic] = members[name]
            collisions.append(dict(original_name=name,preserved_previous_member=historic))
        members[name] = data
    def file(path): add(str(path.relative_to(ROOT)),path.read_bytes())
    for folder in [REPORTS/TRAIN_NAME, REPORTS/FROZEN_NAME]:
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts: continue
            if '.partial.' in path.name:
                final = path.with_name(path.name.replace('.partial.','.'))
                if final.exists() and sha(final) == sha(path):
                    omissions.append(dict(omitted=str(path.relative_to(ROOT)),identical_to=str(final.relative_to(ROOT)),sha256=sha(path)))
                    continue
            file(path)
    for folder in [PARENT,REPORTS/'leg06_pilot8_training_health_20261026']:
        for path in folder.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.name != 'observer.lock': file(path)
    for name in ['policy.pt','checkpoint.pt','job.json','completed.json','INPUTS.json','metrics.json']:
        file(REPORTS/SOURCE_NAME/name)
    file(REPORTS/BASE_NAME/'policy.pt')
    source_frozen = REPORTS/'residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01'
    for name in ['public_and_actions_first2.npz','episodes.json','INPUTS.json','metrics.json','job.json','completed.json','source_hashes.json']:
        file(source_frozen/name)
    # Source snapshots take precedence over newer worktree sources, and preserve
    # every byte needed to identify either native launch independently.
    for folder in [REPORTS/TRAIN_NAME, REPORTS/FROZEN_NAME]:
        with zipfile.ZipFile(folder/'source_snapshot.zip') as snapshot:
            for name,digest in read(folder/'source_hashes.json').items():
                data = snapshot.read(name); assert hashlib.sha256(data).hexdigest() == digest
                add(name,data)
    suite_config = read(ROOT/'configs/g07_full_racket_eval_v4_r3.json')['training_suite']
    suite = ROOT/suite_config['path']
    assert sha(suite) == suite_config['sha256'] and len(read(suite)['cases']) == 128
    file(suite)
    with zipfile.ZipFile(companion) as old:
        manifest = json.loads(old.read('MANIFEST.json'))
        for name,item in manifest['source_files'].items():
            if name.startswith('vendor/rsl_rl/'):
                data = old.read(name); assert hashlib.sha256(data).hexdigest() == item['sha256']
                add(name,data)
        layers = manifest['external_runtime_layers']
    assert len([name for name in members if name.startswith('vendor/rsl_rl/')]) == 30
    for path in HERE.iterdir():
        if path.suffix in ['.py','.md'] and path.is_file(): file(path)
    reproduction = (
        'Fresh leg068: 2048 ground environments, 8 updates,2097152 actual control steps.\n'
        'Source0758 actor/RMS explicitly mapped .03 to .06 leg units, unchanged .15 arms; fresh critic/Adam/RNG/counts. This is NOT a strict source8 resume.\n'
        'Includes all8 diagnostic boundaries, canonical8/public8, source0758 canonical/public, BASE846, all actual raw episodes/contacts/plans, 12 selected100Hz training traces, and original128 frozen50Hz body/ball poses.\n'
        'Frozen100Hz public794/action18 data cover first2 cases with valid masks, not all128.\n'
        'Includes exact native source snapshots, original TRAIN suite and pinned RSL Python/license. Isaac/Kit/Torch and other software remain external.\n'
        'Native runtime requires the companion archive listed in MANIFEST: full USD layers,external Go2 layers,PiPER STLs and walking actor. Restore its runtime_external files to recorded absolute paths or equivalent links.\n'
        'All new MP4 files are excluded pending separate actual visual review. No original formal A/B400 performance or acceptance is claimed.\n'
        'CPU validation: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python reports/G07_full_racket_v4/leg06_completed8_delivery_20261026/check_data.py --root . --output fresh_check.json --require-bundled\n'
    ).encode()
    add('REPRODUCE.txt',reproduction)
    payloads = {name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in members.items()}
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),payloads=payloads,
        actual_independent_check=before, source_version_collisions=collisions, duplicate_omissions=omissions,
        required_native_runtime_companion=dict(name=COMPANION_NAME,sha256=COMPANION_SHA,bytes=companion.stat().st_size,
            url='https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/'+COMPANION_NAME),
        external_runtime_layers=layers, current_unreviewed_video_excluded=True, native_source_bytes_preserved=True,
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    temporary = ARCHIVE.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as output:
        for name,data in members.items(): output.writestr(name,data)
        output.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None and len(set(archive.namelist())) == len(payloads)+1
        for name,item in payloads.items():
            data = archive.read(name)
            assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
    temporary.replace(ARCHIVE)
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),archive=str(ARCHIVE),bytes=ARCHIVE.stat().st_size,
        sha256=sha(ARCHIVE),payloads=len(payloads),all_CRC_and_payload_SHA_verified=True,
        new_native_execution=False,automatic_upload=False,formal_acceptance=False,goal_complete=False)
    (HERE/'ARCHIVE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__': build()
