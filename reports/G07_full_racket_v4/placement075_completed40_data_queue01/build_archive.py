"""Seal completed075 native40/frozen128 data, preserving the verified stage8 data."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT/'reports/G07_full_racket_v4'
PARENT = REPORTS/'centered_placement075_resume32_queue02'
SELECTOR = REPORTS/'centered_placement075_intermediate_selection_queue01'
TRAIN = REPORTS/'residual_ppo_centered_placement075_n2048_resume32_02'
FROZEN = REPORTS/'residual_ppo_centered_placement075_n2048_iter40_fixed_train128_02'
SOURCE = REPORTS/'residual_ppo_centered_placement075_n2048_probe8_01'
ARCHIVE = REPORTS/'G07_placement075_stage8_to40_completed_native_and_frozen_TRAIN128_20261026.zip'
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('completed075_data_native_checks', REPORTS/'residual_resume24_queue01/run.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def read(path):return json.loads(path.read_text())
def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):value.update(block)
    return value.hexdigest()


def verify_completed():
    assert not core.live(1734658, 'centered_placement075_resume32_queue02/run.py'), 'Wait for actual native/fixed/video parent exit'
    state = read(PARENT/'state.json')
    assert state['state'] == 'READY_FOR_VISUAL_REVIEW' and state['selected_training_trace_audit_passed']
    owner_spec = importlib.util.spec_from_file_location('completed075_owner_validation', PARENT/'run.py')
    owner = importlib.util.module_from_spec(owner_spec)
    owner_spec.loader.exec_module(owner)
    receipt = owner.verify_training(core.verify_job(TRAIN, 8388608))
    summary = core.verify_evaluation(FROZEN, TRAIN/'policy.pt')
    assert receipt['checkpoint_sha256'] == read(PARENT/'TRAINING_RECEIPT.json')['checkpoint_sha256']
    audit = read(PARENT/'TRAINING_TRACE_AUDIT.json')
    assert read(PARENT/'TRACE_AUDIT_ATTEMPT.json')['passed'] and audit['all_selected_trajectories_checked']
    assert audit['selected_episodes_checked'] == read(TRAIN/'trajectory_traces/manifest.json')['saved_episode_count']
    assert all(sha(Path(name)) == digest for name, digest in audit['source_sha256'].items())
    assert read(FROZEN/'INPUTS.json')['reset_noise_seed'] == 724201
    return receipt, summary


def build():
    import torch
    assert not ARCHIVE.exists() and not ARCHIVE.with_suffix('.zip.partial').exists()
    assert not (HERE/'ARCHIVE_CHECK.json').exists()
    plan = read(HERE/'PLAN.json')
    for name, digest in plan['main_inputs'].items():assert sha(Path(name)) == digest
    receipt, summary = verify_completed()
    source = torch.load(SOURCE/'checkpoint.pt', map_location='cpu', weights_only=False)
    diagnostics = []
    for iteration in range(9, 41):
        path = TRAIN/'ppo'/f'model_{iteration-1}.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['iter'] == iteration and not state['native']['canonical_restart']
        assert state['native']['total_timesteps'] == iteration*262144
        assert all(torch.isfinite(v).all() for v in state['model_state_dict'].values() if torch.is_tensor(v))
        old, new = source['optimizer_state_dict']['state'], state['optimizer_state_dict']['state']
        assert set(old) == set(new) and len(new) == 17
        assert all(float(new[k]['step'])-float(old[k]['step']) == (iteration-8)*20 for k in old)
        for name in ['actor_obs_normalizer.count', 'critic_obs_normalizer.count']:
            assert int(state['model_state_dict'][name])-int(source['model_state_dict'][name]) == (iteration-8)*262144
        diagnostics.append(dict(iteration=iteration, path=str(path.relative_to(ROOT)), sha256=sha(path), canonical_restart=False))
    parent_path = Path(plan['stage8_archive']['path'])
    assert sha(parent_path) == plan['stage8_archive']['sha256']
    mid_path = Path(plan['CPU16_24_archive']['path'])
    assert sha(mid_path) == plan['CPU16_24_archive']['sha256']
    paths = {Path(name) for name in plan['main_inputs']}
    native_hashes = read(TRAIN/'source_hashes.json')
    omitted = []
    for folder in [TRAIN, FROZEN]:
        duplicates = {}
        for partial, final in [('episodes.partial.json', 'episodes.json'), ('all_first_episodes.partial.npz', 'all_first_episodes.npz')]:
            p, f = folder/partial, folder/final
            if p.exists() and f.exists() and sha(p) == sha(f):
                duplicates[p] = f
                omitted.append(dict(omitted=str(p.relative_to(ROOT)), identical_to=str(f.relative_to(ROOT)), sha256=sha(f)))
        paths.update(p for p in folder.rglob('*') if p.is_file() and p not in duplicates and '__pycache__' not in p.parts)
    directories = [PARENT, REPORTS/'centered_placement075_resume_health_20261026',
        REPORTS/'placement075_first_complete_training_20261026', REPORTS/'placement075_additional_complete_training_20261026',
        REPORTS/'placement075_fourth_complete_training_20261026', REPORTS/'placement075_action_authority_20261026',
        REPORTS/'video_snapshot_placement075_first_complete_20261026',
        REPORTS/'video_snapshot_placement075_additional_complete_20261026',
        REPORTS/'video_snapshot_placement075_fourth_complete_20261026',
        REPORTS/'video_snapshot_centered_placement075_iter40_retry02_20261026']
    for directory in directories:
        paths.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'observer.lock')
    paths.update(p for p in HERE.iterdir() if p.is_file() and p.suffix in ['.py', '.md'] and not p.name.startswith('ARCHIVE_'))
    paths.update([HERE/'PLAN.json', HERE/'ACTUAL_OLD025_COMPLETED_LAYOUT_CHECK.json', SELECTOR/'export_diagnostic.py'])
    for iteration in [16, 24, 32]:
        paths.update([SELECTOR/f'exports/iter{iteration}/policy.pt', SELECTOR/f'exports/iter{iteration}/EXPORT_CHECK.json'])
    # Do not publish the current unreviewed40 MP4 through a data archive.
    paths = {p for p in paths if p.suffix != '.mp4'}
    records, sources, collisions = {}, {}, []
    def add(name, item, origin):
        if name in records and records[name] != item:
            old = 'historical_source_versions/'+records[name]['sha256']+'/'+name
            records[old], sources[old] = records[name], sources[name]
            collisions.append(dict(original_name=name, preserved_previous_member=old))
        records[name], sources[name] = item, origin
    with zipfile.ZipFile(parent_path) as parent, zipfile.ZipFile(mid_path) as middle, zipfile.ZipFile(TRAIN/'source_snapshot.zip') as native:
        prior = json.loads(parent.read('MANIFEST.json'))
        assert len(prior['external_runtime_layers']) == 4 and not prior['required_companion_archives']
        for name in parent.namelist():
            data = parent.read(name)
            if name in prior['source_files']:
                item = prior['source_files'][name]
                assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
            target = 'prior_stage8_evidence/'+name if '/' not in name else name
            add(target, dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()), ('parent', name))
        middle_manifest = json.loads(middle.read('CPU_EXPORT_MANIFEST.json'))
        for name in middle.namelist():
            if name.startswith('vendor/rsl_rl/'):
                data = middle.read(name)
                assert hashlib.sha256(data).hexdigest() == middle_manifest['payloads'][name]['sha256']
                add(name, middle_manifest['payloads'][name], ('middle', name))
        for path in sorted(paths):
            assert path.is_file(), path
            name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else 'runtime_external/'+str(path).lstrip('/')
            add(name, dict(bytes=path.stat().st_size, sha256=sha(path)), ('file', path))
        # The native launch snapshot, not potentially newer unused worktree
        # files, supplies the exact executable source for this completed job.
        for name, digest in native_hashes.items():
            data = native.read(name)
            assert hashlib.sha256(data).hexdigest() == digest, name
            add(name, dict(bytes=len(data), sha256=digest), ('native', name))
        reproduction = (
            'Completed placement075 canonical8-to40 native TRAIN and final40 original TRAIN128 data.\n'
            'Includes all32 diagnostic boundaries9..40, raw closed episodes/contacts, all selected100Hz traces, canonical40/public40 and all128 frozen50Hz body/ball trajectories.\n'
            'Full794 public observations/actions at100Hz for frozen data are stored only for first2 cases, not every case.\n'
            'All previously verified stage8 data/assets, including all10 USD layers, PiPER URDF/STLs and walking actor, are merged; no companion download is needed for DATA.\n'
            'Restore runtime_external files to recorded absolute paths or equivalent links for native execution. Isaac/Kit/Torch/other software remain external prerequisites.\n'
            'Exact native source_snapshot.zip and source_hashes.json belong to each run. Current project Python sources and exact RSL sources/license are also present.\n'
            'Canonical40 is a same-config restart point. model_i diagnostic states are NOT canonical restart points.\n'
            'Only the previously reviewed stage8 video is inherited. The new40 MP4 is excluded pending its own root visual review.\n'
            'Intermediate16/24/32 frozen assessments and original formal A/B400 are not claimed by this archive.\n'
            'Changing-policy TRAIN successes are not frozen-policy performance. Full original50/70/85/NF0 goal remains unproven.\n'
        ).encode()
        records['REPRODUCE.txt'] = dict(bytes=len(reproduction), sha256=hashlib.sha256(reproduction).hexdigest())
        manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), source_files=records,
            actual_completed_training=receipt, independent_final40_original_TRAIN128=summary,
            actual_diagnostic_boundaries=diagnostics, selected_training_traces=read(TRAIN/'trajectory_traces/manifest.json')['saved_episode_count'],
            inherited_stage8_archive=plan['stage8_archive'], required_companion_archives=[],
            external_runtime_layers=prior['external_runtime_layers'], preserved_source_version_collisions=collisions,
            duplicate_omissions_after_byte_equality=omitted, current_unreviewed40_video_excluded=True,
            source_job_complete=True, formal_cases_read=False, formal_acceptance=False, goal_complete=False)
        temporary = ARCHIVE.with_suffix('.zip.partial')
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
            for name, origin in sources.items():
                kind, value = origin
                if kind == 'file':output.write(value, name)
                else:output.writestr(name, {'parent':parent, 'middle':middle, 'native':native}[kind].read(value))
            output.writestr('REPRODUCE.txt', reproduction)
            output.writestr('MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None and len(set(archive.namelist())) == len(records)+1
        for name, item in records.items():
            data = archive.read(name)
            assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'], name
    temporary.replace(ARCHIVE)
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), archive=str(ARCHIVE), bytes=ARCHIVE.stat().st_size,
        sha256=sha(ARCHIVE), source_payloads_checked=len(records), all_CRC_and_payload_SHA_verified=True,
        source_job_complete=True, actual_completed_training=receipt, independent_final40_original_TRAIN128=summary,
        required_companion_archives=[], current_unreviewed40_video_excluded=True,
        source_runtime_custody_ends_when_archive_sealed=True, new_native_execution=False, formal_acceptance=False, goal_complete=False)
    core.atomic(HERE/'ARCHIVE_CHECK.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
