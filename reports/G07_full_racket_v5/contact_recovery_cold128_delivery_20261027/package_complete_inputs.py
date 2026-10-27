"""Seal completed cold128, every held input, and the actual robot USD layers."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WORK = ROOT.parent / 'unitree_smash_contact_recovery'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    run = ROOT / 'reports/G07_full_racket_v5/teacher_contact_recovery_cold128_02'
    queue = ROOT / 'reports/G07_full_racket_v5/contact_recovery_queue02'
    expected = read(queue / 'PLAN.json')['source_sha256']
    assert len(expected) == 931
    result = read(HERE / 'ACTUAL_COMPLETED_COLD128_CHECK.json')
    assert result['actual_completed_new_native']
    assert not result['original128_rows_differing_after_removing_only_new_association']
    assert all(value['exactly_identical'] for value in result['recorded_motion_comparison'].values())
    assert read(run / 'job.json')['status'] == 'SUCCEEDED'
    records, objects = {}, {}

    def add(path, role, wanted=None):
        path = Path(path).absolute()
        assert path.is_file(), str(path)
        canonical = path.resolve(strict=True)
        digest = sha(path)
        if wanted is not None:
            assert digest == wanted, str(path)
        name = str(path)
        if name in records:
            assert records[name]['sha256'] == digest
            records[name]['roles'] = sorted(set(records[name]['roles'] + [role]))
            return
        records[name] = dict(original_path=name, bytes=path.stat().st_size, sha256=digest,
                             link_target_original=str(canonical) if canonical != path else None,
                             roles=[role])
        objects.setdefault(digest, canonical)
        if canonical != path:
            add(canonical, 'resolved_original_link_target', digest)

    for name, digest in expected.items():
        add(Path(name), 'original_held_input', digest)
    precision = ROOT / 'reports/G07_full_racket_v4/origin_precision_after_aim_queue01/PRECISION_ADMISSION.json'
    add(precision, 'original_precision_admission')
    receipt = read(precision)
    for name, digest in receipt['source_sha256'].items():
        add(Path(name), 'precision_admission_hashed_dependency', digest)
    for name in ['baseline_run', 'compact_run']:
        for filename in ['job.json', 'completed.json']:
            add(Path(receipt[name]) / filename, 'actual_precision_probe_receipt')
    layers = read(ROOT / 'configs/full_racket_usd_dependencies_v4.json')['layers']
    for tree in [ROOT, WORK]:
        for layer in layers:
            path = Path(layer['path'])
            add(path if path.is_absolute() else tree / path, 'actual_composed_USD_source_layer', layer['sha256'])
    for tree in [ROOT, ROOT.parent / 'unitree_smash_landing_targets', WORK]:
        for name in ['assets/vendor/piper_isaac_sim/piper_description/urdf/piper_no_gripper_description.urdf',
                     'assets/vendor/go2_flat_baryn20251217/policy.pt',
                     'reports/G07_full_racket_v4/protocol_frozen01/training_cases.json']:
            add(tree / name, 'controller_or_original_TRAIN_runtime_input')
    for path in sorted(run.iterdir()):
        if path.is_file():
            add(path, 'completed_native_cold128')
    for path in sorted(queue.iterdir()):
        if path.is_file() and path.suffix in ('.json', '.py', '.md'):
            add(path, 'current_queue_provenance_at_packaging_time')
    for path in sorted((WORK / 'reports/G07_full_racket_v5/contact_recovery_revision_20261027').iterdir()):
        if path.is_file() and path.suffix in ('.json', '.py', '.md'):
            add(path, 'revision_and_actual_CPU_checks')
    for path in sorted((ROOT / 'reports/G07_full_racket_v5/origin0_measured_swing_velocity_20261027').iterdir()):
        if path.is_file() and path.suffix in ('.json', '.py', '.npz'):
            add(path, 'unexecuted_swing_proposals_and_actual_CPU_checks')
    add(ROOT / 'reports/G00/environment.lock.json', 'pinned_external_software_reference')
    for name in ['restore_inputs.py', 'check_fresh_cpu.py', 'verify_completed_teacher.py',
                 'ACTUAL_COMPLETED_COLD128_CHECK.json', 'SOFTWARE_OBSERVATION.json',
                 'REPRODUCTION.md', 'package_complete_inputs.py']:
        add(HERE / name, 'delivery_program_or_check')
    manifest = dict(schema='g07-original-path-input-bundle-v1', created_at=datetime.now(timezone.utc).isoformat(),
        held_input_count=len(expected), unique_payload_count=len(objects),
        source_path_count=len(records), files=[records[name] for name in sorted(records)],
        held_input_paths=sorted(expected), completed_native_run=str(run),
        revision_worktree=str(WORK), original_teacher_policy_sha256='204ff76f21625d49cd1dc29ec3b6256447a5fa8b2fec935dda986842fc330034',
        current_source_commits={str(tree): subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tree, text=True).strip()
                                for tree in [ROOT, WORK]},
        actual_cold_execution_commit_at_start='e7c8bb3c8abbdc9af4db7afd12215a223763fb68',
        all10_composed_USD_layers_included=True, no_nonlayer_asset_dependencies=True,
        builtin_visual_shader_external='OmniPBR.mdl is supplied by the installed Isaac/Kit environment.',
        live_training_run_included=False, native_execution_in_fresh_extraction=False, formal_acceptance=False)
    archive_path = HERE / 'G07_contact_recovery_v5_cold128_complete_inputs_and_assets_20261027.zip'
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
        for digest, path in sorted(objects.items()):
            assert sha(path) == digest
            archive.write(path, 'objects/' + digest)
        for name in ['restore_inputs.py', 'check_fresh_cpu.py', 'REPRODUCTION.md']:
            archive.write(HERE / name, name)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        for digest in objects:
            assert hashlib.sha256(archive.read('objects/' + digest)).hexdigest() == digest
        members = len(archive.namelist())
    for name, wanted in expected.items():
        assert sha(Path(name)) == wanted
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), archive=archive_path.name,
        bytes=archive_path.stat().st_size, sha256=sha(archive_path), archive_members_CRC_verified=members,
        unique_payload_SHA_verified=len(objects), restored_original_path_count=len(records),
        original_held_inputs_included=len(expected), all_composed_USD_source_layers_included=True,
        original_symlink_targets_preserved_in_manifest=True,
        actual_completed_native_cold128_included=True, source_policy_bytes_unchanged=True,
        package_program_sha256=sha(Path(__file__)),
        no_other_large_runtime_companion_archive_required_for_packaged_USD_or_bound_inputs=True,
        external_installed_Isaac_Python_and_builtin_shader_still_required=True,
        native_reexecution_in_fresh_tree=False, formal_acceptance=False)
    with (HERE / 'ARCHIVE_CHECK.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
