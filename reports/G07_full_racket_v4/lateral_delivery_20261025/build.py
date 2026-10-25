"""Archive completed lateral TRAIN128 evidence, video and exact policy."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[3]
REPORTS=ROOT/'reports/G07_full_racket_v4'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=REPORTS/'G07_lateral_TRAIN128_native_and_video_20261025.zip'
    assert not output.exists()
    run=REPORTS/'train128_lateral_c12_1025'
    completed=json.loads((run/'completed.json').read_text())
    assert completed
    paths=[p for p in run.iterdir() if p.is_file() and 'partial' not in p.name]
    paths += list((REPORTS/'video_snapshot_lateral_train128_20261025').rglob('*'))
    paths += list((REPORTS/'video').glob('Current_effect_lateral_train128_20261025*'))
    for name in ['best_candidate.pt','population_000.pt','episodes_000.json',
                 'generation_000.json','source_hashes.json','job.json','completed.json']:
        paths.append(REPORTS/'cem32_lateral_feedback_1015'/name)
    paths += [REPORTS/'TRAIN128_LATERAL_DIAGNOSTIC_20261025.json',
              REPORTS/'CEM32_LATERAL_RESULT_20261025.json',
              REPORTS/'video_spec_lateral_train128_20261025.json',Path(__file__)]
    paths += [ROOT/name for name in ['scripts/prepare_full_racket_latest_video_v4.py',
        'scripts/render_full_racket_latest_effect_v4.py','scripts/render_full_racket_r3_effect.py',
        'scripts/render_current_v4_progress.py','scripts/evaluate_full_racket_train_v4.py',
        'quadsmash/control/full_racket_train_evaluation_v4.py','quadsmash/evaluation_v4.py']]
    dependencies=json.loads((run/'asset_dependencies.json').read_text())['resolved_layers']
    # Preserve the original absolute directory layout underneath a portable
    # archive prefix. Runtime source snapshots and their original hashes stay intact.
    entries={str(p.relative_to(ROOT)):p for p in paths if p.is_file()}
    for item in dependencies:
        p=Path(item)
        assert p.is_file()
        if p.is_relative_to(ROOT):
            entries[str(p.relative_to(ROOT))]=p
        else:
            entries['external_dependencies/'+str(p.relative_to('/'))]=p
    manifest=dict(scope='Completed CEM32 candidate12 fixed-policy TRAIN128 only, plus exact original policy/population and three selected native video episodes.',
        formal_acceptance=False,formal_cases_read=False,complete_count=0,
        policy_sha256='51a8b19b17e19e60bd7482c7bcb7ed1d4e8b7f50b9317cac3c49007089e8b62d',
        active_training_excluded='Concurrent incomplete CEM33 height training is excluded; no result inherited.',
        runtime_note='Use the source_snapshot.zip from the completed diagnostic for its original runtime code; outer renderer files describe this later video export.',
        dependencies_note='Restore external_dependencies paths to their recorded absolute paths, or explicitly remap USD references; install the separately pinned Isaac runtime. This archive does not bundle Isaac itself.',
        files={name:dict(sha256=digest(p),bytes=p.stat().st_size) for name,p in sorted(entries.items())})
    policy=REPORTS/'cem32_lateral_feedback_1015/best_candidate.pt'
    assert digest(policy)==manifest['policy_sha256']
    text=json.dumps(manifest,indent=2)+'\n'
    (Path(__file__).parent/'MANIFEST.json').write_text(text)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        z.writestr('DELIVERY_MANIFEST.json',text)
        for name,p in sorted(entries.items()):
            z.write(p,name)
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        for name,item in manifest['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==item['sha256']
    result=dict(archive=str(output.relative_to(ROOT)),sha256=digest(output),bytes=output.stat().st_size,
        file_count=len(entries)+1,all_archive_members_rehashed=True,formal_acceptance=False)
    (Path(__file__).parent/'ARCHIVE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
