"""Package both completed128-case samples and the honest current video proof."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent/'unitree_smash'
HERE = Path(__file__).resolve().parent
NATIVE = HERE/'snapshot/prepose_origin0_8_sequential_n128_01'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive = HERE/'G07_origin0_two_completed_candidates_TRAIN128_video_evidence_20261027.zip'
    assert not archive.exists()
    plan = json.loads((HERE/'PLAN.json').read_text())
    assert all(sha(Path(path)) == digest for path, digest in plan['source_sha256'].items())
    files = {'native/'+path.name:path for path in NATIVE.iterdir() if path.is_file()}
    for path in [HERE/name for name in ['PLAN.json', 'ACTUAL_PRODUCER_SNAPSHOT.json', 'NATIVE_EPISODE_CHECK.json',
        'DECODE_AND_NATIVE_INPUT_CHECK.json', 'ROOT_VISUAL_REVIEW.json', 'displayed_candidate_000.pt',
        'displayed_candidate_001.pt', 'prepare.py', 'verify_video.py', 'build_archive.py', 'REPORT.md',
        'snapshot/cases.json']]:
        files['evidence/'+str(path.relative_to(HERE))] = path
    for index in range(3):
        files[f'display/case{index}.npz'] = HERE/f'snapshot/case{index}.npz'
    for path in sorted((HERE/'decoded_samples').glob('*.jpg'))+sorted((HERE/'review').glob('*.jpg')):
        files['evidence/'+str(path.relative_to(HERE))] = path
    stem = 'Current_effect_origin0_actual_failures_20261027'
    for extension in ['.json', '.preview.json']:
        files['video/'+stem+extension] = ROOT/'reports/G07_full_racket_v4/video'/(stem+extension)
    for name in ['render_full_racket_actual_landing_labels_v4.py', 'render_full_racket_r3_effect.py',
                 'render_current_v4_progress.py', 'native_video_framing_v4.py']:
        files['render/scripts/'+name] = ROOT/'scripts'/name
    for name in ['court_net_v4.json', 'g07_full_racket_eval_v4_r3.json', 'g07_full_racket_eval_v4_r3_origin0.json']:
        files['render/configs/'+name] = MAIN/'configs'/name
    precision = MAIN/'reports/G07_full_racket_v4/origin_precision_after_aim_queue01'
    files['precision/PRECISION_ADMISSION.json'] = precision/'PRECISION_ADMISSION.json'
    for run in ['origin16_native_precision_train29_n128_01', 'origin0_native_precision_train29_n128_01']:
        for name in ['native_probe.npz', 'RESULT.json', 'INPUTS.json', 'job.json', 'completed.json']:
            files[f'precision/{run}/{name}'] = MAIN/'reports/G07_full_racket_v4'/run/name
    hashes = {name:dict(bytes=path.stat().st_size, sha256=sha(path)) for name, path in files.items()}
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), payloads=hashes,
        scope='Two actually completed candidates, each original all128 first episodes, from an ongoing origin0 eight-candidate producer; three independent failure display samples.',
        source_native_episode_indices=[0, 0, 3], source_policy_files=['evidence/displayed_candidate_000.pt', 'evidence/displayed_candidate_001.pt'],
        completed_candidate_summaries=json.loads((HERE/'NATIVE_EPISODE_CHECK.json').read_text())['completed_candidate_summaries'],
        dependencies='External pinned IsaacLab/IsaacSim/Torch, full-racket USD/mesh and walking dependencies from the previously verified runtime companion remain required. Preserve or explicitly audit absolute source-path relocation. This is not a standalone simulator or a newly executed physics validation.',
        native_data_unchanged=True, whole_course_finished=False, cold128_confirmation_finished=False,
        formal_cases_read=False, formal_acceptance=False)
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for name, path in files.items():
            zipped.write(path, name)
        zipped.writestr('MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None and len(zipped.namelist()) == len(hashes)+1
        for name, record in hashes.items():
            data = zipped.read(name)
            assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), archive=str(archive), bytes=archive.stat().st_size,
        sha256=sha(archive), members=len(hashes)+1, payloads_SHA_verified=len(hashes), all_CRC_passed=True, formal_acceptance=False)
    (HERE/'ARCHIVE_CHECK.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
