"""Archive the completed candidate's original128 records, policy and video proof."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent/'unitree_smash'
HERE = Path(__file__).resolve().parent
NATIVE = MAIN/'reports/G07_full_racket_v4/contact_aim7_sequential_n128_01'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive = HERE/'G07_contact_aim006_completed_candidate_TRAIN128_weights_and_video_evidence_20261027.zip'
    assert not archive.exists()
    plan = json.loads((HERE.parent/'contact_aim_search_video_20261027/PLAN.json').read_text()) if not (HERE/'PLAN.json').exists() else json.loads((HERE/'PLAN.json').read_text())
    assert all(sha(Path(path)) == digest for path, digest in plan['source_sha256'].items())
    files = {}
    for name in ['source_snapshot.zip','source_hashes.json','dirty.patch','asset_dependencies.json',
                 'INPUTS.json','geometry.json','geometry.npz','episodes_002.json','contacts_002.json',
                 'all_first_episodes_002.npz','public_and_actions_002.npz','initial_ground_states_002.json','causal_plans_002.json']:
        files['native/'+name] = NATIVE/name
    for path in [HERE/'PLAN.json',HERE/'ACTUAL_PARTIAL_JOB_SNAPSHOT.json',HERE/'NATIVE_EPISODE_CHECK.json',
                 HERE/'DECODE_AND_NATIVE_INPUT_CHECK.json',HERE/'ROOT_VISUAL_REVIEW.json',HERE/'displayed_candidate.pt',
                 HERE/'verify_video.py',Path(__file__),HERE/'REPORT.md',HERE/'snapshot/cases.json']:
        files['evidence/'+str(path.relative_to(HERE))] = path
    for name in ['prepare.py','prepare_first_attempt.py']:
        files['evidence/'+name] = HERE.parent/'contact_aim_search_video_20261027'/name
    for i in range(3):
        files[f'display/case{i}.npz'] = HERE/f'snapshot/case{i}.npz'
    for path in sorted((HERE/'decoded_samples').glob('*.jpg'))+sorted((HERE/'review').glob('*.jpg')):
        files['evidence/'+str(path.relative_to(HERE))] = path
    stem = 'Current_effect_contact_aim006_search_complete_and_failures_20261027'
    files['video/'+stem+'.json'] = ROOT/'reports/G07_full_racket_v4/video'/(stem+'.json')
    files['video/'+stem+'.preview.json'] = ROOT/'reports/G07_full_racket_v4/video'/(stem+'.preview.json')
    for name in ['render_full_racket_actual_landing_labels_v4.py','render_full_racket_r3_effect.py',
                 'render_current_v4_progress.py','native_video_framing_v4.py']:
        files['render/scripts/'+name] = ROOT/'scripts'/name
    for name in ['court_net_v4.json','g07_full_racket_eval_v4_r3.json']:
        files['render/configs/'+name] = MAIN/'configs'/name
    hashes = {name:dict(bytes=path.stat().st_size,sha256=sha(path)) for name,path in files.items()}
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), payloads=hashes,
        scope='Completed native candidate2 original all128 first episodes, source policy and chosen three-episode video evidence. Overall producer job was still active at snapshot.',
        original_native_episode_index=2, source_policy_file='evidence/displayed_candidate.pt',
        candidate_original128_summary=json.loads((HERE/'NATIVE_EPISODE_CHECK.json').read_text())['candidate_original128_summary'],
        dependencies='External IsaacLab/IsaacSim/Torch and prior full-racket runtime companion remain required. Native source_snapshot.zip supplies the original program; this package is not a standalone simulator installation.',
        native_data_unchanged=True, cold128_confirmation_pending_at_snapshot=True,
        formal_cases_read=False, formal_acceptance=False)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as zipped:
        for name,path in files.items():
            zipped.write(path,name)
        zipped.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert len(zipped.namelist()) == len(hashes)+1
        for name,record in hashes.items():
            data=zipped.read(name)
            assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),archive=str(archive),bytes=archive.stat().st_size,
        sha256=sha(archive),members=len(hashes)+1,payloads_SHA_verified=len(hashes),all_CRC_passed=True,formal_acceptance=False)
    (HERE/'ARCHIVE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
