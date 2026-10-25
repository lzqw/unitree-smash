"""Preserve the completed intermediate assessments and their final selection.

Only terminal16/24/32 data enters this package. Current075 training is excluded.
The native source diagnostics and original runtime are verified companions.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT/'reports/G07_full_racket_v4'
SELECTOR = REPORTS/'centered_intermediate_selection_queue01'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    output = REPORTS/'G07_centered_intermediate16_24_32_frozen_TRAIN128_and_final_selection_20261026_v2.zip'
    assert not output.exists() and not (HERE/'ARCHIVE_CHECK_V2.json').exists()
    assert not Path('/proc/1697096').exists()
    assert read(SELECTOR/'state.json')['state'] == 'COMPLETED_INTERMEDIATE_SELECTION'
    selection = read(SELECTOR/'RESULT.json')
    assert len(selection['all_candidates']) == 5
    assert {v['iteration'] for v in selection['all_candidates']} == {8,16,24,32,40}
    ranked = max(selection['all_candidates'],key=lambda e:(e['summary']['complete'],e['summary']['attack_outgoing'],
        e['summary']['conditional_recovery_count'],e['summary']['airborne_contacts'],-e['iteration']))
    assert selection['selected'] == ranked
    assert sha(Path(ranked['policy'])) == selection['selected_policy_sha256']
    spec = importlib.util.spec_from_file_location('preserved_terminal_selector',SELECTOR/'run.py')
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    paths,assessments,mandatory,omitted = [],[],[],{}
    for iteration in [16,24,32]:
        run = REPORTS/f'residual_ppo_centered_base8_diag_iter{iteration}_fixed_train128_01'
        policy = SELECTOR/f'exports/iter{iteration}/policy.pt'
        job = read(run/'job.json')
        assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0 and job['charged_env_steps'] == 102400
        assert job['receipt_present'] and read(run/'completed.json')['env_steps'] == 102400
        summary = selector.core.verify_evaluation(run,policy)
        entry = next(e for e in selection['all_candidates'] if e['iteration'] == iteration)
        receipt = read(SELECTOR/f'iter{iteration}_RESULT.json')
        assert receipt['iteration'] == iteration and receipt['policy_sha256'] == sha(policy)
        assert summary == entry['summary'] == receipt['independent_original_TRAIN128']
        assert summary['numerical_failures'] == 0
        assert read(run/'INPUTS.json')['reset_noise_seed'] == read(run/'metrics.json')['noise_seed'] == 724201
        with np.load(run/'all_first_episodes.npz',allow_pickle=False) as poses:
            dimensions = {key:list(poses[key].shape) for key in poses.files}
            assert dimensions == {'link_poses':[400,128,28,7],'ball_position_m':[400,128,3],
                'episode_clock_s':[400,128],'valid_first_episode':[400,128]}
            assert poses['valid_first_episode'].any(axis=0).all()
        with np.load(run/'public_and_actions_first2.npz',allow_pickle=False) as public:
            public_dimensions = {key:list(public[key].shape) for key in public.files}
            assert public_dimensions == {'observations':[800,2,794],'actions':[800,2,18],'valid_first_episode':[800,2]}
        paths += [p for p in run.rglob('*') if p.is_file() and '.partial' not in p.name]
        paths += [p for p in (SELECTOR/f'exports/iter{iteration}').rglob('*') if p.is_file()]
        mandatory += [str((run/name).relative_to(ROOT)) for name in
            ['episodes.json','contacts.json','all_first_episodes.npz','initial_states.json',
             'source_snapshot.zip','source_hashes.json','job.json','completed.json','INPUTS.json','metrics.json']]
        mandatory.append(str(policy.relative_to(ROOT)))
        for p in run.glob('*.partial.*'):
            omitted[str(p.relative_to(ROOT))] = dict(bytes=p.stat().st_size,sha256=sha(p),
                reason='Historical partial observation; the terminal episodes/poses in this package are authoritative. Original partial retained locally.')
        assessments.append(dict(iteration=iteration,public_policy_sha256=sha(policy),summary=summary,
            actual_GPU_hours=job['charged_gpu_hours'],actual_control_env_steps=job['charged_env_steps'],
            terminal_pose_array_dimensions=dimensions,public_and_action_array_dimensions=public_dimensions,
            public_input_scope='All128 native body/ball poses at50Hz until first-episode termination, with validity masks. Full794D public inputs and18D joint actions at100Hz only for first2 cases, also with validity masks.'))
    paths += [p for p in SELECTOR.iterdir() if p.is_file() and p.suffix in ['.json','.py','.csv','.log']]
    paths.append(Path(__file__))
    paths += [HERE/'CORRECTION_FROM_V1.json',HERE/'VALIDATION_NOTES.json']
    protocol = read(ROOT/'configs/g07_full_racket_eval_v4_r3.json')
    paths += [ROOT/'configs/g07_full_racket_eval_v4_r3.json',ROOT/protocol['training_suite']['path'],REPORTS/'MODEL_RULES.json']
    paths += [ROOT/name for name in protocol['physical_configs']]
    companions = []
    for folder,purpose in [
        ('centered40_fixed_video_review_20261026','Completed frozen8/40 original128 data and reviewed actual40 video.'),
        ('centered40_training_delivery_20261026','Completed source8-to40 TRAIN, all32 native diagnostic boundaries and two real training successes.'),
        ('centered_training_capture_video_20261026','Original native runtime USD layers, walking actor, standard PiPER URDF/meshes and canonical source8.')]:
        receipt = read(REPORTS/folder/'ARCHIVE_UPLOAD_RECEIPT.json')
        assert receipt['download_verified']
        companions.append({**{key:receipt[key] for key in ['url','bytes','sha256']},'purpose':purpose})
    members = {str(p.relative_to(ROOT)):p for p in sorted(set(paths)) if p.is_file()}
    assert set(mandatory) <= members.keys()
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Completed actual frozen intermediate16/24/32 on originalTRAIN128 and completed five-source selection. Current075 pilot excluded; no new formal A/B score.',
        original128_seed=724201,reset_noise='none',assessments=assessments,
        final_selection=selection,source_diagnostics_are_not_canonical_restart_files=True,
        supersedes=read(HERE/'CORRECTION_FROM_V1.json'),
        launch_time_runtime_custody='Each native run includes its own original source_snapshot.zip, source_hashes.json and dirty.patch. The current main trainer was deliberately revised only after actual selector exit.',
        required_companions=companions,omitted_partial_files=omitted,
        mandatory_members=mandatory,
        source_files={name:dict(bytes=p.stat().st_size,sha256=sha(p)) for name,p in members.items()},
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,path in members.items():
            archive.write(path,name,compress_type=zipfile.ZIP_STORED if path.suffix in ['.zip','.npz'] else zipfile.ZIP_DEFLATED)
        archive.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(members)+1
        for name,entry in manifest['source_files'].items():
            payload = archive.read(name)
            assert len(payload) == entry['bytes'] and hashlib.sha256(payload).hexdigest() == entry['sha256']
        assert set(mandatory) <= set(archive.namelist())
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),archive=str(output.relative_to(ROOT)),
        bytes=output.stat().st_size,sha256=sha(output),source_payload_members=len(members),total_members=len(members)+1,
        mandatory_native_members_verified=len(mandatory),all_member_CRC_and_source_payload_SHA_verified=True,
        all_three_original128_results_independently_recomputed=True,source_selection_sha256=sha(SELECTOR/'RESULT.json'),
        current075_training_excluded=True,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    with (HERE/'ARCHIVE_CHECK_V2.json').open('x') as file:
        json.dump(result,file,indent=2);file.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
