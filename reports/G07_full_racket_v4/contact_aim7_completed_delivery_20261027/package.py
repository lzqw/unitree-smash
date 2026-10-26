"""Archive completed original-origin Aim7 search, cold128 and actual best weights."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
REPORTS=ROOT/'reports/G07_full_racket_v4'
SEARCH=REPORTS/'contact_aim7_sequential_n128_01'
FROZEN=REPORTS/'contact_aim7_best_fixed_train128_01'
OWNER=REPORTS/'contact_aim_after_body48_queue01'
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_contact_aim_v4 import make_rally_controller
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases,summarize_training


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive=HERE/'G07_contact_aim7_completed_search_cold_TRAIN128_origin16_20261027.zip'
    assert not archive.exists()
    spec=importlib.util.spec_from_file_location('aim_archive_guard',REPORTS/'residual_resume24_queue01/run.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    assert not core.live(1827656,'contact_aim_after_body48_queue01/run.py')
    assert read(OWNER/'state.json')['state']=='COMPLETED_CONTACT_AIM_SEARCH_AND_COLD128'
    core.verify_job(SEARCH,716800);core.verify_job(FROZEN,102400)
    source=SEARCH/'best_candidate.pt';summary=core.verify_evaluation(FROZEN,source)
    assert summary['complete']==3 and summary['airborne_contacts']==60 and summary['conditional_recovery_count']==26
    protocol=read(ROOT/'configs/g07_full_racket_eval_v4_r3.json');protocol['_root']=str(ROOT)
    cases,indices=training_cases(ROOT,protocol)
    history=read(SEARCH/'metrics.json')['history'];assert len(history)==7
    inputs=read(SEARCH/'INPUTS.json');candidate_summaries=[]
    for group,entry in enumerate(history):
        rows=read(SEARCH/f'episodes_{group:03d}.json');initial=read(SEARCH/f'initial_ground_states_{group:03d}.json')
        assert len(rows)==128
        for lane,row in enumerate(rows):
            assert row['env_id']==row['jump']['env_id']==lane and row['candidate_id']==group
            assert row['episode_index']==initial['episode_ids'][lane] and row['ground_init']
            assert row['source_training_index']==lane and row['case_id']==cases[lane]['case_id']
        measured=summarize_candidate(rows,protocol);assert measured==entry['summary']
        candidate_summaries.append(measured)
        with np.load(SEARCH/f'all_first_episodes_{group:03d}.npz') as arrays:
            assert all(np.isfinite(arrays[name]).all() for name in ['link_poses','ball_position_m','episode_clock_s'])
    policy=torch.load(source,map_location='cpu',weights_only=False)
    assert list(policy['aim_uv_m'])==[0.,.06] and policy['actual_native_search']['candidate_id']==2
    default,limits=[torch.tensor(inputs[key],dtype=torch.float32) for key in ['default_joint_position_rad','joint_limits_rad']]
    with np.load(FROZEN/'public_and_actions_first2.npz') as arrays:
        observations,actions,valid=[arrays[key].copy() for key in ['observations','actions','valid_first_episode']]
    controller=make_rally_controller(policy,2,'cpu');maximum=0.
    with torch.inference_mode():
        for public,actual,keep in zip(observations,actions,valid):
            command=controller.action({'policy':torch.from_numpy(public)},default,limits).numpy()
            if keep.any():maximum=max(maximum,float(abs(command[keep]-actual[keep]).max()))
    assert maximum<=5e-4
    audit=dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_search_rows=896,actual_cold_rows=128,
        search_candidate_summaries=candidate_summaries,cold_original128_summary=summary,
        actual_valid_cold_public_rows=int(valid.sum()),CPU_vs_actual_cold_GPU_max_action_error=maximum,
        original_action_tolerance=5e-4,actual_best_policy_sha256=sha(source),all_native_pose_streams_finite=True,
        search5_complete_not_inherited_as_cold3=True,original_origin_spacing_m=16.,
        known_limitation='Measured float32 global-origin drift remains in this original16m experiment. New origin0 variant has separate admission and must remeasure all policy performance.',
        formal_cases_read=False,formal_acceptance=False)
    (HERE/'ACTUAL_COMPLETED_DATA_CHECK.json').write_text(json.dumps(audit,indent=2)+'\n')
    payload={};duplicates=[]
    for label,run in [('search7',SEARCH),('cold128',FROZEN)]:
        for path in sorted(run.iterdir()):
            if not path.is_file() or path.suffix in ('.lock','.tmp'):continue
            if '.partial.' in path.name:
                final=path.with_name(path.name.replace('.partial.','.'))
                if final.exists() and sha(final)==sha(path):
                    duplicates.append(str(path));continue
            payload[label+'/'+path.name]=path
    for name in ['run.py','PLAN.json','ADMISSION_DECISION.json','ACTUAL_SELECTED_SOURCE_COMMAND_CHECK.json',
                 'SEARCH_RECEIPT.json','COLD_TRAIN128_RESULT.json','COLD128_EPISODES.csv','COLD128_EPISODES.json',
                 'BEST_MEASURED_SOURCE.json','state.json']:
        payload['coordinator/'+name]=OWNER/name
    for path in [Path(__file__),HERE/'ACTUAL_COMPLETED_DATA_CHECK.json']:
        payload['delivery/'+path.name]=path
    for name in ['full_racket_contact_aim_v4.py','full_racket_body_residual_v4.py','full_racket_prepose_branch_v4.py']:
        path=ROOT/'quadsmash/control'/name
        if path.exists() and name!='full_racket_prepose_branch_v4.py':payload['program/quadsmash/control/'+name]=path
    for name in ['search_full_racket_contact_aim_v4.py','evaluate_full_racket_contact_aim_train_v4.py']:
        payload['program/scripts/'+name]=ROOT/'scripts'/name
    for name in ['g07_full_racket_eval_v4_r3.json','court_net_v4.json','shuttle_physics_v2.json']:
        payload['program/configs/'+name]=ROOT/'configs'/name
    for name in ['REPORT.md','PLAN.json','RESULT.json','FLOAT32_ORIGIN_COMPARISON.json',
                 'analyze.py','compare_origin_rounding.py','plot.py','MEASURED_DIRECTION_AND_ORIGIN_ERRORS.png']:
        payload['numerical_diagnosis/'+name]=REPORTS/'aim006_contact_direction_diagnostic_20261027'/name
    hashes={name:dict(bytes=path.stat().st_size,sha256=sha(path)) for name,path in payload.items()}
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),payloads=hashes,
        source_scope='Completed seven-way original-origin16 search plus one independent cold128;896+128 real first episodes, not formal A/B.',
        best_policy_file='search7/best_candidate.pt',best_policy_sha256=sha(source),
        search_original128_complete_counts=[item['complete'] for item in candidate_summaries],cold_original128_summary=summary,
        omitted_byte_identical_partial_files=duplicates,
        runtime_dependencies='Original native source snapshots included. Full racket mesh/USD runtime companion and external pinned IsaacLab/IsaacSim/Torch still required. Not a standalone simulator installation.',
        known_numeric_issue=audit['known_limitation'],formal_cases_read=False,formal_acceptance=False)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as zipped:
        for name,path in payload.items():zipped.write(path,name)
        zipped.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None and len(zipped.namelist())==len(payload)+1
        for name,record in hashes.items():
            content=zipped.read(name)
            assert len(content)==record['bytes'] and hashlib.sha256(content).hexdigest()==record['sha256']
    check=dict(checked_at=datetime.now(timezone.utc).isoformat(),archive=str(archive),bytes=archive.stat().st_size,
        sha256=sha(archive),members=len(payload)+1,payloads_SHA_verified=len(payload),all_CRC_passed=True,formal_acceptance=False)
    (HERE/'ARCHIVE_CHECK.json').write_text(json.dumps(check,indent=2)+'\n');print(json.dumps(check),flush=True)


if __name__=='__main__':main()
