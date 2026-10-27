"""Wait for the confirmed owner, then seal completed swing9 and independent128."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

MAIN=Path(__file__).resolve().parents[3]
ROOT=MAIN.parent/'unitree_smash_contact_recovery'
HERE=Path(__file__).resolve().parent
REPORTS=HERE.parent
SEARCH=REPORTS/'measured_swing9_n128_01'
COLD=REPORTS/'measured_swing9_best_cold128_01'
OWNER=REPORTS/'measured_swing_queue01'
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases,summarize_training
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from scripts.initialize_centered_policy_v4 import same_tree


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    with path.open('x') as stream:stream.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    owner=Path('/proc/1951183')
    if owner.exists():
        assert str(OWNER/'run.py').encode() in (owner/'cmdline').read_bytes()
        assert (owner/'stat').read_text().rsplit(')',1)[1].split()[19]=='69568587'
        print(json.dumps(dict(waiting_for_confirmed_owner=1951183,CPU_only=True)),flush=True)
        while owner.exists():
            assert str(OWNER/'run.py').encode() in (owner/'cmdline').read_bytes()
            assert (owner/'stat').read_text().rsplit(')',1)[1].split()[19]=='69568587'
            time.sleep(15)
    assert read(OWNER/'state.json')['state']=='COMPLETED_MEASURED_SWING9_AND_INDEPENDENT_COLD128'
    protocol_path=ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol=read(protocol_path);protocol['_root']=str(ROOT)
    cases,indices=training_cases(ROOT,protocol)
    assert indices==list(range(128))
    source_path=MAIN/'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/best_candidate.pt'
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    search_metrics=read(SEARCH/'metrics.json')
    assert len(search_metrics['history'])==9
    for entry in search_metrics['history']:
        rows=read(SEARCH/f"episodes_{entry['candidate_id']:03d}.json")
        actual=summarize_candidate(rows,protocol)
        assert all(actual[k]==entry['summary'][k] for k in actual)
        assert actual['episodes']==128 and actual['numerical_failures']==0
        policy=torch.load(SEARCH/f"candidate_{entry['candidate_id']:03d}.pt",map_location='cpu',weights_only=False)
        assert same_tree(policy['source_public_policy'],source)
        assert torch.equal(policy['parameters'],source['parameters'])
    selected=max(search_metrics['history'],key=lambda x:tuple(x['summary']['rank']))
    assert selected['candidate_id']==0
    assert sha(SEARCH/'best_candidate.pt')==sha(SEARCH/'candidate_000.pt')
    cold_summary=summarize_training(read(COLD/'episodes.json'),cases,indices,protocol)
    assert all(cold_summary[k]==read(COLD/'metrics.json')['history'][0]['summary'][k]
               for k in ['complete','airborne_contacts','conditional_recovery_count','numerical_failures'])
    initial=read(COLD/'initial_ground_states_000.json')
    assert initial['all_clocks_zero'] and initial['episode_ids']==[0]*128
    assert initial['actual_environment_origins_m']==[[0.,0.,0.]]*128
    hours=steps=0.;snapshot_counts={}
    for label,run,cap in [('search',SEARCH,921600),('cold128',COLD,102400)]:
        job=read(run/'job.json')
        assert job['status']=='SUCCEEDED' and job['exit_code']==0 and job['receipt_present'] and not job['timed_out']
        assert not Path('/proc',str(job['pid'])).exists() and not Path('/proc',str(job['child_pid'])).exists()
        assert 0<job['charged_env_steps']==read(run/'completed.json')['env_steps']<=cap
        if label=='search':assert job['charged_env_steps']==cap
        hours+=job['charged_gpu_hours'];steps+=job['charged_env_steps']
        inputs=read(run/'INPUTS.json')
        assert inputs['training_indices']==list(range(128)) and inputs['seed']==724201
        assert inputs['reset_noise']=='none' and inputs['origin_spacing_m']==0.
        assert inputs['protocol_sha256']==sha(protocol_path) and inputs['formal_cases_read'] is False
        for scope in ['source','execution_source']:
            hashes=read(run/(scope+'_hashes.json'))
            with zipfile.ZipFile(run/(scope+'_snapshot.zip')) as zipped:
                assert zipped.testzip() is None and set(zipped.namelist())==set(hashes)
                assert all(hashlib.sha256(zipped.read(n)).hexdigest()==h for n,h in hashes.items())
            if scope=='execution_source':assert len(hashes)==394 and all(sha(ROOT/n)==h for n,h in hashes.items())
            snapshot_counts[label+'/'+scope]=len(hashes)
    check=dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_search_episodes=1152,actual_cold_episodes=128,
        selected_candidate_id=0,selected_runtime_overrides={},search_history=search_metrics['history'],
        original128_independent_cold_summary=cold_summary,actual_combined_GPU0_hours=hours,
        actual_combined_control_env_steps=steps,actual_source_snapshot_counts=snapshot_counts,
        original_actors_and_normalizers_unchanged=True,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    save(HERE/'ACTUAL_COMPLETED_CHECK.json',check)
    payload={}
    for label,run in [('search',SEARCH),('cold128',COLD)]:
        for path in sorted(run.rglob('*')):
            if not path.is_file() or path.suffix in ['.tmp','.pyc','.lock'] or '__pycache__' in path.parts:continue
            payload[label+'/'+str(path.relative_to(run))]=path
    for name in ['run.py','evaluate_swing.py','PLAN.json','state.json']:
        payload['coordinator/'+name]=OWNER/name
    payload['source/original204.pt']=source_path
    for name in ['check_completed.py','COMPLETED_SEARCH_CHECK.json','REPORT.md']:
        payload['completed_search/'+name]=REPORTS/'measured_swing9_completed_20261027'/name
    for name in ['package_completed.py','replay_downloaded.py','REPRODUCTION.md','ACTUAL_COMPLETED_CHECK.json']:
        payload['delivery/'+name]=HERE/name
    records={};aliases={};content={}
    for name,path in payload.items():
        row=dict(bytes=path.stat().st_size,sha256=sha(path))
        if row['sha256'] in content:
            aliases[name]=content[row['sha256']]
        else:
            content[row['sha256']]=name;records[name]=dict(row,original_path=str(path))
    companion=read(REPORTS/'contact_recovery_cold128_delivery_20261027/UPLOAD_RECEIPT.json')
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),payloads=records,aliases=aliases,
        selected_policy='search/best_candidate.pt',selected_policy_sha256=sha(SEARCH/'best_candidate.pt'),
        original128_independent_cold_summary=cold_summary,
        controller_search_not_new_actor_training=True,original_actors_and_normalizers_unchanged=True,
        required_runtime_companion={k:companion[k] for k in ['asset','url','bytes','sha256']},
        excluded_live_or_future_runs='All early-lateral, windowed-swing, recovery-prior, air-countermotion runs and their outcomes.',
        fresh_native_replay_claimed=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    archive=HERE/'G07_measured_swing9_completed_search_and_independent128_20261027.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=4) as zipped:
        for name,row in records.items():
            path=payload[name];assert sha(path)==row['sha256'];zipped.write(path,name)
        zipped.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None and len(zipped.namelist())==len(records)+1
        assert all(len(zipped.read(name))==row['bytes'] and hashlib.sha256(zipped.read(name)).hexdigest()==row['sha256']
                   for name,row in records.items())
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),archive=str(archive),bytes=archive.stat().st_size,
        sha256=sha(archive),members=len(records)+1,unique_payloads_SHA_verified=len(records),aliases=len(aliases),
        all_CRC_passed=True,required_runtime_companion=manifest['required_runtime_companion'],
        actual_combined_GPU0_hours=hours,original128_independent_cold_summary=cold_summary,
        formal_acceptance=False,goal_complete=False)
    save(HERE/'ARCHIVE_CHECK.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
