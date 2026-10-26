"""Bundle selected actual training evidence; never claim a completed48 course."""
from datetime import datetime,timezone
import hashlib,json,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];MAIN=ROOT.parent/'unitree_smash';HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    directory=HERE/'delivery';directory.mkdir(exist_ok=False)
    output=directory/'G07_body48_learned_phase_selected_training_video_evidence_20261026.zip'
    review=read(HERE/'ROOT_VISUAL_REVIEW.json');assert review['approved_for_authorized_GitHub_upload']
    video=Path(review['video']);assert sha(video)==review['sha256']
    snapshot=HERE/'snapshot_caption_v2'
    files={}
    def add(path,name):
        assert path.is_file() and name not in files and not name.startswith('/') and '..'not in Path(name).parts
        files[name]=path
    for p in snapshot.rglob('*'):
        if p.is_file():add(p,'snapshot/'+str(p.relative_to(snapshot)))
    for p in [video,video.with_suffix('.json'),video.with_suffix('.preview.json')]:add(p,'video/'+p.name)
    for name in ['prepare.py','verify_video.py','PLAN.json','NATIVE_EPISODE_CHECK.json','CAPTION_REVISION.json',
        'DECODE_AND_NATIVE_INPUT_CHECK.json','ROOT_VISUAL_REVIEW.json','ACTUAL_CONTACT_PHASE_MODEL13.json',
        'ACTUAL_TRACE_MANIFEST_SNAPSHOT.json','REPORT.md','build_archive.py']:
        add(HERE/name,'evidence/'+name)
    phase=HERE.parent/'body48_replay_walk_phase_check_20261026'
    for name in ['check.py','PLAN.json','RESULT.json','ACTUAL_JOINT_PROVENANCE.json']:
        add(phase/name,'walk_phase_check/'+name)
    for old_name in ['body48_learned_training_video_20261026','body48_learned_training_video_20261026_v2']:
        old=HERE.parent/old_name
        for name in ['prepare.py','PLAN.json','FAILED_PREPARATION.json','prepare.log']:
            add(old/name,'preserved_CPU_preparation_failures/'+old_name+'/'+name)
    for name in ['render_full_racket_latest_effect_v4.py','render_full_racket_r3_effect.py',
        'render_current_v4_progress.py','native_video_framing_v4.py']:
        add(ROOT/'scripts'/name,'render_source/scripts/'+name)
    protocol=read(MAIN/'configs/g07_full_racket_eval_v4_r3.json')
    for name in set(protocol['physical_configs'])|{'configs/g07_full_racket_eval_v4_r3.json'}:
        add(MAIN/name,'render_source/'+name)
    base=MAIN/'reports/G07_full_racket_v4/body_strike_grid16_n2048_01/best_candidate.pt'
    add(base,'source_policy/grid8_teacher_best_candidate.pt')
    diagnostic=read(HERE/'ACTUAL_CONTACT_PHASE_MODEL13.json');model=Path(diagnostic['source'])
    assert sha(model)==diagnostic['sha256'] and diagnostic['canonical_restart']is False
    add(model,'noncanonical_diagnostic/after_update13_model_12.pt')
    for p in sorted((HERE/'decoded_samples').glob('*.jpg')):add(p,'decoded_samples/'+p.name)
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='One learned-phase selected complete training episode plus two earlier warmup failures, source evidence and actual noncanonical13 diagnostic. Not the completed48 course, a frozen performance score, or a full simulator installation.',
        video_sha256=sha(video),source_teacher_sha256=sha(base),
        payloads={name:dict(bytes=p.stat().st_size,sha256=sha(p))for name,p in sorted(files.items())},
        required_full_simulation_companion=dict(name='G07_placement075_stage8_to40_completed_native_and_frozen_TRAIN128_20261026.zip',
            sha256='44528da938ad422ed6a5515a32c65df3b763421288d2b4ff6f9236ab36c517bb'),
        native_episode_identities=[[1824,3],[23,0],[58,0]],all_source_actions_and_physical_records_preserved=True,
        formal_cases_included=False,formal_acceptance=False,goal_complete=False)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED)as archive:
        for name,p in sorted(files.items()):archive.write(p,name)
        archive.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(output)as archive:
        assert archive.testzip()is None
        stored=json.loads(archive.read('MANIFEST.json'))
        assert len(archive.namelist())==len(stored['payloads'])+1
        for name,item in stored['payloads'].items():
            payload=archive.read(name);assert len(payload)==item['bytes'] and hashlib.sha256(payload).hexdigest()==item['sha256']
    receipt=dict(created_at=datetime.now(timezone.utc).isoformat(),archive=str(output),bytes=output.stat().st_size,
        sha256=sha(output),members=len(files)+1,payload_sha256_verified=len(files),all_CRC_passed=True,
        selected_native_traces_only=True,complete_simulator_installation=False,formal_acceptance=False)
    (HERE/'ARCHIVE_CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
