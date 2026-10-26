"""Decode the delivered clip and match every displayed sample to native data."""
from datetime import datetime,timezone
import hashlib,json,sys
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[3];MAIN=ROOT.parent/'unitree_smash';HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(MAIN))
from scripts.initialize_centered_policy_v4 import same_tree
from quadsmash.evaluation_v4 import independent_episode

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    video=ROOT/'reports/G07_full_racket_v4/video/Current_training_body48_learned_complete_and_early_failures_20261026.mp4'
    snapshot=HERE/'snapshot_caption_v2';manifest=read(snapshot/'cases.json');metadata=read(video.with_suffix('.json'))
    assert not(HERE/'DECODE_AND_NATIVE_INPUT_CHECK.json').exists()
    assert sha(video)==metadata['sha256'] and not metadata['renderer_uses_gpu']
    assert 'llvmpipe' in metadata['renderer'].lower() and not metadata['interpolated_or_generated_motion']
    assert same_tree(manifest['cases'],metadata['cases'])
    assert [(c['env_id'],c['episode_index'])for c in manifest['cases']]==[(1824,3),(23,0),(58,0)]
    protocol=read(MAIN/'configs/g07_full_racket_eval_v4_r3.json');protocol['_root']=str(MAIN)
    comparisons=[]
    for i,case in enumerate(manifest['cases']):
        directory=snapshot/case['run'];stem=f"lane{case['env_id']:04d}_episode{case['episode_index']:06d}"
        original=read(directory/(stem+'.json'));assert same_tree(case['row'],original['raw_episode'])
        assert sha(directory/(stem+'.npz'))==original['data_sha256']==case['source_trace_data_sha256']
        with np.load(directory/(stem+'.npz'))as raw,np.load(snapshot/f'case{i}.npz')as displayed:
            ids=np.asarray(case['selected_source_frames']);assert np.array_equal(ids,np.arange(0,original['samples'],2))
            for key in ['link_poses','ball_position_m','episode_clock_s']:
                assert np.array_equal(displayed[key],raw[key][ids]),key
            assert displayed['valid_selected_episode'].all() and displayed['valid_first_episode'].all()
            assert (displayed['native_episode_index']==case['episode_index']).all()
            assert np.array_equal(displayed['learner_iteration'],raw['learner_iteration'][ids])
        result=independent_episode(case['row'],protocol);assert result['complete']==(i==0)
        if i==0:assert min(case['actor_iterations_encountered'])>=8
        else:assert max(case['actor_iterations_encountered'])<=7
        comparisons.append(dict(env_id=case['env_id'],episode_index=case['episode_index'],
            native_outcome=result,all_displayed_samples_bitwise_match=True))
    assert all(v['all_fitted_points_in_viewport']for c in metadata['camera_framing_checks']for v in c['views'].values())
    for name,digest in metadata['sources'].items():
        assert sha(Path(name)if Path(name).is_absolute()else ROOT/name)==digest,name
    for name,digest in manifest['original_source_hashes'].items():assert sha(Path(name))==digest,name
    selected={0,metadata['frames']-1}
    for name,case_id in [('exact_contact',0),('first_landing',0),('native_terminal',0),
                         ('exact_contact',1),('native_terminal',1),('native_terminal',2)]:
        segment=next(s for s in metadata['segments']if s['name']==name and s.get('case_id')==case_id)
        selected.add(segment['start_frame']+segment['frames']//2)
    out=HERE/'decoded_samples';out.mkdir(exist_ok=False)
    reader=imageio.get_reader(video);count=0
    for frame in reader:
        assert frame.shape==(720,1280,3)
        if count in selected:Image.fromarray(frame).save(out/f'frame_{count:06d}.jpg',quality=95)
        count+=1
    fps=float(reader.get_meta_data()['fps']);reader.close()
    assert count==metadata['frames'] and fps==50 and abs(count/fps-metadata['duration_s'])<1e-9
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),video=str(video),sha256=sha(video),
        bytes=video.stat().st_size,frames=count,fps=fps,duration_s=count/fps,
        all_frames_decoded=True,all_native_rows_and_displayed_motion_exact=True,
        native_comparisons=comparisons,actual_episode3_identity_preserved=True,
        one_learned_phase_complete_and_two_earlier_warmup_failures=True,
        all_source_hashes_and_viewport_checks_passed=True,CPU_renderer=True,
        decoded_frame_numbers=sorted(selected),manual_visual_review_pending=True,
        frozen_policy_performance=False,formal_acceptance=False,goal_complete=False)
    (HERE/'DECODE_AND_NATIVE_INPUT_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k]for k in ['sha256','bytes','frames','duration_s','all_frames_decoded']}))

if __name__=='__main__':main()
