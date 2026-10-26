"""Decode the actual delivered grid8 video and match native rows/poses."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys
import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
MAIN=ROOT.parent/'unitree_smash'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(MAIN))
from scripts.initialize_centered_policy_v4 import same_tree
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
video=ROOT/'reports/G07_full_racket_v4/video/Current_effect_body_grid8_cold128_complete_and_failures_20261026.mp4'
snapshot=MAIN/'reports/G07_full_racket_v4/body8_current_video_20261026/snapshot'
native=MAIN/'reports/G07_full_racket_v4/body_strike_grid16_fixed_train128_01'
manifest,metadata=read(snapshot/'cases.json'),read(video.with_suffix('.json'))
rows=read(native/'episodes.json')
assert [c['env_id'] for c in manifest['cases']]==[13,19,23,107,0,8,58]
assert same_tree(metadata['cases'],manifest['cases'])
assert not metadata['renderer_uses_gpu'] and not metadata['interpolated_or_generated_motion']
assert 'llvmpipe' in metadata['renderer'].lower()
assert sha(video)==metadata['sha256']
for i,c in enumerate(manifest['cases']):
    assert same_tree(c['row'],rows[c['env_id']]) and c['episode_index']==0
    with np.load(snapshot/f'case{i}.npz') as data:
        assert data['valid_first_episode'].all() and np.isfinite(data['link_poses']).all()
    assert 'nan' not in c['limitation'].lower()
    coverage=metadata['camera_framing_checks'][i]
    assert all(v['all_fitted_points_in_viewport'] for v in coverage['views'].values())
assert sum(c['row']['complete_smash'] for c in manifest['cases'])==1
for name,digest in metadata['sources'].items():assert sha(Path(name) if Path(name).is_absolute() else ROOT/name)==digest,name
for name,digest in manifest['original_source_hashes'].items():assert sha(MAIN/name)==digest,name
segments=metadata['segments']
selected={0,metadata['frames']-1}
for name in ['exact_contact','first_landing','native_terminal']:
    segment=next(s for s in segments if s['name']==name and s.get('case_id')==0)
    selected.add(segment['start_frame']+segment['frames']//2)
selected.update({metadata['frames']//3,metadata['frames']//2,2*metadata['frames']//3})
out=HERE/'decoded_samples';out.mkdir(exist_ok=False)
reader=imageio.get_reader(video);count=0
for frame in reader:
    assert frame.shape==(720,1280,3)
    if count in selected:Image.fromarray(frame).save(out/f'frame_{count:06d}.jpg',quality=94)
    count+=1
fps=float(reader.get_meta_data()['fps']);reader.close()
assert count==metadata['frames'] and fps==50 and count/fps==metadata['duration_s']
result=dict(checked_at=datetime.now(timezone.utc).isoformat(),video=str(video),sha256=sha(video),
    bytes=video.stat().st_size,frames=count,fps=fps,duration_s=count/fps,
    all_frames_decoded=True,all_seven_native_rows_deep_matched=True,all_original_source_hashes_verified=True,
    camera_framing_all_views_passed=True,software_CPU_renderer=True,
    missing_events_captioned_without_technical_nan=True,
    selected_decoded_frame_numbers=sorted(selected),
    posthoc_complete_TRAIN13_added_to_same_six_cases=True,
    native_summary=read(MAIN/'reports/G07_full_racket_v4/body_strike_grid16_queue01/COLD_TRAIN128_RESULT.json'),
    manual_visual_review_pending=True,formal_acceptance=False)
(HERE/'DECODE_AND_NATIVE_INPUT_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['sha256','bytes','frames','duration_s','all_frames_decoded']}))
