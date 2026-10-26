"""Decode the film and match every displayed pose to completed native candidate2."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent/'unitree_smash'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(MAIN))
from scripts.initialize_centered_policy_v4 import same_tree
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


video = ROOT/'reports/G07_full_racket_v4/video/Current_effect_contact_aim006_search_complete_and_failures_20261027.mp4'
snapshot = HERE/'snapshot'
manifest, metadata = read(snapshot/'cases.json'), read(video.with_suffix('.json'))
native = MAIN/'reports/G07_full_racket_v4/contact_aim7_sequential_n128_01'
rows = read(native/'episodes_002.json')
protocol = read(MAIN/'configs/g07_full_racket_eval_v4_r3.json'); protocol['_root'] = str(MAIN)
assert summarize_candidate(rows, protocol) == manifest['candidate_original128_summary']
assert same_tree(metadata['cases'], manifest['cases'])
assert [case['env_id'] for case in manifest['cases']] == [12, 13, 0]
assert not metadata['renderer_uses_gpu'] and not metadata['interpolated_or_generated_motion']
assert 'llvmpipe' in metadata['renderer'].lower() and sha(video) == metadata['sha256']
with np.load(native/'all_first_episodes_002.npz') as arrays:
    for i, case in enumerate(manifest['cases']):
        lane = case['env_id']; ids = np.asarray(case['selected_source_frames'])
        assert same_tree(case['row'], rows[lane]) and case['episode_index'] == rows[lane]['episode_index'] == 2
        with np.load(snapshot/f'case{i}.npz') as recorded:
            assert recorded['valid_first_episode'].all()
            assert np.array_equal(recorded['link_poses'], arrays['link_poses'][ids, lane])
            assert np.array_equal(recorded['ball_position_m'], arrays['ball_position_m'][ids, lane])
            assert np.array_equal(recorded['episode_clock_s'], arrays['episode_clock_s'][ids, lane])
            assert (recorded['native_episode_index'] == 2).all()
        assert all(view['all_fitted_points_in_viewport'] for view in metadata['camera_framing_checks'][i]['views'].values())
assert sum(case['row']['complete_smash'] for case in manifest['cases']) == 1
for name, digest in metadata['sources'].items():
    assert sha(Path(name) if Path(name).is_absolute() else ROOT/name) == digest, name
for name, digest in manifest['original_source_hashes'].items():
    assert sha(Path(name)) == digest, name
selected = {0, metadata['frames']-1, metadata['frames']//3, metadata['frames']//2, 2*metadata['frames']//3}
for name in ['exact_contact','first_landing','native_terminal']:
    segment = next(item for item in metadata['segments'] if item['name'] == name and item.get('case_id') == 0)
    selected.add(segment['start_frame']+segment['frames']//2)
out = HERE/'decoded_samples'; out.mkdir(exist_ok=False)
reader = imageio.get_reader(video); count = 0
for frame in reader:
    assert frame.shape == (720, 1280, 3)
    if count in selected:
        Image.fromarray(frame).save(out/f'frame_{count:06d}.jpg', quality=94)
    count += 1
fps = float(reader.get_meta_data()['fps']); reader.close()
assert count == metadata['frames'] and fps == 50 and count/fps == metadata['duration_s']
result = dict(checked_at=datetime.now(timezone.utc).isoformat(), video=str(video), sha256=sha(video),
    bytes=video.stat().st_size, frames=count, fps=fps, duration_s=count/fps, all_frames_decoded=True,
    all_displayed_poses_balls_and_clocks_bitwise_match_native=True, all_three_native_episode_rows_deep_matched=True,
    actual_episode_index2_preserved=True, all_original_source_hashes_verified=True,
    camera_framing_all_views_passed=True, software_CPU_renderer=True,
    selected_decoded_frame_numbers=sorted(selected), manual_visual_review_pending=True,
    actual_candidate_original128_summary=manifest['candidate_original128_summary'],
    cold_confirmation_pending_at_snapshot=True, formal_cases_read=False, formal_acceptance=False)
(HERE/'DECODE_AND_NATIVE_INPUT_CHECK.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({key:result[key] for key in ['sha256','bytes','frames','duration_s','all_frames_decoded']}), flush=True)
