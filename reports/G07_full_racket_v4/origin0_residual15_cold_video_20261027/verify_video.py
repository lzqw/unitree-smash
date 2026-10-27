"""Verify displayed residual15 episodes and decode every actual rendered frame."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent / 'unitree_smash'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(MAIN))
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from scripts.initialize_centered_policy_v4 import same_tree
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE / 'DECODE_AND_NATIVE_INPUT_CHECK.json').exists()
    video = ROOT / 'reports/G07_full_racket_v4/video/Current_effect_origin0_residual15_fixed_cold128_20261027.mp4'
    snapshot = HERE / 'snapshot'
    manifest, metadata = read(snapshot / 'cases.json'), read(video.with_suffix('.json'))
    native = MAIN / 'reports/G07_full_racket_v4/origin0_residual_fresh15_fixed_train128_01'
    protocol = read(MAIN / 'configs/g07_full_racket_eval_v4_r3_origin0.json')
    protocol['_root'] = str(MAIN)
    cases, indices = training_cases(MAIN, protocol)
    rows = read(native / 'episodes.json')
    assert summarize_training(rows, cases, indices, protocol) == manifest['completed_cold_summary']
    assert same_tree(manifest['cases'], metadata['cases'])
    assert {69, 112, 0}.issubset({c['env_id'] for c in manifest['cases']})
    assert not metadata['renderer_uses_gpu'] and not metadata['interpolated_or_generated_motion']
    assert 'llvmpipe' in metadata['renderer'].lower() and sha(video) == metadata['sha256']
    events = read(native / 'contacts.json')
    with np.load(native / 'all_first_episodes.npz') as arrays:
        for case_id, case in enumerate(manifest['cases']):
            lane, ids = case['env_id'], np.asarray(case['selected_source_frames'])
            assert same_tree(case['row'], rows[lane]) and case['episode_index'] == 0
            actual = [e for e in events if e['env_id'] == lane and e['episode_index'] == 0
                      and e['time_s'] == case['row']['contact_time_s']]
            assert same_tree(case['contact'], actual[0] if actual else None)
            with np.load(snapshot / f'case{case_id}.npz') as recorded:
                assert recorded['valid_first_episode'].all()
                assert np.array_equal(recorded['link_poses'], arrays['link_poses'][ids, lane])
                assert np.array_equal(recorded['ball_position_m'], arrays['ball_position_m'][ids, lane])
                assert np.array_equal(recorded['episode_clock_s'], arrays['episode_clock_s'][ids, lane])
                assert (recorded['native_episode_index'] == 0).all()
            state = case['last_recorded_state']
            assert state['time_s'] == float(arrays['episode_clock_s'][ids[-1], lane])
            assert np.array_equal(np.column_stack((state['body_link_positions_m'],
                state['body_link_quaternions_wxyz'])), arrays['link_poses'][ids[-1], lane])
            assert all(v['all_fitted_points_in_viewport'] for v in metadata['camera_framing_checks'][case_id]['views'].values())
    for name, digest in metadata['sources'].items():
        assert sha(Path(name) if Path(name).is_absolute() else ROOT / name) == digest, name
    for name, digest in manifest['original_source_hashes'].items():
        assert sha(Path(name)) == digest, name
    native_hashes = read(native / 'source_hashes.json')
    for name, digest in native_hashes.items():
        assert sha(MAIN / name) == digest, name
    chosen = {0, metadata['frames'] - 1, metadata['frames'] // 2}
    for segment in metadata['segments']:
        if segment['name'] in ['exact_contact', 'first_landing', 'native_terminal']:
            chosen.add(segment['start_frame'] + segment['frames'] // 2)
    samples = HERE / 'decoded_samples'
    samples.mkdir()
    reader = imageio.get_reader(video)
    count = 0
    for frame in reader:
        assert frame.shape == (720, 1280, 3)
        if count in chosen:
            Image.fromarray(frame).save(samples / f'frame_{count:06d}.jpg', quality=94)
        count += 1
    fps = float(reader.get_meta_data()['fps'])
    reader.close()
    assert count == metadata['frames'] and fps == 50 and count / fps == metadata['duration_s']
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), video=str(video), sha256=sha(video),
        bytes=video.stat().st_size, frames=count, fps=fps, duration_s=count / fps,
        all_frames_decoded=True, all_displayed_poses_balls_clocks_bitwise_match_native=True,
        every_displayed_original_episode_row_and_contact_event_deep_matched=True,
        actual_native_source_hashes_checked=len(native_hashes), all_original_source_hashes_verified=True,
        camera_framing_all_views_passed=True, actual_all128_cold_summary=manifest['completed_cold_summary'],
        software_CPU_renderer=True, selected_decoded_frame_numbers=sorted(chosen),
        manual_visual_review_pending=True, new_native_execution=False, formal_cases_read=False, formal_acceptance=False)
    (HERE / 'DECODE_AND_NATIVE_INPUT_CHECK.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['sha256', 'bytes', 'frames', 'duration_s', 'all_frames_decoded']}), flush=True)


if __name__ == '__main__':
    main()
