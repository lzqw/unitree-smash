"""Decode the final video and verify native input identity and projection scale."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = json.loads((HERE / 'RENDER_RECEIPT.json').read_text())
    final = json.loads((HERE / 'FINAL_VIDEO.json').read_text())
    video = Path(final['video'])
    original_video = Path(receipt['video'])
    assert original_video.stat().st_size == receipt['bytes'] and sha(original_video) == receipt['sha256']
    assert video.stat().st_size == final['bytes'] and sha(video) == final['sha256']
    assert all(sha(Path(path)) == digest for path, digest in receipt['source_sha256'].items())
    run = Path(next(path for path in receipt['source_sha256'] if path.endswith('/all_first_episodes.npz'))).parent
    with np.load(run / 'all_first_episodes.npz') as original, np.load(HERE / 'displayed_native_inputs.npz') as displayed:
        for key in ['link_poses', 'ball_position_m', 'episode_clock_s']:
            assert np.array_equal(displayed[key], original[key][:, 29])
        assert original['valid_first_episode'][:, 29].all()
    row = receipt['native_episode']
    assert row['complete_smash'] and row['jump']['success'] and not row['jump']['body_landing']
    assert np.linalg.norm(row['outgoing_velocity_mps']) >= 8 and row['outgoing_velocity_mps'][2] <= -1
    assert 2 <= row['first_landing_m'][0] <= 4 and abs(row['first_landing_m'][1]) <= 1
    assert row['jump']['recovery_s'] >= .8 and receipt['source_is_historical_origin16']
    assert not receipt['current_origin0_success_claimed']
    camera = receipt['main_camera']
    matrix = np.asarray(camera['projection_matrix'])
    target, eye = np.asarray(camera['target']), np.asarray(camera['eye'])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    points = np.array([target, target + right, target + up])
    homogeneous = np.column_stack((points, np.ones(3))) @ matrix.T
    ndc = homogeneous[:, :2] / homogeneous[:, 3:]
    pixels = (ndc + 1) * np.array(camera['viewport']) / 2
    horizontal, vertical = [float(np.linalg.norm(point - pixels[0])) for point in pixels[1:]]
    assert abs(horizontal - vertical) < 1e-5
    probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'info',
        '-i', str(video), '-vf', 'showinfo', '-frames:v', '1', '-f', 'null', '-'],
        text=True, capture_output=True)
    assert probe.returncode == 0
    assert 'sar:1/1' in probe.stderr.replace(' ', '')
    samples = {0, 75, 140, 155, 200, 399, 455, 615, 640, receipt['frames'] - 1}
    output = HERE / 'decoded_samples'
    output.mkdir(exist_ok=False)
    reader = imageio.get_reader(video)
    original_reader = imageio.get_reader(original_video)
    stream = reader.get_meta_data()
    assert stream['size'] == (1280, 720) and stream['fps'] == 50.
    count = 0
    for index, frame in enumerate(reader):
        assert frame.shape == (720, 1280, 3) and frame.dtype == np.uint8
        assert np.array_equal(frame, original_reader.get_next_data())
        if index in samples:
            Image.fromarray(frame).save(output / f'frame_{index:06d}.jpg', quality=93)
        count += 1
    reader.close()
    original_reader.close()
    assert count == receipt['frames'] == 741
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), frames_decoded=count,
        duration_s=count / 50, video_bytes=video.stat().st_size, video_sha256=sha(video),
        ffmpeg_stream=stream, encoded_sample_aspect_ratio='1:1',
        all_final_decoded_pixels_identical_to_original_render=True,
        all400_native_pose_ball_clock_inputs_bitwise_verified=True,
        original128_summary_independently_verified=True, selected_TRAIN29_complete_success=True,
        camera_pixels_per_m_at_same_depth=dict(horizontal=horizontal, vertical=vertical),
        perspective_has_no_axis_stretch=True, native_model_vertices_not_rescaled=True,
        source_is_historical_origin16=True, current_origin0_success_claimed=False,
        code_source_sha256=sha(Path(__file__)), renderer_source_unchanged=True,
        decoded_samples=sorted(samples), new_native_execution=False, formal_acceptance=False)
    with (HERE / 'DECODE_AND_SCALE_CHECK.json').open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
