"""Extract actual decoded video frames after the parent's full decode passed."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    video = ROOT/'reports/G07_full_racket_v4/video/Current_effect_centered_residual_base8_iter40_20261026.mp4'
    meta = read(video.with_suffix('.json'))
    verification_path = video.with_name(video.stem+'_verification.json')
    verified = read(verification_path)
    assert verified['all_frames_decoded'] and verified['frames_decoded'] == meta['frames'] == 4233
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    assert digest == verified['sha256'] == meta['sha256']
    assert [case['env_id'] for case in meta['cases']] == [19, 23, 107, 0, 8, 58]
    assert not any(case['row']['complete_smash'] for case in meta['cases'])
    assert meta['native_recorded_motion'] and not meta['interpolated_or_generated_motion']
    assert not meta['renderer_uses_gpu'] and 'llvmpipe' in meta['renderer'].lower()
    wanted = {25}
    for case in range(6):
        choices = [segment for segment in meta['segments'] if segment.get('case_id') == case]
        focus = next((segment for segment in choices if segment['name'] == 'exact_contact'), None)
        focus = focus or next(segment for segment in choices if segment['name'] == 'slow')
        wanted.add(focus['start_frame']+focus['frames']//2)
    closing = next(segment for segment in meta['segments'] if segment['name'] == 'result')
    wanted.add(closing['start_frame']+closing['frames']//2)
    output = HERE/'decoded_samples'
    assert not output.exists() and len(wanted) == 8
    output.mkdir()
    samples = []
    with imageio.get_reader(video) as reader:
        assert abs(reader.get_meta_data()['fps']-50.) < 1e-9
        for index in sorted(wanted):
            frame = reader.get_data(index)
            assert frame.shape == (720, 1280, 3) and frame.dtype == np.uint8
            path = output/f'frame_{index:06d}.jpg'
            Image.fromarray(frame).save(path, quality=94)
            samples.append(dict(frame_index=index, video_time_s=index/50., path=str(path.relative_to(ROOT)),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), video=str(video.relative_to(ROOT)), sha256=digest,
                  frames=4233, duration_s=84.66, parent_full_decode_verification=str(verification_path.relative_to(ROOT)),
                  parent_full_decode_verification_sha256=hashlib.sha256(verification_path.read_bytes()).hexdigest(),
                  all4233_frames_decoded_by_parent=True, samples=samples, event_previews=meta['previews'],
                  human_visual_review_is_separate=True, formal_acceptance=False, goal_complete=False)
    (HERE/'DECODED_VISUAL_SAMPLES.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
