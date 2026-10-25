"""Fully decode the actual TRAIN video and save review samples from its bytes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    video = ROOT/'reports/G07_full_racket_v4/video/Current_training_centered_first_complete_and_failures_20261026.mp4'
    meta = json.loads(video.with_suffix('.json').read_text())
    output = HERE/'decoded_samples'
    assert not output.exists()
    output.mkdir()
    wanted = {25,100,209,450,612,687,1110,1716,1804}
    frames = 0
    samples = []
    with imageio.get_reader(video) as reader:
        assert abs(reader.get_meta_data()['fps']-50.) < 1e-9
        for index, frame in enumerate(reader):
            assert frame.shape == (720,1280,3) and frame.dtype == np.uint8
            if index in wanted:
                path = output/f'frame_{index:06d}.jpg'
                Image.fromarray(frame).save(path,quality=94)
                samples.append(dict(frame_index=index,video_time_s=index/50.,
                    path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            frames += 1
    assert frames == meta['frames'] == 1854 and len(samples) == len(wanted)
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    assert digest == meta['sha256']
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),video=str(video.relative_to(ROOT)),
        sha256=digest,bytes=video.stat().st_size,decoded_frames=frames,all_frames_decoded=True,
        duration_s=frames/50.,fps=50,resolution=[1280,720],samples=samples,
        human_visual_review_is_separate=True,formal_acceptance=False,goal_complete=False)
    video.with_name(video.stem+'_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    (HERE/'DECODED_VISUAL_SAMPLES.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key]for key in ['decoded_frames','duration_s','sha256','bytes']}))


if __name__ == '__main__':
    main()
