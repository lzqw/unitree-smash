#!/usr/bin/env python3
"""Assemble measured reference/latest fixed cases, with explicit population scores."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new presentation filename')
    source_path = args.input/'SOURCES.json'
    sources = json.loads(source_path.read_text())
    expected = [('reference', 0), ('reference', 1), ('latest', 0), ('latest', 1)]
    if [(s['kind'], s['case']) for s in sources['sources']] != expected:
        raise ValueError('Both preselected fixed cases must be shown for each policy')
    clips = []
    for source in sources['sources']:
        for name, sha in source['raw_files'].items():
            if digest(Path(name)) != sha:
                raise ValueError('Raw evaluation changed while preparing the video')
        derived = Path(source['derived'])
        video = json.loads((derived/'video.json').read_text())
        if 'llvmpipe' not in video['renderer'] or video['fps'] != 50:
            raise ValueError('Expected measured50Hz software rendering')
        clips.append(dict(source, video=video, clip=str(derived.with_suffix('.mp4'))))
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title_font = ImageFont.truetype(font_path, 46)
    text_font = ImageFont.truetype(font_path, 32)
    small_font = ImageFont.truetype(font_path, 23)

    def card(name, title, lines):
        frame = Image.new('RGB', (1280, 720), (18, 28, 43))
        draw = ImageDraw.Draw(frame)
        draw.rectangle((48, 135, 1232, 139), fill=(73, 187, 168))
        draw.text((48, 55), title, font=title_font, fill='white')
        for index, line in enumerate(lines):
            draw.text((48, 195+index*83), line, font=text_font, fill=(225, 231, 241))
        draw.text((48, 653), '实测仿真轨迹 · 完整负载 · 中心1.524 / 侧1.55m实体研究网',
                  font=small_font, fill=(166, 193, 214))
        frame.save(args.input/(name+'.png'))
        output = args.input/(name+'.mp4')
        with imageio.get_writer(str(output), fps=50, codec='libx264', quality=8,
                               macro_block_size=8, output_params=['-movflags', '+faststart']) as writer:
            for _ in range(150):
                writer.append_data(np.asarray(frame))
        return output

    def scores(source):
        rows = json.loads((Path(source['source'])/'episodes.json').read_text())
        count = len(rows)
        air = sum(r['airborne_contact'] for r in rows)
        full = sum(r['complete_smash'] and not r['robot_net_contact'] and r['ground_init'] for r in rows)
        recovery = sum(r['airborne_contact'] and r['recovery_success'] and not r['robot_net_contact'] for r in rows)
        return dict(episodes=count, complete=full, airborne=air, conditional_recovery=recovery,
                    robot_net=sum(r['robot_net_contact'] for r in rows),
                    numerical_failures=sum(r['numerical_failures'] for r in rows))

    reference, latest = scores(clips[0]), scores(clips[2])
    def score_lines(stats):
        return [f"完整扣杀并恢复 {stats['complete']}/{stats['episodes']}（{100*stats['complete']/stats['episodes']:g}%）",
                f"空中触球 {stats['airborne']}/{stats['episodes']}",
                f"条件恢复 {stats['conditional_recovery']}/{stats['airborne']}（{100*stats['conditional_recovery']/max(1,stats['airborne']):.2f}%）"]
    introduction = card('reference_title', '当前参考策略 · 正式 B 组',
                        score_lines(reference)+['预先固定案例0 / 1 · 正常速度 · 全组未达标'])
    latest_title = card('latest_title', '最新速度反馈试验 · 独立扰动组',
                       score_lines(latest)+['固定案例0 / 1 · 正常与慢放 · 未优于参考'])
    ending = card('ending_title', '当前状态：G07 尚未通过',
                  ['主要瓶颈：碰网、首次落点和稳定恢复',
                   '最新试验的成功与失败均保留在统计中',
                   '代码、权重与全部原始数据已保存至GitHub'])
    order = [introduction, Path(clips[0]['clip']), Path(clips[1]['clip']), latest_title,
             Path(clips[2]['clip']), Path(clips[3]['clip']), ending]
    listing = args.input/'concat.txt'
    listing.write_text(''.join("file '"+str(p.resolve()).replace("'", "'\\''")+"'\n" for p in order))
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-f', 'concat',
                             '-safe', '0', '-i', str(listing), '-c', 'copy', '-movflags', '+faststart',
                             str(args.output)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    frames, duration = imageio_ffmpeg.count_frames_and_secs(str(args.output))
    expected_frames = 450+sum(clip['video']['frames'] for clip in clips)
    if frames != expected_frames or abs(duration-expected_frames/50)>.05:
        raise ValueError(f'Decoded length differs: {frames}/{expected_frames}, {duration}s')
    metadata = dict(kind='Current measured effect: reference policy and latest experiment',
                    created_at=datetime.now(timezone.utc).isoformat(), path=str(args.output),
                    bytes=args.output.stat().st_size, sha256=digest(args.output),
                    frames=frames, fps=50, duration_s=duration, resolution=[1280, 720],
                    reference_scores=reference, latest_scores=latest, clips=clips,
                    source_manifest_sha256=digest(source_path), source_evaluations_unchanged=True,
                    renderer='CPU llvmpipe; original measured50Hz link/ball poses',
                    new_simulation=False, new_policy_training=False, gpu_hours_for_render=0,
                    formal_acceptance=False,
                    selection='Cases0 and1 recorded before outcomes, all4 included; latest candidate is not better than reference',
                    segments=[str(path) for path in order])
    args.output.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key:metadata[key] for key in ['path', 'bytes', 'sha256', 'frames', 'duration_s']}), flush=True)


if __name__ == '__main__':
    main()
