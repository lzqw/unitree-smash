#!/usr/bin/env python3
"""Render three frozen native training episodes, including a current failure."""
import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from render_full_racket_r3_effect import ROOT, Scene, sha
from render_current_v4_progress import camera
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    snapshot, output = args.snapshot.resolve(), args.output.resolve()
    if output.exists() and not args.preview_only:
        raise ValueError('Use a fresh output file')
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((snapshot / 'cases.json').read_text())
    cases = manifest['cases']
    assert len(cases) == 3
    assert all(not c['row']['complete_smash'] for c in cases)
    court = json.loads((ROOT / 'configs/court_net_v4.json').read_text())
    scenes = {}
    for case in cases:
        if case['run'] in scenes:
            continue
        if not scenes:
            scene = Scene(snapshot / case['run'], court)
        else:
            # Keep one software GL context, while using each run's own exported geometry.
            scene = copy.copy(next(iter(scenes.values())))
            scene.objects = []
            metadata = json.loads((snapshot / case['run'] / 'geometry.json').read_text())
            with np.load(snapshot / case['run'] / 'geometry.npz') as meshes:
                for item in metadata['meshes']:
                    color = (.065, .085, .11) if '/Go2/' in item['path'] else (.72, .76, .80)
                    if '/Piper/' in item['path'] and any(n in item['path'] for n in ['link2', 'link4', 'link6']):
                        color = (.06, .27, .50)
                    scene.objects.append((scene.geometry(meshes[item['key'] + '_vertices'],
                                                         meshes[item['key'] + '_faces']),
                                          item['body'], item['color'] or color))
        for name, w, h, eye, target, fov in [
            ('wide_side', 760, 480, [1.3, -6., 3.], [1.3, 0., 1.], 49),
            ('wide_top', 504, 480, [1.35, -.001, 5.5], [1.35, 0., 0.], 60),
        ]:
            scene.views[name] = (scene.ctx.simple_framebuffer((w, h), components=3),
                                 camera(eye, target, w / h, fov))
        scenes[case['run']] = scene
    data = [np.load(snapshot / f'case{i}.npz') for i in range(3)]
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title, text, small = [ImageFont.truetype(font, n) for n in [29, 23, 20]]
    headings = ['回合一｜空中回球与合格恢复已连上',
                '回合二｜命中目标，恢复仍按失败保留',
                '回合三｜当前新策略：触球后身体触地失败']
    subtitles = ['上一轮第1代 / 候选6 / TRAIN1｜独立实际回合',
                 '上一轮第2代 / 同参数候选6 / TRAIN1｜另一次独立执行',
                 '正在训练的新一轮 / 候选0 / TRAIN1｜本回合已终止']
    fps, frame_count, segments, previews = 50, 0, [], []

    def frame(case_id, index=None, *, slow=False, exact=False, terminal=False, landing_freeze=False):
        case = cases[case_id]
        row, event, recorded = case['row'], case['contact'], data[case_id]
        scene = scenes[case['run']]
        landing = trail = ball = None
        if exact:
            poses = np.asarray(event['body_link_poses_local'])
            ball, t = np.asarray(event['ball_position_m']), event['time_s']
        elif terminal:
            state = row['terminal_native_state']
            poses = np.column_stack((state['body_link_positions_m'], state['body_link_quaternions_wxyz']))
            t = state['time_s']
        else:
            t = float(recorded['episode_clock_s'][index])
            poses = recorded['link_poses'][index]
            if t >= 2:
                ball = recorded['ball_position_m'][index]
                chosen = np.arange(max(0, index - 25), index + 1)
                mask = recorded['episode_clock_s'][chosen] >= 2
                if row['ground_seen']:
                    mask &= recorded['episode_clock_s'][chosen] <= row['first_landing_time_s']
                trail = recorded['ball_position_m'][chosen[mask]]
        if row['ground_seen'] and (t >= row['first_landing_time_s'] or landing_freeze):
            ball = None
            landing = np.asarray(row['first_landing_m'])
        wide = (case_id < 2 and 2.84 <= t <= row['first_landing_time_s'] + .3) or landing_freeze
        views = ['wide_side', 'wide_top'] if wide else ['side', 'top']
        picture = Image.new('RGB', (1280, 720), (18, 28, 43))
        for view, left in zip(views, [0, 776]):
            pane = scene.render(poses, view, ball=ball, trail=trail, landing=landing)
            draw = ImageDraw.Draw(pane)
            if wide:
                x0, x1 = court['target_x_limits_m']
                y = court['target_y_half_width_m']
                corners = [scene.project(p, view) for p in
                           [(x0, -y, .002), (x1, -y, .002), (x1, y, .002), (x0, y, .002)]]
                if all(p is not None for p in corners):
                    draw.line(corners + [corners[0]], fill=(100, 245, 165), width=2)
                    p = scene.project([(x0 + x1) / 2, 0., .002], view)
                    if p:
                        draw.text((p[0] - 45, p[1] - 10), '原目标区', font=small, fill=(145, 255, 190))
            if exact:
                point = scene.project(event['point_m'], view)
                if point:
                    x, y = point
                    draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline=(255, 245, 80), width=3)
                    draw.text((max(8, x - 90), max(35, y - 40)), '实际拍线接触点', font=small, fill=(255, 245, 80))
            if landing_freeze:
                point = scene.project(row['first_landing_m'], view)
                if point:
                    x, y = point
                    draw.text((max(8, min(x - 130, pane.width - 260)), max(40, y - 45)),
                              '真实首次落点', font=small, fill=(255, 150, 120))
            picture.paste(pane, (left, 106))
        draw = ImageDraw.Draw(picture)
        draw.text((20, 10), headings[case_id], font=title, fill='white')
        draw.text((20, 57), subtitles[case_id], font=text, fill=(187, 214, 231))
        draw.text((18, 113), '侧视｜己方 ← 网 → 对方', font=small, fill='white')
        draw.text((790, 113), '俯视｜同一时刻', font=small, fill='white')
        mode = '原生接触子步定格' if exact else ('终止前原生状态定格' if terminal else ('0.25倍慢放' if slow else '正常速度'))
        if landing_freeze:
            mode = '首次落点事件 + 相邻50Hz全身采样'
        draw.rectangle((16, 548, 760, 584), fill=(18, 28, 43))
        draw.text((24, 552), f't = {t:.4f} s｜{mode}', font=small, fill='white')
        speed = float(np.linalg.norm(row['outgoing_velocity_mps']))
        if terminal:
            if case_id == 0:
                status = f"同回合合格恢复｜最终连续稳定 {row['jump']['recovery_s']:.2f} s｜身体触地0"
                note = '出球速度与向下分量不足；首次落点偏出原目标横向6.7厘米'
            elif case_id == 1:
                status = '目标命中，但原恢复判定为失败｜完整扣杀失败'
                note = '落地后约1毫秒再次离地使合格落足判定失效；最终稳定3.83秒'
            else:
                status = f'身体触地，t={t:.2f} s终止｜未完成越网、落点和恢复'
                note = '当前策略仍在训练；这是已结束的失败回合，不代表整轮最终结果'
        elif landing_freeze:
            status = f"首次落地 t={row['first_landing_time_s']:.4f} s｜x={row['first_landing_m'][0]:.3f}, y={row['first_landing_m'][1]:.3f} m"
            note = ('命中原指定目标区；完整成功仍需攻击出球与合格恢复' if case_id == 1 else
                    '落在对方单打场内，横向偏出原指定目标6.7厘米')
        elif exact:
            status = f"己方空中拍线接触｜高 {row['contact_point_m'][2]:.3f} m｜实际出球 {speed:.2f} m/s"
            note = f"实际出球 vz={row['outgoing_velocity_mps'][2]:+.2f} m/s；原要求速度≥8、vz≤-1"
        elif t < 2:
            status, note = '从地面自主准备与走位', '完整Go2、标准PiPER和675毫米球拍｜原生动力学执行记录'
        elif t < row['contact_time_s']:
            status, note = '实际起跳与来球拦截', '完整拍柄、拍杆和拍头随真实机械臂运动'
        elif row['crossed_net'] and t >= row['crossing_time_s']:
            status = '已实际越网' + ('｜已首次落在对方场内' if row['ground_seen'] and t >= row['first_landing_time_s'] else '')
            note = ('本回合球有触网记录，但未被挡回；机器人触网故障为0' if row['shuttle_net_contact'] else
                    '球落地后的滚动不再显示；首落点事件只计一次')
        else:
            status, note = f'实际空中拍线接触已发生｜出球 {speed:.2f} m/s', '橙线为记录的球心轨迹；光圈与红叉均为标注'
        draw.text((20, 598), status, font=text, fill=(255, 205, 131))
        draw.text((20, 640), note, font=small, fill=(211, 225, 234))
        draw.text((20, 681), '三个独立精选训练回合｜完整扣杀仍为0｜尚未正式评测A/B各200回合',
                  font=small, fill=(163, 188, 204))
        return picture

    landing_indices = [int(np.argmin(np.abs(data[i]['episode_clock_s'] - cases[i]['row']['first_landing_time_s'])))
                       for i in range(2)]
    for name, picture in [('return_contact', frame(0, exact=True)),
                          ('target_landing', frame(1, landing_indices[1], landing_freeze=True)),
                          ('current_contact', frame(2, exact=True)), ('current_failure', frame(2, terminal=True))]:
        path = output.with_name(output.stem + '_' + name + '.jpg')
        picture.save(path, quality=94)
        previews.append(str(path.relative_to(ROOT)))
    if args.preview_only:
        print(json.dumps(dict(previews=previews), ensure_ascii=False), flush=True)
        return

    def card(heading, lines):
        picture = Image.new('RGB', (1280, 720), (18, 28, 43))
        draw = ImageDraw.Draw(picture)
        draw.text((50, 100), heading, font=title, fill='white')
        draw.line((50, 175, 1230, 175), fill=(73, 187, 168), width=3)
        for i, line in enumerate(lines):
            draw.text((50, 225 + i * 80), line, font=text, fill=(214, 229, 238))
        return picture

    with imageio.get_writer(output, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p',
                            ffmpeg_params=['-movflags', '+faststart', '-threads', '2']) as writer:
        def still(picture, seconds, name, **extra):
            nonlocal frame_count
            begin = frame_count
            for _ in range(round(seconds * fps)):
                writer.append_data(np.asarray(picture))
                frame_count += 1
            segments.append(dict(name=name, start_frame=begin, frames=frame_count - begin, **extra))

        def clip(case_id, ids, slow=False, name=None):
            nonlocal frame_count
            begin = frame_count
            for index in ids:
                picture = np.asarray(frame(case_id, int(index), slow=slow))
                for _ in range(4 if slow else 1):
                    writer.append_data(picture)
                    frame_count += 1
            segments.append(dict(name=name, start_frame=begin, frames=frame_count - begin, case_id=case_id,
                                 recorded_start_s=float(data[case_id]['episode_clock_s'][ids[0]]),
                                 recorded_end_s=float(data[case_id]['episode_clock_s'][ids[-1]]), interpolated=False))
            print('Rendered ' + str(name), flush=True)

        still(card('当前实际效果｜回球已出现，完整扣杀尚未成功', [
            '回合一：空中拍线、真实越网、对方场内落地、同回合合格恢复。',
            '回合二：首落点命中目标，但原恢复判定失败。',
            '回合三：正在训练的新策略，触球后身体触地失败。']), 2, 'opening')
        for case_id in range(3):
            ids = np.arange(len(data[case_id]['episode_clock_s']))
            clip(case_id, ids, name=f'case{case_id}_whole_episode')
            times = data[case_id]['episode_clock_s']
            selected = ids[(times >= 2.46) & (times <= 3.04)]
            clip(case_id, selected, slow=True, name=f'case{case_id}_contact_quarter_speed')
            still(frame(case_id, exact=True), 1.5, f'case{case_id}_exact_native_contact',
                  case_id=case_id, event_time_s=cases[case_id]['contact']['time_s'])
            if case_id < 2:
                index = landing_indices[case_id]
                still(frame(case_id, index, landing_freeze=True), 1.5, f'case{case_id}_first_landing',
                      case_id=case_id, event_time_s=cases[case_id]['row']['first_landing_time_s'],
                      body_pose_time_s=float(times[index]))
            still(frame(case_id, terminal=True), 1.5, f'case{case_id}_native_terminal', case_id=case_id,
                  event_time_s=cases[case_id]['row']['terminal_native_state']['time_s'])
        still(card('仍未达到扣杀验收｜GPU0继续训练', [
            '上一轮128次候选首回合：15次空中拍线、1次拍线与合格恢复同回合。',
            '出球≥8 m/s且向下≥1 m/s仍未达到，完整扣杀成功为0。',
            '视频来自实际执行；三段互相独立，不拼成一个成功回合。']), 2.5, 'result')

    protocol = json.loads((ROOT / 'configs/g07_full_racket_eval_v4_r3.json').read_text())
    assert sha(ROOT / protocol['asset']['path']) == protocol['asset']['sha256']
    sources = [p for p in snapshot.rglob('*') if p.is_file()]
    sources += [Path(__file__), ROOT / 'scripts/render_full_racket_r3_effect.py',
                ROOT / 'scripts/render_current_v4_progress.py', ROOT / 'configs/court_net_v4.json',
                ROOT / 'configs/g07_full_racket_eval_v4_r3.json', ROOT / protocol['asset']['path']]
    metadata = dict(created_at=datetime.now(timezone.utc).isoformat(), video=str(output.relative_to(ROOT)),
                    sha256=sha(output), bytes=output.stat().st_size, fps=fps, frames=frame_count,
                    duration_s=frame_count / fps, resolution=[1280, 720], segments=segments, previews=previews,
                    native_recorded_motion=True, interpolated_or_generated_motion=False,
                    renderer_uses_gpu=False, renderers={n: s.ctx.info['GL_RENDERER'] for n, s in scenes.items()},
                    formal_acceptance=False, complete_smash=False, cases_are_independent=True,
                    cases=[{k: c[k] for k in ['id', 'run', 'generation', 'env_id', 'episode_index', 'row',
                                             'whole_source_job_complete']} for c in cases],
                    caveats=['Selected training examples are not formal acceptance scores.',
                             'The current GPU job is still running; its shown episode had already terminated.',
                             'Only contact freezes use exact contact-substep body snapshots; landing poses are adjacent50Hz samples.',
                             'The roughly1ms extra flight in case1 is a native metric event, too short to resolve in50Hz motion.',
                             'Recorded maximum joint speed30.002182rad/s slightly exceeds configured30rad/s through solver overshoot.',
                             'Ball after first ground contact is hidden. No motion or landing is invented.'],
                    sources={str(p.relative_to(ROOT)): sha(p) for p in sources})
    output.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: metadata[k] for k in ['video', 'bytes', 'sha256', 'frames', 'duration_s']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
