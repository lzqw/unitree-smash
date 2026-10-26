#!/usr/bin/env python3
"""Render frozen, independent native episodes with their original outcomes."""
import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_full_racket_r3_effect import ROOT, Scene, sha
from native_video_framing_v4 import PANE_HEIGHT, PANE_TOP, case_views
from quadsmash.evaluation_v4 import FAULTS


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
    court = json.loads((ROOT / 'configs/court_net_v4.json').read_text())
    scenes = {}
    for case in cases:
        if case['run'] in scenes:
            continue
        if not scenes:
            scene = Scene(snapshot / case['run'], court)
        else:
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
            scene.views = dict(scene.views)
        for prefix in ['', 'wide_', 'court_', 'contact_']:
            for name, width in [('side', 760), ('top', 504)]:
                scene.views[prefix + name] = (scene.ctx.simple_framebuffer((width, PANE_HEIGHT), components=3), np.eye(4))
        scenes[case['run']] = scene
    data = [np.load(snapshot / f'case{i}.npz') for i in range(len(cases))]
    for case, recorded in zip(cases, data):
        assert recorded['valid_first_episode'].all()
        assert np.isfinite(recorded['link_poses']).all()
        assert np.allclose(np.diff(recorded['episode_clock_s']), .02, atol=2e-6)
        assert isinstance(case['row']['complete_smash'], bool)
    cameras, framing_checks = [], []
    for case, recorded in zip(cases, data):
        fitted, checked = case_views(snapshot / case['run'], case, recorded, court)
        cameras.append(fitted)
        framing_checks.append(checked)
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title, text, small = [ImageFont.truetype(font, n) for n in [29, 23, 20]]
    fps, frames, segments, previews = 50, 0, [], []

    def frame(case_id, index=None, *, slow=False, exact=False, terminal=False, landing_freeze=False):
        case, recorded = cases[case_id], data[case_id]
        row, event, scene = case['row'], case.get('contact'), scenes[case['run']]
        for name, matrix in cameras[case_id].items():
            framebuffer, _ = scene.views[name]
            scene.views[name] = (framebuffer, matrix)
        ball = trail = landing = None
        if exact:
            poses = np.asarray(event['body_link_poses_local'])
            ball, t = np.asarray(event['ball_position_m']), event['time_s']
        elif terminal:
            state = case.get('last_recorded_state') or row['terminal_native_state']
            poses = np.column_stack((state['body_link_positions_m'], state['body_link_quaternions_wxyz']))
            t = state['time_s']
            if row['launched'] and not row['ground_seen']:
                ball = np.asarray(state['ball_position_m'])
        else:
            t = float(recorded['episode_clock_s'][index])
            poses = recorded['link_poses'][index]
            if t >= 2 and row['launched']:
                ball = recorded['ball_position_m'][index]
                ids = np.arange(max(0, index - 25), index + 1)
                mask = recorded['episode_clock_s'][ids] >= 2
                if row['ground_seen']:
                    mask &= recorded['episode_clock_s'][ids] <= row['first_landing_time_s']
                trail = recorded['ball_position_m'][ids[mask]]
        if row['ground_seen'] and (t >= row['first_landing_time_s'] or landing_freeze):
            ball, landing = None, np.asarray(row['first_landing_m'])
        wide = terminal or t >= 3.5 or landing_freeze or (row['crossed_net'] and t >= row['crossing_time_s']
                                 and (not row['ground_seen'] or t <= row['first_landing_time_s'] + .35))
        views = ['wide_side', 'wide_top'] if wide else ['side', 'top']
        if exact:
            wide, views = False, ['contact_side', 'contact_top']
        # Keep actual incoming/outgoing balls and first-landing markers in
        # both panels, including court-end misses outside the earlier view.
        faulted = any(row[name] for name in FAULTS)
        follow_ball = (not terminal and (not faulted or not row['had_contact']
                       or t <= row['release_time_s']))
        reference = landing if landing_freeze else (ball if follow_ball else None)
        if reference is not None:
            if any(scene.project(reference, view) is None for view in views):
                wide, views = True, ['wide_side', 'wide_top']
            if any(scene.project(reference, view) is None for view in views):
                views = ['court_side', 'court_top']
        picture = Image.new('RGB', (1280, 720), (18, 28, 43))
        for view, left in zip(views, [0, 776]):
            pane = scene.render(poses, view, ball=ball, trail=trail, landing=landing)
            draw = ImageDraw.Draw(pane)
            if wide:
                x0, x1 = court['target_x_limits_m']
                y = court['target_y_half_width_m']
                corners = [scene.project(p, view) for p in [(x0, -y, .002), (x1, -y, .002),
                                                            (x1, y, .002), (x0, y, .002)]]
                if all(p is not None for p in corners):
                    draw.line(corners + [corners[0]], fill=(100, 245, 165), width=2)
            if exact:
                point = scene.project(event['point_m'], view)
                if point:
                    x, y = point
                    draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline=(255, 245, 80), width=3)
            picture.paste(pane, (left, PANE_TOP))
        draw = ImageDraw.Draw(picture)
        draw.text((20, 10), case['title'], font=title, fill='white')
        draw.text((20, 57), case['subtitle'], font=text, fill=(187, 214, 231))
        side_label = '接触斜侧｜30°正交' if exact else '侧视｜正交投影'
        draw.text((18, 98), side_label + '｜己方 ← 网 → 对方', font=small, fill='white')
        draw.text((790, 98), '俯视｜正交投影｜同一时刻', font=small, fill='white')
        mode = '原生接触子步定格' if exact else ('终止前原生状态' if terminal else ('0.25倍慢放' if slow else '正常速度'))
        if terminal and case.get('last_recorded_state'):
            mode = '末次有效50Hz采样｜非精确终止子步'
        if landing_freeze:
            mode = '首次落点事件 + 相邻50Hz全身采样'
        draw.rectangle((16, 548, 760, 584), fill=(18, 28, 43))
        draw.text((24, 552), f't = {t:.4f} s｜{mode}', font=small, fill='white')
        if terminal:
            status, note = case['terminal_status'], case['limitation']
        elif landing_freeze:
            x, y, _ = row['first_landing_m']
            low, high = court['opponent_x_limits_m']
            geometric_court = low <= x <= high and abs(y) <= court['singles_width_m'] / 2
            if row['landed_target']:
                landing_result = '任务有效回球、命中目标'
            elif row['landed_legal_court']:
                landing_result = '任务有效回球、未中目标'
            else:
                landing_result = '几何在场内；任务回球未成立' if geometric_court else '几何落点在对方场外'
            status = f'真实首次落点 x={x:.3f}, y={y:.3f} m｜{landing_result}'
            note = case['limitation']
        elif exact:
            speed = float(np.linalg.norm(row['outgoing_velocity_mps']))
            contact_kind = ('空中拍线' if row['airborne_string_hit'] else '拍线（非合格空中）') if event['string_contact'] else '拍框'
            status = f"{contact_kind}接触｜高 {event['point_m'][2]:.3f} m｜实际出球 {speed:.5f} m/s"
            velocity = np.asarray(row['outgoing_velocity_mps'])
            attack = (row['airborne_string_hit'] and row['string_hit'] and row['legal_initial_contact']
                      and np.isfinite(row['contact_point_m']).all()
                      and row['contact_point_m'][0] < court['net_plane_x_m'] and row['released']
                      and np.isfinite(velocity).all() and speed >= 8 and velocity[0] > 0 and velocity[2] <= -1)
            qualified = '通过' if attack else '未过'
            note = f"实际vz={velocity[2]:+.5f} m/s；出球攻击条件{qualified}（速度≥8、vx>0、vz≤-1）"
        elif t < 2:
            status, note = '从地面自主准备与走位', '完整Go2、标准PiPER和675毫米球拍｜实际动力学执行记录'
        elif row['ground_seen'] and t >= row['first_landing_time_s'] and not row['had_contact']:
            status, note = '来球已落地，未击中｜继续播放机器人恢复', '跳跃恢复与击球分别统计；此回合未完成击球任务'
        elif not row['had_contact'] or t < row['contact_time_s']:
            status, note = '来球拦截与自主动作执行中', '不补造动作、触球或球路；失败按原记录保留'
        elif row['net_blocked_fault'] and t >= case.get('rally_fault_time_s', float('inf')):
            status = '球被网挡住｜本回合回球失败'
            note = '继续播放同一回合的机器人恢复；恢复不代表完整任务成功'
        elif row['crossed_net'] and t >= row['crossing_time_s']:
            status, note = '实际越网已发生｜继续观察首次落点与机器人恢复', '橙线为球心轨迹；红叉为首次落点；均为辅助标注'
        else:
            status = '已发生实际拍线接触' if row['string_hit'] else '已发生拍框接触，未计为拍线成功'
            note = '本回合是否完成越网与恢复，以后续原生事件记录为准'
        draw.text((20, 598), status, font=text, fill=(255, 205, 131))
        draw.text((20, 640), note, font=small, fill=(211, 225, 234))
        footer = ('该独立回合完整扣杀成立｜整体正式A/B各200回合尚未运行'
                  if row['complete_smash'] else
                  '独立精选训练回合｜完整扣杀仍为0｜尚未正式评测A/B各200回合')
        draw.text((20, 681), manifest.get('evaluation_footer',footer),
                  font=small, fill=(163, 188, 204))
        return picture

    for i, case in enumerate(cases):
        pictures = [('terminal', frame(i, terminal=True))]
        if case.get('contact'):
            pictures.append(('contact', frame(i, exact=True)))
        if case['row']['ground_seen']:
            index = int(np.argmin(abs(data[i]['episode_clock_s'] - case['row']['first_landing_time_s'])))
            pictures.append(('landing', frame(i, index, landing_freeze=True)))
        for label, picture in pictures:
            path = output.with_name(output.stem + f'_case{i}_{label}.jpg')
            picture.save(path, quality=94)
            previews.append(str(path.relative_to(ROOT)))
    if args.preview_only:
        check = dict(previews=previews, native_cases=len(cases), camera_framing_checks=framing_checks,
                     snapshot_manifest_sha256=sha(snapshot / 'cases.json'),
                     renderer_sha256=sha(Path(__file__)), framing_helper_sha256=sha(ROOT / 'scripts/native_video_framing_v4.py'),
                     pane_layout=dict(top=PANE_TOP, height=PANE_HEIGHT, time_band_top=548, labels_outside_panes=True),
                     native_motion_unchanged=True, new_native_execution=False, formal_acceptance=False)
        output.with_suffix('.preview.json').write_text(json.dumps(check, indent=2) + '\n')
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
            nonlocal frames
            start = frames
            for _ in range(round(seconds * fps)):
                writer.append_data(np.asarray(picture))
                frames += 1
            segments.append(dict(name=name, start_frame=start, frames=frames-start, **extra))

        def clip(case_id, ids, slow=False):
            nonlocal frames
            start = frames
            for index in ids:
                picture = np.asarray(frame(case_id, int(index), slow=slow))
                for _ in range(4 if slow else 1):
                    writer.append_data(picture)
                    frames += 1
            segments.append(dict(name='slow' if slow else 'whole_episode', case_id=case_id,
                                 start_frame=start, frames=frames-start, interpolated=False))
            print(f'Rendered case{case_id} slow={slow}', flush=True)

        still(card(manifest.get('opening_title', '当前真实效果｜空中回球与恢复已出现'), manifest['opening_lines']), 1.5, 'opening')
        for i, case in enumerate(cases):
            ids = np.arange(len(data[i]['episode_clock_s']))
            clip(i, ids)
            times = data[i]['episode_clock_s']
            focus = (case.get('contact') or {}).get('time_s', min(2.75, float(times[-1])))
            selected = ids[(times >= max(2., focus - .25)) & (times <= focus + .25)]
            if len(selected):
                clip(i, selected, slow=True)
            if case.get('contact'):
                still(frame(i, exact=True), 1.5, 'exact_contact', case_id=i,
                      native_event_time_s=case['contact']['time_s'])
            if case['row']['ground_seen']:
                index = int(np.argmin(abs(times - case['row']['first_landing_time_s'])))
                still(frame(i, index, landing_freeze=True), 1.5, 'first_landing', case_id=i,
                      native_event_time_s=case['row']['first_landing_time_s'], body_pose_time_s=float(times[index]))
            still(frame(i, terminal=True), 1.5, 'native_terminal', case_id=i)
        still(card(manifest.get('closing_title','当前仍未完成合格扣杀'), manifest['closing_lines']), 2, 'result')

    sources = [p for p in snapshot.rglob('*') if p.is_file()]
    sources += [Path(__file__), ROOT / 'scripts/render_full_racket_r3_effect.py',
                ROOT / 'scripts/render_current_v4_progress.py', ROOT / 'configs/court_net_v4.json',
                ROOT / 'configs/g07_full_racket_eval_v4_r3.json', ROOT / 'scripts/native_video_framing_v4.py']
    metadata = dict(created_at=datetime.now(timezone.utc).isoformat(), video=str(output.relative_to(ROOT)),
                    sha256=sha(output), bytes=output.stat().st_size, fps=fps, frames=frames,
                    duration_s=frames/fps, resolution=[1280, 720], segments=segments, previews=previews,
                    native_recorded_motion=True, interpolated_or_generated_motion=False,
                    renderer_uses_gpu=False, renderer=next(iter(scenes.values())).ctx.info['GL_RENDERER'],
                    formal_acceptance=False,
                    complete_smash=any(case['row']['complete_smash'] for case in cases),
                    complete_smash_scope='Any selected independent episode; no population or formal acceptance claim',
                    cases_are_independent=True,
                    cases=cases, caveats=manifest['caveats'] + [
                        'Fixed per-case orthographic cameras fit the actual recorded mesh bounds; exact contact has a separate fitted camera. This is a visual replay update, not a new experiment.',
                        'View labels and time labels sit outside the420px image panes; viewport coverage does not assert that every mesh surface is unoccluded.'],
                    camera_projection='orthographic, side12-degree elevation, exact-contact30-degree oblique and vertical top',
                    pane_layout=dict(top=PANE_TOP, height=PANE_HEIGHT, time_band_top=548, labels_outside_panes=True),
                    camera_framing_checks=framing_checks,
                    sources={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p) for p in sources})
    output.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: metadata[k] for k in ['video', 'bytes', 'sha256', 'frames', 'duration_s']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
