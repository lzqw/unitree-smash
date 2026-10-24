#!/usr/bin/env python3
"""Replay completed native recovery and higher strings-contact episodes on CPU.

The two examples are separate episodes. No motion is reconstructed across resets.
Run with PYTHONPATH=/tmp/unitree_video_deps and the Isaac Lab Python interpreter.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from render_full_racket_r3_effect import Scene, ROOT, sha
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists() and not args.preview_only:
        raise ValueError('Use a fresh output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads((run/'episodes_001.json').read_text())
    contacts = json.loads((run/'contacts_001.json').read_text())
    trace = json.loads((run/'trajectory_001.json').read_text())
    recovery = next(x for x in rows if x['env_id'] == 0)
    contact = next(x for x in rows if x['env_id'] == 62)
    events = [x for x in contacts if x['env_id'] == 62
              and x['episode_index'] == contact['episode_index'] and x['first_racket_contact']]
    assert len(events) == 1
    event = events[0]
    full = np.load(run/'public_and_poses_001.npz')
    window = np.load(run/'near_contact_all_001.npz')
    full_t = full['observations'][::2, 0, 775]
    recovery_ids = np.flatnonzero(full['valid_first_episode'][::2, 0])
    contact_ids = np.flatnonzero(window['valid_first_episode'][:, 62])
    contact_t = window['episode_clock_s'][:, 62]
    assert len(full['link_poses']) == len(trace) == len(full_t)
    assert np.all(np.diff(full_t[recovery_ids]) > 0)
    assert np.all(np.diff(contact_t[contact_ids]) > 0)
    assert recovery['recovery_success'] and not recovery['had_contact']
    assert recovery['jump']['maximum_body_ground_force_n'] == 0
    assert contact['airborne_string_hit'] and contact['legal_initial_contact']
    assert contact['net_blocked_fault'] and not contact['crossed_net']
    assert event['string_contact'] and event['legal_initial_contact']
    assert not any(x['complete_smash'] for x in rows)
    assert np.isfinite(full['link_poses']).all() and np.isfinite(window['link_poses']).all()
    speed = float(np.linalg.norm(contact['outgoing_velocity_mps']))
    court = json.loads((ROOT/'configs/court_net_v4.json').read_text())
    protocol = json.loads((ROOT/'configs/g07_full_racket_eval_v4_r3.json').read_text())
    assert sha(ROOT/protocol['asset']['path']) == protocol['asset']['sha256']
    scene = Scene(run, court)
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title, text, small = [ImageFont.truetype(font, n) for n in [30, 23, 20]]
    fps, frame_count, segments, previews = 50, 0, [], []

    def frame(kind, index=None, *, slow=False, exact=False, terminal=False, overview=False):
        landing = trail = None
        if kind == 'recovery':
            poses = full['link_poses'][index, 0]
            time_s = float(full_t[index])
            heading = '当前效果 ①｜自主走位、起跳挥拍、落地恢复'
            subtitle = '例 A：漏球后继续观察同一次跳跃｜未完成击球任务'
            ended = time_s >= recovery['first_landing_time_s']
            ball = np.array(trace[index]['ball_position_m'][0]) if 2 <= time_s and not ended else None
            if time_s >= 2:
                trail = [x['ball_position_m'][0] for x in trace[max(100, index-20):index+1]
                         if x['time_s'] < recovery['first_landing_time_s']]
            if ended:
                landing = np.array(recovery['first_landing_m'])
            status = (f"腾空 {recovery['jump']['airborne_s']:.3f} s｜无身体触地｜最终稳定 {recovery['jump']['recovery_s']:.2f} s")
            note = '首次落地红叉来自真实球事件；球回合结束后的滚动不再显示' if ended else '自主从地面执行；橙色轨迹来自同回合实际球心记录'
        else:
            heading = '当前效果 ②｜1.70 米高处真实空中拍线触球'
            subtitle = '例 B：另一独立回合｜向下回球，随后被网挡回'
            if exact:
                poses = np.asarray(event['body_link_poses_local'])
                ball = np.asarray(event['ball_position_m'])
                time_s = float(event['time_s'])
            elif terminal:
                state = contact['terminal_native_state']
                poses = np.column_stack((state['body_link_positions_m'], state['body_link_quaternions_wxyz']))
                ball, time_s = np.asarray(state['ball_position_m']), float(state['time_s'])
            else:
                poses = window['link_poses'][index, 62]
                ball, time_s = window['ball_position_m'][index, 62], float(contact_t[index])
                trail = window['ball_position_m'][contact_ids[(contact_ids >= index-20) & (contact_ids <= index)], 62]
            if exact:
                status = f"己方合法拍线接触｜高 {event['point_m'][2]:.3f} m｜实际出球 {speed:.2f} m/s"
                note = '出球 vx=4.82、vz=-1.05 m/s；低于原攻击速度门槛 8 m/s'
            elif terminal:
                heading = '例 B 的真实结果｜球触网挡回，完整扣杀失败'
                status = '实际球速已转向己方｜未越网｜此回合尚无首次落地事件'
                note = '原记录在 3.07 s 因挡网终止；后续稳定恢复未观测'
            else:
                status = '来球飞行中｜等待真实接触' if time_s < event['time_s'] else '已发生空中拍线触球｜出球朝对方并向下'
                note = '本例最终被网挡回；与例 A 的稳定恢复属于不同回合'
        canvas = Image.new('RGB', (1280, 720), (18, 28, 43))
        for view, left in ([('overview', 0)] if overview else [('side', 0), ('top', 776)]):
            pane = scene.render(poses, view, ball=ball, trail=trail, landing=landing)
            if exact:
                point = scene.project(event['point_m'], view)
                if point:
                    x, y = point
                    d = ImageDraw.Draw(pane)
                    d.ellipse((x-12, y-12, x+12, y+12), outline=(255, 245, 80), width=3)
                    d.text((max(8, x-105), max(35, y-42)), '实际接触点', font=small, fill=(255, 245, 80))
            canvas.paste(pane, (left, 106))
        d = ImageDraw.Draw(canvas)
        d.text((20, 10), heading, font=title, fill='white')
        d.text((20, 57), subtitle, font=text, fill=(187, 214, 231))
        d.text((18, 113), '标准尺寸全场概览' if overview else '侧视｜己方 ← 网 → 对方', font=small, fill='white')
        if not overview:
            d.text((790, 113), '俯视｜同一时刻', font=small, fill='white')
        mode = '原生接触子步定格' if exact else ('终止前原生状态定格' if terminal else ('0.25 倍慢放' if slow else '正常速度'))
        d.rectangle((16, 548, 754, 584), fill=(18, 28, 43))
        d.text((24, 552), f't = {time_s:.4f} s｜{mode}', font=small, fill='white')
        d.text((20, 598), status, font=text, fill=(255, 205, 131))
        d.text((20, 640), note, font=small, fill=(211, 225, 234))
        d.text((20, 681), '完整 Go2 + 标准 PiPER + 675 mm 球拍｜实际仿真记录回放｜当前完整成功：0', font=small, fill=(163, 188, 204))
        return canvas

    for label, picture in [('contact', frame('contact', exact=True)),
                           ('recovery', frame('recovery', int(recovery_ids[-1]))),
                           ('net_failure', frame('contact', terminal=True))]:
        path = output.with_name(output.stem+'_'+label+'.jpg')
        picture.save(path, quality=94)
        previews.append(str(path.relative_to(ROOT)))
    if args.preview_only:
        print(json.dumps(dict(previews=previews, renderer=scene.ctx.info['GL_RENDERER']), ensure_ascii=False), flush=True)
        return

    def card(heading, lines):
        pic = Image.new('RGB', (1280, 720), (18, 28, 43))
        d = ImageDraw.Draw(pic)
        d.text((50, 100), heading, font=title, fill='white')
        d.line((50, 175, 1230, 175), fill=(73, 187, 168), width=3)
        for i, line in enumerate(lines):
            d.text((50, 225+i*80), line, font=text, fill=(214, 229, 238))
        return pic

    with imageio.get_writer(output, fps=fps, codec='libx264', quality=8,
                            pixelformat='yuv420p', ffmpeg_params=['-movflags', '+faststart', '-threads', '2']) as writer:
        def still(pic, seconds, name, **extra):
            nonlocal frame_count
            begin = frame_count
            for _ in range(round(seconds*fps)):
                writer.append_data(np.asarray(pic)); frame_count += 1
            segments.append(dict(name=name, start_frame=begin, frames=frame_count-begin, **extra))

        def clip(kind, ids, slow=False):
            nonlocal frame_count
            begin = frame_count
            for index in ids:
                pic = np.asarray(frame(kind, int(index), slow=slow))
                for _ in range(4 if slow else 1):
                    writer.append_data(pic); frame_count += 1
            times = full_t if kind == 'recovery' else contact_t
            row = recovery if kind == 'recovery' else contact
            segments.append(dict(name=kind+('_quarter_speed' if slow else '_normal'),
                start_frame=begin, frames=frame_count-begin, env_id=row['env_id'],
                episode_index=row['episode_index'], generation=1,
                recorded_start_s=float(times[ids[0]]), recorded_end_s=float(times[ids[-1]]), interpolated=False))
            print('Rendered '+segments[-1]['name'], flush=True)

        still(frame('recovery', 0, overview=True), 1.5, 'court_overview_pose_freeze')
        clip('recovery', recovery_ids)
        clip('recovery', recovery_ids[(full_t[recovery_ids] >= 2.4) & (full_t[recovery_ids] <= 3.4)], slow=True)
        still(frame('recovery', int(recovery_ids[-1])), 1.5, 'recovered_pose_freeze')
        still(card('例 B｜真实高位触球：与例 A 是不同回合', [
            '接下来展示最新完成训练中实际发生的空中拍线触球。',
            '侧视与俯视同步；慢放逐帧重复原始记录。',
            '该回合被网挡回，不能与例 A 拼成一次完整成功。']), 2, 'separate_episode_transition')
        clip('contact', contact_ids)
        clip('contact', contact_ids, slow=True)
        still(frame('contact', exact=True), 3, 'exact_contact_substep_freeze', time_s=event['time_s'])
        still(frame('contact', terminal=True), 2.5, 'net_failure_terminal_native_freeze', time_s=contact['terminal_native_state']['time_s'])
        still(card('当前结果｜触球与恢复均有进展，尚未形成完整扣杀', [
            '例 A：同回合自主跳跃并稳定恢复，但漏球。',
            f'例 B：{event["point_m"][2]:.3f} m 空中拍线，出球 {speed:.2f} m/s，随后挡网。',
            '已完成两代共 128 次候选首回合：13 次跳跃恢复，0 完整成功。',
            '上述恢复全部来自漏球回合；独立 A/B 各 200 回合尚未评测。']), 3, 'result_card')

    source_files = [run/n for n in ['episodes_000.json', 'episodes_001.json', 'contacts_001.json',
        'trajectory_001.json', 'public_and_poses_001.npz', 'near_contact_all_001.npz',
        'geometry.json', 'geometry.npz', 'job.json', 'completed.json', 'source_hashes.json',
        'source_snapshot.zip', 'population_001.pt', 'FINAL_TRAIN_SUMMARY.json']]
    source_files += [Path(__file__), ROOT/'scripts/render_full_racket_r3_effect.py',
        ROOT/'scripts/render_current_v4_progress.py', ROOT/'quadsmash/physics/court_v4.py',
        ROOT/'configs/court_net_v4.json', ROOT/'configs/g07_full_racket_eval_v4_r3.json',
        ROOT/protocol['asset']['path']]
    metadata = dict(created_at=datetime.now(timezone.utc).isoformat(), video=str(output.relative_to(ROOT)),
        bytes=output.stat().st_size, sha256=sha(output), fps=fps, frames=frame_count,
        duration_s=frame_count/fps, resolution=[1280,720], renderer=scene.ctx.info['GL_RENDERER'],
        renderer_uses_gpu=False, run=str(run.relative_to(ROOT)), segments=segments, previews=previews,
        poses_are_native_recorded=True, interpolated_or_generated_motion=False,
        contact_freeze_is_exact_native_substep=True, separate_episodes=True, formal_acceptance=False,
        contact=dict(env_id=62, generation=1, episode_index=contact['episode_index'],
            time_s=event['time_s'], point_m=event['point_m'], outgoing_velocity_mps=contact['outgoing_velocity_mps'],
            outgoing_speed_mps=speed, net_blocked=True, complete_smash=False,
            first_landing_observed=False, recovery_observed=False),
        recovery=dict(env_id=0, generation=1, episode_index=recovery['episode_index'],
            had_contact=False, stable_recovery_s=recovery['jump']['recovery_s'],
            first_landing_m=recovery['first_landing_m'], first_landing_time_s=recovery['first_landing_time_s']),
        caveats=['Separate episodes are not a successful combined smash.',
            'Env62 continuous full-body data exists only from 2.2 to 3.06 s; exact terminal state is separately labelled.',
            'No post-termination recovery or first ball landing was recorded for env62.',
            'Recovery-course env0 continues robot motion after the shuttle first landed; ball rally remains dead.',
            'Ball rings and contact circles are visibility annotations, not expanded collision shapes.',
            'Visual floor is 5 mm below physical ground; native body and ball positions are unchanged.',
            'TRAIN results include elite repeats and are not formal evaluation rates.'],
        sources={str(p.relative_to(ROOT)):sha(p) for p in source_files})
    output.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:metadata[k] for k in ['video','bytes','sha256','frames','duration_s']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
