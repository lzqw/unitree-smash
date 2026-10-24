#!/usr/bin/env python3
"""Render native first-episode poses and exact contact snapshots; CPU only."""
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
    rows = json.loads((run/'episodes_000.json').read_text())
    contacts = json.loads((run/'contacts_000.json').read_text())
    trace = json.loads((run/'trajectory_000.json').read_text())
    events = {x['env_id']: x for x in contacts if x['episode_index'] == 0}
    full = np.load(run/'public_and_poses_000.npz')
    window = np.load(run/'near_contact_all_000.npz')
    court = json.loads((ROOT/'configs/court_net_v4.json').read_text())
    protocol = json.loads((ROOT/'configs/g07_full_racket_eval_v4_r3.json').read_text())
    assert sha(ROOT/protocol['asset']['path']) == protocol['asset']['sha256']
    assert len(rows) == 32 and sum(x['airborne_string_hit'] for x in rows) == 3
    assert not any(x['complete_smash'] for x in rows)
    assert rows[0]['airborne_string_hit'] and rows[24]['airborne_string_hit']
    assert rows[20]['frame_hit'] and not rows[20]['string_hit']
    assert all(x['episode_index'] == 0 for x in rows)
    assert np.isfinite(full['link_poses']).all() and np.isfinite(window['link_poses']).all()
    assert len(full['link_poses']) == len(trace)
    scene = Scene(run, court)
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title, text, small = [ImageFont.truetype(font, n) for n in [30, 23, 20]]
    fps = 50
    segments = []
    previews = []
    frame_count = 0

    def indices(env_id, source='window', start=0):
        if source == 'full':
            valid = full['valid_first_episode'][::2, env_id]
            times = full['observations'][::2, env_id, 775]
        else:
            valid = window['valid_first_episode'][:, env_id]
            times = window['episode_clock_s'][:, env_id]
        result = np.flatnonzero(valid & (times >= start))
        assert len(result) and np.all(np.diff(times[result]) > 0)
        return result

    def frame(env_id, index=None, *, source='window', slow=False,
              exact_contact=False, overview=False, landing_summary=False):
        row, event = rows[env_id], events[env_id]
        if exact_contact:
            poses = np.array(event['body_link_poses_local'])
            ball = np.array(event['ball_position_m'])
            time_s = event['time_s']
            trail = None
        elif source == 'full':
            assert full['valid_first_episode'][2*index, env_id]
            poses = full['link_poses'][index, env_id]
            time_s = float(full['observations'][2*index, env_id, 775])
            ball = np.array(trace[index]['ball_position_m'][env_id]) if time_s >= 2 else None
            trail = [x['ball_position_m'][env_id] for x in trace[max(100, index-20):index+1]] if time_s >= 2 else None
        else:
            assert window['valid_first_episode'][index, env_id]
            poses = window['link_poses'][index, env_id]
            ball = window['ball_position_m'][index, env_id]
            time_s = float(window['episode_clock_s'][index, env_id])
            trail = window['ball_position_m'][max(0, index-20):index+1, env_id]
        landing = np.array(row['first_landing_m']) if landing_summary else None
        heading = {0: '当前效果｜自主走位、起跳与真实拍线触球',
                   24: '另一组协调参数｜较高位置的真实拍线触球',
                   20: '失败对照｜拍框误击，首次落点仍在己方'}[env_id]
        if exact_contact:
            heading += ' · 接触瞬间定格'
        if landing_summary:
            heading = '拍框误击后的首次落点｜原生事件记录'
        subtitle = '完整 Go2 + 标准 PiPER + 675 mm 球拍｜Isaac / PhysX 实际记录'
        canvas = Image.new('RGB', (1280, 720), (18, 28, 43))
        views = [('overview', 0)] if overview else [('side', 0), ('top', 776)]
        for view, left in views:
            pane = scene.render(poses, view, ball=ball, trail=trail, landing=landing)
            if exact_contact:
                point = scene.project(event['point_m'], view)
                if point is not None:
                    x, y = point
                    d = ImageDraw.Draw(pane)
                    d.ellipse((x-12, y-12, x+12, y+12), outline=(255, 245, 80), width=3)
                    d.text((max(8, x-105), max(34, y-42)), '实际接触点', font=small, fill=(255, 245, 80))
            canvas.paste(pane, (left, 106))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 10), heading, font=title, fill='white')
        draw.text((20, 57), subtitle, font=text, fill=(187, 214, 231))
        draw.text((18, 113), '全场概览｜己方 ← 网 → 对方' if overview else '侧视｜己方 ← 网 → 对方', font=small, fill='white')
        if not overview:
            draw.text((790, 113), '俯视｜同一时刻', font=small, fill='white')
        label = '原生接触子步定格' if exact_contact else ('0.25 倍慢放' if slow else '正常速度')
        if landing_summary:
            label = '最后有效姿态定格 + 首次落点标记'
        draw.rectangle((16, 548, 752, 583), fill=(18, 28, 43))
        draw.text((24, 552), f't = {time_s:.4f} s｜{label}', font=small, fill='white')
        if landing_summary:
            p = row['first_landing_m']
            status = f"首次落地 t={row['first_landing_time_s']:.4f} s｜x={p[0]:.3f}, y={p[1]:.3f} m｜未过网"
            note = f'红叉来自原生首落地记录；全身姿态停在 t={time_s:.2f} s，并非落地瞬间全身快照'
        elif exact_contact:
            speed = np.linalg.norm(row['outgoing_velocity_mps'])
            status = f"{'拍线' if event['string_contact'] else '拍框'}接触高 {event['point_m'][2]:.3f} m｜己方合法首次接触｜出球 {speed:.2f} m/s"
            note = ('仍未达到攻击出球要求；触球后身体触地，恢复失败' if event['string_contact'] else
                    '拍框接触不计拍线任务成功；出球下落，首次落在己方')
        else:
            before = time_s < event['time_s']
            status = ('准备与接近来球' if time_s < 2 else '来球飞行中｜等待接触') if before else (
                '已发生真实空中拍线触球｜完整扣杀仍失败' if event['string_contact'] else '已发生拍框接触｜不计拍线任务成功')
            note = ('本例最终因身体触地终止；终止后的动作与球轨迹不拼接进本回合' if event['string_contact'] else
                    '本例首次落地在己方；后续弹跳不能重新计分')
        draw.text((20, 598), status, font=text, fill=(255, 205, 131))
        draw.text((20, 640), note, font=small, fill=(211, 225, 234))
        draw.text((20, 681), '标准场地 / 悬空网｜橙圈为球心可见性标记｜本批 32 次训练执行：3 次空中拍线接触，0 完整成功', font=small, fill=(163, 188, 204))
        return canvas

    samples = [('contact0', frame(0, exact_contact=True)),
               ('contact24', frame(24, exact_contact=True)),
               ('before24', frame(24, int(indices(24)[20]), slow=True)),
               ('landing20', frame(20, int(indices(20)[-1]), landing_summary=True))]
    for label, picture in samples:
        path = output.with_name(output.stem+'_'+label+'.jpg')
        picture.save(path, quality=94)
        previews.append(str(path.relative_to(ROOT)))
    if args.preview_only:
        print(json.dumps(dict(previews=previews, renderer=scene.ctx.info['GL_RENDERER']), ensure_ascii=False))
        return

    def card(heading, lines):
        picture = Image.new('RGB', (1280, 720), (18, 28, 43))
        draw = ImageDraw.Draw(picture)
        draw.text((50, 95), heading, font=title, fill='white')
        draw.line((50, 165, 1230, 165), fill=(73, 187, 168), width=3)
        for i, line in enumerate(lines):
            draw.text((50, 225+i*82), line, font=text, fill=(214, 229, 238))
        return np.asarray(picture)

    with imageio.get_writer(str(output), fps=fps, codec='libx264', quality=8, macro_block_size=8,
                           output_params=['-movflags', '+faststart']) as writer:
        def still(picture, seconds, name, **details):
            nonlocal frame_count
            for _ in range(round(seconds*fps)):
                writer.append_data(np.asarray(picture))
            segments.append(dict(name=name, start_frame=frame_count, frames=round(seconds*fps), **details))
            frame_count += round(seconds*fps)

        still(card('最新实际效果｜完整球拍 r3 · 2026-10-24', [
            '最新一批：32 次候选训练执行，记录到 3 次真实空中拍线触球。',
            '下面展示自主准备、起跳、接触慢放与精确接触瞬间。',
            '同时展示拍框误击及其首次落点，保留失败结果。',
            '当前仍无完整扣杀成功；本视频不是正式 400 回合评测。']), 2, 'introduction')

        def clip(env_id, source='window', slow=False, overview=False, start=0):
            nonlocal frame_count
            ids = indices(env_id, source, start)
            copies = 4 if slow else 1
            begin = frame_count
            for index in ids:
                picture = np.asarray(frame(env_id, int(index), source=source, slow=slow, overview=overview))
                for _ in range(copies):
                    writer.append_data(picture)
                    frame_count += 1
            times = full['observations'][::2, env_id, 775] if source == 'full' else window['episode_clock_s'][:, env_id]
            segments.append(dict(name=f'env{env_id}_{source}'+('_quarter_speed' if slow else '_normal')+('_overview' if overview else '_side_top'),
                start_frame=begin, frames=frame_count-begin, env_id=env_id, episode_index=0,
                recorded_start_s=float(times[ids[0]]), recorded_end_s=float(times[ids[-1]]),
                interpolated=False))
            print('Rendered '+segments[-1]['name'], flush=True)

        clip(0, 'full', overview=True)
        clip(0, 'full')
        clip(0, 'full', slow=True, start=2.4)
        still(frame(0, exact_contact=True), 3, 'env0_exact_contact_snapshot', env_id=0, contact_time_s=events[0]['time_s'])
        clip(24, slow=True)
        still(frame(24, exact_contact=True), 3, 'env24_exact_contact_snapshot', env_id=24, contact_time_s=events[24]['time_s'])
        clip(20, slow=True)
        still(frame(20, int(indices(20)[-1]), landing_summary=True), 2.5, 'env20_first_landing_record',
              landing_time_s=rows[20]['first_landing_time_s'], pose_time_s=float(window['episode_clock_s'][indices(20)[-1],20]))
        still(card('当前结果｜已触到拍线，尚未完成扣杀', [
            '拍线触球发生在己方腾空阶段；出球仍较弱且向上。',
            '3 个拍线触球回合均因身体触地终止，未完成稳定恢复。',
            '拍框误击对照首次落在己方，不能算有效回球。',
            '后续继续优化全身协调、拍头速度、出球方向与落地恢复。']), 3, 'ending')

    files = [run/n for n in ['episodes_000.json', 'contacts_000.json', 'trajectory_000.json',
        'public_and_poses_000.npz', 'near_contact_all_000.npz', 'geometry.json', 'geometry.npz',
        'job.json', 'completed.json', 'metrics.json', 'source_hashes.json', 'source_snapshot.zip', 'population_000.pt']]
    files += [Path(__file__), ROOT/'scripts/render_full_racket_r3_effect.py', ROOT/'scripts/render_current_v4_progress.py',
              ROOT/'quadsmash/physics/court_v4.py', ROOT/'configs/court_net_v4.json',
              ROOT/'configs/g07_full_racket_eval_v4_r3.json', ROOT/protocol['asset']['path']]
    meta = dict(created_at=datetime.now(timezone.utc).isoformat(), video=str(output.relative_to(ROOT)),
        bytes=output.stat().st_size, sha256=sha(output), fps=fps, frames=frame_count,
        resolution=[1280,720], duration_s=frame_count/fps, renderer=scene.ctx.info['GL_RENDERER'],
        renderer_uses_gpu=False, asset_version='full-racket-court-v4-r3',
        run=str(run.relative_to(ROOT)), segments=segments, previews=previews,
        poses_are_native_recorded=True, interpolated_or_generated_motion=False,
        contact_freezes_are_exact_native_substep_snapshots=True, formal_acceptance=False,
        batch=dict(executions=len(rows), airborne_string_hits=3, frame_hit_episodes=1,
                   complete_smash=0, numerical_failures=sum(x['numerical_failures'] for x in rows)),
        caveats=['TRAIN executions include repeated reference policies, not independent formal evaluation.',
                 'Training episodes terminate on irreversible failure. Unrecorded post-termination trajectories are not reconstructed.',
                 'Env20 first-landing marker is an exact native event position; displayed body pose is the last valid 50Hz sample at an earlier time.',
                 'Orange ball rings and yellow contact circles are annotations, not enlarged collision geometry.',
                 'Shared renderer ground visual surface is 5mm below physical ground; native recorded robot/ball coordinates are unchanged.'],
        sources={str(p.relative_to(ROOT)):sha(p) for p in files})
    output.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:meta[k] for k in ['video','bytes','sha256','frames','duration_s']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
