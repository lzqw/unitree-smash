"""Replay an independently verified success with an undistorted perspective view."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent / 'unitree_smash'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(MAIN))
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
from quadsmash.evaluation_v4 import independent_episode
sys.path.insert(0, str(ROOT / 'scripts'))
from render_full_racket_r3_effect import Scene
from render_current_v4_progress import camera, pose_matrix
from native_video_framing_v4 import body_mesh_corners, at_poses, fit


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists() and not args.preview_only:
        raise ValueError('Use a fresh output')
    run = MAIN / 'reports/G07_full_racket_v4/contact_aim7_best_fixed_train128_01'
    policy = MAIN / 'reports/G07_full_racket_v4/contact_aim7_sequential_n128_01/best_candidate.pt'
    protocol_path = MAIN / 'configs/g07_full_racket_eval_v4_r3.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(MAIN)
    cases, indices = training_cases(MAIN, protocol)
    rows = read(run / 'episodes.json')
    summary = summarize_training(rows, cases, indices, protocol)
    assert all(read(run / 'metrics.json')[key] == value for key, value in summary.items())
    assert (summary['complete'], summary['airborne_contacts'], summary['conditional_recovery_count']) == (3, 60, 26)
    assert summary['numerical_failures'] == 0
    inputs, job = read(run / 'INPUTS.json'), read(run / 'job.json')
    assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0
    assert inputs['policy_sha256'] == sha(policy) and inputs['protocol_sha256'] == sha(protocol_path)
    assert not any(inputs['reset_noise_amplitudes'].values())
    assert inputs['reset_noise_seed'] == 724201
    successes = [index for index, row in enumerate(rows) if independent_episode(row, protocol)['complete']]
    lane = max(successes, key=lambda index: np.linalg.norm(rows[index]['outgoing_velocity_mps']))
    assert lane == 29
    row = rows[lane]
    event = next(item for item in read(run / 'contacts.json')
                 if item['env_id'] == lane and item['episode_index'] == 0 and item['first_racket_contact'])
    assert event['string_contact'] and event['legal_initial_contact']
    with np.load(run / 'all_first_episodes.npz') as arrays:
        valid = arrays['valid_first_episode'][:, lane]
        assert valid.all() and len(valid) == 400
        poses = arrays['link_poses'][:, lane].copy()
        balls = arrays['ball_position_m'][:, lane].copy()
        clocks = arrays['episode_clock_s'][:, lane].copy()
    assert np.max(abs(np.diff(clocks) - .02)) < 2e-6 and clocks[0] == 0
    assert np.isfinite(poses).all() and np.isfinite(balls).all()
    assert np.max(abs(np.linalg.norm(poses[:, :, 3:], axis=-1) - 1)) < 1e-4
    court = read(MAIN / 'configs/court_net_v4.json')
    scene = Scene(run, court)
    scene.box([-.35, -1.05, .004], [1., .018, .004], (.95, .95, .90))
    width, height = 1280, 720
    bodies, corners, _ = body_mesh_corners(run)
    robot_points = at_poses(poses, bodies, corners).reshape(-1, 3)
    event_points = at_poses(np.asarray(event['body_link_poses_local']), bodies, corners).reshape(-1, 3)
    target_corners = np.array([[x, y, .002] for x in court['target_x_limits_m']
                               for y in [-court['target_y_half_width_m'], court['target_y_half_width_m']]])
    critical = np.concatenate([robot_points, event_points, target_corners,
                               np.array([row['first_landing_m'], [court['net_plane_x_m'], 0, court['net_center_top_m']]])])
    target = np.array([1.5, 0., .85])
    initial_eye = np.array([0., -14., 1.8])
    vertical_fov = 16.
    for distance_scale in np.linspace(1., 1.8, 17):
        eye = target + distance_scale * (initial_eye - target)
        matrix = camera(eye, target, width / height, vertical_fov)
        projected = np.column_stack([critical, np.ones(len(critical))]) @ matrix.T
        ndc = projected[:, :3] / projected[:, 3:]
        if (projected[:, 3] > 0).all() and np.max(abs(ndc[:, :2])) < .94:
            break
    else:
        raise ValueError('Recorded robot and target do not fit the perspective camera')
    scene.views['main'] = (scene.ctx.simple_framebuffer((width, height), components=3), matrix)
    mini_width, mini_height = 320, 180
    court_points = np.array([[court['net_plane_x_m'] + x, y, z]
        for x in [-court['court_length_m'] / 2 - .3, court['court_length_m'] / 2 + .3]
        for y in [-3.35, 3.35] for z in [0., 2.2]])
    mini_matrix, mini_check = fit(court_points, mini_width, mini_height, False)
    scene.views['mini'] = (scene.ctx.simple_framebuffer((mini_width, mini_height), components=3), mini_matrix)
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title_font, text_font, small_font = [ImageFont.truetype(font, size) for size in [28, 22, 18]]
    outgoing_speed = float(np.linalg.norm(row['outgoing_velocity_mps']))

    def pane(state, view, ball):
        framebuffer, vp = scene.views[view]
        framebuffer.use()
        framebuffer.clear(.07, .095, .13, 1., depth=1.)
        scene.program['vp'].write(vp.T.astype('f4').tobytes())
        scene.program['model'].write(scene.identity)
        for vao, color in scene.static:
            scene.ctx.wireframe = color == (.13, .16, .17)
            scene.program['color'].value = color
            vao.render()
        scene.ctx.wireframe = False
        for vao, body, color in scene.objects:
            scene.program['model'].write(pose_matrix(state[body]).T.astype('f4').tobytes())
            scene.program['color'].value = tuple(color)
            vao.render()
        if ball is not None:
            transform = np.eye(4)
            transform[:3, 3] = ball
            scene.program['model'].write(transform.T.astype('f4').tobytes())
            scene.program['color'].value = (1., .35, .02)
            scene.ball.render()
        w, h = framebuffer.size
        return Image.fromarray(np.frombuffer(framebuffer.read(components=3, alignment=1), np.uint8)
                               .reshape(h, w, 3)[::-1].copy())

    def frame(index, slow=False, exact=False, conclusion=False):
        t = event['time_s'] if exact else float(clocks[index])
        state = np.asarray(event['body_link_poses_local']) if exact else poses[index]
        ball = np.asarray(event['ball_position_m']) if exact else (balls[index] if t >= 2 else None)
        image = pane(state, 'main', ball)
        draw = ImageDraw.Draw(image)
        ruler = scene.project([-.35, -1.05, .004], 'main')
        if ruler:
            draw.text((ruler[0]-12,ruler[1]+5),'1 m',font=small_font,fill='white')
        if t >= row['first_landing_time_s']:
            point = scene.project(row['first_landing_m'], 'main')
            if point:
                x, y = point
                draw.line((x - 8, y - 8, x + 8, y + 8), fill=(255, 145, 80), width=3)
                draw.line((x - 8, y + 8, x + 8, y - 8), fill=(255, 145, 80), width=3)
        mini = pane(state, 'mini', ball)
        md = ImageDraw.Draw(mini)
        if ball is not None:
            point = scene.project(ball, 'mini')
            if point:
                x, y = point
                md.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(255, 220, 100), width=1)
        if t >= row['first_landing_time_s']:
            point = scene.project(row['first_landing_m'], 'mini')
            if point:
                x, y = point
                md.line((x - 4, y - 4, x + 4, y + 4), fill=(255, 145, 80), width=2)
                md.line((x - 4, y + 4, x + 4, y - 4), fill=(255, 145, 80), width=2)
        image.paste(mini, (944, 92))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1280, 80), fill=(18, 28, 43))
        draw.text((18, 9), '真实完整扣杀｜正常透视与米制比例', font=title_font, fill='white')
        draw.text((18, 49), '历史实录：上一仿真布局｜TRAIN29｜当前新版尚无完整成功', font=small_font, fill=(205, 220, 230))
        draw.rectangle((944, 275, 1264, 305), fill=(18, 28, 43))
        draw.text((955, 279), '俯视13.4×5.18m｜圈=球心', font=small_font, fill='white')
        draw.rectangle((0, 625, 1280, 720), fill=(18, 28, 43))
        mode = '实际接触子步定格' if exact else ('0.25倍慢放' if slow else '正常速度')
        draw.text((18, 632), f't={t:.4f}s｜{mode}｜网高1.524m · 球拍675mm', font=text_font, fill='white')
        if conclusion:
            status = f'本回合完整成功：出球{outgoing_speed:.2f}m/s · 首落目标区 · 合格恢复{row["jump"]["recovery_s"]:.2f}s'
        elif exact:
            status = f'合法空中拍线触球｜出球{outgoing_speed:.2f}m/s，vz={row["outgoing_velocity_mps"][2]:.2f}m/s'
        elif t < 2:
            status = '从地面自主准备与走位'
        elif t < row['contact_time_s']:
            status = '真实来球与跳跃拦截'
        elif t < row['first_landing_time_s']:
            status = '空中击球后实际越网'
        else:
            status = f'首次落点({row["first_landing_m"][0]:.3f}, {row["first_landing_m"][1]:.3f})m，继续观察落地恢复'
        draw.text((18, 668), status, font=text_font, fill=(255, 218, 136))
        return image

    previews = []
    for name, index, exact in [('ground', 30, False), ('contact', 141, True),
                               ('landing', 160, False), ('recovery', 360, False)]:
        path = HERE / (name + '.jpg')
        frame(index, exact=exact, conclusion=name == 'recovery').save(path, quality=94)
        previews.append(str(path))
    if args.preview_only:
        print(json.dumps(dict(previews=previews, camera_eye=eye.tolist(), distance_scale=float(distance_scale),
                             renderer=scene.ctx.info['GL_RENDERER'])))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = 0
    segments = []
    with imageio.get_writer(str(args.output), fps=50, codec='libx264', quality=8, macro_block_size=8,
        output_params=['-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart']) as writer:
        start = frames
        for index in range(400):
            writer.append_data(np.asarray(frame(index)))
            frames += 1
        segments.append(dict(kind='complete_actual_first_episode', start_frame=start, frames=400))
        start = frames
        for index in range(122, 176):
            picture = np.asarray(frame(index, slow=True))
            for _ in range(4):
                writer.append_data(picture)
                frames += 1
        segments.append(dict(kind='quarter_speed_actual_frame_repeats', start_frame=start, frames=216))
        picture = np.asarray(frame(141, exact=True))
        for _ in range(50):
            writer.append_data(picture)
            frames += 1
        picture = np.asarray(frame(399, conclusion=True))
        for _ in range(75):
            writer.append_data(picture)
            frames += 1
    sources = [run / name for name in ['episodes.json', 'contacts.json', 'all_first_episodes.npz',
        'geometry.json', 'geometry.npz', 'job.json', 'INPUTS.json', 'source_hashes.json']]
    sources += [policy, protocol_path, MAIN / 'configs/court_net_v4.json', Path(__file__),
                ROOT / 'scripts/render_full_racket_r3_effect.py', ROOT / 'scripts/render_current_v4_progress.py']
    metadata = dict(created_at=datetime.now(timezone.utc).isoformat(), video=str(args.output),
        bytes=args.output.stat().st_size, sha256=sha(args.output), frames=frames, fps=50, duration_s=frames / 50,
        selected_source_TRAIN=29, source_policy_sha256=sha(policy), original128_summary=summary,
        native_episode=row, actual_contact=event, selected_source_frames=list(range(400)), segments=segments,
        main_camera=dict(kind='perspective', eye=eye.tolist(), target=target.tolist(), vertical_fov_degrees=vertical_fov,
                         viewport=[1280, 720], square_pixels=True, all_robot_mesh_bounds_and_target_in_view=True,
                         projection_matrix=matrix.tolist(), maximum_body_target_NDC_xy=float(np.max(abs(ndc[:, :2])))),
        mini_camera=dict(kind='orthographic',viewport=[320,180],equal_world_units_per_pixel=True,checks=mini_check),
        original_native_meshes_and_poses_used=True, model_vertices_scaled=False, ball_geometry_radius_m=.025,
        net_visualization='Wireframe of the same suspended equivalent-band geometry; source collisions and dimensions unchanged.',
        main_ball_overlay_enlarged=False, small_top_ball_marker_is_annotation=True,
        source_is_historical_origin16=True, current_origin0_success_claimed=False,
        frames_interpolated=False, new_native_execution=False,
        source_sha256={str(path): sha(path) for path in sources},
        renderer=scene.ctx.info['GL_RENDERER'], formal_acceptance=False)
    (HERE / 'RENDER_RECEIPT.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    np.savez_compressed(HERE / 'displayed_native_inputs.npz', link_poses=poses, ball_position_m=balls, episode_clock_s=clocks)
    print(json.dumps({key:metadata[key] for key in ['video','bytes','sha256','frames','duration_s','renderer']}), flush=True)


if __name__ == '__main__':
    main()
