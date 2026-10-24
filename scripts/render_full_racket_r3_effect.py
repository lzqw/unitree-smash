#!/usr/bin/env python3
"""Render recorded r3 native poses; no simulation, interpolation or invented hits."""
import os
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = '/usr/share/glvnd/egl_vendor.d/50_mesa.json'
os.environ.setdefault('LP_NUM_THREADS', '4')
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import imageio.v2 as imageio
import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.physics.court_v4 import net_band_segments
from render_current_v4_progress import camera, pose_matrix


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Scene:
    def __init__(self, run, court):
        self.ctx = ctx = moderngl.create_standalone_context(backend='egl', device_index=1)
        assert 'llvmpipe' in ctx.info['GL_RENDERER']
        ctx.enable(moderngl.DEPTH_TEST)
        self.program = ctx.program(vertex_shader='''#version 330
uniform mat4 vp; uniform mat4 model;
in vec3 position; in vec3 normal; out vec3 N;
void main(){N=mat3(model)*normal;gl_Position=vp*model*vec4(position,1.);}
''', fragment_shader='''#version 330
uniform vec3 color; in vec3 N; out vec4 output_color;
void main(){float d=abs(dot(normalize(N),normalize(vec3(-.6,-.7,1.5))));
output_color=vec4(pow(color*(.45+.55*d),vec3(1./2.2)),1.);}
''')
        self.objects, self.static = [], []
        metadata = json.loads((run/'geometry.json').read_text())
        with np.load(run/'geometry.npz') as meshes:
            for item in metadata['meshes']:
                color = (.065, .085, .11) if '/Go2/' in item['path'] else (.72, .76, .80)
                if '/Piper/' in item['path'] and any(x in item['path'] for x in ['link2', 'link4', 'link6']):
                    color = (.06, .27, .50)
                self.objects.append((self.geometry(meshes[item['key']+'_vertices'], meshes[item['key']+'_faces']),
                                     item['body'], item['color'] or color))
        center, length, width = court['net_plane_x_m'], court['court_length_m'], court['doubles_width_m']
        self.box([center, 0, -.015], [length+2, width+2, .02], (.035, .15, .12))
        line = court['line_width_m']
        # The target is a visual marking of the frozen evaluation target.
        self.box([3., 0, .0003], [2., 2., .0004], (.11, .23, .16))
        for sign in [-1, 1]:
            for x in [center+sign*(length/2-line/2), center+sign*1.98, center+sign*(length/2-.76)]:
                self.box([x, 0, .001], [line, width, .002], (.92, .94, .92))
            self.box([center+sign*((length/2+1.98)/2), 0, .001], [length/2-1.98, line, .002], (.92, .94, .92))
            for w in [width, court['singles_width_m']]:
                self.box([center, sign*(w/2-line/2), .001], [length, line, .002], (.92, .94, .92))
        for points, faces in net_band_segments(court):
            vertices = np.asarray(points); vertices[:, 0] += center
            triangles = [[f[0], f[i], f[i+1]] for f in faces for i in range(1, len(f)-1)]
            self.static.append((self.geometry(vertices, triangles), (.13, .16, .17)))
            left, right, top = vertices[0, 1], vertices[2, 1], vertices[1, 2]
            self.box([center, (left+right)/2, top-court['net_top_tape_depth_m']/2],
                     [court['net_equivalent_thickness_m']+.001, right-left, court['net_top_tape_depth_m']], (.95, .95, .95))
        for sign in [-1, 1]:
            mesh = trimesh.creation.cylinder(radius=court['post_radius_m'], height=court['post_height_m'], sections=20)
            mesh.apply_translation([center, sign*court['post_y_center_abs_m'], court['post_height_m']/2])
            self.static.append((self.geometry(mesh.vertices, mesh.faces), (.25, .29, .32)))
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=.025)
        self.ball = self.geometry(mesh.vertices, mesh.faces)
        specs = [('side', 760, 480, [.2, -4., 1.9], [.2, 0, .9], 37),
                 ('top', 504, 480, [.35, -.001, 4.6], [.35, 0, 0], 51),
                 ('overview', 1280, 480, [9., -14., 13.], [1.5, 0, 0], 39)]
        self.views = {n: (ctx.simple_framebuffer((w, h), components=3), camera(e, t, w/h, f)) for n, w, h, e, t, f in specs}
        self.identity = np.eye(4, dtype='f4').T.tobytes()

    def geometry(self, vertices, faces):
        triangles = np.asarray(vertices)[np.asarray(faces)]
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        packed = np.concatenate((triangles.reshape(-1, 3), np.repeat(normals, 3, axis=0)), 1).astype('f4')
        return self.ctx.vertex_array(self.program, [(self.ctx.buffer(packed.tobytes()), '3f 3f', 'position', 'normal')])

    def box(self, center, extents, color):
        mesh = trimesh.creation.box(extents=extents); mesh.apply_translation(center)
        self.static.append((self.geometry(mesh.vertices, mesh.faces), color))

    def project(self, point, view):
        framebuffer, vp = self.views[view]
        p = vp@np.r_[point, 1.]
        if p[3] <= 0:
            return None
        q = p[:3]/p[3]
        if np.max(np.abs(q[:2])) > 1 or abs(q[2]) > 1:
            return None
        w, h = framebuffer.size
        return ((q[0]+1)*w/2, (1-q[1])*h/2)

    def render(self, poses, view, ball=None, trail=None, landing=None):
        framebuffer, vp = self.views[view]; framebuffer.use()
        framebuffer.clear(.07, .095, .13, 1., depth=1.)
        self.program['vp'].write(vp.T.astype('f4').tobytes())
        self.program['model'].write(self.identity)
        for vao, color in self.static:
            self.program['color'].value = color; vao.render()
        for vao, body, color in self.objects:
            self.program['model'].write(pose_matrix(poses[body]).T.astype('f4').tobytes())
            self.program['color'].value = tuple(color); vao.render()
        if ball is not None:
            model = np.eye(4); model[:3, 3] = ball
            self.program['model'].write(model.T.astype('f4').tobytes())
            self.program['color'].value = (1., .35, .02); self.ball.render()
        w, h = framebuffer.size
        frame = Image.fromarray(np.frombuffer(framebuffer.read(components=3, alignment=1), np.uint8).reshape(h, w, 3)[::-1].copy())
        draw = ImageDraw.Draw(frame)
        if trail is not None:
            points = [self.project(p, view) for p in trail]
            for a, b in zip(points, points[1:]):
                if a is not None and b is not None: draw.line([a, b], fill=(255, 172, 76), width=2)
        if ball is not None:
            p = self.project(ball, view)
            if p is not None:
                x, y = p; draw.ellipse((x-7, y-7, x+7, y+7), outline=(255, 210, 100), width=2)
        if landing is not None:
            p = self.project(landing, view)
            if p is not None:
                x, y = p; draw.line((x-10, y-10, x+10, y+10), fill=(255, 120, 110), width=3)
                draw.line((x-10, y+10, x+10, y-10), fill=(255, 120, 110), width=3)
        return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists() and not args.preview_only: raise ValueError('Use a fresh output file')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = ROOT/'reports/G07_full_racket_v4'
    fine, near, rally = [root/n for n in ['reach_cem_fine_feedback01', 'reach_near_net_backface01', 'rally_probe_r3_01']]
    court = json.loads((ROOT/'configs/court_net_v4.json').read_text())
    protocol = json.loads((ROOT/'configs/g07_full_racket_eval_v4_r3.json').read_text())
    assert sha(ROOT/protocol['asset']['path']) == protocol['asset']['sha256']
    f, n = np.load(fine/'public_and_poses_000.npz'), np.load(near/'public_and_poses_000.npz')
    rows_f, rows_n = [json.loads((p/'episodes_000.json').read_text()) for p in [fine, near]]
    poses_r = np.load(rally/'link_poses_first2.partial.npy')
    trace = json.loads((rally/'trajectory_first2.partial.json').read_text())
    obs_r = np.load(rally/'public_observations.partial.npy')
    assert len(poses_r) == len(trace) and poses_r.shape[1:] == (2, 28, 7)
    assert np.allclose(obs_r[::2, 1, 775], np.arange(len(trace))*.02, atol=1e-5)
    # Lane 1 is the second recorded development case, with no reset during this partial trace.
    assert np.all(np.diff(obs_r[:, 1, 775]) > 0)
    assert rows_f[1]['recovery'] and not rows_f[1]['robot_net_contact'] and not rows_f[1]['engineering_body_crossing']
    assert rows_n[0]['robot_net_contact'] and not any(x['contact'][1] for x in trace)
    for data in [f, n]:
        assert np.isfinite(data['link_poses']).all() and data['valid_first_episode'].all()
    # Body-local geometry must be identical in all three runs.
    with np.load(fine/'geometry.npz') as a:
        for directory in [near, rally]:
            with np.load(directory/'geometry.npz') as b:
                assert a.files == b.files and all(np.allclose(a[k], b[k], rtol=0, atol=1e-10) for k in a.files)
    scene = Scene(fine, court)
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title, text, small = [ImageFont.truetype(font, n) for n in [30, 24, 20]]
    landing_i = next(i for i, x in enumerate(trace) if x['time_s'] > 2 and not x['active'][1])
    landing_sample = np.array(trace[landing_i]['ball_position_m'][1]); landing_sample[2] = .026
    source_files = [ROOT/'configs/court_net_v4.json', ROOT/'configs/g07_full_racket_eval_v4_r3.json', ROOT/protocol['asset']['path'], Path(__file__)]
    for directory in [fine, near, rally]:
        source_files += [directory/x for x in ['job.json', 'geometry.json', 'geometry.npz', 'source_hashes.json']]
    source_files += [fine/'public_and_poses_000.npz', fine/'episodes_000.json', near/'public_and_poses_000.npz', near/'episodes_000.json', rally/'link_poses_first2.partial.npy', rally/'trajectory_first2.partial.json', rally/'public_observations.partial.npy', rally/'EARLY_STOP_REASON.json']
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_files}
    previews = []

    def frame(kind, index, slow=False, overview=False):
        if kind == 'jump':
            poses, t = f['link_poses'][index, 1], index*.01
            heading = '完整球拍 r3｜走位 → 起跳挥拍 → 落地恢复'
            subtitle = '无球动作课程；此例完成跳跃恢复，尚未达到所需击球高度和速度'
            status = f"腾空 {rows_f[1]['jump']['airborne_s']:.3f} s｜稳定恢复 {rows_f[1]['jump']['recovery_s']:.2f} s｜无触网 / 无身体越网"
            note = '拍线附近虚拟球心高 1.579 m；该时刻点速向己方，不能算完成扣杀'
            ball = trail = landing = None
        elif kind == 'rally':
            poses, row = poses_r[index, 1], trace[index]; t = row['time_s']
            heading = '完整球拍 r3｜真实来球开发试验：漏球样例'
            subtitle = '原生仿真部分记录；拍线触球未发生，来球落在己方'
            status = '回合前准备' if t < 2 else ('来球飞行中｜拍线触球：无' if row['active'][1] else '回合已结束｜后续弹跳不再计分')
            note = '橙色圆圈 / 线：记录的球心轨迹；红叉：首次落地附近的 50 Hz 采样位置'
            ball = np.array(row['ball_position_m'][1]) if t >= 2 else None
            trail = [x['ball_position_m'][1] for x in trace[max(100, index-30):index+1]] if t >= 2 else None
            landing = landing_sample if index >= landing_i else None
        else:
            poses, t = n['link_poses'][index, 0], index*.01
            heading = '完整球拍 r3｜较新近网候选：触网失败'
            subtitle = '无球动作搜索；原生接触力记录确认触网，本例不能作为合法动作'
            status = f"本回合网接触峰值 {rows_n[0]['maximum_net_force_n']:.0f} N｜未完成稳定恢复"
            note = '展示失败以定位走位与挥拍问题；碰撞后的高度不能作为正常起跳能力'
            ball = trail = landing = None
        canvas = Image.new('RGB', (1280, 720), (18, 28, 43))
        if overview:
            canvas.paste(scene.render(poses, 'overview', ball, trail, landing), (0, 106))
        else:
            canvas.paste(scene.render(poses, 'side', ball, trail, landing), (0, 106))
            canvas.paste(scene.render(poses, 'top', ball, trail, landing), (776, 106))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 10), heading, font=title, fill='white')
        draw.text((20, 57), subtitle, font=text, fill=(187, 214, 231))
        draw.text((18, 113), '全场概览｜己方 ← 网 → 对方' if overview else '侧视｜己方 ← 网 → 对方', font=small, fill='white')
        if not overview: draw.text((790, 113), '俯视', font=small, fill='white')
        draw.rectangle((16, 550, 495, 582), fill=(18, 28, 43))
        draw.text((24, 552), f"t = {t:.2f} s｜{'0.25 倍慢放' if slow else '正常速度'}｜实际记录回放", font=small, fill='white')
        draw.text((20, 598), status, font=text, fill=(255, 205, 131))
        draw.text((20, 640), note, font=small, fill=(211, 225, 234))
        draw.text((20, 681), '675 mm 完整球拍｜标准尺寸场地 / 悬空网｜本视频尚无合法完整扣杀成功', font=small, fill=(163, 188, 204))
        return canvas

    for kind, index in [('jump', 283), ('rally', 140), ('rally', landing_i), ('net', 280)]:
        path = args.output.with_name(args.output.stem+f'_{kind}_{index}.jpg')
        frame(kind, index).save(path, quality=92); previews.append(str(path))
    if args.preview_only:
        print(json.dumps(dict(previews=previews, renderer=scene.ctx.info['GL_RENDERER']), ensure_ascii=False), flush=True)
        return

    segments = []
    fps = 50
    def card(heading, lines):
        image = Image.new('RGB', (1280, 720), (18, 28, 43)); draw = ImageDraw.Draw(image)
        draw.text((50, 100), heading, font=title, fill='white')
        draw.line((50, 175, 1230, 175), fill=(73, 187, 168), width=3)
        for i, line in enumerate(lines): draw.text((50, 230+i*80), line, font=text, fill=(214, 229, 238))
        return np.asarray(image)

    with imageio.get_writer(str(args.output), fps=fps, codec='libx264', quality=8, macro_block_size=8,
                           output_params=['-movflags', '+faststart']) as writer:
        intro = card('当前效果｜完整球拍 r3 · 2026-10-24', ['画面来自已记录的 Isaac / PhysX 实际执行。', '包含起跳恢复、真实来球漏球，以及较新候选的触网失败。', '侧视与俯视使用同一时刻的全身刚体姿态。', '当前仍未完成合法跳跃扣杀；新版正式 400 案例尚未运行。'])
        for _ in range(100): writer.append_data(intro)
        segments.append(dict(name='introduction', duration_s=2))
        clips = [('rally', range(0, 166), 1, True),
                 ('jump', range(0, 600, 2), 1, False),
                 ('jump', range(240, 342, 2), 4, False),
                 ('rally', range(0, 226), 1, False),
                 ('rally', range(120, 159), 4, False),
                 ('net', range(0, 600, 2), 1, False),
                 ('net', range(190, 341, 2), 4, False)]
        for kind, indices, copies, overview in clips:
            for index in indices:
                array = np.asarray(frame(kind, index, copies > 1, overview))
                for _ in range(copies): writer.append_data(array)
            segments.append(dict(name=kind+('_overview' if overview else '_side_top')+('_quarter_speed' if copies > 1 else ''),
                                 start_recorded_time_s=indices[0]*(.02 if kind=='rally' else .01),
                                 end_recorded_time_s=indices[-1]*(.02 if kind=='rally' else .01),
                                 duration_s=len(indices)*copies/fps, formal_acceptance=False))
            print('Rendered '+segments[-1]['name'], flush=True)
        ending = card('目前结果与正在推进的工作', ['已有能自主起跳并恢复的无球动作样例。', '真实来球探针未触球；较新近网搜索出现触网，需要继续适配。', '当前继续优化走位、拍头高度、挥拍速度和落地恢复。', '本视频为开发进展，不是新版策略验收通过。'])
        for _ in range(150): writer.append_data(ending)
        segments.append(dict(name='ending', duration_s=3))
    meta = dict(created_at=datetime.now(timezone.utc).isoformat(), video=str(args.output), sha256=sha(args.output),
                bytes=args.output.stat().st_size, fps=fps, resolution=[1280, 720], segments=segments,
                duration_s=sum(s['duration_s'] for s in segments), renderer=scene.ctx.info['GL_RENDERER'],
                renderer_uses_gpu=False, simulation_uses_gpu=0, poses_are_native_recorded=True,
                interpolated_or_generated_motion=False, formal_acceptance=False, asset_version='full-racket-court-v4-r3',
                caveats=['Real-ball clip is a partial diagnostic, not a completed evaluation.',
                         'Fine-feedback no-ball curriculum did not explicitly enforce ball-material parity.',
                         'Near-net candidate has native net contacts and is invalid.',
                         'Ball trajectory ring and first-landing vicinity cross are annotations, not enlarged collision geometry.',
                         'Landing cross uses the first dead-ball 50Hz sample, not an exact substep contact point.'],
                sources=source_hashes, previews=previews)
    args.output.with_suffix('.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: meta[k] for k in ['video', 'sha256', 'bytes', 'duration_s']}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
