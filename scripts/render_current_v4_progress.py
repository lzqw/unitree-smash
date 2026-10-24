#!/usr/bin/env python3
"""Render recorded v4 motion and explicitly labelled legacy policy comparisons."""
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
import imageio_ffmpeg
import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.physics.court_v4 import net_band_segments


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose_matrix(pose):
    w, x, y, z = pose[3:]
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    matrix[:3, 3] = pose[:3]
    return matrix


def camera(eye, target, aspect, field=35):
    eye, target = np.asarray(eye), np.asarray(target)
    z = eye-target
    z /= np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    view = np.eye(4)
    view[:3, :3] = [x, y, z]
    view[:3, 3] = -view[:3, :3]@eye
    f = 1/np.tan(np.deg2rad(field)/2)
    near, far = .03, 80.
    projection = np.zeros((4, 4))
    projection[0, 0], projection[1, 1] = f/aspect, f
    projection[2, 2] = (far+near)/(near-far)
    projection[2, 3] = 2*far*near/(near-far)
    projection[3, 2] = -1
    return projection@view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output filename')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((args.input/'metrics.json').read_text())
    job = json.loads((args.input/'job.json').read_text())
    assert job['status'] == 'SUCCEEDED'
    assert not metrics['formal_acceptance'] and metrics['ground_initialization']
    court = metrics['court']
    meta = json.loads((args.input/'geometry.json').read_text())
    meshes = np.load(args.input/'geometry.npz')
    poses = np.load(args.input/'link_poses.npy')
    trace = json.loads((args.input/'trajectory.json').read_text())
    assert len(poses) == len(trace) and np.isfinite(poses).all()
    assert np.max(np.abs(np.diff([row['time_s'] for row in trace])-.02)) < 1e-8
    assert len(poses) == 200
    current_sources = [args.input/name for name in (
        'geometry.json', 'geometry.npz', 'link_poses.npy', 'trajectory.json',
        'metrics.json', 'job.json', 'source_hashes.json')]
    current_sources += [Path(metrics['asset'])]
    raw_hashes = {str(path): sha(path) for path in current_sources}
    assert sha(Path(metrics['asset'])) == metrics['asset_sha256']

    ctx = moderngl.create_standalone_context(backend='egl', device_index=1)
    assert 'llvmpipe' in ctx.info['GL_RENDERER']
    ctx.enable(moderngl.DEPTH_TEST)
    program = ctx.program(vertex_shader='''#version 330
uniform mat4 vp; uniform mat4 model;
in vec3 position; in vec3 normal; out vec3 N;
void main(){N=mat3(model)*normal;gl_Position=vp*model*vec4(position,1.);}
''', fragment_shader='''#version 330
uniform vec3 color; in vec3 N; out vec4 output_color;
void main(){float d=abs(dot(normalize(N),normalize(vec3(-.6,-.7,1.5))));
output_color=vec4(pow(color*(.45+.55*d),vec3(1./2.2)),1.);}
''')

    def geometry(vertices, faces):
        triangles = np.asarray(vertices)[np.asarray(faces)]
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        packed = np.concatenate((triangles.reshape(-1, 3), np.repeat(normals, 3, axis=0)), 1).astype('f4')
        buffer = ctx.buffer(packed.tobytes())
        return ctx.vertex_array(program, [(buffer, '3f 3f', 'position', 'normal')])

    objects, static = [], []
    for item in meta['meshes']:
        color = (.065, .085, .11) if '/Go2/' in item['path'] else (.72, .76, .80)
        if '/Piper/' in item['path'] and any(link in item['path'] for link in ['link2', 'link4', 'link6']):
            color = (.06, .27, .50)
        objects.append((geometry(meshes[item['key']+'_vertices'], meshes[item['key']+'_faces']),
                        item['body'], item['color'] or color))

    def box(center, extents, color):
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
        static.append((geometry(mesh.vertices, mesh.faces), color))

    center, length, width = court['net_plane_x_m'], court['court_length_m'], court['doubles_width_m']
    box([center, 0, -.015], [length+2, width+2, .02], (.035, .15, .12))
    line = court['line_width_m']
    for sign in [-1, 1]:
        for x in [center+sign*(length/2-line/2), center+sign*1.98,
                  center+sign*(length/2-.76)]:
            box([x, 0, .001], [line, width, .002], (.92, .94, .92))
        box([center+sign*((length/2+1.98)/2), 0, .001],
            [length/2-1.98, line, .002], (.92, .94, .92))
        for w in [width, court['singles_width_m']]:
            box([center, sign*(w/2-line/2), .001], [length, line, .002], (.92, .94, .92))
    for points, faces in net_band_segments(court):
        vertices = np.asarray(points)
        vertices[:, 0] += center
        triangles = [[f[0], f[i], f[i+1]] for f in faces for i in range(1, len(f)-1)]
        static.append((geometry(vertices, triangles), (.13, .16, .17)))
        left, right = vertices[0, 1], vertices[2, 1]
        top = vertices[1, 2]
        box([center, (left+right)/2, top-court['net_top_tape_depth_m']/2],
            [court['net_equivalent_thickness_m']+.001, right-left,
             court['net_top_tape_depth_m']], (.95, .95, .95))
    for sign in [-1, 1]:
        mesh = trimesh.creation.cylinder(radius=court['post_radius_m'], height=court['post_height_m'], sections=20)
        mesh.apply_translation([center, sign*court['post_y_center_abs_m'], court['post_height_m']/2])
        static.append((geometry(mesh.vertices, mesh.faces), (.25, .29, .32)))

    # All three cameras use the same actual body poses and the recorded court.
    view_specs = [
        ('overview', 1280, 470, [11., -13., 13.], [1.5, 0., 0.], 37),
        ('side', 800, 470, [1.2, -3.2, 1.3], [.35, 0., .66], 38),
        ('top', 472, 470, [.3, -.001, 3.6], [.3, 0., 0.], 40)]
    views = {}
    for name, w, h, eye, target, field in view_specs:
        views[name] = (ctx.simple_framebuffer((w, h), components=3),
                       camera(eye, target, w/h, field))
    identity = np.eye(4, dtype='f4').T.tobytes()

    def scene(index, name):
        framebuffer, vp = views[name]
        framebuffer.use()
        framebuffer.clear(.07, .095, .13, 1., depth=1.)
        program['vp'].write(vp.T.astype('f4').tobytes())
        program['model'].write(identity)
        for vao, color in static:
            program['color'].value = color
            vao.render()
        for vao, body, color in objects:
            program['model'].write(pose_matrix(poses[index, body]).T.astype('f4').tobytes())
            program['color'].value = tuple(color)
            vao.render()
        w, h = framebuffer.size
        return Image.fromarray(np.frombuffer(framebuffer.read(components=3, alignment=1), np.uint8).reshape(h, w, 3)[::-1].copy())

    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title = ImageFont.truetype(font, 32)
    text = ImageFont.truetype(font, 25)
    small = ImageFont.truetype(font, 21)
    previews = []

    def current_frame(index, overview=False, slow=False):
        frame = Image.new('RGB', (1280, 720), (18, 28, 43))
        if overview:
            frame.paste(scene(index, 'overview'), (0, 100))
        else:
            frame.paste(scene(index, 'side'), (0, 100))
            frame.paste(scene(index, 'top'), (808, 100))
        draw = ImageDraw.Draw(frame)
        draw.text((22, 12), '当前新版 v4｜完整球拍 + 标准尺寸球场', font=title, fill='white')
        draw.text((22, 60), '实际仿真记录 · 站立与机械臂小幅运动测试 · 本段没有来球或扣杀', font=text, fill=(177, 209, 225))
        draw.text((20, 108), '全场：己方 ← 球网 → 对方' if overview else '侧视', font=text, fill='white')
        if not overview:
            draw.text((828, 108), '俯视', font=text, fill='white')
        draw.text((22, 585), f"t = {trace[index]['time_s']:.2f} s｜{'0.5 倍慢放' if slow else '正常速度'}｜总质量 {metrics['total_mass_kg']:.3f} kg", font=text, fill='white')
        draw.text((22, 626), '整拍 675 mm；法兰至拍线中心约 452 mm；球拍和夹具质量 153 g', font=text, fill=(207, 224, 233))
        draw.text((22, 667), '网宽 6.10 m / 深 0.76 m；中央 1.524 m / 侧边 1.55 m；新版扣杀尚未验证', font=small, fill=(255, 201, 121))
        return frame

    def card(title_text, lines):
        frame = Image.new('RGB', (1280, 720), (18, 28, 43))
        draw = ImageDraw.Draw(frame)
        draw.text((48, 100), title_text, font=title, fill='white')
        draw.rectangle((48, 170, 1232, 174), fill=(73, 187, 168))
        for i, line_text in enumerate(lines):
            draw.text((48, 225+i*90), line_text, font=text, fill=(220, 230, 240))
        return frame

    segments = []
    legacy_hashes = {}
    with imageio.get_writer(str(args.output), fps=50, codec='libx264', quality=8,
                           macro_block_size=8, output_params=['-movflags', '+faststart']) as writer:
        for name, overview, copies in [('current_court_overview', True, 1),
                                        ('current_side_and_top', False, 1),
                                        ('current_half_speed', False, 2)]:
            for index in range(len(poses)):
                frame = current_frame(index, overview, copies == 2)
                for _ in range(copies):
                    writer.append_data(np.asarray(frame))
                if index == 130:
                    preview = args.output.with_name(args.output.stem+'_'+name+'.jpg')
                    frame.save(preview)
                    previews.append(str(preview))
            segments.append(dict(name=name, duration_s=len(poses)*copies/50,
                                 source=str(args.input), formal_acceptance=False))
            print('Rendered '+name, flush=True)
        intro = card('旧模型扣杀试验对照｜以下画面不是新版球拍', [
            '使用之前的短方拍板和 2 m 宽研究网。',
            '展示最新旧版候选的固定案例 0 与 1，均为实际运行。',
            '旧版开发组完整成功 4 / 16；不能继承为新版成绩。'])
        for _ in range(100):
            writer.append_data(np.asarray(intro))
        segments.append(dict(name='legacy_introduction', duration_s=2.))
        for case in [0, 1]:
            clip = args.legacy/f'latest_case{case}.mp4'
            legacy_hashes[str(clip)] = sha(clip)
            reader = imageio.get_reader(str(clip))
            try:
                for index in range(425):
                    frame = Image.fromarray(reader.get_data(index))
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 1280, 88), fill=(43, 32, 22))
                    draw.text((22, 8), f"旧版短方拍板｜固定案例 {case}｜{'成功样例' if case == 0 else '失败样例'}", font=title, fill='white')
                    draw.text((22, 51), '历史版本动作对照 · 2 m 宽研究网 · 正常速度 · 非当前完整球拍结果', font=small, fill=(255, 204, 143))
                    writer.append_data(np.asarray(frame))
                    if index == 200:
                        preview = args.output.with_name(args.output.stem+f'_legacy_case{case}.jpg')
                        frame.save(preview)
                        previews.append(str(preview))
            finally:
                reader.close()
            segments.append(dict(name=f'legacy_fixed_case_{case}', duration_s=8.5,
                                 source=str(clip), asset_version='legacy-short-board'))
            print(f'Appended legacy case {case}', flush=True)
        ending = card('当前进展：新版模型已运行，扣杀能力仍待适配验证', [
            '新版：站立、质量惯量和拍头运动学测试通过。',
            '新版：触球事件与合法性判定仍在排错。',
            '旧版：有完整动作样例，整体成功率未达标。',
            '后续工作：修好触球判定，继续训练和独立评测。'])
        for _ in range(150):
            writer.append_data(np.asarray(ending))
        segments.append(dict(name='current_status', duration_s=3.))

    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(args.output))
    expected = round(sum(segment['duration_s'] for segment in segments)*50)
    assert frames == expected and abs(seconds-expected/50) < .05
    assert all(sha(Path(name)) == digest for name, digest in {**raw_hashes, **legacy_hashes}.items())
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), path=str(args.output),
                    bytes=args.output.stat().st_size, sha256=sha(args.output), frames=frames,
                    fps=50, duration_s=seconds, resolution=[1280, 720], segments=segments,
                    renderer=ctx.info['GL_RENDERER'], source_hashes={**raw_hashes, **legacy_hashes},
                    simulation_job=str(args.input/'job.json'), previews=previews,
                    formal_acceptance=False, new_policy_training=False,
                    description='Measured current v4 mechanical motion plus explicitly labelled legacy success/failure examples; no fabricated action or contact.')
    args.output.with_suffix('.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key: manifest[key] for key in ['path', 'sha256', 'bytes', 'duration_s']}), flush=True)


if __name__ == '__main__':
    main()
