#!/usr/bin/env python3
"""CPU rendering of independently replayed G07 curriculum motion."""
import os
os.environ['LIBGL_ALWAYS_SOFTWARE']='1'
os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']='/usr/share/glvnd/egl_vendor.d/50_mesa.json'
import argparse,json,sys
from pathlib import Path
import numpy as np
import moderngl
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--normal-only',action='store_true');p.add_argument('--fps',type=float,default=50.0);a=p.parse_args()
if a.fps <= 0: p.error('--fps must be positive')
meta=json.load(open(a.input/'geometry.json'));data=np.load(a.input/'geometry.npz');poses=np.load(a.input/'link_poses.npy');trace=json.load(open(a.input/'trajectory.json'));metrics=json.load(open(a.input/'metrics.json'))
learned=metrics.get('controller','').startswith('learned')
net_corrected=metrics.get('net_geometry_version') in {'environment-local-net-v2','environment-local-profiled-net-v3'}
profiled_net=metrics.get('net_geometry_version')=='environment-local-profiled-net-v3'
ctx=moderngl.create_standalone_context(backend='egl',device_index=1)
assert 'llvmpipe' in ctx.info['GL_RENDERER'],ctx.info['GL_RENDERER']
width,height=1280,720
fbo=ctx.simple_framebuffer((width,height),components=3);fbo.use();ctx.enable(moderngl.DEPTH_TEST)
program=ctx.program(vertex_shader='''#version 330
uniform mat4 vp; uniform mat4 model; in vec3 position; in vec3 normal;
out vec3 N; out vec3 P;
void main(){vec4 p=model*vec4(position,1);P=p.xyz;N=mat3(model)*normal;gl_Position=vp*p;}
''',fragment_shader='''#version 330
uniform vec3 color; uniform int floor_mesh; in vec3 N; in vec3 P; out vec4 output_color;
void main(){vec3 n=normalize(N);vec3 l=normalize(vec3(-.6,-.7,1.5));
float diffuse=abs(dot(n,l));vec3 c=color*(.38+.62*diffuse);
if(floor_mesh==1){vec2 g=abs(fract(P.xy*4.-.5)-.5)/fwidth(P.xy*4.);float line=1.-min(min(g.x,g.y),1.);c=mix(vec3(.83,.86,.89),vec3(.68,.73,.78),line*.5);}
output_color=vec4(pow(c,vec3(1./2.2)),1);}
''')
def normalize(x):return x/np.linalg.norm(x)
def camera(eye,target):
 z=normalize(eye-target);x=normalize(np.cross([0,0,1],z));y=np.cross(z,x);view=np.eye(4);view[:3,:3]=np.array([x,y,z]);view[:3,3]=-view[:3,:3]@eye
 f=1/np.tan(np.deg2rad(32)/2);near=.05;far=30.;proj=np.zeros((4,4));proj[0,0]=f/(width/height);proj[1,1]=f;proj[2,2]=(far+near)/(near-far);proj[2,3]=2*far*near/(near-far);proj[3,2]=-1
 return proj@view
program['vp'].write(camera(np.array([5.5,-6.5,3.25]),np.array([1.9,0,.70])).T.astype('f4').tobytes())
def geometry(vertices,faces):
 tri=vertices[faces];norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm/=np.maximum(np.linalg.norm(norm,axis=1,keepdims=True),1e-12)
 packed=np.concatenate((tri.reshape(-1,3),np.repeat(norm,3,axis=0)),1).astype('f4');buf=ctx.buffer(packed.tobytes());return ctx.vertex_array(program,[(buf,'3f 3f','position','normal')])
objects=[]
for m in meta['meshes']:
 vao=geometry(data[m['key']+'_vertices'],data[m['key']+'_faces'])
 path=m['path']
 color=(.08,.11,.15) if '/Go2/' in path else (.78,.81,.85)
 if '/Piper/' in path and any(x in path for x in ['link2','link4','link6']):color=(.07,.3,.58)
 if 'Racket' in path:color=(.12,.42,.52)
 objects.append((vao,m['body'],m['color'] or color))
floor=geometry(np.array([[-6,-6,-.002],[6,-6,-.002],[6,6,-.002],[-6,6,-.002]]),np.array([[0,1,2],[0,2,3]]))
def model(pose):
 x,y,z=pose[:3];w,qx,qy,qz=pose[3:];R=np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*w),2*(qx*qz+qy*w)],[2*(qx*qy+qz*w),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*w)],[2*(qx*qz-qy*w),2*(qy*qz+qx*w),1-2*(qx*qx+qy*qy)]])
 M=np.eye(4);M[:3,:3]=R;M[:3,3]=[x,y,z];return M

# Presentation geometry: frozen standard net dimensions, target and ball radius.
import trimesh
sphere=trimesh.creation.icosphere(subdivisions=2,radius=.025)
ball_mesh=geometry(np.asarray(sphere.vertices),np.asarray(sphere.faces))
net_parts=[]
def box(center,extent):
 shape=trimesh.creation.box(extents=extent);shape.apply_translation(center)
 return geometry(np.asarray(shape.vertices),np.asarray(shape.faces))
for y in np.linspace(-1,1,17):
 h=1.524+(.026*y*y if profiled_net else 0.)
 net_parts.append(box([1.5,y,h/2],[.0025,.0025,h]))
for z in np.linspace(0,1.524,18):net_parts.append(box([1.5,0,z],[.0025,2,.0025]))
if profiled_net:
 for y in np.linspace(-1,1,33):net_parts.append(box([1.5,y,1.524+.026*y*y],[.02,.065,.004]))
else:net_parts.append(box([1.5,0,1.524],[.02,2,.008]))
target=box([3,0,.0001],[2,2,.001])
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
title=ImageFont.truetype(font,27);small=ImageFont.truetype(font,20)
identity=np.eye(4,dtype='f4').tobytes()
def draw(index,slow=False):
 pose=poses[index];r=trace[index]
 fbo.clear(.94,.96,.98);program['floor_mesh'].value=1;program['color'].value=(.8,.8,.8);program['model'].write(identity);floor.render()
 program['floor_mesh'].value=0;program['color'].value=(.25,.58,.37);target.render()
 program['color'].value=(.42,.46,.49)
 for mesh in net_parts:mesh.render()
 for vao,body,color in objects:
  program['model'].write(model(pose[body]).T.astype('f4').tobytes());program['color'].value=tuple(color);vao.render()
 bp=np.eye(4,dtype='f4');bp[:3,3]=r['ball_xyz'];program['model'].write(bp.T.tobytes());program['color'].value=(1.,.35,.04);ball_mesh.render()
 frame=np.frombuffer(fbo.read(components=3,alignment=1),np.uint8).reshape(height,width,3)[::-1].copy()
 img=Image.fromarray(frame);d=ImageDraw.Draw(img);d.rectangle((0,0,width,88),fill=(18,28,43))
 heading=metrics.get('presentation_title') or 'Go2 + PiPER｜'+('正确球网下重验' if net_corrected else '历史网坐标未修正')+' · '+('学习策略' if learned else '教师')
 d.text((24,8),heading,font=title,fill='white')
 net_label='中心1.524 / 侧边1.55 m 网体v3' if profiled_net else '恒高1.524 m矩形网课程' if net_corrected else '历史网位置错误'
 subtitle=metrics.get('presentation_subtitle') or '实测关节连杆与球轨迹 · 原负载与执行器 · '+net_label
 d.text((24,50),subtitle,font=small,fill=(166,193,214))
 d.text((1040,20),'0.5 倍慢放' if slow else '正常速度',font=small,fill=(115,218,206))
 t=r['time_s'];e=metrics['best'];v=np.asarray(r['ball_v'])
 contact_time=e.get('contact_time_s') or float('inf');landing_time=e.get('first_landing_time_s') or float('inf')
 phase=('击球后腾空' if r.get('airborne') else '着地/恢复') if r.get('contact') else ('腾空拦截' if r.get('airborne') else '地面准备/移动')
 d.rectangle((20,104,390,143),fill=(22,39,55));d.text((30,109),f"{t:.2f} s · {phase} · 球速 {np.linalg.norm(v):.2f}",font=small,fill='white')
 d.rectangle((0,623,width,height),fill=(18,28,43))
 out=e.get('outgoing_velocity_mps');landing=e.get('first_landing_position_m');margin=e.get('net_margin_m')
 out_text='未形成稳定出球' if out is None or not np.isfinite(out).all() else f'出球 [{out[0]:.2f}, {out[1]:.2f}, {out[2]:.2f}] m/s'
 margin_text='未记录越网' if margin is None or not np.isfinite(margin) else f'球表面越网净空 {margin*100:.1f} cm'
 landing_text='未记录首次落点' if landing is None or not np.isfinite(landing).all() else f'首次落点 [{landing[0]:.2f}, {landing[1]:.2f}] m'
 d.text((24,632),f"本回合{out_text}｜{margin_text}｜{landing_text}",font=small,fill='white')
 face='是' if e.get('effective_face_contact') else '否'
 yes=lambda value:'是' if value else '否'
 valid=bool(e.get('complete5', e.get('complete_smash')) and not e.get('robot_net_contact') and not e.get('numerical_failures'))
 d.text((24,658),f"本回合完整：{yes(valid)}｜机器人碰网：{yes(e.get('robot_net_contact'))}｜稳定恢复：{yes(e.get('recovery_success'))}｜有效拍面：{face}",font=small,fill=(220,225,235))
 population='正式单面板' if metrics.get('formal_panel_recording') else '开发整组'
 conclusion='完整结论见双面板400回合报告' if metrics.get('formal_panel_recording') else '非正式400回合验收'
 population=metrics.get('presentation_population',population)
 conclusion=metrics.get('presentation_conclusion',conclusion)
 recovery_text=f"条件恢复 {metrics['conditional_recovery_count']}/{metrics['airborne_contact_count']}" if 'conditional_recovery_count' in metrics else f"原始恢复 {metrics['recovery']}"
 d.text((24,687),f"{population} {metrics['episodes']} 回合：完整 {metrics['complete']}，{recovery_text}｜案例 {metrics['selected_video_lane']}｜{conclusion}",font=small,fill=(166,193,214))
 return np.asarray(img)
out=a.output or a.input/('G07_near_net_learned_replay.mp4' if learned else 'G07_near_net_teacher_replay.mp4')
contact_index=min(range(len(trace)),key=lambda i:abs(trace[i]['time_s']-(metrics['best'].get('contact_time_s') or 1.48)))
with imageio.get_writer(str(out),fps=a.fps,codec='libx264',quality=8,macro_block_size=8,output_params=['-movflags','+faststart']) as writer:
 for slow in ((False,) if a.normal_only else (False,True)):
  for i in range(len(poses)):
   frame=draw(i,slow);writer.append_data(frame)
   if slow:writer.append_data(frame)
   if not slow and i in (0,contact_index,max(0,contact_index-5),min(len(poses)-1,contact_index+10)):imageio.imwrite(a.input/f'preview_{i:03d}.jpg',frame)
imageio.imwrite(a.input/'poster.jpg',draw(contact_index))
(a.input/'video.json').write_text(json.dumps(dict(path=str(out),frames=len(poses)*(1 if a.normal_only else 3),fps=a.fps,duration_s=len(poses)/a.fps*(1 if a.normal_only else 3),resolution=[width,height],renderer=ctx.info['GL_RENDERER'],motion=f'Measured {a.fps:g} Hz link poses and ball positions, no invented motion',development_recording=not metrics.get('formal_panel_recording',False),formal_acceptance=False,selected_lane=metrics['selected_video_lane'],segments=['normal speed'] if a.normal_only else ['normal speed','half speed replay']),indent=2)+'\n')
print(out,flush=True)
