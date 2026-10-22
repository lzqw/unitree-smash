#!/usr/bin/env python3
"""CPU rasterization of recorded physical link poses; no invented motion."""
import os
os.environ['LIBGL_ALWAYS_SOFTWARE']='1'
os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']='/usr/share/glvnd/egl_vendor.d/50_mesa.json'
import argparse,json,sys
from pathlib import Path
import numpy as np
import moderngl
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args()
meta=json.load(open(a.input/'geometry.json'));data=np.load(a.input/'geometry.npz');poses=np.load(a.input/'link_poses.npy');trace=json.load(open(a.input/'trace.json'));metrics=json.load(open(a.input/'metrics.json'))
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
program['vp'].write(camera(np.array([1.35,-1.9,1.02]),np.array([.02,0,.4])).T.astype('f4').tobytes())
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
frames=[]
for i,pose in enumerate(poses):
 fbo.clear(.94,.96,.98);program['floor_mesh'].value=1;program['color'].value=(.8,.8,.8);program['model'].write(np.eye(4,dtype='f4').tobytes());floor.render()
 program['floor_mesh'].value=0
 for vao,body,color in objects:
  program['model'].write(model(pose[body]).T.astype('f4').tobytes());program['color'].value=tuple(color);vao.render()
 frame=np.frombuffer(fbo.read(components=3,alignment=1),np.uint8).reshape(height,width,3)[::-1].copy();frames.append(frame)
 if i in [0,65,100,199]:imageio.imwrite(a.input/f'preview_{i:03d}.png',frame)
 if i%50==0:print('Rendered',i,'of',len(poses),flush=True)
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';title=ImageFont.truetype(font,29);small=ImageFont.truetype(font,20);big=ImageFont.truetype(font,35)
def overlay(frame,index,label):
 img=Image.fromarray(frame);d=ImageDraw.Draw(img);d.rectangle((0,0,width,88),fill=(18,28,43));d.text((28,10),'Go2 + PiPER + 球拍｜自主跳跃恢复',font=title,fill='white');d.text((28,53),'真实仿真轨迹 · 离线渲染 · 完整负载 20.65 kg',font=small,fill=(166,193,214));d.text((995,25),label,font=title,fill=(115,218,206))
 t=trace[index]['time_s'];e=metrics['episode'];phase='准备' if t<.8 else ('蹬地起跳' if t<e['takeoff_time_s'] else ('腾空' if t<e['landing_time_s'] else '落地恢复'))
 d.rounded_rectangle((26,110,240,160),radius=9,fill=(22,39,55));d.text((42,117),f'{t:4.2f} s  ·  {phase}',font=small,fill='white')
 d.rectangle((0,635,width,height),fill=(18,28,43));d.text((28,644),f"本回合：起跳后 COM 升高 {e['total_com_rise_m']*100:.1f} cm     持续腾空 {e['airborne_s']:.2f} s     连续恢复 {e['recovery_s']:.2f} s",font=small,fill='white');d.text((28,681),'独立双面板成功率：99.5% / 95.5%     当前展示：预先选定的面板 B / 场景 0',font=small,fill=(166,193,214));return np.asarray(img)
out=a.input/'G04_autonomous_jump_demo.mp4'
with imageio.get_writer(str(out),fps=50,codec='libx264',quality=8,macro_block_size=8,output_params=['-movflags','+faststart']) as writer:
 for i in range(200):writer.append_data(overlay(frames[i],i,'正常速度'))
 for i in range(200):
  img=overlay(frames[i],i,'0.5 倍慢放');writer.append_data(img);writer.append_data(img)
 for _ in range(50):writer.append_data(overlay(frames[-1],199,'恢复完成'))
imageio.imwrite(a.input/'poster.jpg',overlay(frames[65],65,'起跳瞬间'))
(a.input/'video.json').write_text(json.dumps(dict(path=str(out),frames=650,fps=50,duration_s=13,resolution=[width,height],renderer=ctx.info['GL_RENDERER'],motion='Unmodified measured body-link poses at 50 Hz; no interpolation or fabricated motion',geometry='USD visual geometry in body-local coordinates; presentation materials and floor grid added',segments=['4 s real time','8 s half speed repeat','1 s final hold']),indent=2))
print(out,flush=True)
