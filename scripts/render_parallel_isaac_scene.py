#!/usr/bin/env python3
"""Render world poses from one multi-environment Isaac Sim process."""
import argparse,json,os
os.environ['LIBGL_ALWAYS_SOFTWARE']='1'; os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']='/usr/share/glvnd/egl_vendor.d/50_mesa.json'
from pathlib import Path
import numpy as np, moderngl, imageio.v2 as imageio, trimesh
from PIL import Image, ImageDraw, ImageFont
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
meta=json.loads((a.input/'metadata.json').read_text()); geom=json.loads((a.input/'geometry.json').read_text()); data=np.load(a.input/'geometry.npz'); poses=np.load(a.input/'parallel_link_poses.npy',mmap_mode='r'); balls=np.load(a.input/'parallel_ball_positions.npy',mmap_mode='r'); origins=np.load(a.input/'env_origins.npy')
N=poses.shape[1]; W,H=1280,720; TOP,BOTTOM=68,82; RENDER_H=H-TOP-BOTTOM
ctx=moderngl.create_standalone_context(backend='egl',device_index=1); fbo=ctx.simple_framebuffer((W,RENDER_H),components=3); fbo.use(); ctx.enable(moderngl.DEPTH_TEST)
program=ctx.program(vertex_shader='''#version 330
uniform mat4 vp; uniform mat4 model; in vec3 position; in vec3 normal; out vec3 N; out vec3 P;
void main(){vec4 p=model*vec4(position,1);P=p.xyz;N=mat3(model)*normal;gl_Position=vp*p;}''',fragment_shader='''#version 330
uniform vec3 color; uniform int floor_mesh; in vec3 N; in vec3 P; out vec4 output_color;
void main(){vec3 n=normalize(N);vec3 l=normalize(vec3(-.6,-.7,1.5));float diffuse=abs(dot(n,l));vec3 c=color*(.38+.62*diffuse);if(floor_mesh==1){vec2 g=abs(fract(P.xy*4.-.5)-.5)/fwidth(P.xy*4.);float line=1.-min(min(g.x,g.y),1.);c=mix(vec3(.83,.86,.89),vec3(.68,.73,.78),line*.5);}output_color=vec4(pow(c,vec3(1./2.2)),1);}''')
def norm(x): return x/np.linalg.norm(x)
def camera(eye,target):
 z=norm(eye-target); x=norm(np.cross([0,0,1],z)); y=np.cross(z,x); view=np.eye(4); view[:3,:3]=np.array([x,y,z]); view[:3,3]=-view[:3,:3]@eye
 f=1/np.tan(np.deg2rad(42)/2); near=.05; far=100.; proj=np.zeros((4,4)); proj[0,0]=f/(W/RENDER_H); proj[1,1]=f; proj[2,2]=(far+near)/(near-far); proj[2,3]=2*far*near/(near-far); proj[3,2]=-1; return proj@view
center=origins.mean(axis=0); span=max(float(np.ptp(origins[:,0])),float(np.ptp(origins[:,1])),8.)
program['vp'].write(camera(center+np.array([span*.75,-span*1.0,span*.85]),center+np.array([1.5,0,.7])).T.astype('f4').tobytes())
def geometry(vertices,faces):
 tri=vertices[faces]; n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12); packed=np.concatenate((tri.reshape(-1,3),np.repeat(n,3,axis=0)),1).astype('f4'); buf=ctx.buffer(packed.tobytes()); return ctx.vertex_array(program,[(buf,'3f 3f','position','normal')])
def model(p):
 x,y,z=p[:3]; w,qx,qy,qz=p[3:]; R=np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*w),2*(qx*qz+qy*w)],[2*(qx*qy+qz*w),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*w)],[2*(qx*qz-qy*w),2*(qy*qz+qx*w),1-2*(qx*qx+qy*qy)]]); M=np.eye(4); M[:3,:3]=R; M[:3,3]=[x,y,z]; return M
objects=[]
for m in geom['meshes']:
 vao=geometry(data[m['key']+'_vertices'],data[m['key']+'_faces']); path=m['path']; color=(.08,.11,.15) if '/Go2/' in path else (.78,.81,.85)
 if '/Piper/' in path and any(x in path for x in ['link2','link4','link6']): color=(.07,.3,.58)
 if 'Racket' in path: color=(.12,.42,.52)
 objects.append((vao,m['body'],m['color'] or color))
def box(center,extent):
 s=trimesh.creation.box(extents=extent); s.apply_translation(center); return geometry(np.asarray(s.vertices),np.asarray(s.faces))
def floor_mesh(center): return box(center,[4.1,3.1,.01])
def net_meshes(o):
 parts=[]
 for y in np.linspace(-1,1,9): parts.append(box(o+[1.5,y,.76],[.012,.012,1.52]))
 for z in np.linspace(0,1.52,8): parts.append(box(o+[1.5,0,z],[.012,2,.012]))
 return parts
sphere=trimesh.creation.icosphere(subdivisions=2,radius=.045); ball_mesh=geometry(np.asarray(sphere.vertices),np.asarray(sphere.faces))
lanes=[]
for o in origins:
 lanes.append((floor_mesh(o+[1.5,0,0]),net_meshes(o),box(o+[3,0,.005],[2,2,.01])))
identity=np.eye(4,dtype='f4').tobytes(); font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; bold_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
def F(n,b=False):
 try:return ImageFont.truetype(bold_path if b else font_path,n)
 except:return ImageFont.load_default()
def frame(idx):
 fbo.clear(.93,.95,.98); program['floor_mesh'].value=1; program['color'].value=(.8,.8,.8); program['model'].write(identity)
 for floor,nets,target in lanes:
  floor.render(); program['floor_mesh'].value=0; program['color'].value=(.25,.58,.37); target.render(); program['color'].value=(.42,.46,.49)
  for part in nets: part.render()
 for lane in range(N):
  for vao,body,color in objects:
   program['model'].write(model(poses[idx,lane,body]).T.astype('f4').tobytes()); program['color'].value=tuple(color); vao.render()
  bp=np.eye(4,dtype='f4'); bp[:3,3]=balls[idx,lane]; program['model'].write(bp.T.tobytes()); program['color'].value=(1.,.35,.04); ball_mesh.render()
 raw=np.frombuffer(fbo.read(components=3,alignment=1),np.uint8).reshape(RENDER_H,W,3)[::-1].copy(); out=Image.new('RGB',(W,H),(12,22,36)); out.paste(Image.fromarray(raw),(0,TOP)); d=ImageDraw.Draw(out); d.rectangle((0,0,W,TOP),fill=(12,22,36)); d.text((24,9),'Isaac Sim Parallel Training Scene',font=F(27,True),fill='white'); d.text((25,41),f'one Isaac process  ·  {N} parallel environments  ·  GPU PhysX  ·  t={idx/30:.2f}s',font=F(15),fill=(164,210,222)); d.rectangle((0,H-BOTTOM,W,H),fill=(12,22,36)); d.text((24,H-BOTTOM+12),'同一个 Isaac Sim 场景中的并行环境：每个网格都是独立机器狗与球场',font=F(17),fill='white'); d.text((24,H-BOTTOM+43),'true multi-env capture  ·  shared simulation process  ·  10 seconds',font=F(15),fill=(164,210,222)); return np.asarray(out)
a.output.parent.mkdir(parents=True,exist_ok=True); writer=imageio.get_writer(str(a.output),fps=meta['fps'],codec='libx264',quality=8,macro_block_size=8,output_params=['-movflags','+faststart'])
try:
 for i in range(len(poses)): writer.append_data(frame(i))
finally: writer.close()
(a.input/'video.json').write_text(json.dumps(dict(path=str(a.output),duration_s=len(poses)/meta['fps'],fps=meta['fps'],resolution=[W,H],num_envs=N,renderer=ctx.info['GL_RENDERER'],single_isaac_process=True),indent=2)+'\n')
print(a.output)
