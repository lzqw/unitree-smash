#!/usr/bin/env python3
"""Render a measured high-arm PhysX trajectory without inventing motion."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=json.loads((a.input/'trajectory.json').read_text());a.output.parent.mkdir(parents=True,exist_ok=True)
fig=plt.figure(figsize=(11,6.2),dpi=120);ax=fig.add_subplot(111,projection='3d');ax.set_xlim(-.4,2.2);ax.set_ylim(-1.0,1.0);ax.set_zlim(0,3.0);ax.set_xlabel('x / m');ax.set_ylabel('y / m');ax.set_zlabel('z / m');ax.view_init(elev=20,azim=-68);ax.set_title('G07 高臂真实 PhysX 下扣轨迹（测量可视化）')
ax.plot([1.5,1.5],[ -1,1],[0,0],alpha=0) # keep axes stable
netx=np.full(40,1.5);nety=np.linspace(-1,1,40);netz=np.linspace(0,1.524,40);ax.plot(netx,nety,netz,color='gray',lw=2,alpha=.7)
ball,=ax.plot([],[],[],'o',color='#e63946',ms=8);tool,=ax.plot([],[],[],'s',color='#1677ff',ms=7);trail,=ax.plot([],[],[],color='#e63946',lw=1.5,alpha=.65);line,=ax.plot([],[],[],color='#1677ff',lw=1.2,alpha=.5)
text=fig.text(.02,.94,'',fontsize=11,ha='left',va='top',family='DejaVu Sans')
writer=FFMpegWriter(fps=30,metadata={'title':'G07 measured PhysX trajectory'},bitrate=5000)
with writer.saving(fig,str(a.output),dpi=120):
 for i,r in enumerate(rows):
  b=np.asarray(r['ball_xyz']);t=np.asarray(r['tool_xyz']);ball.set_data([b[0]],[b[1]]);ball.set_3d_properties([b[2]]);tool.set_data([t[0]],[t[1]]);tool.set_3d_properties([t[2]])
  hist=np.asarray([q['ball_xyz'] for q in rows[:i+1]]);trail.set_data(hist[:,0],hist[:,1]);trail.set_3d_properties(hist[:,2]);th=np.asarray([q['tool_xyz'] for q in rows[:i+1]]);line.set_data(th[:,0],th[:,1]);line.set_3d_properties(th[:,2]);
  v=np.asarray(r['ball_v']); contact='接触已记录' if r['contact'] else ('下行接近' if v[2]<-1 else '飞行中'); text.set_text(f"t={r['time_s']:.2f}s   球速={np.linalg.norm(v):.2f} m/s   vz={v[2]:+.2f} m/s   {contact}\n红球=PhysX梭球，蓝方块=球拍；灰线=1.524m球网")
  writer.grab_frame()
plt.close(fig)
(a.output.with_suffix('.json')).write_text(json.dumps({'source':str(a.input/'trajectory.json'),'frames':len(rows),'fps':30,'measured_positions':True,'renderer':'matplotlib 3D'},indent=2)+'\n')
