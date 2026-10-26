"""CPU render of one audited real TRAIN completion and two actual failures."""
import datetime,hashlib,json,os,pathlib,subprocess,sys,traceback
ROOT=pathlib.Path(__file__).resolve().parents[3];HERE=pathlib.Path(__file__).resolve().parent
MAIN=ROOT.parent/'unitree_smash';SNAPSHOT=MAIN/'reports/G07_full_racket_v4/video_snapshot_placement075_first_complete_20261026'
AUDIT=MAIN/'reports/G07_full_racket_v4/placement075_first_complete_training_20261026/FIRST_COMPLETE_NATIVE_EPISODE_CHECK.json'
VIDEO=ROOT/'reports/G07_full_racket_v4/video/Current_training_placement075_first_complete_and_failures_20261026.mp4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def write(state,**extra):save(HERE/'state.json',dict(state=state,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_observer_pid=os.getpid(),video=str(VIDEO),new_native_execution=False,renderer_uses_gpu=False,frozen_policy_performance=False,formal_acceptance=False,goal_complete=False,**extra))
def main():
 assert not VIDEO.exists()and not(HERE/'PLAN.json').exists()and not(HERE/'state.json').exists()
 audit=read(AUDIT);assert audit['actual_case_id']=='TRAIN80_LANE208_EP4'and audit['actual_independent_outcome']['complete']
 manifest=read(SNAPSHOT/'cases.json');assert [(c['env_id'],c['episode_index'])for c in manifest['cases']]==[(208,4),(23,2),(8,3)]
 assert manifest['includes_failure']and not manifest['source_job_complete']
 paths=[p for p in SNAPSHOT.rglob('*')if p.is_file()]+[AUDIT,pathlib.Path(__file__)]+[ROOT/'scripts'/n for n in ['render_full_racket_latest_effect_v4.py','native_video_framing_v4.py','render_full_racket_r3_effect.py','render_current_v4_progress.py']]
 sources={str(p):sha(p)for p in paths};save(HERE/'PLAN.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,actual_case_keys=[[208,4],[23,2],[8,3]],selection_after_observed_outcomes=True,not_a_success_rate_estimate=True,source_job_still_running=True,automatic_upload=False,formal_acceptance=False,goal_complete=False))
 env=dict(os.environ,PYTHONPATH='/tmp/unitree_video_deps',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',LP_NUM_THREADS='2',LIBGL_ALWAYS_SOFTWARE='1',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')
 with(HERE/'render.log').open('x')as log:
  child=subprocess.Popen(['/home/autodrive/Isaac/env_isaaclab/bin/python',str(ROOT/'scripts/render_full_racket_latest_effect_v4.py'),'--snapshot',str(SNAPSHOT),'--output',str(VIDEO)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write('RENDERING_AUDITED_TRAIN_CASES_ON_CPU',CPU_renderer_pid=child.pid);code=child.wait(timeout=1800)
 assert code==0
 for p,h in sources.items():assert sha(pathlib.Path(p))==h,p
 m=read(VIDEO.with_suffix('.json'));assert m['complete_smash']and len(m['cases'])==3
 import imageio.v2 as imageio
 from PIL import Image
 write('DECODING_ALL_FRAMES_ON_CPU')
 samples=HERE/'decoded_samples';samples.mkdir(exist_ok=False)
 indices=sorted(set([0,100,m['frames']//4,m['frames']//2,3*m['frames']//4,m['frames']-50]));reader=imageio.get_reader(VIDEO);count=0
 for frame in reader:
  assert frame.shape==(720,1280,3)
  if count in indices:Image.fromarray(frame).save(samples/f'frame_{count:06d}.jpg',quality=94)
  count+=1
 fps=reader.get_meta_data()['fps'];reader.close();assert count==m['frames']and fps==50
 for p,h in m['sources'].items():assert sha(pathlib.Path(p)if pathlib.Path(p).is_absolute()else ROOT/p)==h,p
 check=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),video=str(VIDEO),sha256=sha(VIDEO),bytes=VIDEO.stat().st_size,frames_decoded=count,all_frames_decoded=True,fps=fps,duration_s=count/fps,root_visual_review_pending=True,changing_policy_training_only=True,source_episode_audit_sha256=sha(AUDIT),formal_acceptance=False,goal_complete=False);save(HERE/'DECODE_CHECK.json',check);write('READY_FOR_ROOT_VISUAL_REVIEW',verification=check,automatic_upload_performed=False)
if __name__=='__main__':
 try:main()
 except BaseException as e:(HERE/'error.txt').write_text(traceback.format_exc());write('FAILED_CPU_VIDEO_NATIVE_UNTOUCHED',error=repr(e));raise
