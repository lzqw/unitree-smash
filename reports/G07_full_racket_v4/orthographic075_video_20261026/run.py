"""CPU replay of the same completed075 frozen episodes with fitted views."""
import argparse,datetime,hashlib,json,os,pathlib,signal,subprocess,sys,traceback
ROOT=pathlib.Path(__file__).resolve().parents[3];HERE=pathlib.Path(__file__).resolve().parent
MAIN=ROOT.parent/'unitree_smash';REPORTS=MAIN/'reports/G07_full_racket_v4'
SNAPSHOT=REPORTS/'video_snapshot_centered_placement075_pilot8_20261026'
FROZEN=REPORTS/'residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01'
POLICY=REPORTS/'residual_ppo_centered_placement075_n2048_probe8_01/policy.pt'
VIDEO=ROOT/'reports/G07_full_racket_v4/video/Current_effect_centered_placement075_pilot8_orthographic_v2_20261026.mp4'
PLAN=HERE/'PLAN.json';STATE=HERE/'state.json';PYTHON='/home/autodrive/Isaac/env_isaaclab/bin/python'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def write(state,**extra):
 atomic(STATE,dict(state=state,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),observer_pid=os.getpid(),video=str(VIDEO),renderer_uses_gpu=False,new_native_execution=False,formal_acceptance=False,goal_complete=False,**extra))
def sources():
 paths=[p for p in SNAPSHOT.rglob('*')if p.is_file()]
 paths += [ROOT/'scripts'/n for n in ['render_full_racket_latest_effect_v4.py','native_video_framing_v4.py','render_full_racket_r3_effect.py','render_current_v4_progress.py']]
 paths += [ROOT/'configs/court_net_v4.json',ROOT/'configs/g07_full_racket_eval_v4_r3.json',POLICY,pathlib.Path(__file__)]
 paths += [FROZEN/n for n in ['INPUTS.json','job.json','completed.json','metrics.json','episodes.json']]
 return {str(p):sha(p)for p in paths}
def check_sources(expected):
 for p,h in expected.items():assert sha(pathlib.Path(p))==h,p

def prepare():
 assert not any(p.exists()for p in [PLAN,STATE,VIDEO,VIDEO.with_suffix('.json')])
 preview=ROOT/'reports/G07_full_racket_v4/video/Preview_placement075_orthographic_draft02_20261026.preview.json';p=read(preview)
 assert p['renderer_sha256']==sha(ROOT/'scripts/render_full_racket_latest_effect_v4.py')
 assert p['framing_helper_sha256']==sha(ROOT/'scripts/native_video_framing_v4.py')
 assert p['snapshot_manifest_sha256']==sha(SNAPSHOT/'cases.json')
 assert p['native_cases']==6
 for case,coverage in zip(read(SNAPSHOT/'cases.json')['cases'],p['camera_framing_checks']):
  meshes=read(SNAPSHOT/case['run']/'geometry.json')['meshes']
  assert coverage['rendered_meshes']==len(meshes) and coverage['rendered_bodies']==len({m['body']for m in meshes})
 assert all(v['all_fitted_points_in_viewport'] for c in p['camera_framing_checks']for v in c['views'].values())
 assert p['pane_layout']['top']+p['pane_layout']['height']<p['pane_layout']['time_band_top']
 job=read(FROZEN/'job.json');assert job['status']=='SUCCEEDED'and job['receipt_present']and read(FROZEN/'completed.json')['env_steps']==102400
 assert read(FROZEN/'metrics.json')['policy_sha256']==sha(POLICY)=='66c93b9f2bfd1f65812ca230312a45cda1d11f59e4a6f076ebdd98ae3c88a3b5'
 assert [c['env_id']for c in read(SNAPSHOT/'cases.json')['cases']]==[19,23,107,0,8,58]
 atomic(PLAN,dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),preview_check=str(preview),preview_sha256=sha(preview),same_frozen075_stage8=True,source_policy_sha256=sha(POLICY),actual_frames_expected=4383,duration_s_expected=87.66,changes='Only orthographic framing/layout, extra contact angle and precise outgoing labels; native data and outcomes unchanged.',automatic_upload=False,formal_acceptance=False,goal_complete=False))
 print(json.dumps(dict(prepared=str(PLAN),source_files=len(read(PLAN)['sources']))))

def main():
 assert PLAN.exists()and not STATE.exists()and not VIDEO.exists()
 plan=read(PLAN);check_sources(plan['sources']);environment=dict(os.environ,PYTHONPATH='/tmp/unitree_video_deps',LIBGL_ALWAYS_SOFTWARE='1',LP_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')
 with(HERE/'render.log').open('x')as log:
  child=subprocess.Popen([PYTHON,str(ROOT/'scripts/render_full_racket_latest_effect_v4.py'),'--snapshot',str(SNAPSHOT),'--output',str(VIDEO)],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write('RENDERING_ON_CPU',renderer_pid=child.pid)
  def stop(*_):
   if child.poll()is None:child.terminate()
   raise RuntimeError('CPU video observer stopped; native training untouched')
  signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
  try:code=child.wait(timeout=1800)
  except subprocess.TimeoutExpired:
   child.terminate();child.wait(timeout=30);raise
 assert code==0
 check_sources(plan['sources']);metadata=read(VIDEO.with_suffix('.json'));assert metadata['frames']==4383 and metadata['duration_s']==87.66
 write('DECODING_ALL_ACTUAL_VIDEO_FRAMES_ON_CPU')
 import imageio.v2 as imageio
 import numpy as np
 from PIL import Image
 sys.path.insert(0,str(ROOT));from scripts.initialize_centered_policy_v4 import same_tree
 raw=read(FROZEN/'episodes.json')
 assert all(same_tree(c['row'],raw[c['env_id']])for c in metadata['cases'])
 frames=[0,210,1249,1974,2699,3424,4050,4300];samples=HERE/'decoded_samples';samples.mkdir(exist_ok=False)
 reader=imageio.get_reader(VIDEO);count=0
 for frame in reader:
  assert frame.shape==(720,1280,3)
  if count in frames:Image.fromarray(frame).save(samples/f'frame_{count:06d}.jpg',quality=94)
  count+=1
 fps=float(reader.get_meta_data()['fps']);reader.close();assert count==4383 and fps==50.
 for p,h in metadata['sources'].items():assert sha(pathlib.Path(p)if pathlib.Path(p).is_absolute()else ROOT/p)==h,p
 verification=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),video=str(VIDEO),sha256=sha(VIDEO),bytes=VIDEO.stat().st_size,all_frames_decoded=True,frames_decoded=count,fps=fps,duration_s=count/fps,six_native_rows_deep_matched=True,source_hashes_verified=len(metadata['sources']),root_visual_review_pending=True,new_native_execution=False,formal_acceptance=False,goal_complete=False)
 atomic(HERE/'DECODE_AND_NATIVE_CUSTODY_CHECK.json',verification)
 write('READY_FOR_ROOT_VISUAL_REVIEW',verification=verification,automatic_upload_performed=False)
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
 try:prepare()if args.prepare_only else main()
 except BaseException as e:
  (HERE/'error.txt').write_text(traceback.format_exc())
  if not args.prepare_only:write('FAILED_CPU_VIDEO_ONLY_NATIVE_UNTOUCHED',error=repr(e))
  raise
