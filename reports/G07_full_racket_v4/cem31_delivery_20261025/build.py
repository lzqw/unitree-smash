import pathlib,json,hashlib,zipfile,subprocess,datetime
ROOT=pathlib.Path(__file__).resolve().parents[3]
R=ROOT/'reports/G07_full_racket_v4'
out=R/'G07_CEM31_formal400_native_trajectories_20261025.zip'
assert not out.exists()
files=set()
def add(p):
 p=pathlib.Path(p)
 if not p.is_absolute():p=ROOT/p
 assert p.is_file(),p
 files.add(p)
for folder in ['formal_lock_cem31_candidate5_0915','formal_panel_A_cem31_candidate5_0915','formal_panel_B_cem31_candidate5_0915','cem31_candidate5_diag_0900','video_snapshot_formal_cem31_20261025','lateral_feedback_ready_1015']:
 for p in (R/folder).rglob('*'):
  if p.is_file() and '.partial' not in p.name:add(p)
for name in ['formal_aggregate_cem31_candidate5_1005.json','CEM31_EVALUATION_AND_LATERAL_NEXT_20261025.json','video_spec_formal_cem31_20261025.json']:
 add(R/name)
for p in (R/'video').glob('Current_formal_evaluation_cem31_20261025*'):add(p)
for name in ['job.json','metrics.json','source_hashes.json','source_snapshot.zip','dirty.patch','population_000.pt','population_001.pt','episodes_000.json','episodes_001.json','contacts_000.json','contacts_001.json','all_first_episodes_000.npz','all_first_episodes_001.npz']:
 add(R/'cem31_timing_plane_distribution_0820'/name)
for p in [ROOT/'scripts/prepare_full_racket_latest_video_v4.py',ROOT/'scripts/render_full_racket_latest_effect_v4.py',ROOT/'scripts/prepare_full_racket_lateral_feedback_v4.py',ROOT/'quadsmash/control/full_racket_teacher_v4.py',pathlib.Path(__file__)]:add(p)
manifest=[]
with zipfile.ZipFile(out,'w',allowZip64=True) as z:
 for p in sorted(files):
  payload=p.read_bytes();arc=str(p.relative_to(ROOT));entry=dict(path=arc,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest());manifest.append(entry)
  z.writestr(arc,payload,compress_type=zipfile.ZIP_STORED if p.suffix in ('.zip','.npz','.mp4','.jpg') else zipfile.ZIP_DEFLATED)
 m=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),formal_acceptance=False,
        scope='Candidate5 locked A/B400 failure evidence, all native trajectories, TRAIN128, CEM31 populations and actual video; includes unexecuted CEM32 initial proposals, which inherit no performance.',files=manifest)
 z.writestr('MANIFEST.json',json.dumps(m,indent=2)+'\n')
receipt=dict(asset=out.name,bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),file_count=len(manifest),formal_acceptance=False)
(R/'cem31_delivery_20261025/MANIFEST.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
