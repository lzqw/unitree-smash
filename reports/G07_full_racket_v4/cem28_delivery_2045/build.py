"""Package CEM28 four native generations, two complete milestones, and videos."""
import hashlib,json,re,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];R=ROOT/'reports/G07_full_racket_v4';RUN=R/'contact_lead_gain_normal_candidate43_cem28';OUT=Path(__file__).parent;NAME='G07_CEM28_complete_milestones_and_distribution_diagnostic_20261025_2045';DEST=R/(NAME+'.zip');assert not DEST.exists();entries={}
def add(p,name=None):
 assert p.is_file() and not p.is_symlink(),p;entries[str(p.relative_to(ROOT)) if name is None else name]=p.read_bytes()
for p in sorted(RUN.iterdir()):
 if p.is_file() and (p.name in ['source_snapshot.zip','dirty.patch','geometry.json','geometry.npz','asset_dependencies.json','source_hashes.json','metrics.json','best_candidate.pt','resume.pt','completed.json','job.json'] or re.search(r'_(000|001|002|003)\.(json|npz|pt)$',p.name)):add(p)
for p in sorted(RUN.glob('GENERATION?_SUMMARY.*')):add(p)
for d in ['complete_smash_gen1_cem28_2005','complete_smash_gen3_cem28_2030','cem28_terminal_2030','cem27_plane_terminal_1925','contact_lead_gain_normal_candidate43_bank_ready01_1930','contact_lead_gain_normal_candidate43_bank_cpu01_1930','train128_candidate30_diag_2045']:
 for p in sorted((R/d).rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:add(p)
for p in sorted((R/'video').glob('Current_effect_complete_smash_20261025_20*.json')):add(p)
for p in [R/'video_spec_20261025_2005.json',R/'video_spec_20261025_2030.json',ROOT/'README.md',ROOT/'AGENTS.md',ROOT/'PLAN.md',ROOT/'TRAINING_SPEC.md',ROOT/'ACCEPTANCE.md',ROOT/'status/STATUS.json',ROOT/'status/NEXT.md',Path(__file__)]:add(p)
manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='Four complete native CEM28 generations/512 executions with two independent complete-smash TRAIN milestones, latest complete videos and fixed-policy TRAIN128 diagnostic. Formal A/B remains unrun.',source_commit='10b3d88b568242f153a36109555ea433d5a5161c',native_execution=True,formal_acceptance=False,complete_milestones=[dict(candidate=30,env=61,norm_mps=8.101310022,first_x=3.9766407,recovery_s=3.6859152),dict(candidate=39,env=79,norm_mps=8.011582711,first_x=3.8528538,recovery_s=4.0118918)],videos=['Current_effect_complete_smash_20261025_2005.mp4','Current_effect_complete_smash_20261025_2030.mp4'],files=[dict(path=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in sorted(entries.items())]);m=(json.dumps(manifest,indent=2)+'\n').encode();(OUT/'MANIFEST.json').write_bytes(m)
with zipfile.ZipFile(DEST,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for n,b in sorted(entries.items()):z.writestr(n,b,compress_type=zipfile.ZIP_STORED if Path(n).suffix in ['.npz','.zip','.mp4'] else zipfile.ZIP_DEFLATED)
 z.writestr('CEM28_DELIVERY_MANIFEST.json',m)
with zipfile.ZipFile(DEST) as z:
 assert len(z.namelist())==len(entries)+1
 for rec in manifest['files']:
  b=z.read(rec['path']);assert len(b)==rec['bytes'] and hashlib.sha256(b).hexdigest()==rec['sha256']
record=dict(name=DEST.name,bytes=DEST.stat().st_size,sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),archive_members=len(entries)+1,all_data_members_sha256_verified=True,scope=manifest['scope'],formal_acceptance=False);(R/(NAME+'.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
