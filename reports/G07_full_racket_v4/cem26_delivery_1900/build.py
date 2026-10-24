"""Package complete CEM26b native search and latest candidate43 video."""
import hashlib,json,re,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];R=ROOT/'reports/G07_full_racket_v4';RUN=R/'ball_relative_candidate11_gain10_cem26b';OUT=Path(__file__).parent;NAME='G07_CEM26b_four_generations_candidate43_and_video_20261025_1900';DEST=R/(NAME+'.zip');assert not DEST.exists();entries={}
def add(p,name=None):
 assert p.is_file() and not p.is_symlink(),p;entries[str(p.relative_to(ROOT)) if name is None else name]=p.read_bytes()
for p in sorted(RUN.iterdir()):
 if p.is_file() and (p.name in ['source_snapshot.zip','dirty.patch','geometry.json','geometry.npz','asset_dependencies.json','source_hashes.json','metrics.json','best_candidate.pt','resume.pt','completed.json','job.json'] or re.search(r'_(000|001|002|003)\.(json|npz|pt)$',p.name)):add(p)
for p in sorted(RUN.glob('GENERATION?_SUMMARY.*')):add(p)
for d in ['positive_attack_recovery_gen3_cem26_1900','cem26_terminal_1900','relative_prepose_candidate11_gain10_ready01_1748','relative_prepose_candidate11_bank_cpu01_1748','relative_prepose_two_gen_terminal_1748']:
 for p in sorted((R/d).rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:add(p)
for p in sorted((R/'video').glob('Current_effect_attack_recovery_20261025_1900.*')):add(p)
add(R/'video_spec_20261025_1900.json');add(ROOT/'README.md');add(ROOT/'AGENTS.md');add(ROOT/'PLAN.md');add(ROOT/'TRAINING_SPEC.md');add(ROOT/'ACCEPTANCE.md');add(ROOT/'status/STATUS.json');add(ROOT/'status/NEXT.md');add(Path(__file__))
manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='Four complete native CEM26b generations over TRAIN[1,65], 512 executions/409600 steps, candidate43 native attack+recovery milestone and latest actual replay video. Formal acceptance remains false/full0.',source_commit='5c33b55ff72d6b094960623c668febac8c8df0c3',native_execution=True,formal_acceptance=False,video='Current_effect_attack_recovery_20261025_1900.mp4',video_sha256='a1bb3f841bf5273bffa982f34df9c7b4d876e4a08c4ab9430e948b8b39502fc1',files=[dict(path=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in sorted(entries.items())]);m=(json.dumps(manifest,indent=2)+'\n').encode();(OUT/'MANIFEST.json').write_bytes(m)
with zipfile.ZipFile(DEST,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for n,b in sorted(entries.items()):z.writestr(n,b,compress_type=zipfile.ZIP_STORED if Path(n).suffix in ['.npz','.zip','.mp4'] else zipfile.ZIP_DEFLATED)
 z.writestr('CEM26B_DELIVERY_MANIFEST.json',m)
with zipfile.ZipFile(DEST) as z:
 assert len(z.namelist())==len(entries)+1
 for rec in manifest['files']:
  b=z.read(rec['path']);assert len(b)==rec['bytes'] and hashlib.sha256(b).hexdigest()==rec['sha256']
record=dict(name=DEST.name,bytes=DEST.stat().st_size,sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),archive_members=len(entries)+1,all_data_members_sha256_verified=True,scope=manifest['scope'],formal_acceptance=False);(R/(NAME+'.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
