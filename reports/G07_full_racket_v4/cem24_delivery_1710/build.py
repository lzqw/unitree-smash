"""Package four complete CEM24 native generations and latest actual video."""
import hashlib,json,re,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];R=ROOT/'reports/G07_full_racket_v4';RUN=R/'ball_slew50_near_attack_positive38_cem24';OUT=Path(__file__).parent;NAME='G07_CEM24_four_generations_and_latest_attack_recovery_20261025_1710';DEST=R/(NAME+'.zip');assert not DEST.exists()
entries={}
def add(p,name=None):
 assert p.is_file() and not p.is_symlink(),p
 entries[str(p.relative_to(ROOT)) if name is None else name]=p.read_bytes()
for p in sorted(RUN.iterdir()):
 if p.is_file() and (p.name in ['source_snapshot.zip','dirty.patch','geometry.json','geometry.npz','asset_dependencies.json','source_hashes.json','metrics.json','best_candidate.pt','resume.pt','completed.json'] or re.search(r'_(000|001|002|003)\.(json|npz|pt)$',p.name)):
  add(p)
for p in sorted(RUN.glob('GENERATION?_SUMMARY.*')):add(p)
for d in ['positive_attack_recovery_gen0_1615','positive_attack_recovery_gen3_1710','cem24_terminal_1710','positive38_contact_prefix_1605','positive38_initialization_1600']:
 for p in sorted((R/d).rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:add(p)
for p in sorted((R/'video').glob('Current_effect_attack_recovery_20261025_1710.*')):add(p)
add(R/'video_spec_20261025_1710.json');add(ROOT/'README.md');add(ROOT/'AGENTS.md');add(ROOT/'PLAN.md');add(ROOT/'TRAINING_SPEC.md');add(ROOT/'ACCEPTANCE.md');add(ROOT/'status/STATUS.json');add(ROOT/'status/NEXT.md');add(Path(__file__))
manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='Four complete native CEM24 generations over TRAIN[1,65], 512 candidate executions/409600 control env steps, plus selected actual candidate milestones and latest native replay video. No formal A200+B200 acceptance; complete=0.',source_commit='9e1643d1767f35bd297718e16e8f3153ac7ee17d',native_execution=True,formal_acceptance=False,video='Current_effect_attack_recovery_20261025_1710.mp4',video_sha256='d7a0f6883c71fe6d4708e98ab47a1a6de9282bccc393b331f27d640f656f0b58',resource_note='CEM24 native child completed all four generations. Guard receipt was unavailable after orphaning; cem24_terminal_1710/REPORT.json records the exact completion marker and elapsed single-GPU lower bound; stale protected ledger is preserved.',files=[dict(path=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in sorted(entries.items())])
m=(json.dumps(manifest,indent=2)+'\n').encode();(OUT/'MANIFEST.json').write_bytes(m)
with zipfile.ZipFile(DEST,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for n,b in sorted(entries.items()):z.writestr(n,b,compress_type=zipfile.ZIP_STORED if Path(n).suffix in ['.npz','.zip','.mp4'] else zipfile.ZIP_DEFLATED)
 z.writestr('CEM24_DELIVERY_MANIFEST.json',m)
with zipfile.ZipFile(DEST) as z:
 assert len(z.namelist())==len(entries)+1
 for rec in manifest['files']:
  b=z.read(rec['path']);assert len(b)==rec['bytes'] and hashlib.sha256(b).hexdigest()==rec['sha256']
record=dict(name=DEST.name,bytes=DEST.stat().st_size,sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),archive_members=len(entries)+1,all_data_members_sha256_verified=True,scope=manifest['scope'],formal_acceptance=False)
(R/(NAME+'.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
