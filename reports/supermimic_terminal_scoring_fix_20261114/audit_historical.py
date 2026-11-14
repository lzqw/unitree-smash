"""Read-only scoring comparison on the recorded historical128 (no Isaac)."""
import hashlib
import json
from pathlib import Path
from quadsmash.rl.supermimic_terminal_scoring import score_terminal_episode
ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/home/autodrive/LZQW/unitree_smash/reports/G07_full_racket_v5/frozen_v2_smooth_approach16_frozen128_seed882072_01')
p=json.loads((ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json').read_text());p['_root']=str(ROOT)
rows=json.loads((SOURCE/'episodes.json').read_text())
reference=json.loads((SOURCE/'metrics.json').read_text())['complete5_case_outcomes']
results=[]
for raw in rows:
    # Logging-only fields absent from the historical format remain null. The
    # classifier below uses original jump/rally evidence, never these placeholders.
    row=dict(raw, episode_duration_s=None,episode_timeout=None,
        qualified_airborne_string_contact=None,ball_side_complete_except_recovery=None,
        complete5=None,body_ground_seen=raw['jump']['body_landing'],rally_fault=None)
    s=score_terminal_episode(row,p)
    results.append(dict(training_index=raw['source_training_index'],
        air=s['qualified_airborne_string_contact'],ball5_except_recovery=s['ball_side_complete_except_recovery'],
        recovery=s['recovery_success'],complete5=s['complete5']))
fields=['air','ball5_except_recovery','recovery','complete5']
changes={k:dict(reference=sorted(x['training_index'] for x in reference if x[k]),
                scored=sorted(x['training_index'] for x in results if x[k])) for k in fields}
assert all(x['reference']==x['scored'] for x in changes.values())
print(json.dumps(dict(source=str(SOURCE),source_sha256=hashlib.sha256((SOURCE/'episodes.json').read_bytes()).hexdigest(),
    counts={k:sum(x[k] for x in results) for k in fields},all_case_sets_match=True,sets=changes,
    scope='Historical raw128 rescored offline; not a new native run'),indent=2))
