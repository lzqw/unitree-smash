"""Read-only settings/evidence audit, not a dynamics or policy training run."""
import hashlib, json, subprocess, sys
from pathlib import Path
from datetime import datetime
from quadsmash.rl.supermimic_timebase_v2 import Timebase
root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
paths = [
 'configs/supermimic_task_horizon_v2.json',
 'quadsmash/rl/supermimic_timebase_v2.py',
 'quadsmash/rl/supermimic_horizon_learner_v2.py',
 'quadsmash/rl/supermimic_v1.py',
 'quadsmash/rl/supermimic_sequence_replay_v2.py',
 'quadsmash/tasks/stand/supermimic_env_v1.py',
 'scripts/train_supermimic_horizon_v2.py',
 'reports/supermimic_imagination_task_audit_20261114/final_checkpoint_audit.json',
 'reports/supermimic_terminal_eval4_result_20261114/metrics.json',
 'reports/supermimic_terminal_eval4_result_20261114/comparison.json',
 'reports/supermimic_terminal_eval4_result_20261114/process_result.json']
cfg = Timebase(**json.loads((root/paths[0]).read_text()))
old = json.loads((root/paths[7]).read_text())
result = dict(timestamp=datetime.now().astimezone().isoformat(),
 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 interpreter=sys.executable, default_timebase=cfg.manifest(),
 stopped_run_timebase=old['timebase'], stopped_run_counters=old['counters'],
 evidence_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
 discount_weights={str(s):cfg.gamma**round(s/cfg.control_dt_s) for s in (0.16,0.64,1.28,3.2,8.)},
 gae_td_residual_weights={str(s):(cfg.gamma*cfg.lam)**round(s/cfg.control_dt_s) for s in (0.16,0.64,1.,1.28,3.2,8.)},
 event_accumulation_sensitivity={str(p):{str(n):1-(1-p)**n for n in (16,64,128,320)} for p in (.001,.01)},
 sensitivity_scope='Analytic constant-probability example only; not measured event false-positive rates or a performance prediction.',
 open_loop_scope=old['scope'], open_loop_horizons=old['horizons'],
 reward_correlation=old['reward_correlation'], reward_mae=old['reward_mae'])
(out/'settings_and_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
with (out/'source_evidence.txt').open('w') as f:
 for path, a, b in [(paths[2],85,116),(paths[2],128,170),(paths[3],38,95),(paths[3],254,262),(paths[4],15,26),(paths[4],56,68),(paths[5],138,179),(paths[6],215,234),(paths[6],356,381)]:
  f.write('\n'+path+'\n')
  for n,line in enumerate((root/path).read_text().splitlines(),1):
   if a <= n <= b: f.write(f'{n}: {line}\n')
print(json.dumps({k:result[k] for k in ('source_commit','default_timebase','discount_weights','gae_td_residual_weights','event_accumulation_sensitivity')},indent=2))
