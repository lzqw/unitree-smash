"""One authorized sequential GPU0 continuation; never overlaps the prior guard."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os,subprocess,sys,time,traceback
ROOT=Path('/home/autodrive/LZQW/unitree_smash')
HERE=Path(__file__).resolve().parent
PYTHON='/home/autodrive/Isaac/env_isaaclab/bin/python'
PRIOR=ROOT/'reports/G07_full_racket_v4/ball_upper_strings_cem01'
OUTPUT=ROOT/'reports/G07_full_racket_v4/ball_contact_recovery_cem02'
COMMAND=[PYTHON,'scripts/run_gpu_job.py','--goal','G07','--budget-window','g07-full-racket-unlimited-20261024',
 '--max-seconds','5400','--env-steps-cap','102400','--output',str(OUTPUT.relative_to(ROOT)),
 '--require-artifact','completed.json','--',PYTHON,'scripts/search_full_racket_ball_v4.py',
 '--output',str(OUTPUT.relative_to(ROOT)),'--warm-start','reports/G07_full_racket_v4/contact_recovery_warm06/policy.pt',
 '--candidates','32','--generations','2','--sigma-fraction','.025','--training-indices','1','65',
 '--continue-after-miss','--contact-curriculum','--contact-grid','--recovery-leg-bank','reports/G07_full_racket_v4/contact_recovery_warm06/RECOVERY_LEG_BANK.json','--push-search-range','.30','--headless']
state=dict(worker_pid=os.getpid(),prior_guard_pid=1344215,prior_child_pid=1344227,command=COMMAND,
 prior_output=str(PRIOR.relative_to(ROOT)),output=str(OUTPUT.relative_to(ROOT)),formal_acceptance=False)
def save(status,**extra):
 state.update(status=status,updated_at=datetime.now(timezone.utc).isoformat(),**extra)
 path=HERE/'state.json';temporary=path.with_suffix('.tmp')
 temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(path)
 print(json.dumps(dict(status=status,worker_pid=os.getpid(),**extra)),flush=True)
def live(pid,expected):
 path=Path('/proc')/str(pid)
 if not path.exists():return False
 if path.joinpath('stat').read_text().split()[2]=='Z':return False
 command=path.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
 if expected not in command:raise RuntimeError('Prior PID was reused; refuse automatic GPU launch')
 return True
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
try:
 if OUTPUT.exists():raise ValueError('Fresh output required')
 sources=list(ROOT.glob('quadsmash/**/*.py'))+[ROOT/'scripts/search_full_racket_ball_v4.py',ROOT/'scripts/run_gpu_job.py',
  ROOT/'configs/g07_full_racket_eval_v4_r3.json',ROOT/'reports/G07_full_racket_v4/contact_recovery_warm06/policy.pt',ROOT/'reports/G07_full_racket_v4/contact_recovery_warm06/RECOVERY_LEG_BANK.json']
 hashes={str(p.relative_to(ROOT)):digest(p) for p in sources}
 (HERE/'launch_source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
 started=time.monotonic();save('WAITING_FOR_PRIOR_TERMINAL_RECEIPT')
 while True:
  guard_live=live(1344215,'ball_upper_strings_cem01')
  child_live=live(1344227,'ball_upper_strings_cem01')
  if not guard_live and not child_live:break
  if time.monotonic()-started>5400:raise TimeoutError('Queue wait limit reached; no replacement job launched')
  time.sleep(15)
 receipt=json.loads((PRIOR/'job.json').read_text())
 if receipt['status']!='SUCCEEDED' or not (PRIOR/'completed.json').exists():
  raise RuntimeError('Prior job did not succeed; require diagnosis before continuation')
 if any(digest(ROOT/name)!=expected for name,expected in hashes.items()):
  raise RuntimeError('Control/launch source changed while queued; require a fresh queue')
 with (HERE/'guard_stdout.log').open('x') as log:
  process=subprocess.Popen(COMMAND,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  save('RUNNING_GUARDED_JOB',guard_pid=process.pid,prior_terminal_status=receipt['status'])
  result=process.wait()
 save('SUCCEEDED' if result==0 else 'FAILED',exit_code=result)
 sys.exit(result)
except Exception:
 (HERE/'error.txt').write_text(traceback.format_exc())
 save('FAILED_BEFORE_OR_DURING_CONTINUATION')
 raise
