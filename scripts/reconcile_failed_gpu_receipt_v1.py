"""Finish a stale RUNNING ledger entry from an existing FAILED guard receipt."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt-sha256',required=True);a=p.parse_args()
    output=a.output.resolve();receipt=output/'job.json';raw=receipt.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=a.receipt_sha256:raise ValueError('Terminal receipt changed')
    job=json.loads(raw)
    if job['status']!='FAILED' or not (output/'error.txt').is_file() or (output/'completed.json').exists():
        raise ValueError('Require explicit failed incomplete output')
    for key in ['pid','child_pid']:
        if Path('/proc',str(job[key])).exists():raise ValueError('Recorded process still exists')
    elapsed=(datetime.fromisoformat(job['finished_at'])-datetime.fromisoformat(job['started_at'])).total_seconds()
    if abs(elapsed-job['elapsed_s'])>1 or abs(job['charged_gpu_hours']-job['elapsed_s']/3600)>1e-9:
        raise ValueError('Inconsistent actual-time charge')
    path=ROOT/'status/GPU_LEDGER.json'
    with (ROOT/'status/gpu_job.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=json.loads(path.read_text());rows=[r for r in ledger['runs'] if r.get('output')==str(output) and r.get('status')=='RUNNING']
        if len(rows)!=1:raise ValueError('Exactly one matching stale entry required')
        row=rows[0]
        for key in ['pid','child_pid','command','started_at','budget_window','goal','git_sha']:
            if row[key]!=job[key]:raise ValueError('Ledger identity differs '+key)
        before=dict(row);row.update(job)
        row['charged_env_steps']=max(before['charged_env_steps'],job['charged_env_steps'])
        row['failed_receipt_reconciled_at']=datetime.now(timezone.utc).isoformat()
        row['failed_receipt_sha256']=a.receipt_sha256
        proof=output/'failed_ledger_reconciliation.json'
        if proof.exists():raise ValueError('Preserve previous reconciliation')
        proof.write_text(json.dumps(dict(before=before,after=row,receipt_unchanged=True),indent=2)+'\n')
        tmp=path.with_suffix('.json.tmp')
        with tmp.open('w') as f:
            json.dump(ledger,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
        tmp.replace(path)
        with (ROOT/'status/RUNS.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(status=row['status'],charged_gpu_hours=row['charged_gpu_hours'],charged_env_steps=row['charged_env_steps'])))


if __name__=='__main__':main()
