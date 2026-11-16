#!/usr/bin/env python3
"""Read-only observer. It never sends signals or changes the target process."""
from __future__ import annotations
import argparse, datetime as dt, json, os, pathlib, subprocess, time

def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def read(pid):
    d={'timestamp':now(),'pid':pid}
    try:
        s=pathlib.Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        d.update(state=s[0],ppid=int(s[1]),pgid=int(s[2]),sid=int(s[3]),start_ticks=int(s[19]),utime_ticks=int(s[11]),stime_ticks=int(s[12]))
        d['cmdline']=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip()
        d['cgroup']=pathlib.Path(f'/proc/{pid}/cgroup').read_text().splitlines()
        d['rss_kib']=int(next(x.split()[1] for x in pathlib.Path(f'/proc/{pid}/status').read_text().splitlines() if x.startswith('VmRSS:')))
    except Exception as e: d['alive']=False; d['error']=repr(e); return d
    d['alive']=True
    try:
        q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True,capture_output=True,timeout=3)
        d['gpu_compute']=[x.strip() for x in q.stdout.splitlines() if x.strip()]
    except Exception as e: d['gpu_error']=repr(e)
    return d

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pid-file',type=pathlib.Path,required=True); ap.add_argument('--output',type=pathlib.Path,required=True); ap.add_argument('--interval',type=float,default=5.0); ap.add_argument('--once',action='store_true'); a=ap.parse_args(); a.output.parent.mkdir(parents=True,exist_ok=True)
    while True:
        try: pid=int(a.pid_file.read_text().strip())
        except Exception: pid=None
        row=read(pid) if pid else {'timestamp':now(),'pid':None,'alive':False,'error':'pid file unavailable'}
        with a.output.open('a') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
        if a.once or not row.get('alive'): return 0
        time.sleep(a.interval)
if __name__=='__main__': raise SystemExit(main())
