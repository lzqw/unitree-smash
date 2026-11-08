"""Verify native single-writer wiring and replay the actual causal prefix."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['entry','configuration','run','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();text=a.entry.read_text();ast.parse(text)
    call='action=native_replay.action(tick,obs,action,finished)'
    if text.count(call)!=1 or 'native_pure_control_state_replanning_v16' not in text:raise ValueError('Wrong native owner')
    block=text[text.index(call):text.index('observation, _, done, _ = wrapped.step(action)')]
    if 'action = ' in block or 'action=' in block.split('\n',1)[1]:raise ValueError('Additional final writer')
    cfg=json.loads(a.configuration.read_text())
    subprocess.run([sys.executable,str(ROOT/'scripts/check_state_replanning_v16.py'),
        '--run',str(a.run),'--configuration',str(a.configuration),'--library',cfg['model_replanning_library'],
        '--output',str(a.output)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
    result=json.loads(a.output.read_text());result.update(single_final_native_writer=True,
        entry_sha256=hashlib.sha256(a.entry.read_bytes()).hexdigest())
    tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(a.output)


if __name__=='__main__':main()
