#!/usr/bin/env python3
"""Public G02 entry: analytical tests, native fixtures, full-load reaction."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PYTHON='/home/autodrive/Isaac/env_isaaclab/bin/python'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs',type=int,default=32)
    parser.add_argument('--output',type=Path,default=Path('reports/G02'))
    args=parser.parse_args()
    if args.num_envs<1:
        parser.error('--num-envs must be positive')
    # The registered fixture has 51 simultaneous scenario lanes. The full
    # coupled diagnostic uses 4 paired lanes; these counts are not PPO sizing.
    subprocess.run([sys.executable,'tools/verify_goal.py','--root','.','--goal','G01'],cwd=ROOT,check=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
    run=args.output/f'native_{stamp}'
    run.mkdir(parents=True)
    cpu=run/'cpu_tests.xml'
    subprocess.run([PYTHON,'-m','pytest','tests/test_native_physics_contract.py','tests/test_shuttlecock_physics.py',
        'tests/test_racket_collision.py',f'--junitxml={cpu}','-q'],cwd=ROOT,check=True)
    for label,dt in [('2','.002'),('1','.001'),('05','.0005'),('025','.00025'),('0125','.000125'),('00625','.0000625'),('003125','.00003125')]:
        out=run/f'fixture_dt{label}'
        subprocess.run([sys.executable,'scripts/run_gpu_job.py','--goal','G04','--max-seconds','120',
            '--env-steps-cap','1100','--output',str(out),'--require-artifact','benchmark.json','--',PYTHON,'-u',
            'scripts/isaac_contact_benchmark.py','--dt',dt,'--output',str(out),'--headless'],cwd=ROOT,check=True)
    out=run/'coupled'
    subprocess.run([sys.executable,'scripts/run_gpu_job.py','--goal','G04','--max-seconds','120',
        '--env-steps-cap','100','--output',str(out),'--require-artifact','coupling.json','--',PYTHON,'-u',
        'scripts/isaac_coupled_contact_audit.py','--output',str(out),'--headless'],cwd=ROOT,check=True)
    subprocess.run([PYTHON,'scripts/finalize_g02_reaudit.py','--fixture-prefix',str(run/'fixture_dt'),
        '--coupling-dir',str(out),'--cpu-report',str(cpu),'--output',str(args.output)],cwd=ROOT,check=True)
    if args.output.resolve()==(ROOT/'reports/G02').resolve():
        subprocess.run([sys.executable,'tools/verify_goal.py','--root','.','--goal','G02'],cwd=ROOT,check=True)

if __name__=='__main__':
    main()
