"""Run a pinned, guarded diagnostic without promoting an earlier failed audit.

The earlier native job must have finished normally; its strict comparison
failure remains a failure. This independent run receives the same strict audit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.native_output_reserve_v1 import require_free_space


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text()); out = a.plan.parent
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def write(name, data):
        path = out / name; tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2)+'\n'); tmp.replace(path)
    def status(state, **kw):
        write('status.json', dict(state=state, pid=os.getpid(),
            time=datetime.now(timezone.utc).isoformat(), **kw))
    def check():
        for path, digest in plan['source_sha256'].items():
            if sha(ROOT/path) != digest: raise ValueError('Changed pinned input '+path)
    def run(command, log):
        with (out/log).open('x') as f:
            subprocess.run(command, cwd=ROOT, env=dict(os.environ,
                OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'), stdout=f,
                stderr=subprocess.STDOUT, check=True)
    with (out/'owner.json').open('x') as f:
        json.dump(dict(pid=os.getpid(), plan_sha256=sha(a.plan)), f)
    try:
        check()
        for pid in plan['previous_process_ids']:
            if Path('/proc', str(pid)).exists(): raise ValueError('Prior process still exists')
        prior = json.loads(Path(plan['prior_native_job']).read_text())
        if prior['status'] != 'SUCCEEDED' or prior['exit_code'] != 0:
            raise ValueError('Prior native simulation did not finish normally')
        if plan['automatic_promotion'] or plan['valid_expert_labels']:
            raise ValueError('Diagnostic cannot promote or label an expert')
        command = plan['command']; split = command.index('--')
        guard, native = command[:split], command[split+1:]
        if ('scripts/run_gpu_job.py' not in guard or
                int(guard[guard.index('--max-seconds')+1]) > 7200 or
                int(guard[guard.index('--env-steps-cap')+1]) != 102400 or
                native[native.index('--device')+1] != 'cuda:0'):
            raise ValueError('Require finite GPU0 original-panel diagnostic')
        if Path(plan['output']).exists(): raise ValueError('Preserve prior output')
        require_free_space(ROOT)
        status('RUNNING_INDEPENDENT_DIAGNOSTIC', previous_audit_not_promoted=True)
        run(command, 'guard.log'); check()
        actual = json.loads((Path(plan['output'])/'sequence_entry_provenance.json').read_text())
        if actual['sha256'] != plan['entry_sha256']: raise ValueError('Unexpected executed entry')
        snapshot = json.loads((Path(plan['output'])/'execution_source_hashes.json').read_text())
        for name, digest in plan['source_sha256'].items():
            try: relative = str((ROOT/name).relative_to(ROOT))
            except ValueError: continue
            if relative.startswith(('scripts/', 'quadsmash/', 'configs/')) and relative.endswith(('.py', '.json')):
                if snapshot.get(relative) != digest: raise ValueError('Execution source differs '+relative)
        status('AUDITING_WITH_UNCHANGED_COMPARISON_TOLERANCES')
        for i, command in enumerate(plan['post_commands']): run(command, f'audit{i}.log')
        audit = json.loads((out/'native_audit.json').read_text())
        if not audit['passed_dispatch_integrity']: raise ValueError('Strict comparison failed')
        status('COMPLETED_DIAGNOSTIC', complete5=audit['complete5'],
            automatic_promotion=False, valid_expert_labels=False)
    except BaseException as error:
        status('FAILED', error=repr(error)); raise


if __name__ == '__main__':
    main()
