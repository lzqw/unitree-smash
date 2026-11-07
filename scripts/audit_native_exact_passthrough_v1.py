#!/usr/bin/env python3
"""Verify that rejected teacher plans leave unmodified student joints exact."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','command-audit','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior evidence')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    audit=json.loads(a.command_audit.read_text());inputs=json.loads((a.run/'INPUTS.json').read_text())
    if not audit['passed'] or Path(audit['run']).resolve()!=a.run.resolve() or not inputs['landing_mpc_exact_student_passthrough']:
        raise ValueError('Matching exact-passthrough native run required')
    for f,h in audit['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Changed command evidence')
    with np.load(a.run/'causal_teacher_commands.npz',allow_pickle=False) as z:
        base=z['base_command'].copy();final=z['final_command'].copy()
    rows=json.loads((a.run/'landing_mpc_records.json').read_text());fallback=0;noop=0
    for r in rows:
        if not r['exact_student_passthrough']:raise ValueError('Mixed command interface modes')
        if r.get('applied_first_step'):continue
        fallback+=1;case,tick=r['case'],r['control_tick'];unchanged=np.ones(18,dtype=bool)
        unchanged[r['student_admission_changed_joints']]=False
        if not np.array_equal(base[tick,case,unchanged],final[tick,case,unchanged]):
            raise ValueError('Unmodified fallback joints changed')
        noop+=bool(unchanged.all())
    sources=[Path(__file__),a.command_audit,a.run/'INPUTS.json',a.run/'causal_teacher_commands.npz',a.run/'landing_mpc_records.json']
    result=dict(passed=bool(fallback),rejected_teacher_records=fallback,fully_unchanged_fallback_records=noop,
        command_mapping_already_verified=True,scope=__doc__,native_recovery_gain_claimed=False,
        source_sha256={str(p.resolve()):sha(p) for p in sources})
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))
    return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
