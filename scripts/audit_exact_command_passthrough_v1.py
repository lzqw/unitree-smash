#!/usr/bin/env python3
"""Replay target admission on actual recorded student requests (no simulation)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.control.exact_command_passthrough_v1 import exact_admitted_action
from scripts.command_history_support_qp_v1 import CommandHistoryPDInterface
from scripts.whole_body_support_qp_v1 import PositionPDInterface


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','command-audit','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    if a.output.exists():raise ValueError('Preserve prior evidence')
    read=lambda n:json.loads((a.run/n).read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    audit=json.loads(a.command_audit.read_text())
    if not audit['passed'] or Path(audit['run']).resolve()!=a.run.resolve():raise ValueError('Matching command audit required')
    for f,digest in audit['source_sha256'].items():
        if sha(f)!=digest:raise ValueError('Changed native source')
    native=read('native_command_inputs.json');limits=np.asarray(native['joint_limits_rad'])
    interface=CommandHistoryPDInterface(PositionPDInterface(limits,
        np.r_[np.full(12,25.),np.full(6,80.)],np.r_[np.full(12,.5),np.full(6,4.)],
        np.r_[np.full(12,23.5),np.full(6,100.)],np.r_[np.full(12,30.),np.full(6,5.)],.65),.01,10.)
    with np.load(a.run/'causal_teacher_commands.npz',allow_pickle=False) as z:base=z['base_command'].copy()
    counts=dict(records=0,unchanged_admission=0,changed_admission=0,
                rejected_teacher_unchanged_admission=0,old_roundtrip_changed_noop_bits=0)
    max_target_error=0.;max_old_error=0.;rows=[]
    for r in read('landing_mpc_records.json'):
        case,tick=r['case'],r['control_tick'];original=torch.from_numpy(base[tick,case])
        requested=np.asarray(r['requested_student_joint_reference']);interface.bind_previous_command(r['previous_dispatched_target'])
        low,high=interface.admitted_target_bounds(np.zeros(18),.01)
        admitted=np.clip(requested,low,high);encoded=interface.normalized_action(admitted)
        corrected=exact_admitted_action(original,requested,admitted,encoded)
        mask=admitted!=requested;counts['records']+=1
        counts['changed_admission' if mask.any() else 'unchanged_admission']+=1
        if not mask.any():
            assert torch.equal(corrected,original)
            counts['old_roundtrip_changed_noop_bits']+=not torch.equal(torch.as_tensor(encoded,dtype=original.dtype),original)
            counts['rejected_teacher_unchanged_admission']+=not bool(r.get('applied_first_step'))
        assert torch.equal(corrected[~mask],original[~mask])
        target=limits[:,0]+.5*(corrected.numpy().astype(float)+1.)*(limits[:,1]-limits[:,0])
        max_target_error=max(max_target_error,float(abs(target-admitted).max()))
        if np.any(target<low-5e-6) or np.any(target>high+5e-6):raise ValueError('Existing admission bounds violated')
        if not r.get('applied_first_step'):
            max_old_error=max(max_old_error,float(abs(np.asarray(r['final_dispatched_joint_target_rad'])-
                (limits[:,0]+.5*(np.asarray(encoded,dtype=np.float32).astype(float)+1.)*(limits[:,1]-limits[:,0]))).max()))
        if mask.any():rows.append(dict(case=case,tick=tick,admitted_joints=np.flatnonzero(mask).tolist()))
    if max_target_error>5e-6 or max_old_error>5e-6:raise ValueError('Mapping differs beyond existing audit tolerance')
    sources=[Path(__file__),a.command_audit,ROOT/'quadsmash/control/exact_command_passthrough_v1.py',
             ROOT/'scripts/command_history_support_qp_v1.py',a.run/'causal_teacher_commands.npz',
             a.run/'native_command_inputs.json',a.run/'landing_mpc_records.json']
    result=dict(passed=True,scope=__doc__,native_new_execution=False,counts=counts,
        admitted_target_error_max_rad=max_target_error,old_fallback_mapping_error_max_rad=max_old_error,
        original_admission_tolerance_rad=5e-6,physical_constraints_relaxed=False,
        actual_clipping_records=rows,source_sha256={str(f.resolve()):sha(f) for f in sources})
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','actual_clipping_records']}))


if __name__=='__main__':main()
