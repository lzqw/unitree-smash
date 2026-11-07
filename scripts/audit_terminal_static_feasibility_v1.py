"""Counterfactual constraint diagnosis only; relaxed targets are never executed."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.audit_terminal_gravity_balance_v1 import gravity_target
from scripts.audit_microstep_recovery_v1 import characterize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','geometry','start-report','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve evidence')
    start=json.loads(a.start_report.read_text())
    for name,digest in start['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed seed '+name)
    m=json.loads((a.bundle/'manifest.json').read_text());cfg=m['configuration'];limits=np.asarray(cfg['interface']['limits']);rows=[]
    with np.load(a.bundle/m['bundle_name'],allow_pickle=False) as z:
        for row in start['records']:
            r=row['continuation_check'];case=r['case'];key=f'case_{case:03d}';c=build_controller(cfg)
            c.sequence_geometry=AuthoredGroundClearance(a.geometry,c.qp.model,margin_m=0.)
            state=c.qp.model.state_from_public(z[key+'_query_public'],np.asarray(cfg['posture_target']))
            state.previous_command_target=limits[:,0]+.5*(z[key+'_previous_action']+1.)*np.diff(limits).ravel()
            reference=limits[:,0]+.5*(z[key+'_query_student_action']+1.)*np.diff(limits).ravel()
            checked,endpoint=characterize(c,state,r['commands'],reference)
            if not checked['accepted']:raise ValueError('Require verified prefix')
            probes=[]
            for remove_slew,remove_effort in [(False,False),(True,False),(False,True),(True,True)]:
                diagnostic=build_controller(cfg);interface=diagnostic.qp.interface
                real_effort=interface.effort.copy()
                if remove_slew:
                    interface.admitted_target_bounds=lambda q,dt: (interface.limits[:,0].copy(),interface.limits[:,1].copy())
                if remove_effort:interface.effort=np.full(18,1e6)
                proposal=gravity_target(diagnostic,copy.deepcopy(endpoint))
                probe=dict(remove_arm_slew=remove_slew,remove_motor_effort=remove_effort,
                    solved=proposal['solved'],reason=proposal.get('reason'),executable=False)
                if proposal['solved']:
                    torque=np.asarray(proposal['torque']);probe.update(
                        maximum_real_effort_violation_nm=float(max(0.,np.max(abs(torque)-real_effort))),
                        maximum_arm_target_change_rad=proposal['maximum_arm_target_change_rad'])
                probes.append(probe)
            rows.append(dict(case=case,probes=probes));print(json.dumps(rows[-1]),flush=True)
    files=[Path(__file__),a.start_report,a.bundle/'manifest.json',a.geometry]
    files += [Path(m.__file__) for n,m in sys.modules.items() if n.startswith('scripts.') and getattr(m,'__file__',None)]
    a.output.write_text(json.dumps(dict(scope=__doc__,native_execution=False,executable=False,records=rows,
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}),indent=2)+'\n')

if __name__=='__main__':main()
