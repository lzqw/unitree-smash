"""Test a fixed gravity-preloaded equilibrium target with bounded transitions.

These are model continuations from predicted endpoints, not native recoveries.
QP static feasibility does not replace rate-admitted dynamic execution.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.extend_frictional_recovery_v1 import set_state
from scripts.frictional_static_reference_v1 import sphere_static_reference
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres
from scripts.extend_recovery_to_episode_v1 import kinematic_window
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','sphere-audit','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--endpoint-results',nargs='+',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous feedback audit')
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    manifest=json.loads((a.bundle/'manifest.json').read_text());cfg=manifest['configuration'];c=build_controller(cfg)
    spheres=json.loads(a.sphere_audit.read_text())['spheres'];model=c.qp.model;interface=c.qp.interface
    geometry=AuthoredGroundClearance(ROOT/'configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.)
    protocol=json.loads((ROOT/'configs/evaluation_protocol_v2.json').read_text())['recovery']
    hashes={};rows=[];began=time.monotonic()
    for path in a.endpoint_results:
        data=json.loads(path.read_text())
        if not data['final']['accepted'] or data['native_execution']:raise ValueError('Require model-only accepted endpoint')
        for name,h in data['source_sha256'].items():
            if sha(name)!=h:raise ValueError('Changed model input '+name)
        hashes.update(data['source_sha256']);hashes[str(path.resolve())]=sha(path)
        with np.load(a.bundle/manifest['bundle_name'],allow_pickle=False) as z:
            template=model.state_from_public(z[f"case_{data['case']:03d}_query_public"],np.asarray(cfg['posture_target']))
        end=data['final']['states'][-1]
        initial=set_state(template,model,end['q'],end['v'],end['contacts'],data['final']['commands'][-1])
        equilibrium=sphere_static_reference(c,initial,spheres)
        for ramp_ticks in [0,10,20,40]:
            state=copy.deepcopy(initial);previous=initial.previous_command_target.copy()
            states=[];commands=[];failure=None;records=[]
            for tick in range(160):
                if not equilibrium['solved']:
                    failure=dict(tick=tick,reason='initial_static_reference_not_found',solver=equilibrium);break
                weight=1. if ramp_ticks==0 else min(1.,(tick+1)/ramp_ticks)
                target=(1.-weight)*initial.previous_command_target+weight*np.asarray(equilibrium['target'])
                target=np.clip(target,interface.limits[:,0],interface.limits[:,1])
                target[12:]=np.clip(target[12:],previous[12:]-.01*interface.arm_rate,previous[12:]+.01*interface.arm_rate)
                reference=dict(initial_static_reference_reused=True,ramp_weight=weight)
                result=predict_frictional_spheres(model,interface,state,target,spheres,geometry=geometry)
                state=result['state'];state.previous_command_target=target.copy();previous=target.copy();d=result['diagnostics']
                violations={k:v for k,v in d.items() if (k=='nonfoot_ground_violation_m' and v>1e-6) or
                    (k in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and v>3e-4)}
                records.append(dict(tick=tick,diagnostics=d,static_reference=reference));commands.append(target)
                if violations:failure=dict(tick=tick,reason='dynamic_constraint_violation',violations=violations);break
                states.append(copy.deepcopy(state))
            row=dict(source=str(path),mode='fixed_initial_gravity_preload',
                ramp_ticks=ramp_ticks,initial_static_reference=equilibrium,
                valid_steps=len(states),requested_steps=160,failure=failure,
                model_window=kinematic_window(model,states,protocol,.01),commands=commands,records=records,
                states=[dict(q=s.q,v=s.v,contacts=s.contacts) for s in states],native_execution=False,valid_teacher=False)
            rows.append(row)
            print(json.dumps({k:row[k] for k in ['source','mode','ramp_ticks','valid_steps','failure','model_window']}),flush=True)
    files=[Path(__file__),a.bundle/'manifest.json',a.sphere_audit,model.path,ROOT/'configs/evaluation_protocol_v2.json']
    files += [Path(v.__file__) for k,v in list(sys.modules.items()) if k.startswith('scripts.') and getattr(v,'__file__',None)]
    for f in files:hashes[str(f.resolve())]=sha(f)
    a.output.write_text(json.dumps(plain(dict(scope=__doc__,rows=rows,wall_seconds=time.monotonic()-began,
        source_sha256=hashes,native_execution=False,valid_teacher=False)),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
