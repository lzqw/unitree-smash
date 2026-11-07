"""Test a gravity-balanced held target after an accepted model trajectory.

Static force balance is only a proposal; the unchanged dynamic, collision and
terminal verifier must accept every subsequent 0.4 s block. No native claim.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.dense_active_set_qp_v1 import DenseActiveSetQP
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.audit_microstep_recovery_v1 import characterize
from scripts.audit_joint_sequence_proposal_v1 import plain


def gravity_target(controller,state):
    model=controller.qp.model;interface=controller.qp.interface
    snap=copy.deepcopy(model.snapshot(state.q,np.zeros(model.model.nv)))
    feet=np.flatnonzero(state.contacts);nf=len(feet)
    if nf==0:return dict(solved=False,reason='No observed/predicted support')
    S=np.zeros((24,18));S[model.v_indices,np.arange(18)]=1.
    J=snap['foot_jacobian'][feet].reshape(-1,24);D=np.c_[S,J.T];n=18+3*nf
    previous=copy.deepcopy(interface.previous_command)
    try:
        interface.bind_previous_command(state.previous_command_target)
        lo,hi=interface.admitted_target_bounds(state.joint_position,.01)
    finally:interface.previous_command=previous
    torque_lo=np.maximum(-interface.effort,interface.kp*(lo-state.joint_position))
    torque_hi=np.minimum(interface.effort,interface.kp*(hi-state.joint_position))
    angles=np.arange(controller.qp.settings.friction_facets)*2*np.pi/controller.qp.settings.friction_facets
    cone=np.c_[np.cos(angles),np.sin(angles),np.full(len(angles),-interface.friction*np.cos(np.pi/len(angles)))]
    A=[D,np.c_[np.eye(18),np.zeros((18,3*nf))]]
    low=[snap['nonlinear'],torque_lo];high=[snap['nonlinear'],torque_hi]
    for i in range(nf):
        force=np.zeros((3,n));force[:,18+3*i:18+3*i+3]=np.eye(3)
        A.extend([force[2:3],cone@force]);low.extend([np.array([0.]),np.full(len(cone),-np.inf)])
        high.extend([np.array([np.inf]),np.zeros(len(cone))])
    # Nearest gravity-preloaded target to the last dispatched target.
    P=np.diag(np.r_[1./interface.kp**2,np.full(3*nf,1e-6)])
    q=np.r_[(state.joint_position-state.previous_command_target)/interface.kp,np.zeros(3*nf)]
    solver=DenseActiveSetQP();solver.setup(P=P,q=q,A=np.vstack(A),l=np.concatenate(low),u=np.concatenate(high),eps_abs=1e-7,eps_rel=1e-7)
    solution=solver.solve()
    if solution.info.status_val!=1:return dict(solved=False,reason=solution.info.status)
    x=solution.x;target=state.joint_position+x[:18]/interface.kp
    error=float(np.max(abs(D@x-snap['nonlinear'])))
    if error>1e-6:raise ValueError('Independent static force balance failed')
    return plain(dict(solved=True,target=target,torque=x[:18],force_world=x[18:].reshape(nf,3),
        dynamics_residual=error,maximum_target_change_rad=float(max(abs(target-state.previous_command_target))),
        maximum_arm_target_change_rad=float(max(abs(target[12:]-state.previous_command_target[12:])))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','geometry','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve evidence')
    m=json.loads((a.bundle/'manifest.json').read_text());archive=a.bundle/m['bundle_name']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=m['bundle_sha256']:raise ValueError('Changed model trajectory')
    for name,digest in m['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed model source '+name)
    rows=[]
    with np.load(archive,allow_pickle=False) as z:
        for row in m['rows']:
            c=build_controller(m['configuration']);c.sequence_geometry=AuthoredGroundClearance(a.geometry,c.qp.model,margin_m=0.)
            key=f"case_{row['case']:03d}";s=c.qp.model.state_from_public(z[key+'_query_public'],np.asarray(m['configuration']['posture_target']))
            s.q=z[key+'_predicted_q'][-1].copy();s.v=z[key+'_predicted_v'][-1].copy();s.contacts=z[key+'_predicted_contacts'][-1].copy()
            s.root_rotation_world=pin.Quaternion(s.q[3:7]).matrix();s.root_position_world=s.q[:3].copy()
            s.joint_position=s.q[c.qp.model.q_indices].copy();s.joint_velocity=s.v[c.qp.model.v_indices].copy()
            s.previous_command_target=z[key+'_commands_rad'][-1].copy();proposal=gravity_target(c,s);blocks=[]
            if proposal['solved']:
                target=np.asarray(proposal['target'])
                for block in range(8):
                    result,following=characterize(c,s,np.tile(target,(40,1)),target)
                    blocks.append(result)
                    if not result['accepted'] or not result['joint_settling_diagnostic']:break
                    s=copy.deepcopy(following)
            result=dict(case=row['case'],native_execution=False,proposal=proposal,blocks=blocks,
                verified_held_seconds=.4*sum(b['accepted'] and b['joint_settling_diagnostic'] for b in blocks))
            rows.append(result);print(json.dumps(dict(case=row['case'],static_solved=proposal['solved'],verified_held_seconds=result['verified_held_seconds'])),flush=True)
    files=[Path(__file__),a.bundle/'manifest.json',archive,a.geometry]
    files += [Path(m.__file__) for n,m in sys.modules.items() if n.startswith('scripts.') and getattr(m,'__file__',None)]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(scope=__doc__,native_execution=False,valid_expert_labels=False,records=rows,
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}),indent=2)+'\n')

if __name__=='__main__':main()
