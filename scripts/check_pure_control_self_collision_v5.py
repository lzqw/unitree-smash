"""Actual-state geometry/command checks, not a native landing result."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.pure_control_self_collision_recovery_v5 import PureControlSelfCollisionRecoveryV5
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.full_authority_position_interface_v1 import FullAuthorityCommandArbiterV1
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['configuration','run','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--entry',type=Path)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text())
    if a.output.exists():raise ValueError('Fresh output required')
    if a.entry:
        text=a.entry.read_text();ast.parse(text)
        if (text.count('action=native_replay.action(tick,obs,action,finished)')!=1
            or text.count('native_replay.verify_applied(finished)')!=1
            or 'native_pure_control_self_collision_recovery_v5' not in text or 'replay_state' in text):
            raise ValueError('Wrong native command wiring')
        block=text[text.index('action=native_replay.action(tick,obs,action,finished)'):text.index('observation, _, done, _ = wrapped.step(action)')]
        if 'action = ' in block or 'action=' in block.split('\n',1)[1]:raise ValueError('Second action writer')
    with np.load(a.run/'pure_control_commands.npz') as z:data={k:z[k].copy() for k in z.files}
    diagnostics=[json.loads(line) for line in (a.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    nominal=read(a.run/'native_command_inputs.json')['default_joint_position_rad']
    cfg=read(a.configuration);controller=PureControlSelfCollisionRecoveryV5(PositionPDInterface(**cfg['interface']),nominal)
    model=controller.model;geometry=controller.geometry
    # Retrospective records exercise varied states; the deployed controller has
    # no case ID or absolute event clock gate.
    selections=[]
    for case in [71,11,32,52,101,58]:
        ids=np.flatnonzero(data['lane']==case)
        if not len(ids):raise ValueError('Missing actual-state fixture')
        selections.append(int(ids[0]))
    records=[];rng=np.random.default_rng(884012)
    for index in selections:
        q=data['q'][index];v=data['v'][index]
        state=PublicState(q=q.copy(),v=v.copy(),root_rotation_world=pin.Quaternion(q[3:7]).matrix(),
            root_position_world=q[:3].copy(),joint_position=q[model.q_indices].copy(),
            joint_velocity=v[model.v_indices].copy(),contacts=np.asarray(diagnostics[index]['phase']['ground_contacts'],bool),
            rotation_projection_error=0.)
        state.previous_command_target=data['previous'][index].copy()
        gaps,jac=geometry.evaluate(q,jacobian=True)
        direction=rng.normal(size=24);direction/=np.linalg.norm(direction);eps=1e-5
        plus=geometry.evaluate(pin.integrate(model.model,q,eps*direction))[0]
        minus=geometry.evaluate(pin.integrate(model.model,q,-eps*direction))[0]
        analytic=jac@direction;numeric=(plus-minus)/(2*eps)
        # Far net distances intentionally have no asserted derivative. Keep
        # all ground/pair rows and only nonzero exact nearby net derivatives.
        mask=np.ones(len(gaps),bool);n=len(geometry.ground.shapes);m=len(geometry.net_names)
        mask[n:n+m]=np.linalg.norm(jac[n:n+m],axis=1)>1e-9
        error=float(np.max(abs(analytic[mask]-numeric[mask])))
        if error>3e-3:raise ValueError('Surface derivative mismatch '+str(error))
        c=PureControlSelfCollisionRecoveryV5(PositionPDInterface(**cfg['interface']),nominal)
        before=q.copy();target,diagnostic=c.command(state,diagnostics[index]['phase']['phase'],float(data['time_s'][index]))
        final,mapping=FullAuthorityCommandArbiterV1(c.interface.limits).select(None,target,state.previous_command_target,owns_recovery=True)
        if not np.array_equal(q,before) or not np.array_equal(final,target):raise ValueError('Command was changed after planning')
        if not np.array_equal(c.interface.velocity,controller.interface.velocity):raise ValueError('Physical speed changed')
        plan=c.last_hip_plan
        if plan['accepted'] and plan['minimum_hip_mount_gap_rad'] < -1e-6:
            raise ValueError('Accepted an intersecting hip/mount prediction')
        if not np.array_equal(c.interface.limits,controller.interface.limits):
            raise ValueError('Changed physical command limits')
        records.append(dict(predicted_minimum_hip_mount_gap_rad=plan.get('minimum_hip_mount_gap_rad'),case=int(data['lane'][index]),tick=int(data['tick'][index]),derivative_error=error,
            maximum_arm_target_change_rad=float(np.max(abs(target[12:]-state.previous_command_target[12:]))),
            solver_accepted=diagnostic['solver_accepted'],rejection=diagnostic['rejection'],mapping=mapping))
    result=dict(passed=True,native_execution=False,landing_success_not_inferred=True,records=records,
        source_sha256={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [Path(__file__),a.configuration,a.run/'pure_control_commands.npz']})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,records=[{k:v for k,v in r.items() if k!='mapping'} for r in records])))


if __name__=='__main__':main()
