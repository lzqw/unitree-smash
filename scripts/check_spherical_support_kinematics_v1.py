"""Check rolling kinematics, virtual work and coherent interval execution."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pinocchio as pin
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.spherical_support_kinematics_v1 import spherical_support_kinematics
from scripts.predict_self_contact_spheres_v2 import surface_jacobians
from scripts.pure_control_spherical_support_recovery_v10 import PureControlSphericalSupportRecoveryV10
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.loaded_pinocchio_model_v1 import PublicState


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['configuration','run','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    if a.output.exists():raise ValueError('Fresh checks required')
    cfg=read(a.configuration);nominal=read(a.run/'native_command_inputs.json')['default_joint_position_rad']
    c=PureControlSphericalSupportRecoveryV10(PositionPDInterface(**cfg['interface']),nominal,cfg['model_reference_library'])
    with np.load(a.run/'pure_control_commands.npz') as z:d={k:z[k].copy() for k in z.files}
    supported=np.flatnonzero((d['ground_force'][:,:,2]>5).any(1))
    ids=supported[np.linspace(0,len(supported)-1,min(len(supported),64),dtype=int)]
    max_jac=max_bias=max_work=0.;solves=[];rng=np.random.default_rng(884017)
    for n,i in enumerate(ids):
        q=d['q'][i];v=d['v'][i];model=c.model;spheres=c.geometry.spheres
        k=spherical_support_kinematics(model,q,v,spheres)
        expected=surface_jacobians(model,q,spheres)
        max_jac=max(max_jac,float(np.max(abs(k['jacobian']-expected))))
        eps=1e-6
        plus=surface_jacobians(model,pin.integrate(model.model,q,eps*v),spheres)
        minus=surface_jacobians(model,pin.integrate(model.model,q,-eps*v),spheres)
        max_bias=max(max_bias,float(np.max(abs((plus-minus)@v/(2*eps)-k['jacobian_dot_velocity']))))
        pin.computeJointJacobians(model.model,model.data,q);pin.updateFramePlacements(model.model,model.data)
        for foot,sphere in enumerate(spheres):
            fid=model.body_frames[sphere['body']];pose=model.data.oMf[fid]
            offset=pose.rotation@np.asarray(sphere['center_body_m'])-np.array([0,0,sphere['radius_m']])
            frame=pin.getFrameJacobian(model.model,model.data,fid,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            force=rng.normal(size=3)*100
            moment=np.cross(offset,force)
            max_work=max(max_work,float(np.max(abs(k['jacobian'][foot].T@force-frame.T@np.r_[force,moment]))))
        if n%8==0:
            state=PublicState(q=q.copy(),v=v.copy(),root_rotation_world=pin.Quaternion(q[3:7]).matrix(),
                root_position_world=q[:3].copy(),joint_position=q[model.q_indices].copy(),joint_velocity=v[model.v_indices].copy(),
                contacts=d['ground_force'][i,:,2]>1.,rotation_projection_error=0.)
            state.previous_command_target=d['previous'][i].copy()
            target,diag=c.command(state,'BUFFER_LOAD',100.+n*.01)
            if not np.isfinite(target).all() or np.any(target<c.interface.limits[:,0]) or np.any(target>c.interface.limits[:,1]):raise ValueError('Original interface bounds failed')
            solves.append(dict(case=int(d['lane'][i]),tick=int(d['tick'][i]),solver_accepted=diag['solver_accepted']))
    if max_jac>1e-10 or max_bias>1e-5 or max_work>1e-9:raise ValueError(str(dict(jac=max_jac,bias=max_bias,work=max_work)))
    result=dict(passed=True,samples=len(ids),jacobian_error=max_jac,bias_finite_difference_error=max_bias,
        virtual_work_error=max_work,solves=solves,native_execution=False,landing_success_not_inferred=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
