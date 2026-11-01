"""Read actual PhysX tensors before enabling the rigid-body support prototype."""
import numpy as np

from scripts.loaded_pinocchio_model_v1 import rotation_wxyz
from scripts.whole_body_support_qp_v1 import PositionPDInterface


def observe_native_whole_body(env,stage,model):
    from quadsmash.control.support_admission_v1 import validate_native_support_interface
    support=validate_native_support_interface(env,stage)
    view=env.robot.root_physx_view
    as_numpy=lambda value:value.detach().cpu().numpy().astype(np.float64)
    body_ids=[env.robot.body_names.index(n) for n in model.names]
    joint_ids=list(env.joint_ids)
    joint_names=[env.robot.joint_names[i] for i in joint_ids]
    masses=as_numpy(view.get_masses())[:,body_ids]
    coms=as_numpy(view.get_coms())[:,body_ids]
    inertias=as_numpy(view.get_inertias())[:,body_ids].reshape(env.num_envs,28,3,3).transpose(0,1,3,2)
    expected_mass=np.array([b['mass_kg'] for b in model.record['bodies']])
    expected_com=np.array([b['center_of_mass_m'] for b in model.record['bodies']])
    expected_inertia=np.array([rotation_wxyz(b['principal_axes_wxyz']) @
        np.diag(b['diagonal_inertia_kg_m2']) @ rotation_wxyz(b['principal_axes_wxyz']).T
        for b in model.record['bodies']])
    # API documents get_inertias as a column-major matrix in the rigid-body
    # prim frame at COM, not the principal-axis frame returned by get_coms.
    observed={}
    for key,method in [('stiffness','get_dof_stiffnesses'),('damping','get_dof_dampings'),
                       ('effort','get_dof_max_forces'),('velocity','get_dof_max_velocities'),
                       ('armature','get_dof_armatures')]:
        observed[key]=as_numpy(getattr(view,method)())[:,joint_ids]
    friction=as_numpy(view.get_dof_friction_properties())[:,joint_ids]
    limits=as_numpy(env.robot.data.joint_pos_limits)[:,joint_ids]
    expected={'stiffness':np.r_[np.full(12,25.),np.full(6,80.)],
              'damping':np.r_[np.full(12,.5),np.full(6,4.)],
              'effort':np.r_[np.full(12,23.5),np.full(6,100.)],
              'velocity':np.r_[np.full(12,30.),np.full(6,5.)]}
    errors={'mass_kg':float(np.max(abs(masses-expected_mass))),
            'com_m':float(np.max(abs(coms[...,:3]-expected_com))),
            'inertia_kg_m2':float(np.max(abs(inertias-expected_inertia))),
            'armature':float(np.max(abs(observed['armature']))),
            'joint_friction':float(np.max(abs(friction))),
            'velocity_target':float(np.max(abs(as_numpy(env.robot.data.joint_vel_target))))}
    errors.update({key:float(np.max(abs(observed[key]-expected[key]))) for key in expected})
    tolerances={'mass_kg':2e-5,'com_m':2e-6,'inertia_kg_m2':2e-6,'armature':1e-8,
                'joint_friction':1e-8,'velocity_target':1e-8,
                'stiffness':1e-5,'damping':1e-5,'effort':1e-5,'velocity':1e-5}
    passed=(not env.robot.is_fixed_base and joint_names==model.joint_order and
            not env.robot.has_external_wrench and
            all(np.isfinite(v) and v<=tolerances[k] for k,v in errors.items()) and
            np.isfinite(limits).all() and np.array_equal(limits,np.broadcast_to(limits[0],limits.shape)))
    result=dict(passed=bool(passed),native_execution=True,model_sha256=model.sha256,
        asset_sha256=model.record['asset_sha256'],body_names=model.names,joint_names=joint_names,
        native_masses_kg=masses.tolist(),native_coms_xyzw=coms.tolist(),
        native_inertias_body_frame=inertias.tolist(),native_actuators={k:v.tolist() for k,v in observed.items()},
        native_joint_friction=friction.tolist(),joint_limits_rad=limits.tolist(),errors=errors,tolerances=tolerances,
        material_admission=support,external_wrench_present=bool(env.robot.has_external_wrench),
        fixed_base=bool(env.robot.is_fixed_base),formal_acceptance=False)
    interface=(PositionPDInterface(limits[0],observed['stiffness'][0],observed['damping'][0],
                                  observed['effort'][0],observed['velocity'][0],.65) if passed else None)
    return result,interface
