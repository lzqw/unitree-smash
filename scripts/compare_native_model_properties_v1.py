"""Compare actual initialized PhysX tensors with authored CPU model/interface."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1,rotation_wxyz
from scripts.whole_body_support_qp_v1 import PositionPDInterface


def compare(model,interface,metadata,arrays):
    if set(metadata['body_names'])!=set(model.names) or set(metadata['joint_names'])!=set(model.joint_order):
        raise ValueError('Missing or different body/joint mapping')
    if len(set(metadata['body_names']))!=len(metadata['body_names']) or len(set(metadata['joint_names']))!=len(metadata['joint_names']):
        raise ValueError('Duplicate model mapping')
    if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite native property')
    bodies=[]
    for body in model.record['bodies']:
        j=metadata['body_names'].index(body['name']);P=rotation_wxyz(body['principal_axes_wxyz'])
        expected=P@np.diag(body['diagonal_inertia_kg_m2'])@P.T
        actual=arrays['inertias'][:,j].reshape(-1,3,3).transpose(0,2,1)
        bodies.append(dict(body=body['name'],mass_max_delta_kg=float(np.max(abs(arrays['masses'][:,j]-body['mass_kg']))),
            com_max_delta_m=float(np.max(abs(arrays['coms'][:,j,:3]-body['center_of_mass_m']))),
            inertia_max_delta_kg_m2=float(np.max(abs(actual-expected)))))
    indices=[metadata['joint_names'].index(n) for n in model.joint_order]
    expected={'dof_stiffnesses':interface.kp,'dof_dampings':interface.kd,'dof_max_forces':interface.effort,
              'dof_max_velocities':interface.velocity,'dof_limits':interface.limits}
    differences={key:float(np.max(abs(arrays[key][:,indices]-value))) for key,value in expected.items()}
    return dict(bodies=bodies,interface_max_differences=differences,
        armature_max=float(np.max(abs(arrays['dof_armatures']))),
        friction_max=float(np.max(abs(arrays['dof_friction_coefficients']))),
        runtime_rigid_damping_verified=False,complete_dynamics_equivalence=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inspection',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior property comparison')
    metadata=json.loads((a.inspection/'PROPERTIES.json').read_text())
    archive=a.inspection/'PROPERTIES.npz'
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=metadata['properties_sha256']:raise ValueError('Native tensor drift')
    with np.load(archive,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    interface=PositionPDInterface(**json.loads(Path('reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface'])
    result=compare(LoadedPinocchioModelV1(),interface,metadata,arrays)
    result['native_properties_sha256']=metadata['properties_sha256']
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
