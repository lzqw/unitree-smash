"""Geometry witnesses and derivatives must match independent support queries."""
import json
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance,support_witness
from test_differentiated_sequence_v1 import setup_profile


def test_rotated_primitive_support_witness_matches_closed_form():
    rng=np.random.default_rng(6)
    for kind in ['Sphere','Cylinder','Capsule','Cube']:
        for axis in range(3):
            T=np.eye(4);T[:3,:3]=pin.exp3(rng.normal(size=3));T[3,:3]=rng.normal(size=3)
            R=pin.exp3(rng.normal(size=3));d=(R@T[:3,:3].T)[2]
            s=dict(kind=kind,transform=T,size=.3,radius=.09,height=.4,axis=axis)
            if kind=='Cube':expected=.15*abs(d).sum()
            elif kind=='Sphere':expected=.09*np.linalg.norm(d)
            elif kind=='Cylinder':expected=.2*abs(d[axis])+.09*np.linalg.norm(np.delete(d,axis))
            else:expected=.2*abs(d[axis])+.09*np.linalg.norm(d)
            assert abs((R@support_witness(s,R))[2]-(R@T[3,:3])[2]+expected)<1e-12


def test_surface_gradient_matches_full_configuration_differences(tmp_path):
    c,state,_=setup_profile();q=pin.integrate(c.qp.model.model,state.q,np.random.default_rng(4).normal(0,.02,24))
    shapes=[]
    for body in ['FL_thigh',c.qp.model.names[-1]]:
        for kind in ['Sphere','Cylinder','Capsule','Cube','Mesh']:
            T=np.eye(4);T[3,:3]=[.1,.02,-.03]
            shapes.append(dict(body=body,path=body+kind,kind=kind,transform=T.tolist(),
                radius=.09,height=.4,axis=2,size=.3,points=[[.1,.2,-.1],[-.1,.1,.2],[.2,-.2,.1]]))
    path=tmp_path/'geometry.json';path.write_text(json.dumps(dict(asset_sha256={},shapes=shapes)))
    geometry=AuthoredGroundClearance(path,c.qp.model,margin_m=0.)
    _,jac=geometry.evaluate(q,jacobian=True)
    actual=[]
    for k in range(24):
        d=np.eye(24)[k]*1e-6
        actual.append((geometry.evaluate(pin.integrate(c.qp.model.model,q,d))[0]-
                       geometry.evaluate(pin.integrate(c.qp.model.model,q,-d))[0])/2e-6)
    np.testing.assert_allclose(jac,np.asarray(actual).T,atol=2e-8,rtol=2e-6)


def test_real_racket_penetration_is_rejected_despite_old_dynamic_acceptance():
    from scripts.parallel_landing_solver_v1 import build_controller
    from scripts.landing_joint_sequence_proposal_v1 import verify_sequence
    root=Path(__file__).resolve().parents[1]
    data=json.loads((root/'tests/fixtures/landing_restoration_friction_boundary.json').read_text())
    cfg=data['configuration'];c=build_controller(cfg);limits=np.asarray(cfg['interface']['limits'])
    state=c.qp.model.state_from_public(np.asarray(data['public']),np.asarray(cfg['posture_target']))
    actions=np.asarray(data['actions'],dtype=np.float32)
    state.previous_command_target=limits[:,0]+.5*(actions[0]+1.)*np.diff(limits).ravel()
    commands=np.asarray(data['commands'])
    assert verify_sequence(c,state,commands)['accepted']
    c.sequence_geometry=AuthoredGroundClearance(root/'configs/go2_piper_authored_ground_geometry_v1.json',c.qp.model,margin_m=0.)
    checked=verify_sequence(c,state,commands)
    assert not checked['accepted']
    assert checked['rejection']=='Authored nonfoot geometry penetrates ground'
    assert checked['checks'][-1]['authored_ground_clearance_m']>1e-6


def test_search_elastic_distance_never_authorizes_penetrating_trajectory(tmp_path):
    from scripts.landing_joint_sequence_proposal_v1 import propose_sequence,verify_sequence
    c,state,profile=setup_profile()
    record=dict(asset_sha256={},shapes=[dict(body=c.qp.model.record['root_body'],
        path='deliberately-oversized-test-sphere',kind='Sphere',radius=2.,transform=np.eye(4).tolist())])
    path=tmp_path/'geometry.json';path.write_text(json.dumps(record))
    c.sequence_geometry=AuthoredGroundClearance(path,c.qp.model,margin_m=0.)
    proposal=propose_sequence(c,state,state.joint_position,knots=2,nominal_profile=profile,
        differentiate_dynamics=True,solver_backend='active_set',feasibility_restoration=True)
    assert proposal['linear_proposal_solved']
    assert proposal['search_only_clearance_slack_m']>1.
    assert not proposal['accepted']
    assert not verify_sequence(c,state,proposal['commands'])['accepted']
