"""Ground sensitivities must cover interior microsteps in native coordinates."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
import pinocchio as pin
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.microstep_ground_sequence_v1 import (
    configuration_from_displacement,microstep_clearance_rows,make_proposal)
from scripts.landing_differentiated_sequence_v1 import interval_outputs,interval_derivatives
from scripts.landing_sequence_linearization_v1 import world_displacement
from scripts.implicit_support_prediction_v1 import predict_implicit_support
from test_differentiated_sequence_v1 import setup_profile
import scripts.landing_joint_sequence_proposal_v1 as original

ROOT=Path(__file__).resolve().parents[1]


def test_rotated_configuration_ground_gradient_matches_independent_perturbation():
    model=LoadedPinocchioModelV1()
    with np.load(ROOT/'tests/fixtures/native_geometry_sequence_case15.npz') as z:
        # Fixture names mirror the immutable archived query.
        q=z['case_015_predicted_q'][0].copy()
    initial=SimpleNamespace(q=q,root_rotation_world=pin.Quaternion(q[3:7]).matrix(),
        joint_position=q[model.q_indices].copy())
    geometry=AuthoredGroundClearance(ROOT/'configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.)
    rng=np.random.default_rng(171);db=rng.normal(0,.03,24);dm=rng.normal(0,.02,(24,5))
    distance,jac=microstep_clearance_rows(geometry,model,initial,dict(displacement_bias=db,displacement_map=dm))
    direction=rng.normal(0,1,5);eps=1e-5
    plus=configuration_from_displacement(model,initial,db+eps*dm@direction)
    minus=configuration_from_displacement(model,initial,db-eps*dm@direction)
    actual=(geometry.evaluate(plus)[0]-geometry.evaluate(minus)[0])/(2*eps)
    np.testing.assert_allclose(jac@direction,actual,atol=2e-7,rtol=1e-5)
    np.testing.assert_allclose(world_displacement(model,initial,configuration_from_displacement(model,initial,db)),db,atol=1e-12)


def test_interior_microstep_clearance_derivative_includes_dynamics():
    c,state,profile=setup_profile()
    # An exactly upright box has tied lowest vertices and no unique gradient.
    # Use a tilted state to test the smooth branch of the dynamics chain.
    delta=np.zeros(24);delta[3:6]=[.08,-.13,.04]
    state.q=pin.integrate(c.qp.model.model,state.q,delta)
    state.root_rotation_world=pin.Quaternion(state.q[3:7]).matrix()
    state.root_position_world=state.q[:3].copy()
    state.joint_position=state.q[c.qp.model.q_indices].copy()
    command=state.joint_position.copy()
    following=predict_implicit_support(c.qp.model,c.qp.interface,state,command,
        substeps=16,contact_velocity_decay_s=c.qp.settings.contact_velocity_decay_s)['state']
    node=dict(state=state,following=following,command=command)
    geometry=AuthoredGroundClearance(ROOT/'configs/go2_piper_authored_ground_geometry_v1.json',c.qp.model,margin_m=0.)
    nominal,jac,_=interval_derivatives(c,state,node)
    direction=np.random.default_rng(173).normal(0,.01,66);micro=6;eps=1e-3
    distances,mapping=microstep_clearance_rows(geometry,c.qp.model,state,
        dict(displacement_bias=nominal[micro,42:66],displacement_map=jac[micro,42:66]))
    heights=[]
    for sign in [-1.,1.]:
        values,_=interval_outputs(c,state,node,sign*eps*direction)
        q=configuration_from_displacement(c.qp.model,state,values[micro,42:66])
        heights.append(geometry.evaluate(q)[0])
    np.testing.assert_allclose(mapping@direction,(heights[1]-heights[0])/(2*eps),atol=2e-6,rtol=3e-3)


def test_variant_does_not_change_frozen_proposal_or_files():
    path=Path(original.__file__);before=hashlib.sha256(path.read_bytes()).hexdigest()
    proposal=original.propose_sequence;variant=make_proposal()
    assert variant is not proposal
    assert original.propose_sequence is proposal
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
