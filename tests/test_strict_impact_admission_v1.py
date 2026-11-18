import json
from pathlib import Path
import numpy as np
import pytest
from quadsmash.control.impact_outcome_evaluator_v1 import ImpactOutcomeEvaluator
from quadsmash.control.strict_impact_admission_v1 import StrictImpactAdmissionV1
from quadsmash.physics.court_v4 import load_court
from quadsmash.physics.full_racket_v4 import load_racket

def fixture():
 m=ImpactOutcomeEvaluator(json.loads(Path('configs/shuttle_physics_v2.json').read_text()),load_court(),load_racket())
 return StrictImpactAdmissionV1(m),m

def test_strict_path_requires_full_body_state():
 s,_=fixture()
 with pytest.raises(ValueError,match='robot_state'):
  s.evaluate(shuttle_position=[.7,0,2],shuttle_velocity_before=[-8,0,-3],racket_contact_position=[.675,0,2],racket_contact_velocity=[1,0,-2],racket_normal=[1,0,0],contact_point_local=[.0006,0,.07],robot_state=None)

def test_support_and_flight_modes_are_explicit():
 s,m=fixture(); common=dict(shuttle_position=[.7,0,2],shuttle_velocity_before=[-8,0,-3],racket_contact_position=[.675,0,2],racket_contact_velocity=[1,0,-2],racket_normal=[1,0,0],contact_point_local=[.0006,0,.07])
 M=np.eye(24);J=np.zeros((3,24));J[0,0]=1.;support=np.zeros((1,24));support[0,1]=1.;
 r=s.evaluate(**common,robot_state={'mass_matrix':M,'contact_jacobian_world':J,'support_jacobian':support,'support_mode':'support'})
 assert r['strict_status']=='FINITE_MASS_VERIFIED' and r['finite_mass_verified']
 with pytest.raises(ValueError,match='support Jacobian'):
  s.evaluate(**common,robot_state={'mass_matrix':M,'contact_jacobian_world':J,'support_jacobian':np.ones((1,24)),'support_mode':'flight'})
