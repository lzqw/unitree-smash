from types import SimpleNamespace
import numpy as np
import torch
import pytest
from scripts.causal_frozen_leg_prefix_v1 import CausalFrozenLegReferenceFactory
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from test_contact_first_fullbody_evaluator_v1 import _fixture


class Teacher:
    observations=[]
    def __init__(self):
        self.walk=SimpleNamespace(actor=torch.nn.Identity())
        self.standing=SimpleNamespace(actor=torch.nn.Identity())
        self.strike=torch.nn.Identity()
    def action(self,obs,*args,**kwargs):
        Teacher.observations.append(obs['policy'].clone())
        return torch.zeros((1,18))


def test_committed_final_arm_replaces_base_proposal_and_isolates_candidates():
    public,default,interface=_fixture(0.)
    public[67:69]=1.;public[61:64]=[1.,0.,1.];public[775]=2.;public[759]=.75
    factory=CausalFrozenLegReferenceFactory(Teacher(),torch.zeros(1,1),torch.tensor(public[None]).float(),
        torch.tensor(default).float(),torch.tensor([[-4.,4.]]*18),torch.tensor([2.75]),torch.zeros(1,3),0,
        {'gravity_mps2':9.81,'quadratic_drag_k_per_m':0.})
    # Use actual physics schema for causal propagation.
    import json
    from pathlib import Path
    factory.physics=json.loads((Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json').read_text())
    state=LoadedPinocchioModelV1().state_from_public(public,default)
    reference=factory();reference(state,0)
    with pytest.raises(ValueError,match='not committed'):reference(state,1)
    target=np.zeros(18);target[12:]=.8
    reference.commit_final_command(target,0)
    with pytest.raises(ValueError,match='commit'):reference.commit_final_command(target,0)
    reference(state,1)
    np.testing.assert_allclose(Teacher.observations[-1][0,-18:].numpy(),target/4,atol=1e-7)
    fresh=factory();fresh(state,0)
    np.testing.assert_allclose(Teacher.observations[-1][0,-18:],public[-18:])
