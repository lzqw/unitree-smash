from pathlib import Path
import importlib.util,json,sys
import torch
import pytest
from quadsmash.rl.supermimic_racket_material_contract_v1 import SCHEMA,component_names,digest
from quadsmash.rl.supermimic_full_body_geometry_v1 import MODEL
import quadsmash.rl.supermimic_action_ball_v1 as m

def model():
    names=component_names();c=dict(schema=SCHEMA,asset_sha256=json.loads(MODEL.read_text())['asset_sha256'],component_names=names,effective=[[.08,.08,.72] if n.startswith('Collider/') else [.29,.29,.72] for n in names],native_tensor_evidence={'unit_test_only':True})
    c['sha256']=digest(c)
    return m.MaterialInertiaActionBallV4(material_contract=c)

@pytest.mark.parametrize('key',['component_material','inertia.geometry.mass','mechanical_scale'])
def test_parent_resume_checks_child_contract_before_mutation(key):
    parent=torch.nn.ModuleDict({'action_ball_model':model()});state={k:v.clone() for k,v in parent.state_dict().items()}
    full='action_ball_model.'+key;old=parent.state_dict()[full].clone();state[full].flatten()[0]+=1
    with pytest.raises(ValueError,match='Immutable mechanical'):parent.load_state_dict(state,strict=True)
    torch.testing.assert_close(parent.state_dict()[full],old,atol=0,rtol=0)

def test_parent_resume_preserves_all_valid_model_parameters():
    parent=torch.nn.ModuleDict({'action_ball_model':model()});state={k:v.clone() for k,v in parent.state_dict().items()}
    parent.load_state_dict(state,strict=True)
    for k,v in parent.state_dict().items():torch.testing.assert_close(v,state[k],atol=0,rtol=0)
