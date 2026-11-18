"""CPU mechanics and schema contracts only; no PhysX/learning acceptance."""
import copy,json
from pathlib import Path
import pytest
import torch
from quadsmash.rl.supermimic_racket_material_contract_v1 import component_names,digest,validate_contract,SCHEMA
from quadsmash.rl.supermimic_action_ball_v1 import ImpactTimeActionBallV3,MaterialInertiaActionBallV4,migrate_material_inertia_v4,action_ball_from_checkpoint
from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1,MODEL


def contract():
 names=component_names();asset=json.loads(MODEL.read_text())['asset_sha256']
 c=dict(schema=SCHEMA,asset_sha256=asset,component_names=names,
  effective=[[.08,.08,.72] if n.startswith('Collider/') else [.29,.29,.72] for n in names],
  native_tensor_evidence={'unit_test_only':True})
 c['sha256']=digest(c);return c


def inputs():
 c=contract();g=FullBodyGeometryV1(c['asset_sha256']);x=torch.zeros(2,55);x[:,3]=1;x[:,2]=.7
 x[:,13:31]=torch.linspace(-.3,.3,18);x[:,31:49]=.2
 b=g(x[:,13:31],x[:,31:49],x[:,:3],x[:,3:7],x[:,7:10],x[:,10:13]);rid=g.ids['Racket'];n=b['rotation'][:,rid,:,0]
 x[:,49:52]=b['position'][:,rid]+.03*n;x[:,52:55]=-3*n
 y=x.clone();y[:,13:31]+=.01;y[:,49:52]+=.01*x[:,52:55]
 return g,x,y,y[:,49:].clone(),torch.ones(2),torch.ones(2)


def test_materials_are_by_geometry_component_and_digest_protected():
 c=contract();validate_contract(c,c['asset_sha256'])
 assert len(c['effective'])==97
 for name,value in zip(c['component_names'],c['effective']):
  assert value[0]==(.08 if name.startswith('Collider/') else .29)
 bad=copy.deepcopy(c);bad['effective'][0][0]=.29
 with pytest.raises(ValueError,match='digest'):validate_contract(bad,bad['asset_sha256'])
 with pytest.raises(ValueError,match='asset'):validate_contract(c,'wrong-asset')


def test_explicit_v3_migration_preserves_every_old_tensor_and_prediction():
 torch.manual_seed(8);old=ImpactTimeActionBallV3()
 with torch.no_grad():old.head.net[-1].bias.fill_(.1)
 new=migrate_material_inertia_v4(old,contract());g,x,y,free,present,prob=inputs()
 with torch.no_grad():
  expected=old(x,y,free,g,present,racket_probability=prob)
  actual=new(x,y,free,g,present,racket_probability=prob)
 torch.testing.assert_close(actual,expected,rtol=0,atol=0)
 for key,value in old.state_dict().items():torch.testing.assert_close(new.state_dict()[key],value,rtol=0,atol=0)
 ck=dict(schema=new.schema,material_contract=new.material_contract)
 restored=action_ball_from_checkpoint(ck);restored.load_state_dict(new.state_dict(),strict=True)
 with torch.no_grad():torch.testing.assert_close(restored(x,y,free,g,present,racket_probability=prob),actual,rtol=0,atol=0)
 with pytest.raises(ValueError,match='Missing native material'):action_ball_from_checkpoint(dict(schema=new.schema))


def test_mechanical_features_and_learned_response_keep_input_gradients():
 torch.manual_seed(9);model=MaterialInertiaActionBallV4(material_contract=contract());g,x,y,free,present,prob=inputs()
 y=y.detach().requires_grad_(True)
 with torch.no_grad():model.mechanical_head.net[-1].weight.normal_(0,.01)
 correction,info=model(x,y,free,g,present,racket_probability=prob,details=True)
 grad,=torch.autograd.grad(correction[:,3:].sum(),y)
 assert info['mechanical_features'].shape==(2,24)
 assert torch.isfinite(grad).all() and grad[:,13:49].abs().sum()>0
 assert model.component_material.requires_grad is False
 zero=model(x,y,free,g,torch.zeros_like(present),racket_probability=prob)
 torch.testing.assert_close(zero,torch.zeros_like(zero),atol=0,rtol=0)
 zero=model(x,y,free,g,present,racket_probability=torch.zeros_like(prob))
 torch.testing.assert_close(zero,torch.zeros_like(zero),atol=0,rtol=0)


def test_wrong_geometry_never_silently_falls_back():
 model=MaterialInertiaActionBallV4(material_contract=contract());g,x,*_=inputs();g.asset_sha256='wrong'
 with pytest.raises(ValueError,match='asset mismatch'):model.mechanical_features(x,g)


def test_checkpoint_cannot_override_pinned_material_or_mass_buffers():
 model=MaterialInertiaActionBallV4(material_contract=contract())
 for key in ('component_material','inertia.geometry.mass'):
  state={k:v.clone() for k,v in model.state_dict().items()};state[key].flatten()[0]+=1
  with pytest.raises(ValueError,match='Immutable mechanical'):model.load_state_dict(state,strict=True)
