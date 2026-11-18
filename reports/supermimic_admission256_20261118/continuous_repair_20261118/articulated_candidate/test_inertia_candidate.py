import importlib.util
from pathlib import Path
import numpy as np
import pinocchio as pin
import torch
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from quadsmash.rl.supermimic_full_body_geometry_v1 import rotation
p=Path(__file__).with_name('inertia_candidate.py');spec=importlib.util.spec_from_file_location('candidate',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_named_mass_and_point_response_match_existing_rigid_body_model():
 rigid=LoadedPinocchioModelV1();model=m.ArticulatedInertiaCandidate(rigid.record['asset_sha256'])
 gen=torch.Generator().manual_seed(188);qj=torch.randn(3,18,generator=gen,dtype=torch.float64)*.3
 quat=torch.randn(3,4,generator=gen,dtype=torch.float64);quat=quat/quat.norm(dim=-1,keepdim=True)
 point=torch.randn(3,3,generator=gen,dtype=torch.float64)*.05
 A,info=model(qj,quat,point)
 for i in range(3):
  q=pin.neutral(rigid.model);R=rotation(quat[i]).numpy();q[3:7]=pin.Quaternion(R).coeffs();q[rigid.q_indices]=qj[i].numpy()
  M=pin.crba(rigid.model,rigid.data,q);M=np.triu(M)+np.triu(M,1).T
  pin.computeJointJacobians(rigid.model,rigid.data,q);pin.updateFramePlacements(rigid.model,rigid.data)
  frame=rigid.body_frames['Racket'];J=pin.getFrameJacobian(rigid.model,rigid.data,frame,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).copy()
  arm=rigid.data.oMf[frame].rotation@point[i].numpy();Jp=J[:3]+np.cross(J[3:].T,arm).T
  expected=Jp@np.linalg.solve(M,Jp.T)
  np.testing.assert_allclose(A[i].detach().numpy(),expected,rtol=1e-9,atol=1e-9)
  assert torch.linalg.eigvalsh(info['mass_matrix'][i]).min()>0

def test_effective_inertia_has_state_gradient():
 rigid=LoadedPinocchioModelV1();model=m.ArticulatedInertiaCandidate(rigid.record['asset_sha256'])
 q=torch.linspace(-.3,.3,18,dtype=torch.float64)[None].requires_grad_();quat=torch.tensor([[1.,0.,0.,0.]],dtype=torch.float64)
 point=torch.tensor([[.01,.02,.03]],dtype=torch.float64)
 assert torch.autograd.gradcheck(lambda x:model(x,quat,point)[0],(q,),eps=1e-6,atol=1e-5,rtol=1e-4)

def test_single_contact_restitution_and_coulomb_contract():
 v=torch.tensor([[-5.,2.,0.],[-5.,0.,0.],[1.,2.,0.]],dtype=torch.float64,requires_grad=True)
 n=torch.tensor([[1.,0.,0.]],dtype=torch.float64).expand(3,-1);A=torch.diag(torch.tensor([20.,30.,10.],dtype=torch.float64)).expand(3,-1,-1)
 P,info=m.sphere_impulse(v,n,A,mass=.0052,radius=.025,restitution=.72,friction=.08)
 after=v+(info['relative_inverse_mass']@P[...,None]).squeeze(-1)
 torch.testing.assert_close(after[:2,0],-.72*v[:2,0]);torch.testing.assert_close(P[2],torch.zeros(3,dtype=torch.float64))
 assert (P[:2,1:].norm(dim=-1)<=.08*P[:2,0]+1e-12).all()
 P.square().sum().backward();assert torch.isfinite(v.grad).all() and v.grad.abs().sum()>0
