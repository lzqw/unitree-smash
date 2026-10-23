"""Observable bounded full-body controller for dynamic-reach curriculum data."""
import torch

LOW=[.14,.06,.05,.15,-.70,-.40,-.40,-1.30,-.75,.1,-2.7,0.,.7,.7,-2.2,-1.,-.35,-.4,-.4,-.4,-.4]+[-.45]*6
HIGH=[.38,.30,.40,.65,-.15,1.2,1.2,-.837,.75,1.5,-1.2,.25,1.3,1.3,-1.4,1.,.35,.4,.4,.4,.4]+[.45]*6
INITIAL=[.2621,.1345,.35,.525,-.454,.365,.903,-.95,0.,-.108,-2.02,.124,1.02,1.05,-1.775,-.20,0.,0.,0.,0.,0.]+[0.]*6

def dynamic_reach_action(obs, parameters, default, limits, arm):
  t=obs['policy'][:,775];ct,pt,tt=parameters[:,:3].unbind(1);crouch=(t>=.8)&(t<.8+ct);push=(t>=.8+ct)&(t<.8+ct+pt);tuck=(t>=.8+ct+pt)&(t<.8+ct+pt+tt);recover=t>=.8+ct+pt+tt
  q=default[None,:].repeat(len(t),1)
  for front,sl in [(True,slice(4,6)),(False,slice(6,8))]:
   q[:,sl]=torch.where(crouch[:,None],default[sl]+parameters[:,3:4],torch.where(push[:,None],parameters[:,5 if front else 6,None]+parameters[:,8:9]*(1 if front else -1),torch.where(tuck[:,None],parameters[:,9:10],torch.where(recover[:,None],parameters[:,12 if front else 13,None],default[sl]))))
  q[:,8:12]=torch.where(crouch[:,None],default[8:12]+parameters[:,4:5],torch.where(push[:,None],parameters[:,7:8],torch.where(tuck[:,None],parameters[:,10:11],torch.where(recover[:,None],parameters[:,14:15],default[8:12]))))
  q[:,4:6]+=(recover*(parameters[:,15]*obs['policy'][:,6]+parameters[:,16]))[:,None];q[:,6:8]-=(recover*(parameters[:,15]*obs['policy'][:,6]+parameters[:,16]))[:,None]
  q[:,:12]-=(recover*parameters[:,11])[:,None]*obs['policy'][:,27:39]
  q[:,:4]+=parameters[:,17:21]*((crouch|push|tuck).float()[:,None])
  blend=((t-.25)/.45).clamp(0,1)*((2.05-t)/.35).clamp(0,1);q[:,12:]=torch.lerp(default[12:][None,:],arm[None,:]+parameters[:,21:27],blend[:,None])
  if parameters.shape[1]>=34:
   swing=((t-parameters[:,33])/.06).clamp(0,1)*((2.05-t)/.35).clamp(0,1)
   q[:,12:]+=parameters[:,27:33]*swing[:,None]
  if parameters.shape[1]>=38:
   # Optional landing feedback for forward motion. Only measured body velocity,
   # gravity and the existing recovery phase enter these bounded joint targets.
   q[:,4:8]+=(recover*(-parameters[:,34]*obs['policy'][:,0]
                         +parameters[:,35]*obs['policy'][:,6]))[:,None]
   q[:,:4]+=(recover*(parameters[:,36]*obs['policy'][:,7]
                       +parameters[:,37]*obs['policy'][:,1]))[:,None]
  return (2*(q-limits[:,0])/(limits[:,1]-limits[:,0])-1).clamp(-1,1)
