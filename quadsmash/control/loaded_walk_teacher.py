"""Observable IK trot controller for the unchanged loaded Go2 model.

Joint geometry is read from the pinned Go2 USD: thigh/calf length .213 m,
hip axis X and thigh/calf axes Y. It commands positions only.
"""
import torch

LOW=[.30,.04,.025,.27,.55,.8,-.08,-.8,-.8,0.,1.0]
HIGH=[.95,.32,.12,.38,.82,5.,.08,.8,.8,1.5,3.0]
INITIAL=[.65,.12,.06,.34,.65,2.,0.,.15,.15,.3,2.7]


def loaded_walk_action(obs, parameters, default, limits, *, balanced=False):
    t=obs['policy'][:,775]
    period,stride,lift,height,duty,kp,xbias,roll_gain,pitch_gain,kv,duration=parameters.unbind(1)
    phase=((t[:,None]-.3)/period[:,None]+torch.tensor([0.,.5,.5,0.],device=t.device))%1
    swing=phase>=duty[:,None]
    u=((phase-duty[:,None])/(1-duty[:,None])).clamp(0,1)
    relative_x=torch.where(swing,-.5+u,.5-phase/duty[:,None])
    error=obs['policy'][:,760]
    drive=torch.tanh(kp*error-kv*obs['policy'][:,0])
    ramp=((t-.3)/.35).clamp(0,1)*((duration+.3-t)/.35).clamp(0,1)
    x=xbias[:,None]+relative_x*stride[:,None]*drive[:,None]*ramp[:,None]
    z=-height[:,None]+swing.float()*lift[:,None]*torch.sin(torch.pi*u)*ramp[:,None]
    if balanced:
        # Preserve the audited front/rear stance offsets supporting the load.
        nominal_x=-.213*(torch.sin(default[4:8])+torch.sin(default[4:8]+default[8:12]))
        nominal_z=-.213*(torch.cos(default[4:8])+torch.cos(default[4:8]+default[8:12]))
        x=x+nominal_x[None,:]
        z=z+nominal_z[None,:]-nominal_z.mean()
    calf=-torch.acos(((x.square()+z.square()-2*.213**2)/(2*.213**2)).clamp(-.9999,.9999))
    thigh=torch.atan2(-x,-z)-.5*calf
    q=default[None,:].repeat(len(t),1)
    q[:,:4]+=roll_gain[:,None]*obs['policy'][:,7:8]
    q[:,4:8]=thigh+pitch_gain[:,None]*obs['policy'][:,6:7]*torch.tensor([1.,1.,-1.,-1.],device=t.device)
    q[:,8:12]=calf
    if balanced:
        blend=((t-.15)/.35).clamp(0,1)*((duration+.9-t)/.6).clamp(0,1)
        q=torch.lerp(default[None,:].expand_as(q),q,blend[:,None])
    return (2*(q-limits[:,0])/(limits[:,1]-limits[:,0])-1).clamp(-1,1)
