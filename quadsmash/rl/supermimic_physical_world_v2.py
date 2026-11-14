"""Physical dynamics with separately supervised contact transitions.

Contact is a predicted observation flag, not a certified ground constraint.
Dynamics fitting can stop gradients into this supervised branch; inference
retains gradients to state/action inputs for future BPTT use.
"""
import torch
from torch import nn

from .supermimic_physical_world_v1 import PhysicalWorldV1
from .supermimic_physical_dataset_v1 import decode_physical, valid_windows
from .supermimic_full_body_geometry_v1 import FullBodyGeometryV1
from .supermimic_v1 import MLP


class PhysicalWorldV2(PhysicalWorldV1):
    schema = 'physical55-separated-contact-transitions-v2'

    def __init__(self, asset_sha256, hidden=(1024,1024,1024), contact_hidden=(256,256)):
        super().__init__(hidden=hidden)
        self.geometry=FullBodyGeometryV1(asset_sha256)
        self.contact=MLP(119,contact_hidden,8)
        # A transition head preserves persistence as an explicit prior.
        nn.init.zeros_(self.contact.net[-1].weight)
        nn.init.constant_(self.contact.net[-1].bias,-3.)

    def import_v1(self, source):
        self.body.load_state_dict(source.body.state_dict(),strict=True)
        self.flight.load_state_dict(source.flight.state_dict(),strict=True)
        with torch.no_grad():
            for name in ('input_mean','input_scale','delta_mean','delta_scale','normalization_ready'):
                getattr(self,name).copy_(getattr(source,name))

    def contact_probabilities(self,state,targets,previous_action,feet,present,elapsed,launch):
        feet=feet.to(state.dtype)
        normalized=(self.features(state,targets,previous_action,feet,present,elapsed,launch)-self.input_mean)/self.input_scale
        p=decode_physical(state)
        geometry=self.geometry(p['joint_position'],p['joint_velocity'],p['root_position_w'],
            p['root_quaternion_w'],p['root_link_velocity_w'],p['root_angular_velocity_w'])
        geometry_features=torch.cat((geometry['feet'][:,:,2]/.05,
            geometry['velocity'][:,self.geometry.feet].flatten(1)),dim=-1)
        rates=self.contact(torch.cat((normalized,geometry_features),-1)).sigmoid()
        load,unload=rates[:,:4],rates[:,4:]
        next_probability=feet*(1.-unload)+(1.-feet)*load
        return next_probability.clamp(1e-6,1.-1e-6)

    def forward(self,state,targets,previous_action,feet,present,elapsed,launch,*,detach_contacts=False):
        args=(state,targets,previous_action,feet,present,elapsed,launch)
        predicted,_,active=super().forward(*args)
        if detach_contacts:
            # Used ONLY while fitting physical dynamics. Actor inference uses
            # the differentiable branch below with model parameters frozen.
            with torch.no_grad():probability=self.contact_probabilities(*args)
        else:
            probability=self.contact_probabilities(*args)
        return predicted,torch.logit(probability),active


def event_window_pools(data,lanes,steps):
    valid=valid_windows(data['episode'],data['terminated'],data['timeout'],steps)
    allowed=torch.zeros(valid.shape[1],dtype=torch.bool);allowed[lanes]=True
    valid &= allowed[None]
    delta=data['next_event_state']>data['event_state']
    names=('jump','qualified_contact','cross','target','recovery','body_ground','frame','fault')
    pools={'uniform':valid.nonzero()}
    for i,name in enumerate(names):
        mask=delta[:,:,i].unfold(0,steps,1).any(-1)&valid
        if mask.any():pools[name]=mask.nonzero()
    foot=(data['next_foot_contact']!=data['foot_contact']).any(-1).unfold(0,steps,1).any(-1)
    if (foot&valid).any():pools['foot_transition']=(foot&valid).nonzero()
    return pools


def sample_event_windows(pools,count,generator,iteration=0):
    names=[name for name in pools if name!='uniform']
    uniform=count//2 if names else count
    schedule=['uniform']*uniform
    schedule += [names[(i+iteration)%len(names)] for i in range(count-uniform)]
    result=[];counts={name:0 for name in pools}
    for name in dict.fromkeys(schedule):
        n=schedule.count(name);pool=pools[name]
        if not len(pool):raise ValueError('Empty sampling category: '+name)
        result.append(pool[torch.randint(len(pool),(n,),generator=generator)])
        counts[name]=n
    return torch.cat(result),counts
