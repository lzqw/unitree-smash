"""Supervised task-memory transition SURROGATE for physical imagination.

Inputs are current memory and model-predicted physical transition. Native
future memory/events are labels only. No scalar reward is freely predicted.
This module must be fitted and audited before any policy update. It is not
the native task engine and its outputs must never enter formal scoring.
"""
import torch
from torch import nn
from .supermimic_v1 import MLP
from .supermimic_event_hazard_v1 import event_hazard_loss,advance_event_latches


class NativeTaskTransitionSurrogateV1(nn.Module):
    schema='physical-transition-native-task-memory-surrogate-v1'
    supervision_contract='task-memory-numeric-and-boolean-logits-v2'

    def __init__(self,memory_schema,torque_dim,hidden=(512,512)):
        super().__init__()
        self.memory_schema=memory_schema
        self.torque_dim=int(torque_dim)
        width=memory_schema.width
        self.net=MLP(55*2+18*2+4+1+2*width,hidden,2*width+torque_dim+10)
        self.register_buffer('memory_mean',torch.zeros(width))
        self.register_buffer('memory_scale',torch.ones(width))
        self.register_buffer('normalization_ready',torch.tensor(False))
        self.register_buffer('fit_updates',torch.tensor(0,dtype=torch.long))
        kinds=[]
        for field in memory_schema.fields:kinds.extend([field.kind=='bool']*field.width)
        self.register_buffer('bool_columns',torch.tensor(kinds,dtype=torch.bool))

    @torch.no_grad()
    def set_normalization(self,values,validity):
        count=validity.sum(0).clamp_min(1.)
        mean=(values*validity).sum(0)/count
        var=((values-mean).square()*validity).sum(0)/count
        self.memory_mean.copy_(mean)
        self.memory_scale.copy_(var.sqrt().clamp_min(.01))
        self.normalization_ready.fill_(True)

    def forward(self,state,next_state,targets,previous_action,feet,elapsed,memory):
        values,validity=self.memory_schema.encode(memory)
        result=self.forward_values(state,next_state,targets,previous_action,feet,elapsed,values,validity)
        next_memory=self.memory_schema.decode_surrogate(result['numeric'],result['valid_logits'].sigmoid(),memory)
        next_memory['paid_events']=advance_event_latches(memory['paid_events'],result['hazards'])
        return dict(result,memory=next_memory)

    def forward_values(self,state,next_state,targets,previous_action,feet,elapsed,values,validity):
        if not bool(self.normalization_ready):raise RuntimeError('Task surrogate normalization missing')
        normalized=(values-self.memory_mean)/self.memory_scale
        x=torch.cat((state,next_state,targets,previous_action,feet.to(state.dtype),
                     elapsed.reshape(-1,1),normalized*validity,validity),-1)
        out=self.net(x);width=self.memory_schema.width
        delta,valid_logits,torque,hazards,terminal,complete=torch.split(out,
            [width,width,self.torque_dim,8,1,1],dim=-1)
        valid_logits=valid_logits+torch.logit(validity.clamp(1e-4,1-1e-4))
        numeric=values+delta*self.memory_scale
        # Soft boolean predictions carry the previous flag as a persistence
        # prior. This is a learned approximation, not exact event certification.
        boolean_logits=self.boolean_logits(values,delta)
        probability=torch.sigmoid(boolean_logits)
        numeric=torch.where(self.bool_columns,probability,numeric)
        return dict(numeric=numeric,boolean_logits=boolean_logits,valid_logits=valid_logits,
                    torque=torque,hazards=hazards,terminal_logit=terminal,
                    complete_logit=complete)

    def boolean_logits(self,values,delta):
        return torch.logit(values.clamp(1e-4,1-1e-4))+delta

    def supervision_loss(self,output,before,after,torque,terminated,complete):
        target,valid=self.memory_schema.encode(after)
        return self.supervision_loss_values(output,target,valid,before['paid_events'],after['paid_events'],torque,terminated,complete)

    def supervision_loss_values(self,output,target,valid,paid_before,paid_after,torque,terminated,complete,*,per_row=False):
        def bce(logits, labels):
            return nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none').mean(-1)
        loss=self.memory_supervision_loss(output,target,valid,per_row=True)
        loss=loss+.1*bce(output['valid_logits'],valid)
        loss=loss+event_hazard_loss(output['hazards'],paid_before,paid_after,per_row=True)
        loss=loss+.1*((output['torque']-torque)/20.).square().mean(-1)
        loss=loss+bce(output['terminal_logit'],terminated.float().reshape(-1,1))
        loss=loss+bce(output['complete_logit'],complete.float().reshape(-1,1))
        return loss if per_row else loss.mean()

    def memory_supervision_loss(self,output,target,valid,*,per_row=False):
        """Native booleans use logit BCE, not near-saturated probability MSE.

        Forward and checkpoint parameter layout are unchanged. This explicit
        WM supervision adaptation keeps real-valued memory in its existing
        normalized units. Unknown fields never become negative examples.
        """
        continuous=valid*(~self.bool_columns).to(valid.dtype)
        categorical=valid*self.bool_columns.to(valid.dtype)
        numeric=((output['numeric']-target)/self.memory_scale).square()
        binary=nn.functional.binary_cross_entropy_with_logits(
            output['boolean_logits'],target.clamp(0.,1.),reduction='none')
        # Preserve the whole batch's known-value denominator when attributing loss to rows.
        rows = ((numeric*continuous).sum(-1) * (len(target)/continuous.sum().clamp_min(1.))
                +(binary*categorical).sum(-1) * (len(target)/categorical.sum().clamp_min(1.)))
        return rows if per_row else rows.mean()


class AbsoluteBooleanTaskSurrogateV2(NativeTaskTransitionSurrogateV1):
    """Explicit candidate: absolute Bernoulli logits, continuous deltas intact.

    Previous boolean state remains an input. Unlike V1, an observed False
    does not impose a fixed -9.21 output prior that rare transitions must
    overcome. This changes surrogate parameter semantics, never native task
    state/scoring. Old checkpoints must be explicitly migrated or dispatched
    to V1; neither conversion nor model quality is inferred from tensor shape.
    """
    schema='physical-transition-task-absolute-boolean-v2'
    supervision_contract='absolute-boolean-logits-native-memory-v3'

    def boolean_logits(self,values,delta):
        return delta

    @torch.no_grad()
    def initialize_boolean_outputs(self,training_values,training_validity):
        """Explicit migration initializer; caller supplies training rows only."""
        if (training_values.shape!=training_validity.shape or training_values.ndim!=2
                or training_values.shape[-1]!=self.memory_schema.width):
            raise ValueError('Boolean initialization needs aligned training memory rows')
        columns=torch.where(self.bool_columns)[0]
        target=training_values[:,columns];valid=training_validity[:,columns]
        if not ((valid==0)|(valid==1)).all() or not torch.isfinite(target).all():
            raise ValueError('Invalid native boolean initialization values')
        if not (((target==0)|(target==1)) | (valid==0)).all():
            raise ValueError('Initialization requires native binary labels')
        count=valid.sum(0)
        if (count==0).any():
            raise ValueError('Boolean field has no known training labels')
        frequency=((target*valid).sum(0)/count).clamp(1e-4,1-1e-4)
        self.net.net[-1].weight[columns]=0.
        self.net.net[-1].bias[columns]=torch.logit(frequency)
        return dict(known_training_counts=count.tolist(),boolean_frequency=frequency.tolist())


def task_model_for_schema(schema,memory_schema,torque_dim,**kwargs):
    classes={cls.schema:cls for cls in (NativeTaskTransitionSurrogateV1,AbsoluteBooleanTaskSurrogateV2)}
    if schema not in classes:
        raise ValueError('Unknown task transition checkpoint schema: '+str(schema))
    return classes[schema](memory_schema,torque_dim,**kwargs)
