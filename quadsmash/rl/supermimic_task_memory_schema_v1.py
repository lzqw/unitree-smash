"""Named task-memory tensor contract for a contextual critic/task predictor.

This is an encoding, not a substitute dynamics model. Required leaves and
their validity masks come from a native start. Missing leaves are errors.
Integer counters and boolean latches remain distinguishable from floats.
"""
from dataclasses import dataclass, asdict
import torch


GROUPS = ('rally','recovery','auxiliary/contact_recovery','auxiliary/jump',
          'auxiliary/environment','auxiliary/contact_episodes')


def at_path(tree,path):
    for part in path.split('/'):tree=tree[part]
    return tree


@dataclass(frozen=True)
class MemoryField:
    path: str
    shape: tuple
    kind: str
    width: int


class TaskMemorySchemaV1:
    version = 'native-named-task-memory-value-valid-v1'
    feedback_contract = 'schema-decode-encode-equivalent-feedback-v1'

    def __init__(self,fields):
        self.fields=tuple(fields)
        self.width=sum(f.width for f in self.fields)

    @classmethod
    def from_native(cls,memory,count):
        fields=[]
        for path in GROUPS:
            values=at_path(memory,path)
            if not isinstance(values,dict) or not values:raise ValueError('Missing memory group '+path)
            for name,tensor in sorted(values.items()):
                if not isinstance(tensor,torch.Tensor) or tensor.ndim<1 or len(tensor)!=count:
                    raise ValueError('Non-lane task field '+path+'/'+name)
                kind='bool' if tensor.dtype==torch.bool else 'float' if tensor.is_floating_point() else 'int'
                fields.append(MemoryField(path+'/'+name,tuple(tensor.shape[1:]),kind,tensor[0].numel()))
        return cls(fields)

    def encode(self,memory):
        numeric=[];validity=[]
        count=None
        for f in self.fields:
            value=at_path(memory,f.path)
            if tuple(value.shape[1:])!=f.shape:raise ValueError('Task shape changed: '+f.path)
            count=len(value) if count is None else count
            if len(value)!=count:raise ValueError('Task lane count mismatch')
            valid=torch.isfinite(value) if value.is_floating_point() else torch.ones_like(value,dtype=torch.bool)
            numeric.append(torch.where(valid,value,torch.zeros_like(value)).reshape(count,-1).float())
            validity.append(valid.reshape(count,-1).float())
        return torch.cat(numeric,-1),torch.cat(validity,-1)

    def manifest(self):
        return dict(schema=self.version,numeric_width=self.width,critic_width=2*self.width,
                    fields=[asdict(f) for f in self.fields],
                    absent_timestamps='explicit validity mask; not a real zero timestamp',
                    future_transition='not implemented by this encoding')

    def feedback_values(self,values,validity):
        """Exact value feedback of encode(decode_surrogate(...)).

        Fitting must not feed soft validity back when imagination thresholds
        missing float fields. Boolean/counter predictions stay finite current
        state values; the validity threshold is the existing discrete branch.
        No future native field or event is introduced by this conversion.
        """
        if values.shape!=validity.shape or values.ndim!=2 or values.shape[-1]!=self.width:
            raise ValueError('Task surrogate feedback width mismatch')
        # Schema-only masks are immutable; cache by device rather than
        # copying them from CPU at every autoregressive step. No causal state
        # or per-lane execution history is cached here.
        cache=getattr(self,'_feedback_masks',None)
        if cache is None:self._feedback_masks={};cache=self._feedback_masks
        key=str(values.device)
        if key not in cache:
            boolean=[f.kind=='bool' for f in self.fields for _ in range(f.width)]
            floating=[f.kind=='float' for f in self.fields for _ in range(f.width)]
            cache[key]=(torch.tensor(boolean,device=values.device,dtype=torch.bool),
                        torch.tensor(floating,device=values.device,dtype=torch.bool))
        boolean,floating=cache[key]
        numeric=torch.where(boolean,values.clamp(0.,1.),values)
        valid=torch.isfinite(numeric)&(~floating | (validity>.5))
        return torch.where(valid,numeric,torch.zeros_like(numeric)).float(),valid.float()

    def decode_surrogate(self,values,validity,template):
        """Decode predicted fields; validity selection is explicitly discrete.

        Boolean values remain probabilities for soft reward/context use. The
        observation phase adapter must explicitly threshold them; these are
        never native contact labels or formal score inputs.
        """
        from .supermimic_physical_imagination_state_v1 import tensor_copy
        result=tensor_copy(template,values.device)
        if values.shape!=validity.shape or values.shape[-1]!=self.width:
            raise ValueError('Task surrogate width mismatch')
        offset=0
        for f in self.fields:
            piece=values[:,offset:offset+f.width].reshape(len(values),*f.shape)
            valid=validity[:,offset:offset+f.width].reshape_as(piece)
            if f.kind=='bool':piece=piece.clamp(0.,1.)
            elif f.kind=='float':piece=torch.where(valid>.5,piece,torch.full_like(piece,float('nan')))
            target=result
            parts=f.path.split('/')
            for name in parts[:-1]:target=target[name]
            target[parts[-1]]=piece
            offset+=f.width
        return result
