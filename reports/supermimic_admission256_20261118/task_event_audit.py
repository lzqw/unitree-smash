"""Read-only event timing/gradient audit; no optimizer or native execution."""
import json
from pathlib import Path
import torch
from quadsmash.rl.supermimic_task_memory_schema_v1 import TaskMemorySchemaV1, MemoryField
from quadsmash.rl.supermimic_task_transition_model_v1 import NativeTaskTransitionSurrogateV1
from quadsmash.rl.supermimic_event_hazard_v1 import event_hazard_loss

torch.set_num_threads(2)
root=Path('/home/autodrive/LZQW/experiments/supermimic_admission256_20261118_s5lbit6k')
destination=root/'task_event_timing_gradient_audit.json'
if destination.exists(): raise FileExistsError(destination)
package=torch.load('/home/autodrive/LZQW/experiments/supermimic_frozen_closure_20261117_223243/packed/data.pt',map_location='cpu',weights_only=True)
d=package['tensors']
checkpoint=torch.load(root/'task_fit_split/task.pt',map_location='cpu',weights_only=True)
schema=TaskMemorySchemaV1([MemoryField(**dict(f,shape=tuple(f['shape']))) for f in package['task_schema']['fields']])
model=NativeTaskTransitionSurrogateV1(schema,d['torque'].shape[-1]);model.load_state_dict(checkpoint['model'],strict=True);model.eval()
split=json.loads((root/'fixed_split.json').read_text())
names=('jump','hit','cross','target','recovery','body','frame','fault')
def predict(t,l):
    return model.forward_values(d['state'][t,l],d['next_state'][t,l],d['targets'][t,l],d['previous_action'][t,l],d['feet'][t,l],d['elapsed'][t,l],d['values'][t,l],d['valid'][t,l])
probs=torch.empty(800,len(split['validation_lanes']),8)
with torch.no_grad():
    for tick in range(800):
        probs[tick]=predict(tick,split['validation_lanes'])['hazards'].sigmoid()
timing={}
for j,name in enumerate(names):
    rows=[]
    for col,lane in enumerate(split['validation_lanes']):
        times=(d['next_event_state'][:,lane,j]>d['event_state'][:,lane,j]).nonzero().flatten()
        for tick in times.tolist():
            indexes=torch.arange(max(0,tick-4),min(800,tick+5))
            indexes=indexes[d['episode'][indexes,lane]==d['episode'][tick,lane]]
            eligible=d['event_state'][indexes,lane,j]==0
            values=probs[indexes,col,j]
            maximum=values[eligible].max() if eligible.any() else torch.tensor(float('nan'))
            rows.append(dict(lane=lane,tick=tick,offsets=(indexes-tick).tolist(),probabilities=values.tolist(),eligible=eligible.tolist(),eligible_max=float(maximum),exact_probability=float(probs[tick,col,j])))
    timing[name]=dict(count=len(rows),exact_over_half=sum(r['exact_probability']>=.5 for r in rows),neighborhood_over_half=sum(r['eligible_max']>=.5 for r in rows),rows=rows)

# Fixed training-only batch: half uniform, half evenly drawn observed edges.
generator=torch.Generator().manual_seed(20261118)
lanes=torch.tensor(split['train_lanes']); t=torch.randint(800,(128,),generator=generator);l=lanes[torch.randint(len(lanes),(128,),generator=generator)]
edges=(d['next_event_state'][:,lanes]>d['event_state'][:,lanes]).any(-1).nonzero()
chosen=edges[torch.randint(len(edges),(128,),generator=generator)]
t=torch.cat((t,chosen[:,0]));l=torch.cat((l,lanes[chosen[:,1]]))
out=predict(t,l);valid=d['next_valid'][t,l]
losses=dict(memory=(((out['numeric']-d['next_values'][t,l])/model.memory_scale).square()*valid).sum()/valid.sum().clamp_min(1),
    validity=.1*torch.nn.functional.binary_cross_entropy_with_logits(out['valid_logits'],valid),
    hazards=event_hazard_loss(out['hazards'],d['event_state'][t,l],d['next_event_state'][t,l]),
    torque=.1*((out['torque']-d['torque'][t,l])/20).square().mean(),
    terminal=torch.nn.functional.binary_cross_entropy_with_logits(out['terminal_logit'],d['terminated'][t,l].float().reshape(-1,1)),
    complete=torch.nn.functional.binary_cross_entropy_with_logits(out['complete_logit'],d['complete_reward'][t,l].float().reshape(-1,1)))
shared=list(model.net.net[0].parameters())
gradient={}
for name,loss in losses.items():
    grads=torch.autograd.grad(loss,shared,retain_graph=True)
    gradient[name]=dict(loss=float(loss.detach()),shared_first_layer_gradient_norm=float(torch.sqrt(sum(g.square().sum() for g in grads))))
result=dict(actor_updates=0,model_updates=0,scope='Recorded native transition inputs only; event probabilities are surrogate diagnostics, not scoring. Fixed training-only gradient audit; unchanged heldout lanes.',timing=timing,gradient=gradient)
destination.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(path=str(destination),gradient=gradient,timing={k:{key:r[key] for key in ('count','exact_over_half','neighborhood_over_half')} for k,r in timing.items()}),indent=2))
