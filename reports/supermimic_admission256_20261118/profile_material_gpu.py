"""Bounded read-only GPU minibatch throughput, no optimizer or native evaluation."""
import argparse,json,time,hashlib
from pathlib import Path
import torch
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
from quadsmash.rl.supermimic_action_ball_v1 import action_ball_from_checkpoint
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition,ball_error
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from quadsmash.rl.supermimic_physical_dataset_v1 import valid_windows
p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True);p.add_argument('--packed',type=Path,required=True);p.add_argument('--split',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
a.output.mkdir();torch.set_num_threads(2);torch.manual_seed(20261118);device='cuda:0'
ck={k:torch.load(a.fit/f,map_location='cpu',weights_only=True) for k,f in [('physical','world.pt'),('ball','ball.pt'),('action','action_ball_candidate.pt')]}
w=PhysicalWorldV2(ck['physical']['source_manifest']['task_contract']['asset']['sha256']).to(device);w.load_state_dict(ck['physical']['model'],strict=True)
b=BallResidualV2().to(device);b.load_state_dict(ck['ball']['model'],strict=True)
s=action_ball_from_checkpoint(ck['action']).to(device);s.load_state_dict(ck['action']['model'],strict=True)
initial=[tensor_digest(x.state_dict().items()) for x in (w,b,s)]
package=torch.load(a.packed,map_location='cpu',weights_only=True);cpu=package['tensors']
keys=('state','next_state','targets','previous_action','feet','present','elapsed','next_feet')
data={k:cpu[k].to(device) for k in keys}
mask=valid_windows(cpu['episode'],cpu['terminated'],cpu['timeout'],16);train=json.loads(a.split.read_text())['train_lanes'];allowed=torch.zeros(128,dtype=torch.bool);allowed[train]=True
# Same finite recorded-contact neighbourhood for all throughput measurements.
mask&=allowed[None];mask&=(cpu['elapsed'][:len(mask)]>=2.5)&(cpu['elapsed'][:len(mask)]<3.0)
starts=torch.nonzero(mask).to(device)
contract=ck['physical']['source_manifest']['task_contract'];suite=Path(contract['suite']['path'])
assert hashlib.sha256(suite.read_bytes()).hexdigest()==contract['suite']['sha256']
launch=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in json.loads(suite.read_text())['cases']],device=device)
records=[]
for batch in (128,512,1024,2048):
    selected=starts[torch.arange(batch,device=device)%len(starts)];t,l=selected.unbind(1)
    trials=[]
    for repeat in range(2):
        for model in (w,b,s):model.zero_grad(set_to_none=True)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();fwd=time.perf_counter()
        state=data['state'][t,l];feet=data['feet'][t,l].float();present=data['present'][t,l];loss=state.new_zeros(())
        for step in range(16):
            state,feet,present=ball_transition(w,b,state,data['targets'][t+step,l],data['previous_action'][t+step,l],feet,present,data['elapsed'][t+step,l],launch[l],action_model=s)
            truth=data['next_state'][t+step,l]
            loss+=(((state[:,:49]-truth[:,:49])/w.delta_scale[:49]).square().mean()+ball_error(state[:,49:],truth[:,49:]))/16
        torch.cuda.synchronize();forward_s=time.perf_counter()-fwd;back=time.perf_counter();loss.backward();torch.cuda.synchronize()
        backward_s=time.perf_counter()-back
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite profile loss')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for model in (w,b,s) for p in model.parameters()):raise FloatingPointError('Nonfinite profile gradient')
        trials.append(dict(repeat=repeat,forward_s=forward_s,backward_s=backward_s,elapsed_s=time.perf_counter()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),loss=float(loss.detach())))
        del state,feet,present,loss
    entry=dict(batch=batch,window=16,trials=trials,supervised_positions_per_second=batch*16/trials[-1]['elapsed_s'],optimizer_steps=0)
    records.append(entry);print(json.dumps(entry),flush=True)
    (a.output/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
    for model in (w,b,s):model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
assert initial==[tensor_digest(x.state_dict().items()) for x in (w,b,s)]
(a.output/'completed.json').write_text(json.dumps(dict(scope=__doc__,device=torch.cuda.get_device_name(),records=records,weights_unchanged=True,optimizer_steps=0,actor_updates=0,limitations='Model-only16-step workload, no task model or native Isaac allocation; not measured full64-step actor BPTT or combined native training peak. No hyperparameter or checkpoint selection.'),indent=2)+'\n')
