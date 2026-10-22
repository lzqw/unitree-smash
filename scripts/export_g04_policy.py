#!/usr/bin/env python3
"""Export an autonomous actor only, with CPU parity against its native checkpoint."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.tasks.stand.observation_contract import OBSERVATION_SIZE,FRAME_FIELDS

class Actor(torch.nn.Module):
    def __init__(self,policy):
        super().__init__();self.normalizer=policy.actor_obs_normalizer;self.actor=policy.actor
    def forward(self,observations):
        return self.actor(self.normalizer(observations))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--observations',type=Path,required=True,help='Actual rollout dataset for export parity')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    config=state['native']['training_config'];kw=dict(config['policy']);kw.pop('class_name')
    policy=ActorCritic(TensorDict({'policy':torch.zeros(1,OBSERVATION_SIZE)},batch_size=[1]),config['obs_groups'],18,**kw)
    policy.load_state_dict(state['model_state_dict']);policy.eval()
    data=torch.load(a.observations,map_location='cpu',weights_only=False)['observations'].flatten(0,1)
    obs=data[::max(1,len(data)//4096)].contiguous()
    module=Actor(policy).eval()
    with torch.inference_mode():
        expected=policy.act_inference(TensorDict({'policy':obs},batch_size=[len(obs)]))
        exported=torch.jit.trace(module,obs[:2]);exported.save(str(a.output/'actor.pt'))
        actual=torch.jit.load(str(a.output/'actor.pt'))(obs)
        error=float((actual-expected).abs().max())
        if not torch.allclose(actual,expected,atol=1e-6,rtol=1e-6):raise RuntimeError('export parity failed')
    metadata=dict(checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        export_sha256=hashlib.sha256((a.output/'actor.pt').read_bytes()).hexdigest(),
        interface=state['native']['interface'],frame_fields=FRAME_FIELDS,observation_shape=['batch',OBSERVATION_SIZE],
        action_shape=['batch',18],execution='deterministic learned neural actor; no teacher or expert prefix',
        action_postprocessing='Apply the original JumpEnv time-dependent action scale, [-1,1] command clipping, joint limits and fixed actuator model.',
        export_parity=dict(source=str(a.observations),samples=len(obs),maximum_absolute_error=error),
        scope='G04 jump recovery only; future striking requires further authorized training and validation')
    (a.output/'interface.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata))
if __name__=='__main__':main()
