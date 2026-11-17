"""Read-only paired-prefix capture. Only an explicit diagnostic evaluation uses it."""
import hashlib,json
from pathlib import Path
import torch
from .supermimic_physical_capture_v1 import capture_state,capture_controller,tensor_tree_copy
from .supermimic_physical_dataset_v1 import encode_physical

BRANCHES=('nominal_a','nominal_b','minus','plus')

def tree_digest(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            t=x.detach().cpu().contiguous();h.update(str((t.dtype,tuple(t.shape))).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for key in sorted(x):h.update(str(key).encode());visit(x[key])
        elif isinstance(x,(tuple,list)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()

def validate_spec(spec):
    if spec.get('schema')!='native-paired-prefix-spec-v1':raise ValueError('Unknown probe spec')
    if type(spec.get('tick'))is not int or not 1<=spec['tick']<799:raise ValueError('Probe tick must be within original episode')
    if type(spec.get('lane'))is not int or not 0<=spec['lane']<128:raise ValueError('Probe lane outside original128')
    direction=torch.tensor(spec['request_direction'],dtype=torch.float64)
    if direction.shape!=(18,) or not torch.isfinite(direction).all() or not direction.norm()>0:raise ValueError('Need finite nonzero18D request direction')
    epsilon=spec['epsilon_request']
    if not isinstance(epsilon,(int,float)) or not 0<epsilon<=2:raise ValueError('Declare a positive request perturbation, no actuator changes')
    if spec.get('head') not in ('recovery','approach'):raise ValueError('Unknown head')
    return spec

class NativePairCapture:
    def __init__(self,spec,branch):
        self.spec=validate_spec(spec);self.branch=branch
        if branch not in BRANCHES:raise ValueError('Unknown probe branch')
        self.prefix=hashlib.sha256();self.start=None;self.raw=None
    def before(self,tick,env,observation,request):
        if tick==self.spec['tick']:
            state=capture_state(env)
            self.start=tensor_tree_copy(dict(before=state,observation=observation,executor_before=capture_controller(env._sm_residual_controller)),cpu=True)
            self.prefix.update(tree_digest(self.start).encode())
            action=request.clone();sign={'nominal_a':0.,'nominal_b':0.,'minus':-1.,'plus':1.}[self.branch]
            action[self.spec['lane']]+=sign*self.spec['epsilon_request']*request.new_tensor(self.spec['request_direction'])
            self.request_before=request.detach().cpu().clone();self.request_after=action.clamp(-1,1).detach().cpu().clone()
            return action.clamp(-1,1)
        self.prefix.update(tree_digest(dict(observation=observation,request=request,physical=capture_state(env)['physical'])).encode())
        return request
    def after(self,tick,env,done):
        if tick<self.spec['tick']:
            self.prefix.update(tree_digest(dict(targets=env._sm_joint_targets,final=env._sm_final_action,done=done)).encode())
        elif tick==self.spec['tick']:
            if done.bool().any():raise ValueError('Probe crosses terminal/reset; reject counterfactual')
            self.raw=tensor_tree_copy(dict(start=self.start,next_state=encode_physical(capture_state(env)['physical']),
                final_joint_targets=env._sm_joint_targets,final_normalized=env._sm_final_action,request_before=self.request_before,
                request_after=self.request_after,task_after=capture_state(env)['task']),cpu=True)
    def save(self,output,actor,task_contract,policy_execution,physics,seed):
        if self.raw is None:raise ValueError('Native probe did not finish')
        from .supermimic_evidence_binding_v1 import runtime_identity
        static=runtime_identity(actor,torch.nn.Identity(),task_contract,policy_execution,0)
        value=dict(execution_code_sha256=static['execution_code_sha256'],schema='native-paired-prefix-run-v1',branch=self.branch,spec=self.spec,raw=self.raw,
            prefix_sha256=self.prefix.hexdigest(),executor_before_sha256=tree_digest(self.start['executor_before']),
            actor_state=tensor_tree_copy(actor.state_dict(),cpu=True),task_contract=task_contract,policy_execution=policy_execution,
            physics=physics,seed=seed,restore_mode='from-ground-identical-prefix',actor_updates=0,world_updates=0,critic_updates=0)
        path=Path(output)/'native_probe.pt';tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
        (Path(output)/'native_probe_manifest.json').write_text(json.dumps(dict(schema=value['schema'],branch=self.branch,
            spec=self.spec,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scope='diagnostic only; no score; requires four-branch comparison'),indent=2)+'\n')
