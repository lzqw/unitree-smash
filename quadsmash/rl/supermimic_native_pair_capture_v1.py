"""Read-only paired-prefix capture. Only an explicit diagnostic evaluation uses it."""
import hashlib,json
from pathlib import Path
import torch
from .supermimic_physical_capture_v1 import capture_state,capture_controller,tensor_tree_copy
from .supermimic_physical_dataset_v1 import encode_physical

BRANCHES=('nominal_a','nominal_b','minus','plus')

def probe_lanes(spec):
    return [spec['lane']] if spec['schema']=='native-paired-prefix-spec-v1' else list(spec['lanes'])

def probe_valid_lanes(done,lane):
    valid=~done.detach().bool().reshape(-1)
    lanes=[lane] if isinstance(lane,int) else list(lane)
    if not lanes or any(not 0<=i<len(valid) or not bool(valid[i]) for i in lanes):
        raise ValueError('Target probe lane crosses terminal/reset; reject counterfactual')
    return valid

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
    if spec.get('schema') not in ('native-paired-prefix-spec-v1','native-paired-prefix-batch-spec-v2'):raise ValueError('Unknown probe spec')
    if type(spec.get('tick'))is not int or not 1<=spec['tick']<799:raise ValueError('Probe tick must be within original episode')
    action_tick = spec.get('action_tick', spec['tick'])
    if type(action_tick) is not int or not 1 <= action_tick <= spec['tick']:
        raise ValueError('Action tick must be an integer at or before the event tick')
    action_hold_ticks = spec.get('action_hold_ticks', 1)
    if type(action_hold_ticks) is not int or action_hold_ticks < 1:
        raise ValueError('Action hold ticks must be a positive integer')
    if action_tick + action_hold_ticks - 1 > spec['tick']:
        raise ValueError('Action hold window must end at or before the event tick')
    if spec['schema']=='native-paired-prefix-spec-v1':
        if 'lanes' in spec or 'request_directions' in spec:raise ValueError('Mixed probe schemas')
        if type(spec.get('lane'))is not int or not 0<=spec['lane']<128:raise ValueError('Probe lane outside original128')
        direction=torch.tensor(spec['request_direction'],dtype=torch.float64).unsqueeze(0)
    else:
        if 'lane' in spec or 'request_direction' in spec:raise ValueError('Mixed probe schemas')
        lanes=spec.get('lanes')
        if not isinstance(lanes,list) or not lanes or any(type(i)is not int or not 0<=i<128 for i in lanes) or len(set(lanes))!=len(lanes):
            raise ValueError('Batch probe needs unique original128 lane IDs')
        direction=torch.tensor(spec['request_directions'],dtype=torch.float64)
    if direction.shape!=(len(probe_lanes(spec)),18) or not torch.isfinite(direction).all() or not (direction.norm(dim=-1)>0).all():
        raise ValueError('Need finite nonzero18D request direction per lane')
    if spec.get('capture_wm_window') and (
            spec.get('data_role') != 'training_action_identification' or spec['tick'] - action_tick + 1 != 16):
        raise ValueError('Mixed WM fitting requires a declared 16-step training intervention')
    if 'capture_substeps' in spec and type(spec['capture_substeps']) is not bool:
        raise ValueError('capture_substeps must be an explicit boolean')
    if 'capture_substep_ticks' in spec:
        ticks = spec['capture_substep_ticks']
        if (not spec.get('capture_substeps') or not isinstance(ticks, list) or not ticks
                or any(type(t) is not int or not action_tick <= t <= spec['tick'] for t in ticks)
                or ticks != sorted(set(ticks)) or ticks[-1] != spec['tick']):
            raise ValueError('Substep trace ticks must be ordered, unique and inside the action-to-event window including its endpoint')
    epsilon=spec['epsilon_request']
    if not isinstance(epsilon,(int,float)) or not 0<epsilon<=2:raise ValueError('Declare a positive request perturbation, no actuator changes')
    if spec.get('head') not in ('recovery','approach'):raise ValueError('Unknown head')
    return spec

def perturb_request(request,spec,branch):
    """Diagnostic intervention at one shared tick; original actuator runs next.

    No lane-specific policy rule. Full pre-intervention prefix is identical for
    all selected lanes, and no previous perturbation is hidden in their state.
    """
    validate_spec(spec)
    if branch not in BRANCHES:raise ValueError('Unknown probe branch')
    if request.shape!=(128,18):raise ValueError('Probe requires original128 request shape')
    direction=([spec['request_direction']] if spec['schema']=='native-paired-prefix-spec-v1' else spec['request_directions'])
    action=request.clone();sign={'nominal_a':0.,'nominal_b':0.,'minus':-1.,'plus':1.}[branch]
    action[probe_lanes(spec)]+=sign*spec['epsilon_request']*request.new_tensor(direction)
    return action.clamp(-1,1)

class TrainingWindowRecorder:
    """Record true pre-reset WM supervision exclusively on intervened training lanes."""
    def __init__(self, lanes):
        self.lanes = list(lanes)
        self.schema = None
        self.before_state = None
        self.rows = []

    def begin(self, env, request, public):
        if self.before_state is not None:
            raise RuntimeError('Missing end of previous probe transition')
        self.before_state = capture_state(env)
        self.public = public.detach().clone()
        self.request = request.detach().clone()

    def finish(self, env, reward):
        from .supermimic_physical_loop_replay_v1 import row_tensors
        from .supermimic_task_memory_schema_v1 import TaskMemorySchemaV1

        before = self.before_state
        after = capture_state(env)
        if (before is None or after['control_tick'] != before['control_tick'] + 1
                or not torch.equal(after['episode_index'], before['episode_index'])):
            raise RuntimeError('Native WM labels must precede reset at the next control boundary')
        if self.schema is None:
            memory = dict(rally=before['task'], recovery=before['recovery'], auxiliary=before['auxiliary'])
            self.schema = TaskMemorySchemaV1.from_native(memory, env.num_envs)
        row = row_tensors(dict(before=before, after=after, final_joint_targets=env._sm_joint_targets,
                               terminated=env.reset_terminated, timeout=env.reset_time_outs,
                               reward_components=env._sm_reward_components), self.schema)
        row.update(observation=self.public, requested_action=self.request, final_action=env._sm_final_action,
                   applied_joint_torque=after['applied_torque'])
        self.rows.append({key: value[self.lanes].detach().cpu() for key, value in row.items()})
        self.before_state = None

    def payload(self):
        if self.before_state is not None or len(self.rows) != 16:
            raise ValueError('Incomplete native WM window')
        data = {key: torch.stack([r[key] for r in self.rows]) for key in self.rows[0]}
        if (data['terminated'] | data['timeout']).any() or not (data['episode'] == data['episode'][0]).all():
            raise ValueError('Perturbation crosses a selected episode boundary')
        return dict(schema=self.schema.manifest(), lanes=self.lanes, tensors=data)


def select_substep_lanes(trace, lane_ids):
    """Retain the selected original cases and remap sparse native contact indices to their saved rows."""
    tensors=trace['tensors']
    num_lanes=tensors['shuttle_state_w'].shape[1]
    device=tensors['shuttle_state_w'].device
    selected=torch.as_tensor(lane_ids,dtype=torch.long,device=device)
    mapping=torch.full((num_lanes,),-1,dtype=torch.long,device=device)
    mapping[selected]=torch.arange(len(selected),device=device)
    out=dict(trace,original_lanes=list(lane_ids),tensors={
        key:x if key=='physics_frame' else x.index_select(1,selected)
        for key,x in tensors.items()})
    for name in ('manifold_frames','racket_friction_frames'):
        frames=[]
        for frame in trace[name]:
            local=mapping[frame['lane']]
            keep=local>=0
            frames.append({k:(local[keep] if k=='lane' else v[keep]) for k,v in frame.items()})
        out[name]=frames
    return tensor_tree_copy(out,cpu=True)


class NativePairCapture:
    def __init__(self,spec,branch):
        self.spec=validate_spec(spec);self.branch=branch
        if branch not in BRANCHES:raise ValueError('Unknown probe branch')
        self.prefix=hashlib.sha256();self.start=None;self.raw=None;self.substep_trace=None;self.wm_recorder=None
        self.step_records=[]
        self.trace_ticks=set(spec.get('capture_substep_ticks',[spec['tick']])) if spec.get('capture_substeps') else set()
        self.earlier_traces={}
    def before(self,tick,env,observation,request):
        action_tick = self.spec.get('action_tick', self.spec['tick'])
        event_tick = self.spec['tick']
        action_hold_ticks = self.spec.get('action_hold_ticks', 1)
        if tick in self.trace_ticks:
            from .supermimic_native_substep_trace_v1 import NativeSubstepTraceV1
            self.substep_trace=NativeSubstepTraceV1(env)
        if tick==action_tick:
            if self.spec.get('capture_wm_window'):
                if env.physical_transition_archive is not None:
                    raise ValueError('A native recorder already owns this environment')
                self.wm_recorder = TrainingWindowRecorder(probe_lanes(self.spec))
                env.physical_transition_archive = self.wm_recorder
            state=capture_state(env)
            self.start=tensor_tree_copy(dict(before=state,observation=observation,executor_before=capture_controller(env._sm_residual_controller)),cpu=True)
            self.step_records=[]
            self.prefix.update(tree_digest(self.start).encode())
            action=perturb_request(request,self.spec,self.branch)
            self.request_before=request.detach().cpu().clone();self.request_after=action.detach().cpu().clone()
            return action
        if action_tick < tick < action_tick + action_hold_ticks:
            # Hold the same signed intervention over a short control window.
            # The policy still produces the nominal request each tick, while
            # the probe applies a causal, bounded offset before execution.
            return perturb_request(request, self.spec, self.branch)
        if tick < action_tick:
            self.prefix.update(tree_digest(dict(observation=observation,request=request,physical=capture_state(env)['physical'])).encode())
        return request
    def after(self,tick,env,done):
        action_tick = self.spec.get('action_tick', self.spec['tick'])
        event_tick = self.spec['tick']
        trace=None
        if tick in self.trace_ticks:
            if self.substep_trace is None:
                raise RuntimeError('Missing native substep trace at selected control tick')
            trace=self.substep_trace.finish()
            self.substep_trace=None
            if tick != event_tick:
                self.earlier_traces[tick]=select_substep_lanes(trace,probe_lanes(self.spec))
        if action_tick <= tick <= event_tick and self.start is not None:
            current=capture_state(env)
            self.step_records.append(tensor_tree_copy(dict(
                next_state=encode_physical(current['physical']),
                final_action=current['physical']['final_action'],
                final_joint_targets=env._sm_joint_targets,
                foot_contact=current['physical']['foot_contact'],
                ball_present=current['physical']['ball_present'],
                elapsed_s=current['physical']['elapsed_s'],
                done=done,
            ), cpu=True))
        if tick < action_tick:
            self.prefix.update(tree_digest(dict(targets=env._sm_joint_targets,final=env._sm_final_action,done=done)).encode())
        elif tick==event_tick:
            valid=probe_valid_lanes(done,probe_lanes(self.spec))
            captured=capture_state(env)
            self.raw=tensor_tree_copy(dict(start=self.start,next_state=encode_physical(captured['physical']),
                final_joint_targets=env._sm_joint_targets,final_normalized=env._sm_final_action,request_before=self.request_before,
                request_after=self.request_after,task_after=captured['task'],recovery_after=captured['recovery'],
                next_feet=captured['physical']['foot_contact'],command_valid_lanes=valid),cpu=True)
            self.raw['step_sequence']=tensor_tree_copy(self.step_records,cpu=True)
            if self.wm_recorder is not None:
                self.raw['wm_window'] = self.wm_recorder.payload()
                env.physical_transition_archive = None
            if trace is not None:self.raw['substep_trace']=tensor_tree_copy(trace,cpu=True)
            if self.earlier_traces:self.raw['earlier_substep_traces']=self.earlier_traces
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
