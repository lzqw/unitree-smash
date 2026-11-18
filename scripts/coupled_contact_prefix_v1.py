"""Bounded offline joint leg/arm position-and-rate prefix refinement.

Arm correction is a cubic endpoint basis: zero position/rate at the current
state, independently variable terminal position and velocity. Parameters are
in radians; the last six yield terminal rate offsets parameter / horizon_s.
All robot motion comes from unchanged PD and fullbody contact dynamics.
"""
import argparse,copy,gzip,hashlib,inspect,json,sys,time
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.terminal_push_refinement_v1 import TerminalPushRollout,refine_terminal
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1,PublicState
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.experimental_sphere_admission_v1 import ExperimentalSphereAdmissionV1


def arm_basis(s):
    s=np.asarray(s,float)
    if not np.isfinite(s).all() or np.any(s<0) or np.any(s>1):raise ValueError('Explicit normalized event time required')
    return 3*s*s-2*s*s*s,s*s*s-s*s


class CoupledContactRollout(TerminalPushRollout):
    radii=np.r_[np.full(12,.1),np.full(6,.1),np.full(6,.05)]
    rate_window_ticks=None

    def basis_at_nodes(self,nodes):
        nodes=np.asarray(nodes,float);count=len(self.commands)
        f,g=arm_basis((nodes+1)/count)
        if self.rate_window_ticks is not None:
            window=self.rate_window_ticks
            if not isinstance(window,int) or not 1<=window<=count:raise ValueError('Explicit valid local rate window required')
            _,g=arm_basis(np.clip((nodes+1-(count-window))/window,0.,1.))
        return f,g

    def validate_amplitude(self,amplitude):
        if amplitude.shape!=(24,) or not np.isfinite(amplitude).all() or np.any(abs(amplitude)>self.radii+1e-10):
            raise ValueError('Bounded 24-parameter prefix required')

    def command_at_node(self,amplitude,reference,node,weight):
        f,g=self.basis_at_nodes(node)
        delta=f*amplitude[12:18]+g*amplitude[18:24]
        command=reference.copy();command[:12]+=weight*amplitude[:12];command[12:]+=delta
        arm_reference=self.arm_reference[node+1]+delta
        if np.any(arm_reference<self.interface.limits[12:,0]-1e-6) or np.any(arm_reference>self.interface.limits[12:,1]+1e-6):
            raise ValueError('Original arm reference position limit at node '+str(node))
        return command,arm_reference

    def amplitude_bounds(self):
        bounds=[*super().amplitude_bounds()]
        f,g=self.basis_at_nodes(np.arange(len(self.commands)))
        for basis in (f,g):
            mask=abs(basis)>1e-12
            for j in range(6):
                lo,hi=self.interface.limits[12+j];low=-np.inf;high=np.inf
                for sequence in (self.commands[:,12+j],self.arm_reference[1:,j]):
                    a=(lo-sequence[mask])/basis[mask];b=(hi-sequence[mask])/basis[mask]
                    low=max(low,float(np.max(np.minimum(a,b))));high=min(high,float(np.min(np.maximum(a,b))))
                bounds.append([min(low,0.),max(high,0.)])
        return np.asarray(bounds)

    def combined_arm_constraints(self):
        rows=[];low=[];high=[]
        f,g=self.basis_at_nodes(np.arange(len(self.commands)))
        for sequence in (self.commands[:,12:],self.arm_reference[1:]):
            for node in range(len(sequence)):
                for j in range(6):
                    row=np.zeros(24);row[12+j]=f[node];row[18+j]=g[node]
                    rows.append(row);low.append(self.interface.limits[12+j,0]-sequence[node,j])
                    high.append(self.interface.limits[12+j,1]-sequence[node,j])
        return np.array(rows),np.array(low),np.array(high)


class LateRateContactRollout(CoupledContactRollout):
    """Same amplitude budget, C1 rate correction in the last 80ms only.

    No new samples/parameters. Actual speed/torque remain prediction gates;
    the endpoint reference derivative is not claimed realized velocity.
    """
    rate_window_ticks=8


def evaluation_signature(evaluator=None):
    """Exact-response cache excludes optimizer code, not rollout semantics."""
    root=Path(__file__).resolve().parents[1]
    methods=(arm_basis,CoupledContactRollout.basis_at_nodes,CoupledContactRollout.command_at_node,CoupledContactRollout.validate_amplitude,
        TerminalPushRollout.__init__,TerminalPushRollout.__call__,selected_arm_reference,main)
    code='\n'.join(inspect.getsource(m) for m in methods)
    files=[p for directory in ('scripts','quadsmash','configs') for p in (root/directory).rglob('*')
        if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts
        and p.name not in ('coupled_contact_prefix_v1.py','terminal_push_refinement_v1.py')]
    files.extend(root/p for p in ('reports/codex_teacher_joint_eval_20261116/controller.json',
        'reports/prediction_model_validation_20261118/AUTHORED_SPHERES.json'))
    signature=dict(rollout_method_sha256=hashlib.sha256(code.encode()).hexdigest(),
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
        rate_window_ticks=getattr(evaluator,'rate_window_ticks',None),radii=CoupledContactRollout.radii.tolist(),python=sys.executable,numpy=np.__version__,pinocchio=pin.__version__)
    if evaluator is not None:
        context=dict(state=evaluator.initial.__dict__,commands=evaluator.commands,
            reference=evaluator.arm_reference,metadata=evaluator.metadata,model=evaluator.model.sha256,
            interface={name:getattr(evaluator.interface,name) for name in ('limits','kp','kd','effort','velocity','friction')},
            predictor_iterations=evaluator.predictor.iterations)
        encoded=json.dumps(context,sort_keys=True,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))
        signature['derived_context_sha256']=hashlib.sha256(encoded.encode()).hexdigest()
    return signature


def selected_arm_reference(record,source,arm_q_indices):
    """Recover reference history, never replace it with measured/nominal q.

    Legacy migration is restricted to a single original-reference iteration
    and checked against its recorded tracking errors. New results carry the
    complete reference explicitly and support repeated re-linearization.
    """
    selected=record['trials'][record['selected_index']]['result']
    trace=selected['trajectory'];n=len(trace)
    original=np.asarray(source['arm_sequence_rad'],float)[:n+1].copy()
    if not selected.get('path_valid') or n!=source['metadata']['contact_index']:
        raise ValueError('Complete verified prefix required for reference continuation')
    if 'selected_arm_reference_rad' in record:
        reference=np.asarray(record['selected_arm_reference_rad'],float).copy();mode='explicit'
    elif all('planned_arm_reference_rad' in s for s in trace):
        reference=np.vstack([original[0],[s['planned_arm_reference_rad'] for s in trace]]);mode='per_node'
    else:
        baseline=record['trials'][0]['result']['trajectory']
        old_commands=np.asarray([s['final_target_rad'] for s in baseline],float)
        original_commands=np.asarray(source['result']['replayed_target_sequence_rad'],float)
        if old_commands.shape!=(n,18) or original_commands.shape!=(n,18) or np.max(abs(old_commands[:,12:]-original_commands[:,12:]))>2e-6:
            raise ValueError('Missing non-original arm reference history')
        command=np.asarray([s['final_target_rad'] for s in trace],float)
        reference=original.copy();reference[1:]+=command[:,12:]-old_commands[:,12:]
        mode='legacy_additive_offset_verified'
    if reference.shape!=(n+1,6) or not np.isfinite(reference).all() or not np.array_equal(reference[0],original[0]):
        raise ValueError('Invalid or changed initial arm reference')
    differences=[]
    for k,s in enumerate(trace):
        if 'planned_arm_reference_rad' in s and not np.array_equal(reference[k+1],np.asarray(s['planned_arm_reference_rad'])):
            raise ValueError('Explicit reference disagrees with per-node history')
        error=float(np.max(abs(np.asarray(s['q'])[arm_q_indices]-reference[k+1])))
        recorded=float(s['arm_tracking_error_rad'])
        if not np.isfinite(recorded) or abs(error-recorded)>1e-9:
            raise ValueError('Reference does not reproduce recorded tracking error')
        differences.append(abs(error-recorded))
    return reference,dict(mode=mode,max_tracking_reconstruction_error_rad=max(differences),steps=n)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reuse-responses',type=Path);p.add_argument('--previous',type=Path,default=Path('reports/terminal_push_20261118/RESULTS.json'));p.add_argument('--backtracking',action='store_true');p.add_argument('--protect-terminal-tasks',action='store_true');a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve attempts')
    root=Path(__file__).resolve().parents[1]
    initial_path=root/'reports/push_off_task_20261118/DIAGNOSED.json'
    previous_path=(root/a.previous).resolve()
    original_path=root/'reports/experimental_fullbody_gate_20261118/REPLAY.json.gz'
    initial=json.loads(initial_path.read_text())['rows'][0]
    old=json.loads(previous_path.read_text());previous=old['trials'][old['selected_index']]['result']
    if previous_path.with_suffix('.sha256').exists() and hashlib.sha256(previous_path.read_bytes()).hexdigest()!=previous_path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Previous prefix integrity check failed')
    if not previous['path_valid']:raise ValueError('Complete verified model prefix required')
    calls=json.load(gzip.open(original_path,'rt'))['physical_calls'];matching=[c for c in calls if c['lane']==initial['lane']]
    if len(matching)!=1:raise ValueError('Unique diagnostic event required')
    source=matching[0];model=LoadedPinocchioModelV1();q=np.array(initial['initial_q']);v=np.array(initial['initial_v'])
    for path in (initial_path,original_path):
        if old['input_sha256'].get(str(path.relative_to(root)))!=hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Previous prefix initial-state or contact-source drift')
    reference,reference_audit=selected_arm_reference(old,source,model.q_indices[12:])
    state=PublicState(q.copy(),v.copy(),pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),q[model.q_indices].copy(),v[model.v_indices].copy(),np.array(initial['baseline_initial_contacts'],bool),0.)
    state.previous_command_target=np.array(initial['previous_final_command'])
    interface=PositionPDInterface(**json.loads((root/'reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface'])
    geometry=AuthoredGroundClearance(root/'configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.)
    if geometry.minimum_witness(state.q)['gap_m'] < -1e-6:raise ValueError('Initial nonfoot geometry violation')
    predictor=ExperimentalSphereAdmissionV1(model,geometry,root/'reports/prediction_model_validation_20261118/AUTHORED_SPHERES.json')
    evaluator=CoupledContactRollout(model,interface,predictor,state,[s['final_target_rad'] for s in previous['trajectory']],reference,source['metadata'])
    signature=evaluation_signature(evaluator)
    inputs={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (initial_path,previous_path,original_path)}
    cache={}
    if a.reuse_responses:
        if hashlib.sha256(a.reuse_responses.read_bytes()).hexdigest()!=a.reuse_responses.with_suffix('.sha256').read_text().strip():
            raise ValueError('Cached result integrity check failed')
        prior=json.loads(a.reuse_responses.read_text())
        signed=json.loads(a.reuse_responses.with_suffix('.evaluation_signature.json').read_text())
        if prior['input_sha256']!=inputs or signed!=signature:raise ValueError('Cached response inputs or evaluator changed')
        cache={np.asarray(t['amplitude_rad'],float).tobytes():t['result'] for t in prior['trials'] if t['label']=='baseline' or t['label'].startswith('derivative_')}
    count=0;fresh=0
    def evaluate(amplitude):
        nonlocal count,fresh
        key=amplitude.tobytes();cached=key in cache
        if cached:result=copy.deepcopy(cache[key])
        else:result=evaluator(amplitude);fresh+=1
        count+=1
        if count==1:
            if not result['path_valid']:raise ValueError('Baseline replay failed')
            for current,oldrow in zip(result['trajectory'],previous['trajectory']):
                if not np.array_equal(current['q'],oldrow['q']) or not np.array_equal(current['v'],oldrow['v']):raise ValueError('Baseline drift')
            result['baseline_exact_replay']=True
        result['response_from_exact_cache']=cached
        print(json.dumps(dict(call=count,cached=cached,path_valid=result['path_valid'],errors=result.get('terminal_errors'),failure=result.get('failure'))),flush=True)
        return result
    start=time.perf_counter()
    result=refine_terminal(evaluate,amplitude_bounds=evaluator.amplitude_bounds(),parameter_radii=evaluator.radii,
        linear_constraints=evaluator.combined_arm_constraints(),backtracking=a.backtracking,protect_terminal_tasks=a.protect_terminal_tasks)
    result.update(seconds=time.perf_counter()-start,parameterization='12 support leg radians, 6 arm endpoint radians, 6 arm rate-basis radians',
        horizon_s=len(evaluator.commands)*.01,input_sha256=inputs,fresh_model_calls=fresh,cached_calls=count-fresh,
        base_arm_reference_rad=reference,reference_continuation_audit=reference_audit)
    chosen=result['trials'][result['selected_index']]['result']
    if chosen.get('path_valid'):
        result['selected_arm_reference_rad']=np.vstack([reference[0],[s['planned_arm_reference_rad'] for s in chosen['trajectory']]])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n')
    a.output.with_suffix('.sha256').write_text(hashlib.sha256(a.output.read_bytes()).hexdigest()+'\n')
    a.output.with_suffix('.evaluation_signature.json').write_text(json.dumps(signature,indent=2)+'\n')
    print(json.dumps(dict(improved=result['improved'],selected=result['selected_index'],errors=result['trials'][result['selected_index']]['result'].get('terminal_errors'),seconds=result['seconds'])))
if __name__=='__main__':main()
