"""Bounded command-sequence MPC using measured state and frictional contacts.

Validated plans may be applied by a native adapter. Failure returns no action.
CPU wall-clock limits do not certify real-time hardware control or safety.
"""
import copy
import time
from collections import Counter
import numpy as np
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.frictional_candidate_pool_v1 import FrictionalCandidatePoolV1
from scripts.frictional_static_reference_v1 import sphere_static_reference
from scripts.sphere_recontact_reference_v1 import propose_recontact
from scripts.search_frictional_recovery_cem_v1 import key


def admit_commands(commands, previous, interface):
    commands = np.asarray(commands,float)
    if commands.ndim != 2 or commands.shape[1] != 18 or len(commands) not in (40,80) or not np.isfinite(commands).all():
        raise ValueError('Finite complete recovery commands required')
    result = np.clip(commands,interface.limits[:,0],interface.limits[:,1]).copy()
    prior = np.asarray(previous,float).copy()
    if prior.shape != (18,) or not np.isfinite(prior).all():
        raise ValueError('Finite actual previous command required')
    for row in result:
        row[12:] = np.clip(row[12:],prior[12:]-.01*interface.arm_rate,prior[12:]+.01*interface.arm_rate)
        prior = row.copy()
    return result


class FrictionalRecedingRecoveryV1:
    def __init__(self, configuration, spheres, *, horizon=80, population=32, iterations=4,
                 time_limit_s=240., seed=882100, workers=4):
        if horizon not in (40,80) or not 8 <= population <= 64 or not 1 <= iterations <= 12 or not 0 < time_limit_s <= 600:
            raise ValueError('Bounded MPC settings required')
        self.configuration = copy.deepcopy(configuration)
        self.controller = build_controller(configuration)
        self.spheres = copy.deepcopy(spheres)
        self.horizon = horizon; self.population = population; self.iterations = iterations
        self.time_limit_s = time_limit_s; self.workers = workers
        self.rng = np.random.default_rng(seed)
        self.seed = seed; self.pool = None

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool = None

    def solve(self, observed_state, student_reference, warm_commands):
        started = time.monotonic()
        state = copy.deepcopy(observed_state)
        reference = np.asarray(student_reference,float)
        if reference.shape != (18,) or not np.isfinite(reference).all():
            raise ValueError('Student reference must be an absolute joint target')
        H = self.horizon; interface = self.controller.qp.interface
        warm = admit_commands(warm_commands,state.previous_command_target,interface)
        if len(warm) != H:
            raise ValueError('Warm horizon differs')
        if self.pool is None:
            self.pool = FrictionalCandidatePoolV1(self.configuration,self.spheres,
                workers=self.workers,timeout_s=min(180.,self.time_limit_s))
        zero = np.zeros((H,18))
        def evaluate(seed, corrections, save=False, lane_offset=0):
            remaining = self.time_limit_s-(time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError('Recovery solve budget exhausted')
            self.pool.timeout_s = min(180.,remaining)
            requests = [dict(lane=i+lane_offset,state=state,seed=seed,correction=x,save_states=save)
                for i,x in enumerate(corrections)]
            result = self.pool.batch(requests)
            for row in result:
                if (not np.array_equal(row['input_q'],state.q) or not np.array_equal(row['input_v'],state.v)
                        or not np.array_equal(row['input_previous_command'],state.previous_command_target)):
                    raise ValueError('Worker used a different observation or command history')
            return result
        def compact(row):
            return {k:v for k,v in row.items() if k not in ['states','commands','input_q','input_v','input_previous_command']}
        records = []; best = None; chosen_seed = warm; best_dense = zero
        try:
            seeds = [('shifted_verified_plan',warm),('student_hold',
                admit_commands(np.repeat(reference[None],H,axis=0),state.previous_command_target,interface))]
            gravity = sphere_static_reference(self.controller,state,self.spheres)
            if gravity['solved']:
                seeds.append(('surface_equilibrium',admit_commands(np.repeat(np.asarray(gravity['target'])[None],H,axis=0),state.previous_command_target,interface)))
            foot = propose_recontact(self.controller,state,self.spheres)
            for item in foot['seeds']:
                proposal = item['commands']
                if H > len(proposal):proposal = np.vstack([proposal,np.repeat(proposal[-1][None],H-len(proposal),axis=0)])
                seeds.append((item['name'],admit_commands(proposal,state.previous_command_target,interface)))
            for label,seed in seeds:
                candidate = evaluate(seed,[zero])[0]
                records.append(dict(seed=label,result=compact(candidate)))
                if best is None or key(candidate) < key(best):
                    best = candidate; chosen_seed = seed; best_dense = zero.copy()
                if best['accepted']:
                    break
            mean = np.zeros((4,18))
            std = np.tile(np.r_[np.full(12,.02),np.full(6,.01)],(4,1))
            times = np.array([0.,5.,min(25.,H-2.),float(H-1)])
            expand = lambda k: np.column_stack([np.interp(np.arange(H),times,k[:,j]) for j in range(18)])
            best_knots = mean.copy()
            for iteration in range(self.iterations if not best['accepted'] else 0):
                knots = mean+self.rng.normal(size=(self.population,4,18))*std;knots[0] = best_knots
                dense = [expand(k) for k in knots]
                values = evaluate(chosen_seed,dense)
                order = sorted(range(len(values)),key=lambda i:key(values[i]))
                elite = knots[order[:max(4,self.population//6)]]
                mean = .2*mean+.8*elite.mean(0);std = np.maximum(.0005,.3*std+.7*elite.std(0))
                if key(values[order[0]]) < key(best):
                    best = values[order[0]];best_knots = knots[order[0]].copy();best_dense = dense[order[0]].copy()
                failures = Counter(row.get('error','constraint_rejection') for row in values if not row['accepted'])
                records.append(dict(iteration=iteration,best=compact(best),failure_reasons=dict(failures)))
                if best['accepted']:
                    break
            if best['accepted']:
                checked = evaluate(chosen_seed,[best_dense],save=True,lane_offset=1)[0]
                if not checked['accepted'] or key(checked) != key(best):
                    raise ValueError('Separate worker verification differs')
                commands = np.asarray(checked['commands'])
                if not np.allclose(admit_commands(commands,state.previous_command_target,interface),commands,atol=1e-14,rtol=0):
                    raise ValueError('Verified plan violates actual command history')
                return dict(accepted=True,commands=commands,records=records,verification=compact(checked),
                    predicted_states=checked['states'],initial_q=state.q.copy(),initial_v=state.v.copy(),
                    previous_command=state.previous_command_target.copy(),used_observed_state=True,
                    horizon_steps=H,wall_seconds=time.monotonic()-started,native_execution=False,
                    valid_expert_labels=False,per_call_hard_deadline=False,worker_response_deadlines=True,
                    external_process_guard_required=True,teacher_model='frictional_spheres_v1')
        except (TimeoutError,EOFError,ConnectionError,RuntimeError) as error:
            self.close()
            records.append(dict(worker_failure=repr(error)))
        return dict(accepted=False,commands=None,records=records,used_observed_state=True,
            initial_q=state.q.copy(),initial_v=state.v.copy(),previous_command=state.previous_command_target.copy(),
            reason='No independently verified frictional sequence from current observation',
            horizon_steps=H,wall_seconds=time.monotonic()-started,native_execution=False,
            valid_expert_labels=False,per_call_hard_deadline=False,worker_response_deadlines=True,
            external_process_guard_required=True,teacher_model='frictional_spheres_v1')
