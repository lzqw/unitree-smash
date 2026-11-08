"""Search the remaining recovery from a recorded actual native state.

Keeps the original reference clock and state origin, but initializes dynamics
from the observed q/v/contacts and actual previous command. Full-remainder
shooting is an offline planning experiment; it is not online MPC execution.
No candidate is executable evidence or a positive demonstration without a
ground-start native replay of its exact preceding controller.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context, state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.terminal_attitude_feedback_v1 import TerminalAttitudeFeedbackV1
from scripts.constrained_recovery_suffix_v1 import evaluate_suffix
from scripts.probe_constrained_support_domain_v1 import identity
from scripts.audit_joint_sequence_proposal_v1 import plain


class SuffixCommandCorrectionV1:
    """Bounded correction to the mapped feedback command with explicit handoff.

The baseline already includes command mapping. A final clipping/slew pass
constrains the combined request, and prediction uses exactly that request.
This is not an inverse residual label or a force-command transformation.
"""
    def __init__(self, base, control_tick, previous_command, knots):
        self.base = base; self.interface = base.interface
        self.control_tick = control_tick
        self.previous_at_start = np.asarray(previous_command, float).copy()
        knots = np.asarray(knots, float)
        if (knots.shape != (6,18) or not np.isfinite(knots).all()
                or np.max(abs(knots[:,:12])) > .3+1e-12
                or np.max(abs(knots[:,12:])) > .12+1e-12):
            raise ValueError('Finite bounded command-space search knots required')
        self.curve = PchipInterpolator([0.,.04,.12,.30,.80,1.8,3.5],
            np.vstack([np.zeros(18),knots]),axis=0,extrapolate=False)

    def action(self, offset, state, previous):
        tick = offset-80+280
        if type(tick) is not int or not self.control_tick <= tick < 800:
            raise ValueError('Original control clock required')
        if tick == self.control_tick:
            if not np.array_equal(previous,self.previous_at_start):
                raise ValueError('Initial actual command history changed')
            return previous.copy(), 'suffix_continuous_handoff'
        target,_ = self.base.action(offset,state,previous)
        t = .01*(tick-self.control_tick); u = min(t/.04,1.)
        blend = u*u*(3.-2.*u)
        target = (1.-blend)*self.previous_at_start+blend*target+self.curve(min(t,3.5))
        target = np.clip(target,self.interface.limits[:,0],self.interface.limits[:,1])
        target[12:] = np.clip(target[12:],previous[12:]-.01*self.interface.arm_rate,
            previous[12:]+.01*self.interface.arm_rate)
        return target, 'state_conditioned_suffix_command_candidate'


def candidate(label, corrections, tick, query, original_knots, audit, substeps, deadline):
    w = _WORKER; controller = w['controller']; state = state_from_row(query)
    base = TerminalAttitudeFeedbackV1(w['factory'](controller.qp.model,controller.qp.interface,
        w['initial'],original_knots),proportional=.3,damping_s=.04,maximum_joint_delta_rad=.08)
    policy = SuffixCommandCorrectionV1(base,tick,state.previous_command_target,corrections)
    geometry = HipMountPathConstraintV1(w['geometry'],audit,margin_rad=.02)
    result = evaluate_suffix(controller,state,policy,w['spheres'],geometry,w['protocol'],
        control_tick=tick,substeps=substeps,deadline=deadline,preceding_stable_ticks=0,save=substeps==160)
    if not all(np.array_equal(result[k],query[q]) for k,q in
        [('initial_q','q'),('initial_v','v'),('initial_contacts','contacts'),('initial_previous_command','target')]):
        raise ValueError('Candidate used a different observed state or history')
    return dict(label=label,corrections_rad=corrections,result=result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['source','native-run','suffix-check','output']: p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--tick',type=int,default=350); p.add_argument('--max-seconds',type=float,default=1800.)
    p.add_argument('--wait-pid',type=int); p.add_argument('--wait-start-ticks')
    a = p.parse_args(); sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    read = lambda path: json.loads(Path(path).read_text())
    if a.output.exists() or not 320 <= a.tick <= 600 or not 900 <= a.max_seconds <= 2400:
        raise ValueError('Fresh output and finite search over the original recovery interval required')
    own = sha(__file__)
    if bool(a.wait_pid) != bool(a.wait_start_ticks): raise ValueError('Explicit wait identity required')
    if a.wait_pid:
        until = time.monotonic()+1200
        print(json.dumps(dict(state='WAITING_FOR_SUFFIX_CHECK',pid=a.wait_pid)),flush=True)
        while True:
            current = identity(a.wait_pid)
            if current is None: break
            if current != a.wait_start_ticks: raise ValueError('Waiting process identity changed')
            if time.monotonic() >= until: raise TimeoutError('Suffix check wait limit')
            time.sleep(5)
    if sha(__file__) != own: raise ValueError('Waiting solver changed')
    check = read(a.suffix_check)
    if not check['all_checks_passed']: raise ValueError('Executed suffix checks required')
    for path,h in check['source_sha256'].items():
        if sha(path) != h: raise ValueError('Changed checked dependency '+path)
    source,original_knots,_,audit,initargs = source_context(a.source)
    job = read(a.native_run/'job.json'); meta = read(a.native_run/'sequence_native_result.json')
    if job['status'] != 'SUCCEEDED' or meta['case'] != 3 or not meta['query_match']['passed']:
        raise ValueError('Completed, query-matched ground-start native record required')
    ticks = np.asarray([r['control_tick'] for r in meta['records']]); matches = np.flatnonzero(ticks==a.tick)
    if len(matches)!=1: raise ValueError('Unique current-time record required')
    i = int(matches[0])
    with np.load(a.native_run/'sequence_native_trace.npz',allow_pickle=False) as z:
        if i == 0 or ticks[i-1] != a.tick-1 or not np.array_equal(z['previous_target_rad'][i],z['final_target_rad'][i-1]):
            raise ValueError('Recorded actual command history is discontinuous')
        query = dict(q=z['q'][i].copy(),v=z['v'][i].copy(),contacts=z['contacts'][i].copy(),
            target=z['previous_target_rad'][i].copy(),measured_foot_force_world_n=z['foot_force_world_n'][i].copy())
    files = [a.source.resolve(),a.suffix_check.resolve(),Path(__file__).resolve(),
        a.native_run/'job.json',a.native_run/'sequence_native_result.json',
        a.native_run/'sequence_native_trace.npz',a.native_run/'INPUTS.json']
    hashes = dict(check['source_sha256']); hashes.update({str(path.resolve()):sha(path) for path in files})
    for path,h in hashes.items():
        if sha(path)!=h: raise ValueError('Changed input '+path)
    rng = np.random.default_rng(882121); mean = np.zeros((6,18))
    std = np.tile(np.r_[np.full(12,.045),np.full(6,.012)],(6,1))
    floor = np.tile(np.r_[np.full(12,.008),np.full(6,.003)],(6,1))
    bound = np.r_[np.full(12,.3),np.full(6,.12)]
    began = time.monotonic(); deadline = began+a.max_seconds; search_end = began+.85*a.max_seconds
    coarse=[]; fine=[]; best=None; replay=None; seen=set(); generations=[]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def progress():
        data = dict(best=best,coarse_evaluations=len(coarse),fine_evaluations=len(fine),generations=generations)
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(plain(data),allow_nan=False)+'\n')
        tmp.replace(a.output.with_suffix('.partial.json'))
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=initargs) as pool:
        def submit(label,delta,resolution,limit):
            return pool.submit(candidate,label,delta,a.tick,query,original_knots,audit,resolution,limit)
        best=submit('zero_correction_handoff',mean,160,deadline).result();fine.append(best);progress()
        for generation in range(5):
            if best['result']['accepted'] or time.monotonic() >= search_end: break
            proposals=[]
            for j in range(32):
                center=mean if j<16 else np.asarray(best['corrections_rad'])
                proposals.append(np.clip(center+std*(.75*rng.normal(size=(1,18))+
                    .5*rng.normal(size=(6,18))),-bound,bound))
            proposals[0]=np.asarray(best['corrections_rad']);proposals[1]=mean.copy()
            futures=[submit(f'g{generation}_coarse_{j}',delta,32,search_end) for j,delta in enumerate(proposals)]
            rows=[f.result() for f in as_completed(futures)];coarse.extend(rows)
            rows=[r for r in rows if (r['result']['failure'] or {}).get('reason')!='declared_compute_budget']
            rows.sort(key=lambda r:r['result']['rank'])
            if not rows: break
            elite=np.stack([r['corrections_rad'] for r in rows[:8]])
            mean=.3*mean+.7*elite.mean(0);std=np.maximum(floor,.3*std+.7*elite.std(0))
            chosen=[]
            for row in rows:
                key=np.asarray(row['corrections_rad']).tobytes()
                if key in seen: continue
                seen.add(key);chosen.append(row)
                if len(chosen)==4: break
            futures=[submit(r['label'].replace('coarse','fine'),r['corrections_rad'],160,search_end) for r in chosen]
            for future in as_completed(futures):
                row=future.result();fine.append(row)
                if ((row['result']['failure'] or {}).get('reason') != 'declared_compute_budget'
                        and row['result']['rank']<best['result']['rank']): best=row
                print(json.dumps(plain(dict(label=row['label'],accepted=row['result']['accepted'],
                    steps=row['result']['completed_model_steps'],failure=row['result']['failure']))),flush=True)
            generations.append(dict(generation=generation,best=best['label']));progress()
        if best['result']['accepted'] and time.monotonic()<deadline:
            replay=submit('independent_full_suffix',best['corrections_rad'],160,deadline).result()
    verified=False
    if replay and replay['result']['accepted']:
        x,y=best['result']['trajectory'],replay['result']['trajectory']
        verified=len(x)==len(y)==800-a.tick and all(np.array_equal(a[k],b[k])
            for a,b in zip(x,y) for k in ['q','v','target','contacts'])
        if not verified: raise ValueError('Full suffix independent replay differs')
    for path,h in hashes.items():
        if sha(path)!=h: raise ValueError('Changed source during search '+path)
    result=dict(scope=__doc__,configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        source_sha256=hashes,query=query,original_knots_rad=original_knots,coarse_rows=coarse,fine_rows=fine,
        best=best,independent_replay=replay,full_suffix_verified=verified,wall_seconds=time.monotonic()-began,
        expired_coarse=sum((r['result']['failure'] or {}).get('reason')=='declared_compute_budget' for r in coarse),
        expired_fine=sum((r['result']['failure'] or {}).get('reason')=='declared_compute_budget' for r in fine),
        preceding_stable_ticks=0,native_execution=False,online_mpc=False,valid_expert_labels=False,
        limitations=['Recorded prefix has unresolved cross-lane isolation differences; no new success claim.',
            'Only audited hip/mount pairs and authored ground clearance are checked, not full collision response.',
            'Coarse scores guide search; only complete 160-substep replay admits a model candidate.',
            'Future native execution must replay the exact preceding controller and verify this actual query.'])
    a.output.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
