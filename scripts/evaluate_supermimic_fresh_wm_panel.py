#!/usr/bin/env python3
"""Compare two frozen WMs on predeclared unseen launches under recorded native commands."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical, valid_windows
from quadsmash.rl.supermimic_physical_observation_v1 import capture_native_inputs


def digest(path):
    with Path(path).open('rb') as stream:
        hasher = hashlib.sha256()
        for block in iter(lambda: stream.read(8 << 20), b''):
            hasher.update(block)
        return hasher.hexdigest()


def check_panel(panel, protocol_path, root, checkpoints):
    protocol = json.loads(protocol_path.read_text())
    if panel['protocol_sha256'] != digest(protocol_path) or panel['policy_training']:
        raise ValueError('Holdout protocol binding failed')
    suites = [protocol['training_suite'], *protocol['formal_suites'].values()]
    reserved = set()
    keys = lambda rows: [tuple(r['launch']['position_m'] + r['launch']['velocity_mps']) for r in rows]
    for suite in suites:
        path = root / suite['path']
        if digest(path) != suite['sha256']:
            raise ValueError('Reserved suite hash mismatch')
        reserved.update(keys(json.loads(path.read_text())['cases']))
    launches = keys(panel['cases'])
    if (set(launches) & reserved or len(set(launches)) != len(launches)
            or len(set(c['case_id'] for c in panel['cases'])) != len(launches)):
        raise ValueError('Repeated or previously reserved launch')
    for directory, artifacts in panel.get('online_world_comparison',{}).items():
        names={str(Path(directory)/name) for name in ('fit.json','online_world.pt')}
        if set(artifacts)!=names or any(digest(path)!=value for path,value in artifacts.items()):
            raise ValueError('Prospective online WM artifacts changed')
    prospective = panel.get('prospective_body_comparison')
    if prospective is not None:
        if not prospective['bodies']:
            raise ValueError('No declared body checkpoint')
        for path, expected in prospective['bodies'].items():
            if digest(path) != expected:
                raise ValueError('Prospective Body WM checkpoint changed')
        for path, expected in prospective.get('ball_candidates',{}).items():
            if digest(path) != expected:
                raise ValueError('Prospective ball checkpoint changed')
        if digest(prospective['calibration_path']) != prospective['calibration_sha256']:
            raise ValueError('Prospective contact calibration changed')
        for path, expected in prospective['previous_panels'].items():
            previous = json.loads(Path(path).read_text())
            if digest(path) != expected or set(launches).intersection(keys(previous['cases'])):
                raise ValueError('Overlapping or changed previous native panel')
    if panel['weights'] != {str(p.resolve()): digest(p) for p in checkpoints}:
        raise ValueError('Frozen panel was declared for different checkpoints')


class TransitionRecorder:
    """Observe the native pre-reset reward hook. Store only compact WM inputs."""
    def __init__(self):
        self.rows = []
        self.pending = None

    def begin(self, env, request, public):
        if self.pending is not None:
            raise RuntimeError('Unfinished native transition')
        before = capture_native_inputs(env)
        self.pending = dict(state=encode_physical(vars(before)), episode=env.episode_index.clone(),
                            previous_action=before.final_action, feet=before.foot_contact,
                            elapsed=before.elapsed_s, present=before.ball_present)
        self.tick = int(env.common_step_counter)

    def finish(self, env, reward):
        if self.pending is None or self.tick + 1 != int(env.common_step_counter):
            raise RuntimeError('Nonconsecutive native transition')
        if not torch.equal(self.pending['episode'], env.episode_index):
            raise RuntimeError('Native reward hook crossed a reset')
        after = capture_native_inputs(env)
        self.pending.update(next_state=encode_physical(vars(after)), targets=env._sm_joint_targets,
                            next_feet=after.foot_contact, next_present=after.ball_present,
                            terminated=env.reset_terminated, timeout=env.reset_time_outs,
                            contact=env._sm_reward_components['qualified_contact'] > 0)
        self.rows.append({k: v.detach().to(device='cpu', copy=True) for k,v in self.pending.items()})
        self.pending = None

    def pack(self):
        data = {k:torch.stack([r[k] for r in self.rows]) for k in self.rows[0]}
        first = data['episode']==data['episode'][0]
        if not ((data['terminated']|data['timeout']) & first).any(0).all():
            raise RuntimeError('Incomplete first native episodes')
        return data


@torch.no_grad()
def assess(data, cases, anchor, adapter, horizons, device, *, contact_step=None, predictors=('anchor','candidate'), flight_attribution=False):
    from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
    from quadsmash.rl.supermimic_toi_racket_objective import racket_features

    horizon = max(horizons)
    valid = valid_windows(data['episode'], data['terminated'], data['timeout'], horizon)
    valid &= data['episode'][:len(valid)] == data['episode'][0]
    # Same time quantiles per case plus each first qualified strike. Stratify
    # reporting so extra event starts do not change the uniform-panel metrics.
    pairs, groups = [], []
    for lane in range(len(cases)):
        starts = torch.where(valid[:,lane])[0]
        if not len(starts):
            raise ValueError(f'Case {lane} has no complete {horizon}-step window')
        for t in starts[torch.linspace(0,len(starts)-1,10).long()]:
            pairs.append((int(t),lane)); groups.append('uniform')
        hits = torch.where(data['contact'][:,lane] & (data['episode'][:,lane]==data['episode'][0,lane]))[0]
        if len(hits):
            for offset in (-1,0,1):
                t = int(hits[0]) + offset
                if 0<=t<len(valid) and valid[t,lane]:
                    pairs.append((t,lane));groups.append(f'contact{offset:+d}')
    ticks,lanes = torch.tensor(pairs).T
    launch=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in cases],
                        dtype=torch.float32,device=device)[lanes]
    buffer={k:v[ticks[None]+torch.arange(horizon)[:,None],lanes].to(device)
            for k,v in data.items() if k in ('state','next_state','targets','previous_action','feet','present','elapsed')}
    metrics={}
    attribution={}
    window_errors={}
    if flight_attribution:
        contact=data['contact'].bool()
        same_episode=data['episode']==data['episode'][0]
        earlier=(contact&same_episode).long().cumsum(0)-(contact&same_episode).long()
        indices=ticks[None]+torch.arange(horizon)[:,None]
        hit=contact[indices,lanes].any(0)
        launched=data['present'][ticks,lanes].bool()
        ends_present=data['next_present'][indices[-1],lanes].bool()
        seen=earlier[ticks,lanes]>0
        strata=[('qualified_contact_in_window' if h else 'previously_qualified' if s else
                 'launch_in_window' if not b and e else 'before_launch' if not b else 'present_without_qualified_contact')
                for h,s,b,e in zip(hit.tolist(),seen.tolist(),launched.tolist(),ends_present.tolist(),strict=True)]
    if not predictors or set(predictors)-{'anchor','candidate'}:
        raise ValueError('Unknown WM predictor')
    for name in predictors:
        s=buffer['state'][0];feet=buffer['feet'][0].float();present=buffer['present'][0]
        metrics[name]={}
        patch = torch.zeros_like(present, dtype=torch.bool)
        for i in range(horizon):
            inputs=(s,buffer['targets'][i],buffer['previous_action'][i],feet,present,buffer['elapsed'][i],launch)
            if name=='anchor':
                s,logits,present=anchor(*inputs);feet=logits.sigmoid()
            elif contact_step is not None:
                s,feet,present,patch=contact_step(inputs,buffer['next_state'][i],patch)
            else:
                s,feet,present=ball_transition(adapter.physical_model,adapter.ball_model,*inputs,action_model=adapter.action_ball_model)
            if i+1 not in horizons:
                continue
            target=buffer['next_state'][i];err=s-target
            # q and -q encode the same rotation.
            err[:,3:7]=torch.where((s[:,3:7]*target[:,3:7]).sum(-1,keepdim=True)<0,-s[:,3:7]-target[:,3:7],err[:,3:7])
            fk=racket_features(s,adapter.physical_model.geometry)-racket_features(target,adapter.physical_model.geometry)
            result={}
            for group in sorted(set(groups)):
                ix=torch.tensor([g==group for g in groups],device=device)
                result[group]=dict(windows=int(ix.sum()),body_rmse=float(err[ix,:49].square().mean().sqrt()),
                    joint_position_mae=float(err[ix,13:31].abs().mean()),root_angular_mae=float(err[ix,10:13].abs().mean()),
                    ball_velocity_mae=float(err[ix,52:].abs().mean()),
                    ball_velocity_rmse=float(err[ix,52:].square().mean().sqrt()),
                    ball_forward_sign_error=float(((s[ix,52]>0)!=(target[ix,52]>0)).float().mean()),
                    ball_position_p95=float(err[ix,49:52].norm(dim=-1).quantile(.95)),
                    racket_position_mae=float(fk[ix,:3].abs().mean()),racket_velocity_mae=float(fk[ix,3:6].abs().mean()),
                    racket_angular_mae=float(fk[ix,9:12].abs().mean()))
            if flight_attribution and i+1==horizon:
                energy=err[:,52:].square().sum(-1).detach().cpu()
                window_errors[name]=[dict(tick=int(ticks[j]),lane=int(lanes[j]),history=strata[j],squared_error=float(energy[j]))
                                     for j in range(len(groups)) if groups[j]=='uniform']
                by_group={}
                for j,kind in enumerate(strata):
                    if groups[j]!='uniform':continue
                    row=by_group.setdefault(kind,dict(windows=0,squared_error_sum=0.))
                    row['windows']+=1
                    row['squared_error_sum']+=float(energy[j])
                total=sum(x['squared_error_sum'] for x in by_group.values())
                for row in by_group.values():
                    row['share_of_squared_error']=row['squared_error_sum']/total if total>0 else None
                    row['component_rmse']=(row['squared_error_sum']/(3*row['windows']))**.5
                attribution[name]=by_group
            metrics[name][i+1]=result
    report=dict(metrics=metrics,windows=[dict(tick=t,lane=l,group=g) for (t,l),g in zip(pairs,groups)])
    if flight_attribution:report.update(uniform_horizon_attribution=attribution,uniform_window_errors=window_errors)
    return report


def run(args):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from quadsmash.rl.supermimic_physical_world_v1 import PhysicalWorldV1
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop,tensor_digest
    from quadsmash.rl.supermimic_policy_execution_v1 import contract,policy_forward
    from quadsmash.rl.supermimic_run_lifecycle import close_resource,stage
    from quadsmash.rl.supermimic_task_contract import make_task_cfg, require_resume_contract, resolve_task_contract, zero_panel_reset_noise
    from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv
    from quadsmash.tasks.stand.supermimic_env_v1 import SuperMimicEnvV1

    class PanelEnv(SuperMimicEnvV1, FrozenPanelEnv):
        pass

    panel=json.loads(args.panel.read_text())
    root=Path(__file__).resolve().parents[1]
    check_panel(panel,root/'configs/g07_full_racket_eval_v4.json',root,[args.anchor,args.candidate])
    ck=torch.load(args.candidate,map_location='cpu',weights_only=False,mmap=True)
    meta=ck['metadata']; identity=meta['physical_imagination']; paths=identity['checkpoint_paths']
    actor,adapter,_,_,m=load_physical_loop(archive=args.archive,physical_checkpoint=paths['physical'],
        ball_checkpoint=paths['ball'],task_checkpoint=paths['task'],action_ball_checkpoint=paths.get('action_ball'),
        execution_root=identity['g07_execution_root'],device=args.device,safety_mapping=args.safety_mapping)
    if identity['checkpoint_sha256']!=adapter.manifest['checkpoint_sha256'] or meta['policy_execution']!=contract():
        raise ValueError('Frozen policy or WM component binding mismatch')
    actor.load_state_dict(ck['actor'],strict=True);adapter.load_state_dict(ck['world'],strict=True)
    actor.eval();adapter.eval()
    anchor=PhysicalWorldV1().to(args.device).eval()
    ac=torch.load(args.anchor,map_location='cpu',weights_only=False)
    if ac['schema']!=anchor.schema:raise ValueError('Anchor schema mismatch')
    anchor.load_state_dict(ac['model'],strict=True)
    source_path=Path(paths['source']);source=torch.load(source_path,map_location='cpu',weights_only=False)
    exec_root=Path(identity['g07_execution_root'])
    task,protocol,_,receipt=resolve_task_contract(source,source_path,execution_root=exec_root,root=exec_root)
    require_resume_contract(ck,task)
    if task!=m['task_contract']:raise ValueError('Base native task contract mismatch')
    cases=panel['cases'];seed=panel['seed']
    cfg=make_task_cfg(len(cases),task,protocol,receipt);cfg.seed=seed;cfg.sim.device=args.device
    env=None;failed=True
    try:
        env=PanelEnv(cfg,cases=cases,reset_noise=zero_panel_reset_noise(len(cases),seed),
            residual_policy_checkpoint=source,g07_execution_root=exec_root,contact_flight_recovery=task['contact_flight_recovery'])
        wrapper=RslRlVecEnvWrapper(env,clip_actions=None)
        obs,_=wrapper.reset()
        rec=TransitionRecorder();env.physical_transition_archive=rec
        stage(args.output,'FRESH_PANEL_RESET',num_envs=len(cases),seed=seed)
        with torch.no_grad():
            for tick in range(800):
                action=policy_forward(actor,obs['policy'],env.rally,sample=False)[0]
                obs,_,_,_=wrapper.step(action)
                if (tick+1)%64==0:stage(args.output,'CAPTURE_PROGRESS',ticks=tick+1)
        data=rec.pack()
        env.physical_transition_archive=None
        path=args.output/'native.pt'
        torch.save(data,path.with_suffix('.partial'));path.with_suffix('.partial').replace(path)
        report=assess(data,cases,anchor,adapter,panel['horizons'],args.device)
        report.update(panel_sha256=digest(args.panel),data_sha256=digest(path),weights=panel['weights'],
            actor_sha256=tensor_digest(actor.state_dict().items()),seed=seed,num_envs=len(cases),ticks=len(rec.rows),
            first_episodes=len(cases),actor_updates=0,wm_updates=0,checkpoint_promoted=False)
        result=args.output/'metrics.json'
        result.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        failed=False
    finally:
        close_resource(args.output,'ENV',env,primary_failed=failed)


def main():
    from isaaclab.app import AppLauncher
    from quadsmash.rl.supermimic_run_lifecycle import execute_scoped_native
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('panel','anchor','candidate','archive','safety-mapping','output'):
        p.add_argument('--'+arg,type=Path,required=True)
    AppLauncher.add_app_launcher_args(p)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    app=AppLauncher(args,fast_shutdown=False).app
    execute_scoped_native(args.output,lambda:run(args),app)


if __name__=='__main__':
    from quadsmash.rl.supermimic_run_lifecycle import native_process_main
    native_process_main(main)
