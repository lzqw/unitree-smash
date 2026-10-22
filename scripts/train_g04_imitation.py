#!/usr/bin/env python3
"""Actual ground-rollout behavior cloning with independent closed-loop checks.

Teacher is used only to collect labels. Each reported learner rollout and the
saved native checkpoint execute the learned actor from t=0 without a prefix.
"""
import argparse,csv,hashlib,json,signal,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--teacher',type=Path,required=True)
p.add_argument('--candidate',type=int,default=0)
p.add_argument('--config',type=Path,default=ROOT/'configs/g04_imitation.json')
p.add_argument('--num-envs',type=int,default=256)
p.add_argument('--updates',type=int,default=3000)
p.add_argument('--rounds',type=int,default=3)
p.add_argument('--seed',type=int,default=421001)
p.add_argument('--init-checkpoint',type=Path)
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
app=AppLauncher(args).app
import torch
from tensordict import TensorDict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.jump_env import JumpEnv,JumpEnvCfg
from quadsmash.control.jump_teacher import teacher_action
from quadsmash.rl.native_ppo import NativeRunner
env=None;stop=False
def request_stop(*_):
    global stop
    stop=True
signal.signal(signal.SIGTERM,request_stop)

def summarize(rows):
    return dict(episodes=len(rows),successes=sum(r['success'] for r in rows),
        qualified=sum(r['qualified_jump_landed'] for r in rows),body_contacts=sum(r['body_landing'] for r in rows),
        max_com_rise=max(r['total_com_rise_m'] for r in rows),
        mean_recovery=sum(r['recovery_s'] for r in rows)/len(rows))

try:
    cfg=JumpEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed;cfg.sim.device='cuda:0'
    env=JumpEnv(cfg);wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
    config=json.loads(args.config.read_text())
    runner=NativeRunner(wrapped,config['runner'],log_dir=str(args.output/'ppo'),device=env.device)
    if args.init_checkpoint:runner.load(str(args.init_checkpoint),load_optimizer=False)
    policy=runner.alg.policy
    teacher_doc=json.loads(args.teacher.read_text())
    parameters=torch.tensor(teacher_doc['elite'][args.candidate]['parameters'],device=env.device).repeat(env.num_envs,1)
    (args.output/'teacher.json').write_text(json.dumps(dict(source=str(args.teacher),sha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest(),parameters=parameters[0].tolist()),indent=2))
    env_steps=0

    def rollout(seed,teacher=False,collect=False):
        global env_steps
        g=torch.Generator(device=env.device).manual_seed(seed)
        with torch.inference_mode():
            env._episode_has_stepped.zero_();env._metrics_started=False;wrapped.reset();env.episode_records.clear()
            q=env.robot.data.default_joint_pos.clone();v=torch.zeros_like(q)
            q[:,env.joint_ids]+=(2*torch.rand((env.num_envs,18),generator=g,device=env.device)-1)*.025
            v[:,env.joint_ids]=(2*torch.rand((env.num_envs,18),generator=g,device=env.device)-1)*.04
            root=env.robot.data.root_state_w.clone();angle=(2*torch.rand(env.num_envs,generator=g,device=env.device)-1)*.08
            root[:,3]=torch.cos(angle/2);root[:,4:6]=0;root[:,6]=torch.sin(angle/2);root[:,7:]=0
            root[:,:2]+=(2*torch.rand((env.num_envs,2),generator=g,device=env.device)-1)*.04
            env.robot.write_root_state_to_sim(root);env.robot.write_joint_state_to_sim(q,v);env.sim.forward()
            env.policy_history.reset(torch.arange(env.num_envs,device=env.device));obs=wrapped.get_observations()
            policy.eval();xs=[];ys=[];safe=[];alive=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
            for tick in range(400):
                target=teacher_action(obs['policy'],parameters)
                if collect:
                    xs.append(obs['policy'].clone());ys.append(target.clone())
                    safe.append(alive&(~env.body_ground_seen))
                action=target if teacher else policy.act_inference(obs)
                if not torch.isfinite(action).all():raise RuntimeError('nonfinite actor output')
                obs,_,done,_=wrapped.step(action);alive&=~done.bool();env_steps+=env.num_envs
            first={}
            for r in env.episode_records:first.setdefault(r['env_id'],r)
            rows=[first[i] for i in range(env.num_envs)]
            if collect:
                # Materialize outside inference mode before optimizer operations.
                data=(torch.stack(xs),torch.stack(ys),torch.stack(safe))
            else:data=None
        return rows,data

    rows,data=rollout(args.seed,teacher=True,collect=True)
    report=dict(method='teacher-guided behavior cloning; not pure PPO',teacher=summarize(rows),
                seed=args.seed,learner_checks=[],weights_updated=True,env_steps=env_steps)
    print('TEACHER_COLLECTION='+json.dumps(report['teacher']),flush=True)
    with (args.output/'teacher_episodes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    successful=torch.tensor([r['success'] for r in rows],device=env.device)
    if int(successful.sum())<8:raise RuntimeError('fewer than eight successful independent teacher episodes; improve teacher before BC')
    x,y,safe=(t.clone() for t in data);del data
    # Entire held-out episodes, never adjacent-frame random validation.
    validation=torch.arange(env.num_envs,device=env.device)%5==0
    trainmask=safe&successful[None,:]&(~validation)[None,:]
    valmask=safe&successful[None,:]&validation[None,:]
    train_x=x[trainmask];train_y=y[trainmask];val_x=x[valmask];val_y=y[valmask]
    torch.save(dict(observations=x.cpu(),actions=y.cpu(),safe=safe.cpu(),successful=successful.cpu(),
                    validation_episode=validation.cpu(),seed=args.seed),args.output/'teacher_dataset.pt')
    del x,y,safe
    # Freeze normalization after real training-episode data, exclude validation data.
    if args.init_checkpoint is None:
        policy.train();policy.update_normalization(TensorDict({'policy':train_x},batch_size=[len(train_x)]))
    policy.eval()
    optimizer=torch.optim.Adam(policy.actor.parameters(),lr=3e-4)
    best=-1
    for round_id in range(args.rounds):
        active=torch.where((train_x[:,775]>=.75)&(train_x[:,775]<=1.65))[0]
        losses=[]
        for update in range(args.updates):
            if stop:break
            # 65% preparation/push/flight, 35% complete-episode distribution.
            idx=torch.cat((active[torch.randint(len(active),(666,),device=env.device)],
                           torch.randint(len(train_x),(358,),device=env.device)))
            pred=policy.act_inference(TensorDict({'policy':train_x[idx]},batch_size=[len(idx)]))
            loss=(pred-train_y[idx]).square().mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.);optimizer.step()
            if update%250==0:losses.append(float(loss))
        with torch.no_grad():
            vl=float((policy.act_inference(TensorDict({'policy':val_x},batch_size=[len(val_x)]))-val_y).square().mean()) if len(val_x) else None
        runner.save_restart(args.output/f'bc_{round_id:02d}.pt',infos=dict(method='behavior cloning',round=round_id))
        rows,data=rollout(args.seed+100+round_id,collect=True)
        check=dict(round=round_id,**summarize(rows),validation_mse=vl,training_mse_samples=losses)
        report['learner_checks'].append(check);report['env_steps']=env_steps
        with (args.output/f'learner_episodes_{round_id:02d}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        if check['successes']>best:
            best=check['successes'];runner.save_restart(args.output/'checkpoint.pt',infos=dict(method='teacher-guided BC',round=round_id));report['best_round']=round_id
        (args.output/'metrics.json').write_text(json.dumps(report,indent=2))
        print('BC_PROGRESS='+json.dumps(check),flush=True)
        if stop or check['successes']/env.num_envs>=.97:break
        # DAgger labels are this bounded state-feedback teacher evaluated on
        # actual learner observations, only while the first episode remains safe.
        x,y,safe=(t.clone() for t in data);del data
        mask=safe&(~validation)[None,:]
        train_x=torch.cat((train_x,x[mask]));train_y=torch.cat((train_y,y[mask]))
        torch.save(dict(observations=x.cpu(),teacher_actions=y.cpu(),safe=safe.cpu(),seed=args.seed+100+round_id),args.output/f'dagger_{round_id:02d}.pt')
        del x,y,safe
    report['best_success_rate']=best/env.num_envs;report['checkpoint']=str(args.output/'checkpoint.pt')
    report['env_steps']=env_steps
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2))
except BaseException:
    (args.output/'error.txt').write_text(traceback.format_exc());raise
finally:
    if env is not None:env.close()
    app.close(skip_cleanup=True)
