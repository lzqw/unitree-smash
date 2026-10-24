#!/usr/bin/env python3
"""Native contact fixtures: real tool parts, suspended net and net posts."""
import argparse,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.full_racket_env_v4 import full_racket_cfg,FullRacketEnvV4
from quadsmash.physics.native import quat_rotate


class ContactFixtureEnv(FullRacketEnvV4):
    def _setup_scene(self):
        # JumpEnv deliberately defaults to a floating robot. Fixed root is
        # enabled only for this explicitly non-policy contact fixture.
        self.cfg.robot.spawn.articulation_props.fix_root_link=True
        super()._setup_scene()


def main():
    args.output.mkdir(parents=True,exist_ok=True)
    cfg=full_racket_cfg(8,.0000625);cfg.seed=720151;cfg.episode_length_s=.35
    cfg.robot.spawn.articulation_props.fix_root_link=True
    cfg.robot.init_state.pos=(0.,0.,1.1)
    # The default downward racket is partly occluded by the dog's head.
    # Raise it using a real joint pose so the strings fixture has a clear ray;
    # this changes this diagnostic's initialization, never the task asset.
    arm=json.loads((ROOT/'reports/G07_full_racket_v4/ik_seeds01/seeds.json').read_text())['seeds'][0]['joint_position_rad']
    cfg.robot.init_state.joint_pos.update({f'joint{i+1}':q for i,q in enumerate(arm)})
    env=ContactFixtureEnv(cfg);wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
    try:
        with torch.inference_mode():
            observation,_=wrapped.reset();ids=torch.arange(8,device=env.device)
            assert env.robot.is_fixed_base
            safety=quat_rotate(env.robot.data.body_link_quat_w[:,env.safety_body_ids,None,:],env.safety_points)+env.robot.data.body_link_pos_w[:,env.safety_body_ids,None,:]-env.scene.env_origins[:,None,None,:]
            assert torch.isfinite(safety).all() and safety.abs().max()<3.
            assert safety[...,0].max()<env.court_config['net_plane_x_m'], 'Actual initial robot must remain on own half'
            (args.output/'initial_safety_geometry.json').write_text(json.dumps(dict(
                robot_roots_m=(env.robot.data.root_pos_w-env.scene.env_origins).cpu().tolist(),
                proxies=[dict(path=path,body=env.robot.body_names[int(env.safety_body_ids[j])],
                    local_corners_m=env.safety_points[0,j].cpu().tolist(),
                    initial_min_m=safety[:,j].amin(1).cpu().tolist(),
                    initial_max_m=safety[:,j].amax(1).cpu().tolist())
                    for j,path in enumerate(env.safety_proxy_paths)]),indent=2)+'\n')
            tool=env.robot.data.body_link_pos_w[:,env.racket_body_id]-env.scene.env_origins
            q=env.robot.data.body_link_quat_w[:,env.racket_body_id]
            normal=quat_rotate(q,torch.tensor([1.,0,0],device=env.device).expand(8,-1))
            points=tool.clone();velocity=-5*normal
            points[:4]=tool[:4]+.08*normal[:4]
            points[1]+=quat_rotate(q[1:2],torch.tensor([[0.,.087,.109]],device=env.device))[0]
            points[2]+=quat_rotate(q[2:3],torch.tensor([[0.,0.,-.24]],device=env.device))[0]
            points[3]=tool[3]+quat_rotate(q[3:4],torch.tensor([[-.0225,0.,-.45]],device=env.device))[0]-.25*normal[3]
            velocity[3]=5*normal[3]
            points[4:]=torch.tensor([[1.1,1.5,.5],[1.1,1.5,1.0],[1.1,1.5,1.8],[1.1,3.075,1.0]],device=env.device)
            velocity[4:]=torch.tensor([5.,0.,0.],device=env.device)
            env.initialize_rally_launch(ids,points,velocity)
            limits=env.robot.data.joint_pos_limits[0,env.joint_ids]
            default=env.robot.data.default_joint_pos[0,env.joint_ids]
            action=(2*(default-limits[:,0])/(limits[:,1]-limits[:,0])-1).expand(8,-1)
            net_seen=torch.zeros(8,dtype=torch.bool,device=env.device)
            racket_seen=net_seen.clone();strings_seen=net_seen.clone();body_seen=net_seen.clone()
            maximum_x=points[:,0].clone()
            trace=[]
            for tick in range(35):
                maximum_x=torch.maximum(maximum_x,env.shuttle.data.root_pos_w[:,0]-env.scene.env_origins[:,0])
                net_seen|=env.rally.shuttle_net_contact
                racket_seen|=env.rally.had_contact
                strings_seen|=env.rally.string_hit
                body_seen|=env.rally.body_ball_fault
                trace.append(dict(time_s=tick*.01,ball_position_m=(env.shuttle.data.root_pos_w-env.scene.env_origins).cpu().tolist(),
                    tool_position_m=(env.robot.data.body_link_pos_w[:,env.racket_body_id]-env.scene.env_origins).cpu().tolist(),
                    tool_quaternion_wxyz=env.robot.data.body_link_quat_w[:,env.racket_body_id].cpu().tolist(),
                    ball_string_gap_m=env._tool_gap(env.shuttle.data.root_pos_w,env.robot.data.body_link_pos_w[:,env.racket_body_id],env.robot.data.body_link_quat_w[:,env.racket_body_id]).cpu().tolist(),
                    racket_force_n=env.shuttle_contact.data.force_matrix_w[:,0,0].norm(dim=-1).cpu().tolist(),
                    net_ball_force_n=env.scene.sensors['inverse_net'].data.force_matrix_w[:,0,-1].norm(dim=-1).cpu().tolist(),
                    in_play=env.rally.in_play.cpu().tolist(),launch_ready=env.launch_signals_ready.cpu().tolist(),
                    contact_count_raw=env.contact_events.count.cpu().tolist(),
                    faults={name:getattr(env.rally,name).cpu().tolist() for name in ['robot_net_fault','body_ball_fault','wrong_side_fault','racket_invasion_fault','body_over_net_fault','engineering_body_crossing','bad_return_fault','net_blocked_fault','ground_seen']}))
                observation,_,_,_=wrapped.step(action)
        rows=env.rally_records
        result=dict(kind='Fixed-base/nonground physical contact fixtures only; no policy acceptance claim',
            names=['string-center','outside-string-corner','shaft','fixture-or-arm','under-net','net-band','above-net','net-post'],
            net_contact_seen=net_seen.cpu().tolist(),racket_contact_seen=racket_seen.cpu().tolist(),
            string_contact_seen=strings_seen.cpu().tolist(),body_contact_seen=body_seen.cpu().tolist(),
            maximum_ball_x_m=maximum_x.cpu().tolist(),episodes=rows,
            actual_contacts=env.contact_records,numerical_failures=sum(r['numerical_failures'] for r in rows),
            physics_dt_s=cfg.sim.dt,fixture_arm_position_rad=arm,
            formal_acceptance=False,ground_initialization=False,fixed_base=True)
        (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        (args.output/'trace.json').write_text(json.dumps(trace)+'\n')
        assert strings_seen[0] and racket_seen[0], 'Actual center-string contact must register'
        assert not strings_seen[1] and not strings_seen[2] and not strings_seen[3]
        assert body_seen[3], 'Fixture/arm touch must be a body-contact fault'
        assert not net_seen[4] and maximum_x[4]>1.6, 'Suspended band must allow under-net physical passage'
        assert net_seen[5] and net_seen[7], 'Band and post must physically contact the shuttle'
        assert not net_seen[6] and maximum_x[6]>1.6, 'Above-net physical passage must remain possible'
        assert result['numerical_failures']==0, 'Fine-step native string sweeps must agree with actual contacts'
        (args.output/'completed.json').write_text(json.dumps(dict(env_steps=280))+'\n')
        print(json.dumps(result),flush=True)
    finally:env.close()


try:main()
except BaseException:
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'error.txt').write_text(traceback.format_exc());raise
finally:app.close(skip_cleanup=True)
