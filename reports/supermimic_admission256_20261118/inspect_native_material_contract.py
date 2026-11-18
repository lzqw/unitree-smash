"""Original128 native setup, read material tensors once, zero control steps.
No actor/world/critic optimizer is constructed; no dynamics or material mutation.
"""
import argparse,json,hashlib,os,sys,traceback,subprocess
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
def write(n,v):(a.output/n).write_text(json.dumps(v,indent=2)+'\n')
write('startup.json',dict(argv=sys.argv,cwd=os.getcwd(),executable=sys.executable,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),pid=os.getpid(),control_ticks=0,scope=__doc__))
from quadsmash.rl.supermimic_run_lifecycle import execute_scoped_native,close_resource

def run():
 import torch
 from pxr import Usd,UsdPhysics,UsdShade
 from isaacsim.core.utils.stage import get_current_stage
 from quadsmash.rl.supermimic_task_contract import resolve_task_contract,make_task_cfg,zero_panel_reset_noise
 from quadsmash.tasks.stand.supermimic_env_v1 import SuperMimicEnvV1
 from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv
 class Env(SuperMimicEnvV1,FrozenPanelEnv):pass
 source=Path('/home/autodrive/LZQW/unitree_smash/reports/G07_full_racket_v5/frozen_v2_smooth_approach16_env5120_01/policy.pt');root=Path('/home/autodrive/LZQW/unitree_smash')
 if hashlib.sha256(source.read_bytes()).hexdigest()!='8f510fe9f218beef791ac21675a19ed25d11f510a83fe247cddf2893bee08399':raise ValueError('Source changed')
 ck=torch.load(source,map_location='cpu',weights_only=False);contract,protocol,cases,receipt=resolve_task_contract(ck,source,execution_root=root)
 cfg=make_task_cfg(128,contract,protocol,receipt);cfg.seed=882072;cfg.sim.device='cuda:0';env=None;failed=False
 try:
  env=Env(cfg,cases=cases,reset_noise=zero_panel_reset_noise(128,882072),residual_policy_checkpoint=ck,g07_execution_root=root,contact_flight_recovery=contract['contact_flight_recovery']);env.reset()
  robot=env.robot.root_physx_view.get_material_properties().clone();shuttle=env.shuttle.root_physx_view.get_material_properties().clone()
  if not torch.isfinite(robot).all() or not torch.isfinite(shuttle).all():raise ValueError('Nonfinite actual material tensor')
  stage=get_current_stage();rows=[]
  for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0/Robot/Racket'),Usd.TraverseInstanceProxies()):
   if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
   mat,rel=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
   attrs={}
   if mat:
    for key in ('physics:staticFriction','physics:dynamicFriction','physics:restitution','physxMaterial:frictionCombineMode','physxMaterial:restitutionCombineMode'):
     attrs[key]=mat.GetPrim().GetAttribute(key).Get()
   rows.append(dict(path=str(prim.GetPath()),bound_material=str(mat.GetPath()) if mat else None,attributes=attrs))
  torch.save(dict(robot=robot.cpu(),shuttle=shuttle.cpu()),a.output/'native_material_tensors.pt')
  write('material_contract.json',dict(task_contract=contract,default_scene_material=cfg.sim.physics_material.to_dict(),racket_collision_bindings=rows,robot_tensor_shape=list(robot.shape),shuttle_tensor_shape=list(shuttle.shape),robot_material_values=torch.unique(robot[0],dim=0).tolist(),shuttle_material_values=torch.unique(shuttle[0],dim=0).tolist(),robot_all_lanes_equal=bool((robot==robot[:1]).all()),shuttle_all_lanes_equal=bool((shuttle==shuttle[:1]).all()),control_ticks=0,actor_updates=0,world_updates=0,critic_updates=0,formal_evaluation=False))
 except BaseException:
  failed=True;write('failure.json',dict(traceback=traceback.format_exc()));raise
 finally:close_resource(a.output,'ENV',env,primary_failed=failed)

os.environ['QUADSMASH_SKIP_KIT_CLOSE']='1'
try:
 app=AppLauncher(a,fast_shutdown=False).app
 execute_scoped_native(a.output,run,app)
except BaseException:
 (a.output/'failure_traceback.txt').write_text(traceback.format_exc());traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:sys.stdout.flush();sys.stderr.flush();os._exit(0)
