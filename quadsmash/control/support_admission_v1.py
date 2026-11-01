"""Read-only admission of the actual actuator/material assumptions."""
import math

import torch

from .centroidal_support_v1 import LEG_KP,LEG_KD,LEG_TORQUE_LIMIT,FRICTION


def validate_native_support_interface(env,stage):
    from pxr import Usd,UsdPhysics,UsdShade
    if list(env.feet_contact.body_names)!=['FL_foot','FR_foot','RL_foot','RR_foot']:
        raise ValueError('Observed foot contact order differs from support model')
    actuator=env.robot.actuators['legs']
    actuator_record={}
    for name,expected in [('stiffness',LEG_KP),('damping',LEG_KD),('effort_limit',LEG_TORQUE_LIMIT)]:
        actual=torch.as_tensor(getattr(actuator,name))
        if not torch.isfinite(actual).all() or not torch.allclose(actual,torch.full_like(actual,expected)):
            raise ValueError('Native leg actuator differs from support model: '+name)
        actuator_record[name]=dict(minimum=float(actual.min()),maximum=float(actual.max()))
    if not torch.allclose(env.robot.data.joint_effort_limits[:,env.joint_ids[:12]],
                          torch.full_like(env.robot.data.joint_effort_limits[:,env.joint_ids[:12]],LEG_TORQUE_LIMIT)):
        raise ValueError('Simulator joint effort limit differs from support model')
    def material(prim):
        api=UsdPhysics.MaterialAPI(prim)
        return dict(path=str(prim.GetPath()),static=float(api.GetStaticFrictionAttr().Get()),
                    dynamic=float(api.GetDynamicFrictionAttr().Get()),
                    combine=str(prim.GetAttribute('physxMaterial:frictionCombineMode').Get() or 'average'))
    default=material(stage.GetPrimAtPath(env.cfg.sim.physics_prim_path+'/defaultMaterial'))
    ground=[material(p) for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Ground')) if p.HasAPI(UsdPhysics.MaterialAPI)]
    if len(ground)!=1:raise ValueError('Require one actual flat-ground physics material')
    robot=stage.GetPrimAtPath(env.robot.cfg.prim_path.replace('.*','0'))
    feet=[]
    for p in Usd.PrimRange(robot,Usd.TraverseInstanceProxies()):
        if '_foot/' not in str(p.GetPath()) or not p.HasAPI(UsdPhysics.CollisionAPI):continue
        binding=UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial('physics')[0]
        record=material(binding.GetPrim()) if binding else default
        if record['combine']!='average' or ground[0]['combine']!='average':
            raise ValueError('Friction combination differs from support model')
        effective=(record['dynamic']+ground[0]['dynamic'])/2
        if not math.isclose(effective,FRICTION,abs_tol=1e-6):
            raise ValueError('Actual foot/ground friction differs from support model')
        feet.append(dict(collider=str(p.GetPath()),material=record,effective_dynamic_friction=effective))
    if len(feet)!=4:raise ValueError('Require the original four foot colliders')
    return dict(legs=actuator_record,default_material=default,ground_material=ground[0],feet=feet,
                physical_parameters_changed=False,external_force_applied=False)
