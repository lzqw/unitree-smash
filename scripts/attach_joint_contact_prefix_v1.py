"""Wire live native truth, causal frozen legs and common-frame terminal checks."""
import numpy as np
import torch
from scripts.causal_frozen_leg_prefix_v1 import CausalFrozenLegReferenceFactory
from scripts.contact_first_fullbody_evaluator_v1 import ContactFirstFullBodyEvaluatorV1
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from quadsmash.control.full_racket_rally_controller_v4 import planning_world_pose


def attach_joint_contact_prefix(env, controller, interface_config):
    cfg=controller.contact_first_planner.config
    if not cfg.require_joint_prefix or not cfg.evaluate_impact_outcome:
        raise ValueError('Explicit joint replay and impact flags required')
    default=env.robot.data.default_joint_pos[0,env.joint_ids]
    limits=env.robot.data.joint_pos_limits[0,env.joint_ids]
    def factory(lane,public,q,qd):
        model=LoadedPinocchioModelV1()
        state=model.state_from_native(env,lane)
        rotation,origin=planning_world_pose(public,[1.1,0.,1.8])
        reconstructed=model.state_from_public(public,default.detach().cpu().numpy())
        controller.contact_first_state_audit.append(dict(lane=lane,clock_s=float(public[775]),
            native_root_position=state.root_position_world.tolist(),
            public_root_position=reconstructed.root_position_world.tolist(),
            root_error_m=float(np.linalg.norm(state.root_position_world-reconstructed.root_position_world)),
            previous_final_target_rad=(limits[:,0]+.5*(env.last_action[lane]+1)*(limits[:,1]-limits[:,0])).detach().cpu().tolist(),
            state_source='native truth for physical admission; causal public input for ball planner'))
        legs=CausalFrozenLegReferenceFactory(controller.teacher,controller.parameters,
            controller.contact_first_current_public,default,limits,controller.event_time,
            controller.target_offset_world,lane,controller.physics,cfg.dt_s,controller.contact_first_prediction_counts)
        def measure(s,metadata):
            with torch.inference_mode():
                arm=torch.as_tensor(s.joint_position[12:],device=default.device,dtype=default.dtype)[None]
                offset=controller.teacher.contact_offset.clone()
                if 'contact_point_local' in metadata:offset[1:]=offset.new_tensor(metadata['contact_point_local'][1:])
                point,normal,jp,_=controller.teacher.kinematics(arm,offset)
            point,normal,jp=[x[0].detach().cpu().numpy().astype(float) for x in (point,normal,jp)]
            R=s.root_rotation_world
            world_position=s.root_position_world+R@point
            world_velocity=R@(s.v[:3]+np.cross(s.v[3:6],point)+jp@s.joint_velocity[12:])
            return rotation.T@(world_position-origin),rotation.T@R@normal,rotation.T@world_velocity
        def impact_state(s,metadata):
            # Full point Jacobian in WORLD, generalized velocity convention of M.
            offset=controller.teacher.contact_offset.detach().clone()
            if 'contact_point_local' in metadata:offset[1:]=offset.new_tensor(metadata['contact_point_local'][1:])
            offset[0]-=controller.teacher.face_sign*controller.physics['collision_radius_m']
            with torch.inference_mode():
                arm=torch.as_tensor(s.joint_position[12:],device=default.device,dtype=default.dtype)[None]
                point,_,jp,_=controller.teacher.kinematics(arm,offset)
            point=point[0].cpu().numpy();jp=jp[0].cpu().numpy()
            R=s.root_rotation_world
            skew=np.array([[0.,-point[2],point[1]],[point[2],0.,-point[0]],[-point[1],point[0],0.]])
            J=np.zeros((3,24));J[:,:3]=R;J[:,3:6]=-R@skew
            J[:,model.v_indices[12:]]=R@jp
            snapshot=model.snapshot(s.q,s.v)
            return dict(mass_matrix=snapshot['mass_matrix'].tolist(),contact_jacobian_world=J.tolist(),
                support_jacobian=snapshot['foot_jacobian'][s.contacts].reshape(-1,24).tolist(),
                support_model='sticking measured foot-body translational constraints' if s.contacts.any() else 'free_floating')
        return ContactFirstFullBodyEvaluatorV1(interface_config=interface_config,
            model=model,initial_state=state,leg_reference_factory=legs,
            contact_prediction='unilateral_release',record_state_trace=True,racket_measure_fn=measure,measurement_uses_metadata=True,impact_state_fn=impact_state)
    controller.contact_first_prediction_counts={}
    controller.contact_first_state_audit=[]
    controller.set_contact_first_evaluator_factory(factory)
    return controller
