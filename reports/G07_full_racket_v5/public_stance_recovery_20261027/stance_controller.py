"""Public post-hit stance-leg position correction over an unchanged source.

This grounded kinematic proposal requests finite joint positions. It does not
apply ground forces, change PD/torque limits, or prescribe actual body motion.
The supported feet are selected only from the observed public contact bits.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
import torch

HERE=Path(__file__).resolve().parent
SCHEMA='g07-public-stance-recovery-proposal-v5'


def candidates():
    return [dict(candidate_id=i,name=name,gain=gain,horizontal_damping=xy) for i,(name,gain,xy) in
        enumerate([('unchanged_source',0.,False),('stance_rotation_half',.5,False),
                   ('stance_rotation_full',1.,False),('stance_rotation_and_xy_full',1.,True)])]


class StanceLegRequest:
    def __init__(self,asset_sha256,gain,horizontal_damping):
        if gain not in [0.,.5,1.] or type(horizontal_damping) is not bool:
            raise ValueError('Require a declared stance gain and damping switch')
        model_dir=HERE.parent/'centroidal_recovery_model_20261027'
        record=json.loads((model_dir/'ACTUAL_MODEL_CHECK.json').read_text())
        for p in [model_dir/'known_model.py',model_dir/'MODEL.json']:
            if hashlib.sha256(p.read_bytes()).hexdigest()!=record['source_sha256'][str(p)]:
                raise ValueError('Validated physical model changed')
        spec=importlib.util.spec_from_file_location('stance_known_model',model_dir/'known_model.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.model=module.KnownModel()
        if self.model.record['asset_sha256']!=asset_sha256:
            raise ValueError('Source and known model must use the same actual physical asset')
        self.feet=[self.model.indices[name+'_foot'] for name in ['FL','FR','RL','RR']]
        self.legs=[np.array([i,i+4,i+8]) for i in range(4)]
        self.gain,self.horizontal=float(gain),horizontal_damping
        self.rotation_position_gain=.20
        self.rotation_velocity_gain_s=.02
        self.horizontal_velocity_gain_s=.025
        self.maximum_virtual_rotation_rad=.08
        self.maximum_virtual_translation_m=.02
        self.maximum_joint_delta_rad=.15
        self.velocity_preview_s=.03
        self.jacobian_regularization_m_per_rad=.03

    def foot_jacobians(self,q):
        q=np.asarray(q,dtype=float)
        position=self.model.forward(q)['position'][:,self.feet]
        columns=[]
        for j in range(12):
            qd=np.zeros_like(q);qd[:,j]=1.
            columns.append(self.model.forward(q,qd)['linear'][:,self.feet])
        return position,np.stack(columns,axis=-1)

    def request(self,public,defaults,limits,base_target,hit_seen):
        public=np.asarray(public,dtype=float)
        defaults,limits,base_target=map(np.asarray,[defaults,limits,base_target])
        if public.ndim!=2 or public.shape[1]!=794 or defaults.shape!=(18,) or limits.shape!=(18,2):
            raise ValueError('Require public794D and original18joint defaults/limits')
        if base_target.shape!=(len(public),18) or np.asarray(hit_seen).shape!=(len(public),):
            raise ValueError('Require source targets and observed-hit latch per lane')
        if not all(np.isfinite(x).all() for x in [public,defaults,limits,base_target]):
            raise ValueError('Finite inputs required')
        up=-public[:,6:9]
        height=1.80-np.sum(up*public[:,760:763],axis=1)
        contacts=public[:,57:61]>.5
        active=(np.asarray(hit_seen,dtype=bool)&contacts.any(1)&(height>=.14)&(up[:,2]>.25)&(self.gain>0.))
        result=np.zeros((len(public),12),dtype=np.float32)
        rows=np.flatnonzero(active);diagnostics=[]
        if not len(rows):return result,active,diagnostics
        obs=public[rows]
        q=defaults[None]+obs[:,9:27]
        position,jacobian=self.foot_jacobians(q)
        state=self.model.public_world_state(obs,defaults)
        root_rotation=state['root_rotation_world']
        for i,lane in enumerate(rows):
            u=up[lane]/max(np.linalg.norm(up[lane]),1e-8)
            omega=public[lane,3:6].copy();omega[2]=0.
            root_v_world=root_rotation[i]@public[lane,:3]
            # A stricter deadband than the later stable criterion avoids
            # continually disturbing an already settled source trajectory.
            if np.linalg.norm(np.cross([0.,0.,1.],u))<=np.sin(np.deg2rad(5.)) and np.linalg.norm(omega)<=.5 and np.linalg.norm(root_v_world[:2])<=.15:
                active[lane]=False
                diagnostics.append(dict(lane=int(lane),settled_deadband=True))
                continue
            angle=self.gain*(self.rotation_position_gain*np.cross([0.,0.,1.],u)-self.rotation_velocity_gain_s*omega)
            angle*=min(1.,self.maximum_virtual_rotation_rad/max(np.linalg.norm(angle),1e-12))
            translation_world=np.zeros(3)
            if self.horizontal:
                translation_world[:2]=-self.gain*self.horizontal_velocity_gain_s*root_v_world[:2]
                translation_world*=min(1.,self.maximum_virtual_translation_m/max(np.linalg.norm(translation_world),1e-12))
            translation=root_rotation[i].T@translation_world
            predictions=[]
            for foot,ids in enumerate(self.legs):
                if not contacts[lane,foot]:continue
                # Rigid stance-foot condition: dq should compensate the
                # small virtual root translation and rotation. Whether the
                # ground actually permits it requires a native trial.
                desired=-translation-np.cross(angle,position[i,foot])
                j=jacobian[i,foot][:,ids]
                measured=public[lane,27:39][ids]
                lower=np.maximum(-self.maximum_joint_delta_rad,np.maximum(limits[ids,0]-base_target[lane,ids],
                    self.velocity_preview_s*(-30.-measured)))
                upper=np.minimum(self.maximum_joint_delta_rad,np.minimum(limits[ids,1]-base_target[lane,ids],
                    self.velocity_preview_s*(30.-measured)))
                if np.any(lower>=upper):
                    predictions.append(dict(foot=foot,solved=False,reason='Original position/velocity box infeasible'))
                    continue
                a=np.vstack((j,self.jacobian_regularization_m_per_rad*np.eye(3)))
                b=np.r_[desired,np.zeros(3)]
                fit=lsq_linear(a,b,bounds=(lower,upper),method='bvls',tol=1e-10)
                if not fit.success or not np.isfinite(fit.x).all():
                    raise RuntimeError('Stance-leg bounded solve failed')
                before=float(np.linalg.norm(desired));after=float(np.linalg.norm(j@fit.x-desired))
                if after>before+1e-9:
                    predictions.append(dict(foot=foot,solved=False,reason='No local kinematic improvement'))
                    continue
                result[lane,ids]=fit.x.astype(np.float32)
                predictions.append(dict(foot=foot,solved=True,joint_indices=ids.tolist(),
                    desired_foot_delta_body_m=desired.tolist(),linearized_foot_delta_body_m=(j@fit.x).tolist(),
                    requested_joint_delta_rad=fit.x.tolist(),hypothetical_total_joint_velocity_rad_s=(measured+fit.x/self.velocity_preview_s).tolist(),
                    foot_kinematic_error_before_m=before,foot_kinematic_error_after_m=after))
            if not result[lane].any():active[lane]=False
            diagnostics.append(dict(lane=int(lane),settled_deadband=False,virtual_root_rotation_body_rad=angle.tolist(),
                virtual_root_translation_body_m=translation.tolist(),observed_supported_feet=contacts[lane].tolist(),legs=predictions))
        return result,active,diagnostics


class StanceRecoveryController:
    def __init__(self,source_controller,asset_sha256,count,device,gain,horizontal_damping):
        self.source=source_controller
        self.requester=StanceLegRequest(asset_sha256,gain,horizontal_damping)
        self.hit_seen=torch.zeros(count,dtype=torch.bool,device=device)
        self.last_diagnostic=None

    @property
    def reference_controller(self):return self.source.reference_controller
    @property
    def logs(self):return self.source.logs
    @logs.setter
    def logs(self,value):self.source.logs=value

    def reset(self,ids=None):
        self.source.reset(ids)
        if ids is None:
            self.hit_seen.zero_();self.last_diagnostic=None
        else:self.hit_seen[ids]=False

    @torch.inference_mode()
    def action(self,observation,default,limits):
        public=observation['policy']
        base=self.source.action(observation,default,limits)
        if self.requester.gain==0.:return base
        # Phase3 is airborne and contact_seen. Latch it through grounded
        # recovery, without consulting true rally legality or case labels.
        self.hit_seen|=public[:,773]>.5
        base_target=limits[:,0]+.5*(base+1.)*(limits[:,1]-limits[:,0])
        delta,active,predictions=self.requester.request(public.detach().cpu().numpy(),default.detach().cpu().numpy(),
            limits.detach().cpu().numpy(),base_target.detach().cpu().numpy(),self.hit_seen.detach().cpu().numpy())
        result=base.clone()
        rows=torch.tensor(np.flatnonzero(active),device=public.device,dtype=torch.long)
        if len(rows):
            change=torch.as_tensor(delta,device=public.device,dtype=public.dtype)
            result[rows,:12]=(base[rows,:12]+2.*change[rows]/(limits[:12,1]-limits[:12,0])).clamp(-1.,1.)
        self.last_diagnostic=dict(active=active,hit_seen=self.hit_seen.clone(),requested_joint_delta_rad=delta,
            model_predictions=predictions,arm_commands_bitwise_preserved=True)
        return result
