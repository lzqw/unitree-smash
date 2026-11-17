"""Same-input historical search versus current position-only anchor. CPU only."""
import argparse,hashlib,json,subprocess,types,sys
from pathlib import Path
import numpy as np
import torch
from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4
from quadsmash.control.position_anchored_contact_search_v1 import position_anchor
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.whole_body_support_qp_v1 import PositionPDInterface


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve evidence')
    root=Path(__file__).resolve().parents[1]
    source=subprocess.check_output(['git','show','53299e8:quadsmash/control/contact_first_trajectory_search_v1.py'])
    name='quadsmash.control._historical_contact53299';module=types.ModuleType(name);module.__package__='quadsmash.control';sys.modules[name]=module
    exec(compile(source,'git:53299e8/contact_first_trajectory_search_v1.py','exec'),module.__dict__)
    obs=np.load(a.input/'public_observations.npy');checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    controller=FullRacketRallyControllerV4(checkpoint,obs.shape[1],'cpu')
    cfg=json.loads((root/'reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface'];limits=torch.tensor(PositionPDInterface(**cfg).limits,dtype=torch.float32)
    model=LoadedPinocchioModelV1();authored=json.loads((root/'reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json').read_text())['defaults_before_neutral_arm_override']
    default=np.array([authored[n] for n in model.joint_order]);default[12:]=json.loads((root/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad'];default=torch.tensor(default,dtype=torch.float32)
    settings={k:v for k,v in checkpoint['seed']['contact_first_search'].items() if k in module.ContactFirstSearchConfig.__dataclass_fields__};settings['require_joint_prefix']=False
    historical=module.ContactFirstTrajectorySearchV1(module.ContactFirstSearchConfig(**settings));rows=[]
    def compare(**kw):
        kw.pop('kinematics_at_time_fn',None);kw['evaluator']=None;kw['impact_evaluator']=None
        try:
            old=historical.plan(**kw)
        except RuntimeError as error:
            rows.append(dict(historical_rejection=str(error),diagnostics=historical.last_diagnostics));raise
        t=old.contact_time_s;base=kw['base_state_fn'](t);R=base['rotation_current_from_future']
        target=kw['target_state_fn'](t)['position_world_m'];target=R.T@(target-base['translation_current_m'])
        fk=kw['kinematics_fn'];anchor=position_anchor(fk,[kw['q_rad']],kw['lower_rad'],kw['upper_rad'],target)
        preserve=position_anchor(fk,[kw['q_rad'],old.q_rad[old.contact_index]],kw['lower_rad'],kw['upper_rad'],target)
        rows.append(dict(contact_time_s=t,frame='same predicted future body frame',target_position=target,
            original_q_rad=old.q_rad[old.contact_index],original_position_error_m=float(np.linalg.norm(fk(old.q_rad[old.contact_index])[0]-target)),
            measured_seed_anchor=anchor,existing_candidate_preserved_anchor=preserve,physical_certification=False))
        raise RuntimeError('Diagnostic only: do not install historical/anchor candidates')
    controller.contact_first_planner.plan=compare
    with torch.inference_mode():
        for tick in range(202):controller.action({'policy':torch.tensor(obs[tick])},default,limits)
    a.output.write_text(json.dumps(dict(historical_source_sha256=hashlib.sha256(source).hexdigest(),
        comparison_scope='Same recorded input, same old selected time, same FK geometry and future base. No inverse-goal substitution.',rows=rows),
        indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())+'\n')
    print(json.dumps([dict(old=r.get('original_position_error_m'),anchor=r.get('measured_seed_anchor',{}).get('position_error_anchor_m')) for r in rows]))
if __name__=='__main__':main()
