"""Read-only chronological native takeoff replay; no case-conditioned control."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.unilateral_implicit_prediction_v1 import predict_unilateral_support


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    obs=np.load(a.input/'public_observations.npy');actions=np.load(a.input/'public_actions.npy')
    cfg=json.loads((root/'reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface']
    interface=PositionPDInterface(**cfg);model=LoadedPinocchioModelV1()
    authored=json.loads((root/'reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json').read_text())['defaults_before_neutral_arm_override']
    order=json.loads((root/'configs/go2_piper_racket_landing_model_v1.json').read_text())['joint_order']
    default=np.array([authored[n] for n in order]);default[12:]=json.loads((root/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad']
    rows=[]
    for lane in range(obs.shape[1]):
        contacts=obs[:,lane,57:61]>.5
        released=np.flatnonzero(contacts[:-1].any(1)&~contacts[1:].any(1))
        # First observed release only; no success labels used for sample selection.
        for tick in (int(released[0]),) if len(released) else ():
            for offset in (-1,0,1):
                k=max(0,tick+offset);state=model.state_from_public(obs[k,lane],default)
                target=interface.limits[:,0]+.5*(np.clip(actions[k,lane],-1,1)+1)*np.diff(interface.limits,axis=1)[:,0]
                for warm in (False,True):
                    trace=[];row=dict(lane=lane,tick=k,warm_start=warm,initial_contacts=state.contacts.tolist(),previous_source='recorded native public state',target_rad=target.tolist())
                    try:
                        result=predict_unilateral_support(model,interface,state,target,substeps=4,iteration_trace=trace,separating_warm_start=warm)
                        row.update(solved=True,diagnostics=result['diagnostics'],contacts=result['state'].contacts.tolist(),foot_forces_world_n=result['foot_forces_world_n'].tolist())
                    except (ValueError,np.linalg.LinAlgError) as error:
                        row.update(solved=False,error=str(error))
                    row['active_set_iterations']=trace;rows.append(row)
    from scripts.contact_first_fullbody_evaluator_v1 import ContactFirstFullBodyEvaluatorV1
    joint_replays=[]
    for lane in range(obs.shape[1]):
        contact=obs[:,lane,57:61]>.5
        release=np.flatnonzero(contact[:-1].any(1)&~contact[1:].any(1))
        if not len(release): continue
        tick=int(release[0]);end=min(tick+6,len(obs))
        target=interface.limits[:,0]+.5*(np.clip(actions[tick:end,lane],-1,1)+1)*np.diff(interface.limits,axis=1)[:,0]
        initial=model.state_from_public(obs[tick,lane],default)
        check=ContactFirstFullBodyEvaluatorV1(interface_config=cfg,initial_state=initial,
            model=model,substeps=4,contact_prediction='unilateral_release',
            leg_target_sequence_rad=target[:,:12],record_state_trace=True)
        # Recorded commands are replay data, never future observations supplied online.
        result=check(target[:,12:],np.zeros_like(target[:,12:]),len(target)-1,{})
        joint_replays.append(dict(lane=lane,start_tick=tick,result=result))
    record=dict(joint_replays=joint_replays,scope='CPU replay of actual historical native states, not native controller performance',
        input_sha256={n:hashlib.sha256((a.input/n).read_bytes()).hexdigest() for n in ('public_observations.npy','public_actions.npy')},rows=rows)
    a.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(queries=len(rows),solved=sum(r['solved'] for r in rows),output=str(a.output))))
if __name__=='__main__':main()
