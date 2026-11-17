"""Field-wise saved-native/public reconstruction audit, not independent truth."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.predicted_observation_frame_v1 import predicted_frame
from quadsmash.tasks.stand.observation_contract import FRAME_FIELDS


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior results')
    root=Path(__file__).resolve().parents[1];model=LoadedPinocchioModelV1()
    authored=json.loads((root/'reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json').read_text())['defaults_before_neutral_arm_override']
    default=np.array([authored[n] for n in model.joint_order]);default[12:]=json.loads((root/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad']
    obs=np.load(a.input,mmap_mode='r');rows=[]
    for tick in (200,201,202):
        for lane in range(obs.shape[1]):
            public=obs[tick,lane];s=model.state_from_public(public,default);R=s.root_rotation_world
            ball=s.root_position_world+R@public[61:64]
            velocity=R@(public[64:67]+s.v[:3]+np.cross(s.v[3:6],public[61:64]))
            frame=predicted_frame(model,s,default,ball,velocity,bool(public[67]>.5))
            errors={};start=0
            for name,size in FRAME_FIELDS.items():
                errors[name]=float(np.max(np.abs(frame[start:start+size]-public[start:start+size])));start+=size
            rows.append(dict(tick=tick,lane=lane,field_max_abs_error=errors,
                native_tool_angular=public[54:57].tolist(),reconstructed_tool_angular=frame[54:57].tolist()))
    a.output.write_text(json.dumps(dict(scope='Saved native observation compared with reconstructed public state; root and ball are not independent truth',rows=rows),indent=2)+'\n')
    print(json.dumps({k:max(r['field_max_abs_error'][k] for r in rows) for k in FRAME_FIELDS}))
if __name__=='__main__':main()
