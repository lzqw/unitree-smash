"""Recompute the stability window including the final model state at 8 s.

The older feedback log reports the phase sampled before each control command.
This read-only audit includes the state after its last physical interval. It
does not rerun dynamics, change thresholds, or certify native execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.event_recovery_manager_v1 import EventRecoveryManagerV1,RecoveryPhaseMemory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['search','feedback','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();read=lambda f:json.loads(Path(f).read_text())
    result=read(a.feedback/'result.json')
    if a.output.exists() or result['state']!='COMPLETED_REPLANNED_FEEDBACK_MODEL':
        raise ValueError('Fresh audit of completed feedback required')
    for directory in [a.search,a.feedback]:
        for f,h in read(directory/'inputs.json')['source_sha256'].items():
            if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:raise ValueError('Changed source '+f)
    model=LoadedPinocchioModelV1();out=[];files=[Path(__file__),a.feedback/'result.json']
    for item in result['results']:
        key=item['state_key'];name='state_'+key.replace(':','_')+'.json'
        source=a.search/name;path=a.feedback/name;files += [source,path]
        initial=read(source)['initial_actual_state'];feedback=read(path);rows=feedback['trajectory']
        if feedback['failure'] is not None or len(rows)!=800-initial['tick']:
            out.append(dict(state_key=key,complete_model_feedback=False,
                reason='Incomplete or physically rejected feedback path'));continue
        phase=initial['phase'];manager=EventRecoveryManagerV1()
        manager.memory=RecoveryPhaseMemory(episode=0,last_time_s=initial['time_s'],phase=phase['phase'],
            release_time_s=phase['release_time_s'],first_ground_time_s=phase['first_ground_time_s'],
            settled_duration_s=phase['phase_settled_duration_s'],
            final_stable_duration_s=0.,ground_contacts=np.array(phase['ground_contacts'],bool))
        for i,row in enumerate(rows):
            tick=initial['tick']+i
            if row['tick']!=tick:raise ValueError('Non-contiguous feedback trajectory')
            q=np.asarray(row['q']);v=np.asarray(row['v']);r=pin.Quaternion(q[3:7]).matrix()
            time_s=float(np.float32(tick+1)*np.float32(.01))
            phase=manager.update(episode=0,time_s=time_s,had_racket_contact=True,released=True,
                racket_touching=False,separated_now=True,ground_normal_force_n=np.asarray(row['ground_force'])[:,2],
                roll_pitch_rad=[np.arctan2(r[2,1],r[2,2]),np.arcsin(np.clip(-r[2,0],-1,1))],
                root_velocity_world=r@v[:3],root_angular_velocity_world=r@v[3:6],
                joint_velocity=v[model.v_indices],nonfoot_contact_now=False)
        if time_s!=8.:raise ValueError('Original episode-end observation required')
        stable=phase['control_sample_stable_duration_s']
        out.append(dict(state_key=key,complete_model_feedback=True,end_time_s=time_s,
            previous_precommand_stable_s=feedback['final_model_stable_s'],
            end_observation_stable_s=stable,model_terminal_window_passed=stable>=.8,
            native_success=False,valid_expert_labels=False))
    files += [Path(m.__file__) for n,m in list(sys.modules.items())
              if n.startswith('scripts.') and getattr(m,'__file__',None)]
    data=dict(results=out,source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        acceptance_changed=False,native_success=False,valid_expert_labels=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps(out))


if __name__=='__main__':main()
