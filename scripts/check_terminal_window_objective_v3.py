"""Check search guidance against the unchanged recovery manager at boundaries."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.search_actual_terminal_window_v3 import stable_window_violations
from scripts.event_recovery_manager_v1 import EventRecoveryManagerV1


def main():
    cases=[]
    for dimension,limit in enumerate([np.deg2rad(15.),.3,.1,1.,1.]):
        for factor in [0.,.999,1.,1.001]:
            values=np.zeros(5);values[dimension]=limit*factor
            cases.append((values,np.array([True,False,False,False])))
    cases += [(np.zeros(5),np.zeros(4,bool)),
              (np.zeros(5),np.array([True,False,True,False]))]
    for values,contacts in cases:
        angle,horizontal,vertical,omega,joint_speed=values
        manager=EventRecoveryManagerV1()
        observation=dict(episode=0,had_racket_contact=True,released=True,
            racket_touching=False,separated_now=True,ground_normal_force_n=contacts*10.,
            roll_pitch_rad=np.array([angle,0.]),root_velocity_world=np.array([horizontal,0.,vertical]),
            root_angular_velocity_world=np.array([0.,0.,omega]),joint_velocity=np.full(18,joint_speed),
            nonfoot_contact_now=False)
        manager.update(time_s=0.,**observation)
        result=manager.update(time_s=.01,**observation)
        penalty=stable_window_violations(observation['roll_pitch_rad'],
            observation['root_velocity_world'],omega,joint_speed,contacts)
        assert (np.all(penalty==0.)) == (result['control_sample_stable_duration_s']>0.), (values,contacts)
    actual=json.loads((ROOT/'reports/pure_control_actual_full_tail_search_20261109/state_12_550.json').read_text())['fine']['terminal']
    penalty=stable_window_violations(np.asarray(actual['roll_pitch']),
        np.asarray(actual['root_velocity_world']),actual['omega'],actual['joint_speed'],actual['contacts'])
    assert penalty[1]>0. and penalty[2]>0. and penalty[4]>0.
    print(json.dumps(dict(passed=True,boundary_cases=len(cases),
        actual_terminal_violations=penalty.tolist(),acceptance_changed=False)))


if __name__=='__main__':main()
