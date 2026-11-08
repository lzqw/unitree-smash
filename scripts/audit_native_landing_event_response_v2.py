"""Measure model response errors on actual landing commands across phases.

Only completed native observations and consecutive commanded intervals are
used. This is a diagnostic, not a new execution, model calibration, or proof
of long-horizon safety. It never changes running controllers or tolerances.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.pure_control_reference_clock_v12 import PureControlReferenceClockV12
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record
from scripts.audit_settled_stance_model_v2 import ForceRecorder
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain

CONTROL = None


def initialize(cfg, nominal):
    global CONTROL
    CONTROL = PureControlReferenceClockV12(PositionPDInterface(**cfg['interface']),
        nominal,cfg['model_reference_library'])


def predict(sample, deadline):
    if time.monotonic() > deadline:
        raise TimeoutError('Bounded model-response audit expired')
    c = CONTROL
    record, actual = sample['initial'], sample['following']
    state = state_from_record(c.model,record)
    row = {key:value for key,value in sample.items() if key not in ['initial','following']}
    row.update(predicted=False,actual_contacts=actual['phase']['ground_contacts'])
    recorder = ForceRecorder(c.model)
    try:
        result = predict_self_contact_spheres(c.model,c.interface,state,
            record['final'],c.geometry.spheres,substeps=160,
            geometry=c.path_geometry,collision_response=recorder)
        y = result['state']
        q,v = actual['q'],actual['v']
        rp,ra = pin.Quaternion(y.q[3:7]).matrix(),pin.Quaternion(q[3:7]).matrix()
        difference = pin.difference(c.model.model,q,y.q)
        initial_contacts = np.asarray(record['phase']['ground_contacts'],bool)
        observed = np.where(initial_contacts,recorder.last_force[:,2]>.5,recorder.last_force[:,2]>1.)
        row.update(predicted=True,
            position_error_m=float(np.linalg.norm(y.q[:3]-q[:3])),
            attitude_error_rad=float(np.linalg.norm(difference[3:6])),
            max_joint_position_error_rad=float(np.max(abs(y.q[c.model.q_indices]-q[c.model.q_indices]))),
            linear_velocity_error_m_s=float(np.linalg.norm(rp@y.v[:3]-ra@v[:3])),
            angular_velocity_error_rad_s=float(np.linalg.norm(rp@y.v[3:6]-ra@v[3:6])),
            max_joint_velocity_error_rad_s=float(np.max(abs(y.v[c.model.v_indices]-v[c.model.v_indices]))),
            predicted_observed_contacts=observed,
            mismatching_feet=int(np.sum(observed!=actual['phase']['ground_contacts'])),
            model_next_normal_force_n=recorder.last_force[:,2],
            actual_next_normal_force_n=np.asarray(actual['ground_force'])[:,2],
            model_nonfoot_violation_m=result['diagnostics']['nonfoot_ground_violation_m'],
            diagnostics=result['diagnostics'])
    except (ValueError,np.linalg.LinAlgError) as error:
        row['rejection'] = repr(error)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run','configuration','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    a = parser.parse_args()
    read = lambda path:json.loads(Path(path).read_text())
    sha = lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if a.output.exists() or read(a.run/'job.json')['status']!='SUCCEEDED':
        raise ValueError('Fresh output and completed native evidence required')
    cfg = read(a.configuration)
    nominal = read(a.run/'native_command_inputs.json')['default_joint_position_rad']
    initialize(cfg,nominal)
    with np.load(a.run/'pure_control_commands.npz') as z:
        arrays = {key:z[key].copy() for key in z.files}
    with np.load(a.run/'pure_control_ground_forces.npz') as z:
        peaks = z['interval_peak_normal_n'].copy()
    names = read(a.run/'pure_control_ground_forces.json')['body_names']
    nonfoot = [i for i,name in enumerate(names) if not name.endswith('_foot')]
    diagnostics = [json.loads(line) for line in (a.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    if len(diagnostics)!=len(arrays['lane']):
        raise ValueError('Misaligned recorded observations')
    samples,missing = [],[]
    def record(index):
        return dict({key:arrays[key][index].copy() for key in ['q','v','previous','final','ground_force']},
            phase=diagnostics[index]['phase'])
    for case in sorted(set(arrays['lane'].tolist())):
        ids = np.flatnonzero(arrays['lane']==case)
        lookup = {int(arrays['tick'][i]):int(i) for i in ids}
        first = min(lookup)
        contact = np.flatnonzero(np.any(peaks[first:,case][:,nonfoot]>0.,axis=1))
        bad_tick = first+int(contact[0]) if len(contact) else None
        grounded = [tick for tick,index in lookup.items() if any(diagnostics[index]['phase']['ground_contacts'])]
        touchdown = min(grounded) if grounded else None
        requests = [('release',first)]
        if touchdown is not None:
            requests += [('touchdown_interval',touchdown-1),('first_supported_interval',touchdown),
                ('support_after_200ms',touchdown+20),('support_after_1s',touchdown+100)]
        if bad_tick is not None:
            requests.append(('first_nonfoot_interval',bad_tick))
        else:
            requests.append(('late_clean_interval',798))
        for label,tick in requests:
            if tick not in lookup or tick+1 not in lookup or (bad_tick is not None and tick>bad_tick):
                missing.append(dict(case=case,label=label,tick=tick,reason='No clean consecutive pre-failure interval'))
                continue
            i,j = lookup[tick],lookup[tick+1]
            if (diagnostics[i]['tick'],diagnostics[i]['lane'])!=(tick,case) or arrays['episode'][i]!=arrays['episode'][j]:
                raise ValueError('Misaligned or reset-crossing interval')
            samples.append(dict(case=case,tick=tick,label=label,phase=diagnostics[i]['phase']['phase'],
                actual_nonfoot_in_interval=bool(np.any(peaks[tick,case,nonfoot]>0.)),
                initial=record(i),following=record(j)))
    files = [Path(__file__),a.configuration,Path(cfg['model_reference_library'])] + [a.run/name for name in
        ['job.json','native_command_inputs.json','pure_control_commands.npz','pure_control_diagnostics.jsonl',
         'pure_control_ground_forces.npz','pure_control_ground_forces.json']]
    files += [Path(module.__file__) for name,module in list(sys.modules.items())
        if name.startswith('scripts.') and getattr(module,'__file__',None)]
    pins = {str(path.resolve()):sha(path) for path in files}
    a.output.mkdir(parents=True)
    def write(name,data):
        path=a.output/name;temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(plain(data),indent=2,allow_nan=False)+'\n');temp.replace(path)
    write('inputs.json',dict(source_sha256=pins,samples=len(samples),workers=8,max_seconds=1800,
        substeps=160,actual_final_commands=True,native_rerun=False,acceptance_changed=False))
    started=time.monotonic();deadline=started+1800;rows=[]
    write('status.json',dict(state='RUNNING_EVENT_RESPONSE_AUDIT',pid=os.getpid(),completed=0,total=len(samples)))
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(cfg,nominal)) as pool:
        futures=[pool.submit(predict,sample,deadline) for sample in samples]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows)%40==0:
                write('status.json',dict(state='RUNNING_EVENT_RESPONSE_AUDIT',pid=os.getpid(),completed=len(rows),total=len(samples)))
    keys=['position_error_m','attitude_error_rad','max_joint_position_error_rad',
        'linear_velocity_error_m_s','angular_velocity_error_rad_s','max_joint_velocity_error_rad_s']
    groups={}
    for label in sorted({r['label'] for r in rows}):
        group=[r for r in rows if r['label']==label];valid=[r for r in group if r['predicted']]
        groups[label]=dict(intervals=len(group),responses=len(valid),rejections=len(group)-len(valid),
            actual_nonfoot_intervals=sum(r['actual_nonfoot_in_interval'] for r in group),
            mismatching_feet=sum(r['mismatching_feet'] for r in valid),contact_comparisons=4*len(valid),
            errors={key:dict(zip(['median','p90','maximum'],np.quantile([r[key] for r in valid],[.5,.9,1.]))) for key in keys} if valid else {})
    for filename,digest in pins.items():
        if sha(filename)!=digest:raise ValueError('Changed audit source '+filename)
    result=dict(state='COMPLETED_EVENT_RESPONSE_AUDIT',groups=groups,
        rows=sorted(rows,key=lambda r:(r['case'],r['tick'],r['label'])),missing=missing,
        wall_seconds=time.monotonic()-started,source_sha256=pins,native_rerun=False,
        model_calibrated=False,acceptance_changed=False,long_horizon_accuracy_not_inferred=True,
        force_caveat='Model final microstep force and sampled native ground-normal force are not interval-integrated impulses.')
    write('result.json',result);write('status.json',dict(state=result['state'],groups=groups))
    print(json.dumps(plain(groups)))


if __name__=='__main__':main()
