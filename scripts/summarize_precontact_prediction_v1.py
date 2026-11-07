"""Separate native prediction error before and after observed nonfoot impact.

First nonfoot contact uses the original three monitored force groups. It is
not an all-collider or geometric collision oracle. This is retrospective
diagnosis, never a teacher trigger or an action-selection label.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recorded_nonfoot_contact_history_v1 import RecordedNonfootContactHistory


def phase(first_contact_s,tick,dt=.01):
    if not np.isfinite(dt) or dt<=0 or tick<0 or tick!=int(tick):
        raise ValueError('Require a valid control tick and timestep')
    if np.isnan(first_contact_s) or first_contact_s<0:
        raise ValueError('Invalid first-contact timestamp')
    if np.isposinf(first_contact_s):return 'no_recorded_nonfoot_event'
    if first_contact_s<=tick*dt+1e-9:return 'after_or_at_observed_nonfoot'
    if first_contact_s<=(tick+1)*dt+1e-9:return 'interval_crosses_observed_nonfoot'
    return 'before_observed_nonfoot'


def stats(values):
    x=np.asarray(values,dtype=float)
    return None if not len(x) else dict(n=len(x),median=float(np.median(x)),p95=float(np.quantile(x,.95)),maximum=float(x.max()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['audit','run','baseline','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve diagnostic')
    audit=json.loads(a.audit.read_text())
    if Path(audit['source']).resolve()!=a.run.resolve():raise ValueError('Prediction source differs')
    for file,digest in audit['source_sha256'].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=digest:raise ValueError('Changed prediction evidence '+file)
    records=json.loads((a.run/'landing_mpc_records.json').read_text())
    logs={}
    for record in records:
        if record.get('episode_index')!=0:continue
        key=(record['case'],record['control_tick'])
        if key in logs:raise ValueError('Duplicate first-episode controller record')
        logs[key]=record
    metrics=json.loads((a.run/'metrics.json').read_text());old=json.loads((a.baseline/'metrics.json').read_text())
    current={r['training_index']:r for r in metrics['complete5_case_outcomes']}
    prior={r['training_index']:r for r in old['complete5_case_outcomes']}
    history=RecordedNonfootContactHistory(a.run);annotated=[]
    for row in audit['records']:
        record=logs[(row['case'],row['tick'])]
        if bool(record['applied_first_step'])!=bool(row['applied']):raise ValueError('Applied label differs')
        row=dict(row,contact_phase=phase(history.first_frame[row['case']]*history.dt,row['tick']),
            full_horizon_accepted=bool(record['accepted']),support_feet=sum(record['support_contacts']),
            old_success_lost=bool(prior[row['case']]['complete5'] and not current[row['case']]['complete5']))
        annotated.append(row)
    selectors={
        'applied_all':lambda r:r['applied'],
        'applied_before_any_recorded_nonfoot':lambda r:r['applied'] and r['contact_phase'] in ('before_observed_nonfoot','no_recorded_nonfoot_event'),
        'applied_after_or_crossing_nonfoot':lambda r:r['applied'] and r['contact_phase'] in ('after_or_at_observed_nonfoot','interval_crosses_observed_nonfoot'),
        'accepted_horizon_before_any_recorded_nonfoot':lambda r:r['applied'] and r['full_horizon_accepted'] and r['contact_phase'] in ('before_observed_nonfoot','no_recorded_nonfoot_event'),
        'first_step_only_before_any_recorded_nonfoot':lambda r:r['applied'] and not r['full_horizon_accepted'] and r['contact_phase'] in ('before_observed_nonfoot','no_recorded_nonfoot_event'),
        'lost_success_before_nonfoot':lambda r:r['applied'] and r['old_success_lost'] and r['contact_phase'] in ('before_observed_nonfoot','no_recorded_nonfoot_event'),
        'fallback_before_nonfoot':lambda r:not r['applied'] and r['contact_phase'] in ('before_observed_nonfoot','no_recorded_nonfoot_event')}
    groups={};fields=['leg_position_error_max_rad','arm_position_error_max_rad','leg_velocity_error_max_rad_s',
        'arm_velocity_error_max_rad_s','base_position_error_m','base_linear_velocity_error_mps','base_angular_velocity_error_rad_s']
    for name,select in selectors.items():
        rows=[r for r in annotated if select(r)];models={}
        for model in audit['summary']:
            valid=[r['models'][model] for r in rows if r['models'][model]['ok']]
            models[model]=dict(failed=len(rows)-len(valid),errors={k:stats([r[k] for r in valid]) for k in fields})
        groups[name]=dict(samples=len(rows),cases=sorted(set(r['case'] for r in rows)),models=models)
    early=[r for r in annotated if selectors['applied_before_any_recorded_nonfoot'](r) and r['models']['implicit_16']['ok']]
    worst=sorted(early,key=lambda r:r['models']['implicit_16']['leg_position_error_max_rad'],reverse=True)[:20]
    compact=[]
    for row in worst:
        record=logs[(row['case'],row['tick'])]
        compact.append({k:row[k] for k in ['case','tick','contact_phase','observed_contact_change','full_horizon_accepted','support_feet','old_success_lost']}|
            dict(errors=row['models']['implicit_16'],foot_force_world_n=record['measured_foot_force_world'],
                first_nonfoot_s=None if not np.isfinite(history.first_frame[row['case']]) else float(history.first_frame[row['case']]*history.dt)))
    files=[Path(__file__),a.audit,a.run/'landing_mpc_records.json',a.run/'metrics.json',a.baseline/'metrics.json',
        a.run/'contact_telemetry_metadata.json',a.run/'first_nonfoot_contact_details.json']
    result=dict(scope=__doc__,run=str(a.run.resolve()),native_new_execution=False,student_updated=False,
        sample_stride_control_ticks=20,nominal_dt_s=.01,groups=groups,worst_precontact_leg_errors=compact,
        velocity_error_coordinate_note='Existing audit compares local base components; position errors are coordinate-independent joint errors.',
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({name:dict(samples=g['samples'],leg_p95=None if not g['samples'] else g['models']['implicit_16']['errors']['leg_position_error_max_rad']['p95']) for name,g in groups.items()}))


if __name__=='__main__':main()
