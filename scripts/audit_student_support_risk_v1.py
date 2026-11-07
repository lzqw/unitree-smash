#!/usr/bin/env python3
"""Case-held-out prediction of future nonfoot contact under the fixed student.

Offline diagnostic only. Fit/calibration/test cases are disjoint per fold;
future sensor events are labels, never features or runtime triggers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.control.support_risk_features_v1 import support_risk_features
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.recorded_nonfoot_contact_history_v1 import RecordedNonfootContactHistory


def fit_logistic(x,y,cases):
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-3)
    z=np.clip((x-mean)/scale,-20,20)
    _,counts=np.unique(cases,return_counts=True)
    count=dict(zip(np.unique(cases),counts))
    weights=np.array([1./count[c] for c in cases])
    for label in [0,1]:weights[y==label]/=weights[y==label].sum()
    weights*=.5
    def objective(theta):
        logits=z@theta[:-1]+theta[-1]
        loss=float(weights@(np.logaddexp(0,logits)-y*logits)+.005*(theta[:-1]@theta[:-1]))
        residual=weights*(expit(logits)-y)
        return loss,np.r_[z.T@residual+.01*theta[:-1],residual.sum()]
    fit=minimize(objective,np.zeros(x.shape[1]+1),jac=True,method='L-BFGS-B',
                 options=dict(maxiter=1000,ftol=1e-11,gtol=1e-7))
    if not fit.success:raise ValueError('Risk fit did not converge: '+fit.message)
    return dict(mean=mean,scale=scale,theta=fit.x,iterations=int(fit.nit))


def predict(model,x):
    return np.clip((x-model['mean'])/model['scale'],-20,20)@model['theta'][:-1]+model['theta'][-1]


def auc(y,score):
    n=int(y.sum());m=len(y)-n
    return float((rankdata(score)[y==1].sum()-n*(n+1)/2)/(n*m)) if n and m else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous risk evidence')
    started=time.perf_counter();r=a.baseline
    read=lambda n:json.loads((r/n).read_text())
    if read('job.json')['status']!='SUCCEEDED':raise ValueError('Completed real baseline required')
    metrics=read('metrics.json')
    if metrics['complete5']!=27 or metrics['landing_config'] is not None:
        raise ValueError('Require declared original27 student reference')
    outcomes={x['training_index']:x for x in metrics['complete5_case_outcomes']}
    history=RecordedNonfootContactHistory(r)
    manifest=read('fixed_policy_traces/manifest.json')
    entries={int(e['env_id']):e for e in manifest['episodes']}
    sources=[Path(__file__),ROOT/'quadsmash/control/support_risk_features_v1.py']
    sources += [r/n for n in ['job.json','metrics.json','contact_telemetry_metadata.json',
        'first_nonfoot_contact_details.json','fixed_policy_traces/manifest.json']]
    arrays=[];labels=[];cases=[];ticks=[];lead=[];coverage=[]
    evaluation=[];evaluation_cases=[];evaluation_ticks=[];evaluation_lead=[]
    for case in range(128):
        path=r/'fixed_policy_traces'/f'case_{case:03d}.npz';sources.append(path)
        with np.load(path,allow_pickle=False) as z:
            public=unpack_public(z['packed_public']);action=z['dispatched_actions'].copy()
        if hashlib.sha256(public.tobytes()).hexdigest()!=entries[case]['full_public794_sha256']:
            raise ValueError('Recorded public reconstruction changed')
        np.testing.assert_allclose(public[:,775],np.arange(len(public))*.01,rtol=0,atol=2e-6)
        hit=False;active=False;eligible=[]
        for tick,row in enumerate(public):
            active=active or (hit and row[774]>.5)
            hit=hit or row[773]>.5
            if active and tick%5==0:eligible.append(tick)
        event_s=history.first_frame[case]*history.dt
        alive=[tick for tick in eligible if event_s-.01*tick>1e-9]
        if alive:
            evaluation.append(support_risk_features(public[alive],action[alive]))
            evaluation_cases.extend([case]*len(alive));evaluation_ticks.extend(alive)
            evaluation_lead.extend([event_s-.01*tick for tick in alive])
        selected=[];target=[];times=[];censored=0
        for tick in eligible:
            # Exclude already observed nonfoot contact. Do not label censored
            # windows negative when the complete 0.8 seconds was not observed.
            remaining=event_s-.01*tick
            if remaining<=1e-9:continue
            positive=remaining<=.8+1e-9
            if not positive and tick+80>=len(public):censored+=1;continue
            selected.append(tick);target.append(int(positive));times.append(remaining)
        coverage.append(dict(case=case,eligible_samples=len(eligible),samples=len(selected),
            positive_samples=sum(target),censored_samples=censored,
            first_nonfoot_s=float(event_s) if np.isfinite(event_s) else None))
        if not selected:continue
        arrays.append(support_risk_features(public[selected],action[selected]))
        labels.extend(target);cases.extend([case]*len(selected));ticks.extend(selected);lead.extend(times)
    x=np.concatenate(arrays);y=np.asarray(labels);c=np.asarray(cases);t=np.asarray(ticks);lead=np.asarray(lead)
    ex=np.concatenate(evaluation);ec=np.asarray(evaluation_cases)
    et=np.asarray(evaluation_ticks);el=np.asarray(evaluation_lead)
    evaluation_scores=np.full(len(ec),np.nan);evaluation_thresholds=np.full(len(ec),np.nan)
    predictions=np.full(len(y),np.nan);thresholds=np.full(len(y),np.nan);folds=[]
    # Case-based modulo allocation is predeclared, not selected for performance.
    for fold in range(5):
        test=c%5==fold;cal=c%5==(fold+1)%5;train=~(test|cal)
        model=fit_logistic(x[train],y[train],c[train])
        evaluation_cal=ec%5==(fold+1)%5
        evaluation_test=ec%5==fold
        cal_score=predict(model,ex[evaluation_cal]);cal_cases=ec[evaluation_cal]
        safe=[int(i) for i in np.unique(cal_cases) if not np.isfinite(history.first_frame[i])]
        if not safe:raise ValueError('No fully observed no-nonfoot calibration cases')
        # Conservative operating point: no calibration episode alarms. This
        # is not a safety guarantee; held-out false alarms remain reportable.
        threshold=float(np.nextafter(max(float(cal_score[cal_cases==i].max()) for i in safe),np.inf))
        predictions[test]=predict(model,x[test]);thresholds[test]=threshold
        evaluation_scores[evaluation_test]=predict(model,ex[evaluation_test])
        evaluation_thresholds[evaluation_test]=threshold
        folds.append(dict(fold=fold,train_cases=np.unique(c[train]).tolist(),
            calibration_cases=np.unique(c[cal]).tolist(),test_cases=np.unique(c[test]).tolist(),
            safe_calibration_cases=safe,threshold_logit=threshold,fit_iterations=model['iterations'],
            test_auc=auc(y[test],predictions[test])))
    assert np.isfinite(predictions).all() and np.isfinite(evaluation_scores).all()
    alarm=predictions>=thresholds;bycase=[]
    evaluation_alarm=evaluation_scores>=evaluation_thresholds
    for case in np.unique(ec):
        mask=ec==case;alert=mask&evaluation_alarm
        in_warning=alert&(el>=.2)&(el<=.8+1e-9)
        bycase.append(dict(case=int(case),complete5=outcomes[case]['complete5'],
            ball5_except_recovery=outcomes[case]['ball5_except_recovery'],
            nonfoot_event=bool(np.isfinite(history.first_frame[case])),
            any_alarm=bool(alert.any()),alarm_samples=int(alert.sum()),
            first_alarm_s=float(et[alert].min()*.01) if alert.any() else None,
            warning_200_to_800ms=bool(in_warning.any()),
            earliest_warning_lead_s=float(el[in_warning].max()) if in_warning.any() else None))
    old=[v for v in bycase if v['complete5']]
    failed=[v for v in bycase if v['ball5_except_recovery'] and not v['complete5'] and v['nonfoot_event']]
    result=dict(scope=__doc__,native_execution=False,online_gate_installed=False,
        teacher_effectiveness_predicted=False,score_is_calibrated_probability=False,
        baseline=str(r.resolve()),horizon_s=.8,sample_interval_s=.05,features=158,
        model='case/class-balanced L2 logistic; lambda .01; no hyperparameter search',
        samples=len(y),positive_samples=int(y.sum()),cases=len(np.unique(c)),
        full_eligible_pre_event_evaluation_samples=len(ec),
        held_out_auc=auc(y,predictions),positive_sample_recall=float(alarm[y==1].mean()),
        negative_sample_false_alarm_rate=float(alarm[y==0].mean()),
        old_success_cases_with_samples=len(old),old_success_cases_alarm=sum(v['any_alarm'] for v in old),
        ball_qualified_nonfoot_failures_with_samples=len(failed),
        ball_qualified_nonfoot_failures_warned_200_to_800ms=sum(v['warning_200_to_800ms'] for v in failed),
        coverage=coverage,folds=folds,case_results=bycase,runtime_s=time.perf_counter()-started,
        limitations=['Fixed128 development traces only; not unseen-feed generalization.',
            'Frame classification excludes censored windows; case alarm checks include the full eligible pre-event episode.',
            'Predicts nonfoot contact under the original frozen student, not teacher rescue capability.',
            'Cannot identify stable-window failure without nonfoot contact.',
            'No model export or online intervention; results do not imply recovery improvement.',
            'Correlated ticks are never split across fit, calibration and held-out cases.'],
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','coverage','folds','case_results']}))


if __name__=='__main__':main()
