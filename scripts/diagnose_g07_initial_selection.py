#!/usr/bin/env python3
"""Grouped cross-validation of public-state controller selection.

This is a retrospective development diagnostic, never a physical rollout or
formal acceptance score. Each outcome is paired with that controller's own
recorded observation. A learned selector still needs a new native evaluation.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def features(public, current=False, feature_set='full-state-v1'):
    # No environment ID, reset seed, case label, or post-initial observation.
    # Log compression handles the explicitly long-distance model launches.
    ball = public[:, 61:67].astype(np.float64)
    ball = np.sign(ball)*np.log1p(np.abs(ball))
    if current:
        if feature_set=='body-ball-v1':
            return np.c_[ball, public[:, :9], public[:, 760:766]]
        if feature_set=='body-ball-arm-v1':
            return np.c_[ball, public[:, :9], public[:, 21:27], public[:, 39:45], public[:, 760:766]]
        return np.c_[ball, public[:, :45], public[:, 760:766]]
    return np.c_[ball, public[:, 6:9], public[:, 9:27], public[:, 763:766]]


def kernel(left, right, scale):
    difference = left[:, None, :]-right[None, :, :]
    return np.exp(-np.square(difference).sum(2)/scale)


def metrics(rows):
    air = sum(r['airborne_contact'] for r in rows)
    return dict(sample_count=len(rows), full=sum(r['complete_smash'] and not r['robot_net_contact'] and r['ground_init'] for r in rows)/len(rows),
                airborne=air/len(rows), recovery=sum(r['airborne_contact'] and r['recovery_success'] and not r['robot_net_contact'] for r in rows)/max(air, 1),
                robot_net=sum(r['robot_net_contact'] for r in rows), numerical=sum(r['numerical_failures'] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--context-time', type=float)
    parser.add_argument('--candidate-ids', type=int, nargs='+')
    parser.add_argument('--features', choices=['full-state-v1','body-ball-v1','body-ball-arm-v1'], default='full-state-v1')
    args = parser.parse_args()
    manifest = json.loads((args.run/'eval_manifest.json').read_text())
    if manifest.get('formal_evaluation_eligible'):
        raise ValueError('Only development outcomes may be used')
    rows = json.loads((args.run/'episodes.json').read_text())
    public = np.load(args.run/'initial_public_observation.npy')
    if args.context_time is not None:
        snapshots = json.loads((args.run/'preparation_snapshots.json').read_text())
        matches = [s for s in snapshots if abs(s['time_s']-args.context_time)<1e-6]
        if len(matches) != 1 or not all(matches[0]['active_first_episode']):
            raise ValueError('Requires one snapshot with every original episode active')
        public = np.zeros_like(public)
        public[:, :69] = matches[0]['public_frame']
        public[:, 759:776] = matches[0]['public_command_and_clock']
    if args.candidate_ids is not None:
        rows = [r for r in rows if r['candidate_id'] in args.candidate_ids]
    candidates = sorted({r['candidate_id'] for r in rows})
    cases = sorted({r['source_scenario'] for r in rows})
    by = {(r['candidate_id'], r['source_scenario']): r for r in rows}
    if len(by) != len(candidates)*len(cases):
        raise ValueError('Requires one outcome for every candidate and case')
    raw = {c: features(public[[by[c, s]['scenario'] for s in cases]], args.context_time is not None, args.features) for c in candidates}
    # Pair canonical and perturbed copies of one launch in the same fold.
    groups = [cases.index(s) % (len(cases)//2) for s in cases]
    reports = []
    for ridge in (1., 10., 100.):
        for width in (8., 32., 128.):
            selected = []
            for fold in range(4):
                train = np.array([i for i, g in enumerate(groups) if g % 4 != fold])
                test = np.array([i for i, g in enumerate(groups) if g % 4 == fold])
                population = np.concatenate([raw[c][train] for c in candidates])
                mean = population.mean(0)
                std = np.maximum(population.std(0), .01)
                scores = []
                for c in candidates:
                    x = (raw[c]-mean)/std
                    observed = [by[c, cases[i]] for i in train]
                    y = np.array([10*bool(r['complete_smash'] and not r['robot_net_contact'] and not r['numerical_failures'])
                                  + 4*bool(r['recovery_success'] and not r['robot_net_contact'] and not r['numerical_failures'])
                                  - 8*bool(r['robot_net_contact'])-10*bool(r['numerical_failures']) for r in observed])
                    intercept = y.mean()
                    alpha = np.linalg.solve(kernel(x[train], x[train], width)+ridge*np.eye(len(train)), y-intercept)
                    scores.append(intercept+kernel(x[test], x[train], width)@alpha)
                choices = np.array(candidates)[np.argmax(scores, axis=0)]
                selected.extend(by[int(c), cases[i]] for c, i in zip(choices, test))
            reports.append(dict(ridge=ridge, width=width, groups={g: metrics([r for r in selected if r['development_group']==g])
                                                              for g in sorted({r['development_group'] for r in selected})},
                                selected_candidates={str(c): sum(r['candidate_id']==c for r in selected) for c in candidates}))
    result = dict(kind='retrospective grouped CV diagnostic only; not native success or independent validation',
                  context='each option uses its own recorded public observation at the declared time; independent native validation required',
                  feature_indices='log-compressed public61:67, public6:9, public9:27, public763:766',
                  context_time_s=args.context_time, candidate_subset=args.candidate_ids,
                  feature_set=args.features,
                  current_features='log-compressed public61:67, public0:45, public760:766' if args.context_time is not None else None,
                  fold='launch pair kept together; source index used only to split training, never a model input',
                  source=str(args.run), reports=reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
