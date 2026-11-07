"""Outcome-filtered labels for replacing a teacher-corrected recovery head.

The replacement coordinate is (executed - original frozen base)/2 in normalized
native position commands. Teacher-only increments are retained as measurements,
never substituted for that total label. This format is not a PPO rollout.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from quadsmash.rl.recovery_command_dataset_v1 import recovery_eligibility,compose_native_recovery_delta
from quadsmash.rl.episode_trace_v4 import unpack_public
from quadsmash.control.learned_recovery_command_v1 import command_features

SCHEMA='verified-paired-native-recovery-command-v1'
FEATURES='public794+original-frozen-base18+actual-final-history18+elapsed1'


def paired_labels(public,base,student,final,previous,active,limits,*,recovery_success,complete5):
    if type(recovery_success) is not bool or type(complete5) is not bool or (complete5 and not recovery_success):
        raise ValueError('Consistent independently verified episode outcomes required')
    eligible,elapsed=recovery_eligibility(public)
    arrays=[np.asarray(x,dtype=np.float64) for x in (base,student,final,previous)]
    base,student,final,previous=arrays;limits=np.asarray(limits,dtype=np.float64)
    active=np.asarray(active)
    if (any(x.shape!=(len(public),18) or not np.isfinite(x).all() or np.max(abs(x))>1.+1e-6 for x in arrays) or
        active.dtype!=np.bool_ or active.shape!=eligible.shape or not np.array_equal(active,eligible) or
        limits.shape!=(18,2) or not np.isfinite(limits).all() or np.any(limits[:,0]>=limits[:,1])):
        raise ValueError('Invalid native paired arrays or causal eligibility')
    if not np.allclose(previous[1:],final[:-1],atol=1e-7,rtol=0.):
        raise ValueError('Previous command is not the actual final command')
    if not np.array_equal(final[~eligible],base[~eligible]) or not np.array_equal(student[~eligible],base[~eligible]):
        raise ValueError('Commands changed outside the causal recovery window')
    total=(final-base)/2.;teacher=(final-student)/2.;learned=(student-base)/2.
    reconstructed=compose_native_recovery_delta(base,total,previous,eligible,limits)
    error=float(np.max(abs(reconstructed-final)*(limits[:,1]-limits[:,0])/2.))
    if error>5e-6:raise ValueError('Target cannot be represented by the original native interface')
    mask=eligible & recovery_success
    return dict(supervision_mask=mask,recovery_eligible=eligible,elapsed_since_recovery_s=elapsed,
        replacement_delta=np.where(mask[:,None],total,0.).astype(np.float32),
        measured_teacher_increment=teacher.astype(np.float32),measured_student_delta=learned.astype(np.float32),
        teacher_changed=(abs(final-student)*(limits[:,1]-limits[:,0])/2.).max(axis=1)>1e-5,
        reconstruction_error_max_rad=error)


def load_paired_episodes(directory):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['schema']!=SCHEMA or manifest['feature_semantics']!=FEATURES:
        raise ValueError('Explicit paired replacement-head coordinates required')
    limits=np.asarray(manifest['joint_limits_rad']);episodes=[]
    for entry in manifest['episodes']:
        path=directory/entry['data_file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['data_sha256']:
            raise ValueError('Paired episode changed')
        with np.load(path,allow_pickle=False) as z:
            public=unpack_public(z['packed_public'])
            if hashlib.sha256(public.tobytes()).hexdigest()!=entry['full_public794_sha256']:
                raise ValueError('Observation history changed')
            labels=paired_labels(public,z['original_base_command'],z['student_command'],z['final_command'],
                z['previous_final_command'],z['recovery_eligible'],limits,recovery_success=True,complete5=entry['complete5'])
            mask=labels['supervision_mask']
            if not mask.any() or not np.array_equal(mask,z['supervision_mask']):raise ValueError('Invalid positive mask')
            if not np.array_equal(labels['replacement_delta'],z['replacement_delta']):raise ValueError('Wrong total correction label')
            convert=lambda x:torch.tensor(x,dtype=torch.float32)
            b=convert(z['original_base_command'][mask]);previous=convert(z['previous_final_command'][mask])
            episodes.append(dict(case=entry['case'],outcome=entry['outcome'],
                corrected=bool(labels['teacher_changed'][mask].any()),
                features=command_features(convert(public[mask]),b,previous,convert(labels['elapsed_since_recovery_s'][mask])),
                base=b,previous=previous,target=convert(z['final_command'][mask]),delta=convert(labels['replacement_delta'][mask])))
    if not episodes or len(episodes)!=manifest['positive_episodes']:raise ValueError('No complete positive paired dataset')
    return manifest,episodes,torch.tensor(limits,dtype=torch.float32)
