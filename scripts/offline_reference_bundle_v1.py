"""Immutable candidate handoff for future from-reset PhysX replay.

This file validates/exports commands, not simulator state cloning. CPU evidence
can never set PHYSX_REFERENCE_VALID or the native success stages.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


NATIVE_STAGES=('PHYSX_REFERENCE_VALID','NATIVE_ADOPTED','NATIVE_HIT','RECOVERY_SUCCESS','COMPLETE5')


def require_final_trajectory_admission(planner_record, physical_result, *,
                                       contact_limit_m=.003):
    """Fail closed until a SELECTED plan has both physical and ball-side proof.

    An individual 18D evaluator callback may accept a trajectory which later
    fails the actual terminal collision and return checks. Such an early
    result is not a complete offline teacher reference.
    """
    if not np.isfinite(contact_limit_m) or not 0 < contact_limit_m <= .003:
        raise ValueError('Cannot relax the final 3mm contact gate')
    if (not isinstance(planner_record, dict) or
            planner_record.get('accepted') is not True or
            not isinstance(physical_result, dict) or
            physical_result.get('accepted') is not True):
        raise ValueError("Selected planner and physical prefix must both accept")
    if planner_record.get('physical_evaluator') is not True:
        raise ValueError("Missing full 18D physical evaluator")
    flags=planner_record.get('final_impact_checks')
    required=('contact_feasible','legal_return_predicted',
              'attack5_predicted','target_landing_predicted')
    if not isinstance(flags,dict) or not all(flags.get(k) is True for k in required):
        raise ValueError("Final physical-terminal analytic impact not verified")
    error=planner_record.get('final_contact_position_error_m')
    if (not isinstance(error,(int,float,np.integer,np.floating)) or
            not np.isfinite(error) or error<0 or error>contact_limit_m):
        raise ValueError("Final 3mm contact gate not verified")
    if (physical_result.get('leg_reference')!='causal_frozen_base' or
            not isinstance(physical_result.get('replayed_target_sequence_rad'),list)):
        raise ValueError("Selected full 18D executable prefix missing")
    return True


def write_candidate(directory, *, final_targets_rad, prior_final_target_rad,
                    state_q, state_v, control_tick, contact_index, limits,
                    input_hashes, controller_parameters, prediction_contract,
                    cpu_prediction_valid, state_source, planner_record=None, physical_result=None):
    directory=Path(directory)
    arrays={k:np.array(v,dtype=np.float64,copy=True) for k,v in dict(
        final_targets_rad=final_targets_rad,prior_final_target_rad=prior_final_target_rad,
        state_q=state_q,state_v=state_v,limits=limits).items()}
    q=arrays['final_targets_rad'];bounds=arrays['limits']
    if (q.ndim!=2 or q.shape[1]!=18 or len(q)<1 or bounds.shape!=(18,2)
            or arrays['prior_final_target_rad'].shape!=(18,) or arrays['state_q'].shape!=(25,)
            or arrays['state_v'].shape!=(24,) or not all(np.isfinite(v).all() for v in arrays.values())
            or np.any(bounds[:,0]>=bounds[:,1]) or np.any(q<bounds[:,0]-1e-7) or np.any(q>bounds[:,1]+1e-7)):
        raise ValueError('Complete finite original-bounded18D commands and measured state required')
    if abs(np.linalg.norm(arrays['state_q'][3:7])-1.)>1e-6:
        raise ValueError('Native generalized state must contain a unit quaternion')
    if type(control_tick) is not int or control_tick<0 or type(contact_index) is not int or contact_index!=len(q):
        raise ValueError('Complete prefix must terminate at its declared contact event')
    if not input_hashes or any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in input_hashes.values()):
        raise ValueError('Pin every input with SHA256')
    if state_source not in ('native','public_reconstruction'):
        raise ValueError('Explicit measured-state information boundary required')
    admission=None
    if cpu_prediction_valid:
        require_final_trajectory_admission(planner_record,physical_result)
        verified=np.asarray(physical_result['replayed_target_sequence_rad'],dtype=float)
        if verified.shape!=q.shape or not np.isfinite(verified).all() or np.max(abs(verified-q))>2e-6:
            raise ValueError('Export commands differ from the finally admitted prefix')
        admission=dict(final_impact_checks=planner_record['final_impact_checks'],
            final_contact_position_error_m=planner_record['final_contact_position_error_m'],
            full18_target_sha256=hashlib.sha256(q.astype('<f8').tobytes()).hexdigest())
    # Serialize first, so invalid metadata cannot leave a seemingly complete bundle.
    metadata=dict(schema='offline_reference_candidate_v1',control_tick=control_tick,contact_index=contact_index,
        control_dt_s=.01,requires_replay_from_frozen_initial_conditions=True,
        state_source=state_source,input_hashes=input_hashes,controller_parameters=controller_parameters,
        final_cpu_admission=admission,
        prediction_contract=prediction_contract,FULLBODY_PREDICTION_VALID=bool(cpu_prediction_valid),
        **{stage:'NOT_RUN' for stage in NATIVE_STAGES})
    json.dumps(metadata,allow_nan=False)
    directory.mkdir(parents=True,exist_ok=False)
    path=directory/'commands_and_state.npz';np.savez_compressed(path,**arrays)
    metadata['archive_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (directory/'CANDIDATE.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    return metadata


def load_candidate(directory):
    directory=Path(directory);meta=json.loads((directory/'CANDIDATE.json').read_text())
    archive=directory/'commands_and_state.npz'
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=meta['archive_sha256']:
        raise ValueError('Reference archive digest mismatch')
    if meta.get('schema')!='offline_reference_candidate_v1' or any(meta.get(k)!='NOT_RUN' for k in NATIVE_STAGES):
        raise ValueError('A candidate bundle cannot self-certify native results')
    with np.load(archive,allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
    return meta,arrays


def verify_replay_handoff(directory, *, current_q,current_v,previous_final_target_rad,input_hashes):
    """Call only after from-reset replay; this check does not restore a simulator."""
    meta,arrays=load_candidate(directory)
    if input_hashes!=meta['input_hashes']:raise ValueError('Frozen inputs drifted')
    if meta['state_source']!='native':raise ValueError('Native handoff needs an independently recorded native state')
    for name,current,tolerance in [('state_q',current_q,1e-5),('state_v',current_v,1e-5),
                                    ('prior_final_target_rad',previous_final_target_rad,2e-6)]:
        current=np.asarray(current,dtype=float)
        if current.shape!=arrays[name].shape or not np.isfinite(current).all() or np.max(abs(current-arrays[name]))>tolerance:
            raise ValueError('Replay handoff mismatch: '+name)
    return arrays['final_targets_rad'].copy()
