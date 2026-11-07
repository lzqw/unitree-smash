"""Compare actual-command predictions with aligned physics-frame normal forces.

Accepts an immutable executed prefix of a live job, not a complete-run claim.
Native forces are sampled normal forces, not measured total ground forces.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.audit_native_frictional_recovery_v1 import state_at
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts import predict_frictional_spheres_v1 as prediction


def interval_indices(frames, tick, substeps=160):
    """State at tick precedes its new command; forces use following frames."""
    start = int(np.searchsorted(frames, tick*substeps))
    stop = start+substeps
    if stop >= len(frames) or not np.array_equal(
            frames[start:stop+1], np.arange(tick*substeps, (tick+1)*substeps+1)):
        raise ValueError('Missing complete physical interval')
    return start, stop, slice(start+1, stop+1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'prefix', 'bundle', 'sphere-audit', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise ValueError('Preserve previous audit')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    meta = json.loads((a.run/'contact_microtrace.json').read_text())
    prefix = json.loads(a.prefix.read_text())
    trace_path = a.prefix.parent/prefix['trace_file']
    micro_path = a.run/'contact_microtrace.npz'
    if (sha(trace_path) != prefix['trace_sha256'] or sha(micro_path) != meta['data_sha256']
            or not meta['complete_window'] or meta['physics_dt_s'] != .01/160
            or prefix['case'] != meta['lane'] or prefix['valid_expert_labels']):
        raise ValueError('Require complete hashed physics window and executed prefix')
    with np.load(trace_path, allow_pickle=False) as z:
        trace = {k: z[k].copy() for k in z.files}
    with np.load(micro_path, allow_pickle=False) as z:
        micro = {k: z[k].copy() for k in z.files}
    manifest = json.loads((a.bundle/'manifest.json').read_text())
    c = build_controller(manifest['configuration']); m = c.qp.model
    if meta['joint_names'] != m.joint_order or meta['foot_names'] != m.feet:
        raise ValueError('Different joint or foot ordering')
    spheres = json.loads(a.sphere_audit.read_text())['spheres']
    geometry = AuthoredGroundClearance(ROOT/'configs/go2_piper_authored_ground_geometry_v1.json', m, margin_m=0.)
    frames = micro['physics_frame']; dt = meta['physics_dt_s']
    rows = []
    ticks = range(int(frames[0]//160), int(frames[-1]//160))
    original = prediction.frictional_velocity_step
    for tick in ticks:
        i = tick-prefix['start_tick']
        if (not 0 <= i < len(trace['q'])-1 or trace['control_tick'][i] != tick
                or trace['control_tick'][i+1] != tick+1):
            raise ValueError('Missing executed endpoint')
        start, stop, interval = interval_indices(frames, tick)
        state, actual = state_at(m, trace, i), state_at(m, trace, i+1)
        target = trace['final_target_rad'][i].copy()
        for j, endpoint in [(start, state), (stop, actual)]:
            if (np.max(abs(micro['joint_position_rad'][j]-endpoint.joint_position)) > 2e-6
                    or np.max(abs(micro['joint_velocity_rad_s'][j]-endpoint.joint_velocity)) > 2e-5
                    or not np.array_equal(micro['observed_foot_contact'][j], endpoint.contacts)):
                raise ValueError('Physics/control state alignment failed')
        target_error = float(np.max(abs(micro['joint_target_buffer_rad'][interval]-target)))
        if target_error > 2e-6 or np.unique(micro['episode_index'][start:stop+1]).size != 1:
            raise ValueError('Command changed within interval or episode reset')
        normal = np.asarray(micro['foot_normal_force_world_n'][interval], float)
        native_flags = micro['observed_foot_contact'][interval]
        observed = state.contacts.copy(); reconstructed = []
        for force in normal:
            observed = np.where(observed, np.linalg.norm(force, axis=1) > .5,
                                np.linalg.norm(force, axis=1) > 1.)
            reconstructed.append(observed.copy())
        if not np.array_equal(reconstructed, native_flags):
            raise ValueError('Recorded flag is not described by force-norm hysteresis')
        model_force = []; model_flags = []; observed = state.contacts.copy()
        def observe(*args, **kwargs):
            nonlocal observed
            answer = original(*args, **kwargs)
            force = answer['impulse'].reshape(4, 3)/args[8]
            model_force.append(force.copy())
            observed = np.where(observed, abs(force[:, 2]) > .5, abs(force[:, 2]) > 1.)
            model_flags.append(observed.copy())
            return answer
        prediction.frictional_velocity_step = observe
        try:
            pred = prediction.predict_frictional_spheres(m, c.qp.interface, state, target,
                                                        spheres, substeps=160, geometry=geometry)
        finally:
            prediction.frictional_velocity_step = original
        model_force = np.asarray(model_force); model_flags = np.asarray(model_flags)
        if model_force.shape != (160, 4, 3):
            raise ValueError('Unexpected prediction duration')
        mean_native = normal.mean(axis=0)[:, 2]; mean_model = model_force.mean(axis=0)[:, 2]
        difference = pin.difference(m.model, actual.q, pred['state'].q)
        rows.append(dict(tick=tick, mode=prefix['records'][i]['mode'],
            actual_target_buffer_error_rad=target_error,
            mean_native_normal_z_n=mean_native.tolist(), mean_model_normal_z_n=mean_model.tolist(),
            mean_normal_error_n=(mean_model-mean_native).tolist(),
            native_normal_impulse_z_ns=(normal[:, :, 2].sum(axis=0)*dt).tolist(),
            model_normal_impulse_z_ns=(model_force[:, :, 2].sum(axis=0)*dt).tolist(),
            native_endpoint_normal_z_n=normal[-1, :, 2].tolist(),
            native_contact_fraction=native_flags.mean(axis=0).tolist(),
            model_contact_fraction=model_flags.mean(axis=0).tolist(),
            native_contact_switches=np.count_nonzero(np.diff(np.vstack([state.contacts, native_flags]), axis=0), axis=0).tolist(),
            model_contact_switches=np.count_nonzero(np.diff(np.vstack([state.contacts, model_flags]), axis=0), axis=0).tolist(),
            mismatching_physics_frames_per_foot=np.count_nonzero(model_flags != native_flags, axis=0).tolist(),
            joint_position_error_rad=difference[m.v_indices].tolist(),
            joint_velocity_error_rad_s=(pred['state'].v[m.v_indices]-actual.v[m.v_indices]).tolist(),
            diagnostics=pred['diagnostics']))
    errors = np.asarray([r['mean_normal_error_n'] for r in rows])
    vel = np.asarray([r['joint_velocity_error_rad_s'] for r in rows])
    files = [Path(__file__), a.prefix, trace_path, micro_path, a.run/'contact_microtrace.json',
             a.bundle/'manifest.json', a.sphere_audit, ROOT/'quadsmash/tasks/stand/coupled_env.py',
             ROOT/'configs/go2_piper_authored_ground_geometry_v1.json']
    files += [Path(v.__file__) for k, v in list(sys.modules.items())
              if k.startswith('scripts.') and getattr(v, '__file__', None)]
    result = dict(scope=__doc__, case=prefix['case'], foot_order=m.feet,
        interval_s=[frames[0]*dt, frames[-1]*dt], intervals=len(rows), rows=rows,
        mean_normal_force_rmse_per_foot_n=np.sqrt(np.mean(errors**2, axis=0)).tolist(),
        mean_normal_force_bias_per_foot_n=errors.mean(axis=0).tolist(),
        joint_velocity_rmse_rad_s=dict(zip(m.joint_order, np.sqrt(np.mean(vel**2, axis=0)).tolist())),
        original_force_hysteresis_reconstructed_exactly=True,
        source_sha256={str(p.resolve()): sha(p) for p in files},
        job_completion_required=False, full_run_admission=False, native_rerun=False,
        valid_expert_labels=False,
        limitation='20 short transitions in one case, mostly student fallback. Contact sensor normal force only; no measured tangential force or full recovery claim.')
    if sha(trace_path) != prefix['trace_sha256'] or sha(micro_path) != meta['data_sha256']:
        raise ValueError('Source evidence changed during audit')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ['intervals', 'mean_normal_force_rmse_per_foot_n', 'mean_normal_force_bias_per_foot_n', 'joint_velocity_rmse_rad_s']}))


if __name__ == '__main__':
    main()
