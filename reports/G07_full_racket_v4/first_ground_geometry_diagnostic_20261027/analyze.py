"""Separate terminal geometry from force attribution using actual native records."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from quadsmash.physics.full_racket_v4 import load_racket, parts, string_strips


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def up_row(poses):
    q = np.asarray(poses)[..., 3:7]
    norms = np.linalg.norm(q, axis=-1)
    assert np.max(abs(norms - 1)) < 1e-5
    w, x, y, z = np.moveaxis(q / norms[..., None], -1, 0)
    return np.stack((2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)), -1)


def racket_clearance(poses, config):
    rz, height = up_row(poses), np.asarray(poses)[..., 2]
    minima, names = [], []
    for component in parts(config):
        if not component['collision']:
            continue
        center = height + rz @ component['center']
        if component['kind'] == 'box':
            extent = abs(rz) @ (component['size']/2)
        elif component['kind'] in ['capsule', 'cylinder']:
            along = abs(rz @ component['axis']).clip(0, 1)
            radial = np.ones_like(along) if component['kind'] == 'capsule' else np.sqrt(1 - along**2)
            extent = component['length']/2*along + component['radius']*radial
        elif component['kind'] == 'string_prism':
            extent = np.maximum.reduce([-rz @ np.asarray(s['center']) + abs(rz) @ (np.asarray(s['size'])/2)
                                        for s in string_strips(config)])
        else:
            raise ValueError(component['kind'])
        minima.append(center - extent)
        names.append(component['name'])
    values = np.stack(minima, -1)
    return values.min(-1), np.asarray(names)[values.argmin(-1)]


def mesh_minima(poses, vertices):
    rz = up_row(poses)
    result = np.full(np.asarray(poses).shape[:-1], np.inf)
    for body, points in vertices.items():
        result[..., body] = poses[..., body, 2] + (rz[..., body, :] @ points.T).min(-1)
    return result


def main():
    assert not (HERE / 'RESULT.json').exists()
    reports = ROOT / 'reports/G07_full_racket_v4'
    run = reports / 'gain_body8_sequential_n128_01'
    rows = read(run / 'episodes_000.json')
    geometry = read(run / 'geometry.json')
    names = geometry['body_names']
    racket_id = names.index('Racket')
    config = load_racket()
    with np.load(run / 'geometry.npz') as meshes:
        vertices = {body: np.concatenate([meshes[m['key'] + '_vertices'] for m in geometry['meshes']
                                         if m['body'] == body]) for body in range(len(names))
                    if any(m['body'] == body for m in geometry['meshes'])}
    legs = [i for i, name in enumerate(names) if name.endswith(('_hip', '_thigh', '_calf'))]
    arms = [names.index(name) for name in ['base_link', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6']]
    monitored = [names.index('base'), *legs, *arms, names.index('MountPlate'), racket_id]
    terminal = np.asarray([np.column_stack((row['terminal_native_state']['body_link_positions_m'],
        row['terminal_native_state']['body_link_quaternions_wxyz'])) for row in rows])
    minimum = mesh_minima(terminal, vertices)
    exact_racket, limiting_part = racket_clearance(terminal[:, racket_id], config)
    tilt = np.degrees(np.arccos(up_row(terminal[:, 0])[:, 2].clip(-1, 1)))
    nearest = [names[monitored[i]] for i in minimum[:, monitored].argmin(-1)]
    failures = [i for i, row in enumerate(rows) if row['jump']['body_landing']]
    tally = {name: sum(nearest[i] == name for i in failures) for name in sorted(set(nearest))}
    all_records = [dict(TRAIN_index=i, actual_episode_index=rows[i]['episode_index'],
        terminal_control_step_time_s=rows[i]['terminal_native_state']['time_s'],
        aggregate_nonfoot_ground_flag=rows[i]['jump']['body_landing'], nearest_monitored_visual_body=nearest[i],
        nearest_visual_clearance_m=float(minimum[i, monitored].min()),
        racket_collision_clearance_m=float(exact_racket[i]), limiting_racket_collision_part=str(limiting_part[i]),
        base_visual_clearance_m=float(minimum[i, 0]), leg_visual_clearance_m=float(minimum[i, legs].min()),
        arm_visual_clearance_m=float(minimum[i, arms].min()), root_tilt_deg=float(tilt[i])) for i in range(128)]
    air = [i for i, row in enumerate(rows) if row['airborne_string_hit']]
    assert air == [69, 100, 112, 123]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.3), constrained_layout=True)
    prefix_records = []
    with np.load(run / 'all_first_episodes_000.npz') as stream:
        for axis, lane in zip(axes.ravel(), air):
            valid = stream['valid_first_episode'][:, lane]
            clock = stream['episode_clock_s'][:, lane]
            ids = np.flatnonzero(valid & (clock >= 2.4))
            assert len(ids) and np.all(np.diff(ids) == 1)
            poses = stream['link_poses'][ids, lane].astype(float)
            times = clock[ids]
            clearances = mesh_minima(poses, vertices)
            racket, _ = racket_clearance(poses[:, racket_id], config)
            angles = np.degrees(np.arccos(up_row(poses[:, 0])[:, 2].clip(-1, 1)))
            columns = np.column_stack((times, clearances[:, 0], clearances[:, legs].min(-1),
                clearances[:, arms].min(-1), racket, angles))
            with (HERE / f'TRAIN{lane}_actual_prefix.csv').open('w') as handle:
                writer = csv.writer(handle)
                writer.writerow(['time_s', 'base_visual_clearance_m', 'leg_visual_clearance_m',
                                 'arm_visual_clearance_m', 'racket_collision_clearance_m', 'root_tilt_deg'])
                writer.writerows(columns)
            prefix_records.append(dict(TRAIN_index=lane, actual_episode_index=rows[lane]['episode_index'],
                last_actual50Hz_time_s=float(times[-1]), root_tilt_at_last_sample_deg=float(angles[-1]),
                racket_collision_clearance_at_last_sample_m=float(racket[-1]),
                first_90deg_tilt_sample_s=float(times[np.flatnonzero(angles >= 90)[0]]) if (angles >= 90).any() else None,
                terminal_control_step_time_s=rows[lane]['terminal_native_state']['time_s']))
            for values, label, color in [(clearances[:, 0], 'Base mesh', '#225ea8'),
                (clearances[:, legs].min(-1), 'Non-foot leg meshes', '#238b45'),
                (clearances[:, arms].min(-1), 'Arm meshes', '#756bb1'),
                (racket, 'Racket collision primitives', '#cb181d')]:
                axis.plot(times, values, label=label, color=color, lw=1.5)
            axis.axhline(0, color='black', lw=.7)
            axis.axvline(rows[lane]['contact_time_s'], color='#969696', ls=':', lw=1)
            axis.axvline(rows[lane]['terminal_native_state']['time_s'], color='#cb181d', ls='--', lw=1)
            axis.set(title=f'TRAIN{lane}: actual first-episode prefix', xlabel='Episode time (s)',
                     ylabel='Geometric lowest height (m)', ylim=(-.035, 1.05))
            secondary = axis.twinx()
            secondary.plot(times, angles, color='#e08214', ls='--', lw=1.1, label='Root tilt')
            secondary.set(ylabel='Root tilt (deg)', ylim=(0, 180))
            axis.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
    fig.suptitle('Actual native poses: clearance and tilt before the first terminal control step\n'
                 'Mesh proximity is not a contact-force attribution; dotted line = racket contact', fontsize=11)
    fig.savefig(HERE / 'ACTUAL_GROUND_CLEARANCE_AND_TILT.png', dpi=160)
    plt.close(fig)
    sources = [run / name for name in ['episodes_000.json', 'all_first_episodes_000.npz',
        'geometry.json', 'geometry.npz', 'contacts_000.json']]
    sources += [ROOT / 'quadsmash/tasks/stand/jump_env.py', ROOT / 'quadsmash/physics/full_racket_v4.py',
                ROOT / 'configs/full_racket_v4_r3.json', Path(__file__)]
    held = read(reports / 'origin0_residual_after_gain_body_queue01/PLAN.json')['source_sha256']
    assert all(sha(Path(path)) == digest for path, digest in held.items())
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_search_reference_rows=128,
        body_ground_flag_count=len(failures), nearest_monitored_visual_body_at_flagged_terminal=tally,
        all128_terminal_geometry=all_records, four_airborne_prefix_observations=prefix_records,
        bodies_without_exported_visual_geometry=[name for i, name in enumerate(names) if i not in vertices],
        measured_sensor_definition='Aggregate filtered ground forces from Go2 base/hip/thigh/calf, Piper links, MountPlate and Racket; feet excluded. Exact sensor-resolved first-contact force histories were not logged.',
        limitations=['Visual meshes for base/legs/arm approximate geometry and do not identify actual PhysX force-producing links.',
            'Racket primitive support uses the unchanged physical configuration and actual poses, not a new physical run.',
            'Terminal states are the final control-step states before reset, not the exact first ground-contact substep.',
            'Prefix curves use actual50Hz samples. Contacts between samples and causal order need sensor-resolved native evidence.',
            'No success, recovery, hardware, physical model, terminal condition or original criterion changed.'],
        source_sha256={str(path): sha(path) for path in sources}, held482_unchanged=True,
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False)
    (HERE / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(flagged=len(failures), nearest=tally, air_prefixes=prefix_records)), flush=True)


if __name__ == '__main__':
    main()
