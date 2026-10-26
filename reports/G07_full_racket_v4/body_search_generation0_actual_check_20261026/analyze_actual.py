"""Compare actual equal-parameter references, not a formal or cold-policy score."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent/'unitree_smash'
HERE = Path(__file__).resolve().parent
REPORTS = MAIN/'reports/G07_full_racket_v4'
NATIVE = REPORTS/'fixed_actor_body_cem1280_gen2_01'
SOURCE = REPORTS/'residual_ppo_centered_placement075_n2048_probe8_01'
COLD = REPORTS/'residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01'
sys.path.insert(0,str(MAIN))
from quadsmash.control.full_racket_residual_v4 import make_rally_controller
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,value):
    p = HERE/name; assert not p.exists(),str(p)
    p.write_text(json.dumps(value,indent=2)+'\n')


def main():
    assert not (HERE/'RESULT.json').exists()
    save('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        actual_native_pid=1794497,wait_for='Completed native generation0 population and metrics, never a partial subset.',
        source_sha256=sha(Path(__file__)),CPU_only=True,new_native_execution=False,
        scope=__doc__,formal_cases_read=False,formal_acceptance=False))
    while not ((NATIVE/'population_000.pt').is_file() and (NATIVE/'metrics.json').is_file()):
        proc = Path('/proc/1794497')
        assert proc.is_dir() and b'search_full_racket_fixed_actor_body_v4.py' in (proc/'cmdline').read_bytes()
        assert (proc/'stat').read_text().split(') ',1)[1].split()[0] not in ('Z','X')
        assert not (NATIVE/'error.txt').exists(), 'Preserve native error; no automatic retry'
        time.sleep(5)
    paths = [Path(__file__),NATIVE/'population_000.pt',NATIVE/'episodes_000.json',
        NATIVE/'public_and_actions_000.npz',NATIVE/'all_first_episodes_000.npz',
        NATIVE/'initial_ground_states_000.json',NATIVE/'INPUTS.json',NATIVE/'geometry.json',
        SOURCE/'policy.pt',SOURCE/'trajectory_traces/manifest.json',COLD/'all_first_episodes.npz',
        COLD/'public_and_actions_first2.npz',COLD/'episodes.json']
    paths += sorted(MAIN.glob('quadsmash/**/*.py'))
    fingerprints = {str(path):sha(path) for path in paths}
    population = torch.load(NATIVE/'population_000.pt',map_location='cpu',weights_only=False)
    source = torch.load(SOURCE/'policy.pt',map_location='cpu',weights_only=False)
    assert torch.equal(population['parameters'][0],source['parameters'])
    assert torch.equal(population['parameters'][1],source['parameters'])
    protocol = read(MAIN/'configs/g07_full_racket_eval_v4_r3.json');protocol['_root'] = str(MAIN)
    rows = read(NATIVE/'episodes_000.json');assert len(rows) == 1280
    summaries = [summarize_candidate(rows[i*128:(i+1)*128],protocol) for i in range(10)]
    assert summaries == population['summaries']
    references = []
    for candidate in range(2):
        group = rows[candidate*128:(candidate+1)*128]
        references.append(dict(candidate=candidate,actual_summary=summaries[candidate],
            body_landing=sum(row['jump']['body_landing'] for row in group),
            stable_recovery=sum(row['jump']['stable_recovery'] for row in group),
            native_env_ids=[row['env_id'] for row in group],
            source_training_indices=[row['source_training_index'] for row in group]))
    with np.load(NATIVE/'public_and_actions_000.npz') as arrays:
        obs = arrays['observations'][:,:4].copy()
        actions = arrays['actions'][:,:4].copy()
        valid = arrays['valid_first_episode'][:,:4].astype(bool).copy()
        assert arrays['native_env_ids'][:4].tolist() == [0,1,128,129]
    controller = make_rally_controller(source,4,'cpu')
    provenance = read(SOURCE/'trajectory_traces/manifest.json')['provenance']
    default = torch.tensor(provenance['default_joint_position_rad'],dtype=torch.float32)
    limits = torch.tensor(provenance['joint_limits_rad'],dtype=torch.float32)
    previous = None;maximum = 0.;per_stream = np.zeros(4)
    for public,actual,keep in zip(obs,actions,valid):
        tensor = torch.from_numpy(public)
        if previous is not None:
            ids = torch.where(tensor[:,775] < previous-1e-5)[0]
            if len(ids):controller.reset(ids)
        predicted = controller.action({'policy':tensor},default,limits).numpy()
        for i in range(4):
            if keep[i]:per_stream[i] = max(per_stream[i],float(abs(predicted[i]-actual[i]).max()))
        previous = tensor[:,775].clone()
    maximum = float(per_stream.max())
    replay = dict(actual_saved_streams=4,actual_valid_public_rows=int(valid.sum()),
        original_source_controller_CPU_vs_actual_GPU_max_normalized_action_error=maximum,
        per_stream_max_error=per_stream.tolist(),tolerance=5e-4,passed=maximum<=5e-4,
        meaning='Real new reference commands replayed through the original source controller; CPU/GPU floating tolerance, no same-trajectory guarantee.')
    old_rows = read(COLD/'episodes.json')
    old_summary = dict(body_landing=sum(row['jump']['body_landing'] for row in old_rows),
        stable_recovery=sum(row['jump']['stable_recovery'] for row in old_rows))
    with np.load(COLD/'public_and_actions_first2.npz') as arrays:
        old_public = arrays['observations'].copy();old_actions = arrays['actions'].copy()
        old_public_valid = arrays['valid_first_episode'].astype(bool).copy()
    before_launch = obs[:,:,775] < 2.
    pair_errors = []
    for group,ids in [('new_reference0',[0,1]),('new_reference1',[2,3])]:
        for case,stream in enumerate(ids):
            n = min(len(obs),len(old_public))
            keep = valid[:n,stream] & old_public_valid[:n,case]
            pre = keep & before_launch[:n,stream]
            difference = np.abs(obs[:n,stream]-old_public[:n,case]).max(1)
            command_error = np.abs(actions[:n,stream]-old_actions[:n,case]).max(1)
            pair_errors.append(dict(group=group,TRAIN_index=case,common_valid_samples=int(keep.sum()),
                initial794D_max_difference=float(difference[0]),
                max_public_difference_before2s=float(difference[pre].max()),
                max_action_difference_before2s=float(command_error[pre].max()),
                first_common_valid_public_difference_over1e4_s=(float(obs[np.flatnonzero(keep & (difference>1e-4))[0],stream,775])
                    if (keep & (difference>1e-4)).any() else None)))
    names = read(NATIVE/'geometry.json')['body_names'];base = names.index('base')
    with np.load(NATIVE/'all_first_episodes_000.npz') as arrays:
        new_z = arrays['link_poses'][:,[0,1,128,129],base,2].copy()
        new_t = arrays['episode_clock_s'][:,[0,1,128,129]].copy()
        new_keep = arrays['valid_first_episode'][:,[0,1,128,129]].astype(bool).copy()
    with np.load(COLD/'all_first_episodes.npz') as arrays:
        old_z = arrays['link_poses'][:,:2,base,2].copy()
        old_t = arrays['episode_clock_s'][:,:2].copy()
        old_keep = arrays['valid_first_episode'][:,:2].astype(bool).copy()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(12,4),sharey=True)
    for case,ax in enumerate(axes):
        ax.plot(old_t[:,case],np.where(old_keep[:,case],old_z[:,case],np.nan),label='Prior cold128 source')
        for candidate in range(2):
            k=case+2*candidate
            ax.plot(new_t[:,k],np.where(new_keep[:,k],new_z[:,k],np.nan),label=f'New native reference {candidate}')
        ax.set(xlabel='Actual episode time (s)',title=f'TRAIN {case}: same policy parameters',xlim=(0,4.2))
        ax.grid(alpha=.3)
    axes[0].set_ylabel('Recorded robot base height (m)');axes[1].legend(fontsize=8)
    fig.suptitle('Actual first-episode records; curves stop at termination. Causes unproven.')
    fig.tight_layout();fig.savefig(HERE/'Actual_reference_base_height_20261026.png',dpi=160);plt.close(fig)
    assert all(sha(Path(name)) == digest for name,digest in fingerprints.items())
    save('RESULT.json',dict(checked_at=datetime.now(timezone.utc).isoformat(),source_sha256=fingerprints,
        all1280_rows_independently_recomputed=True,summaries=summaries,actual_equal_parameter_references=references,
        original_cold128_body_summary=old_summary,original_source_action_replay=replay,
        actual_public_and_action_comparisons=pair_errors,
        interpretation='Reference outcomes vary across actual batch/instance placements. This changes uncertainty about CEM ranking; no physical cause is established. Original128 cold reevaluation remains necessary.',
        no_new_native_execution=True,no_formal_cases_read=True,formal_acceptance=False,goal_complete=False))
    print(json.dumps(dict(actual_generation0_complete=True,reference_summaries=references,
        source_action_replay=replay)),flush=True)


if __name__ == '__main__':main()
