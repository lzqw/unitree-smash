"""Leave one complete real episode out; never select an epoch on its labels."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

import fit_success_residual_bc_v4 as fitting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fitted-witness', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not args.output.exists()
    declaration = fitting.read(args.fitted_witness / 'PREDECLARED_CPU_FIT.json')
    assert declaration['config'] == fitting.CONFIG
    for path, digest in declaration['source_hashes'].items():
        assert fitting.sha(path) == digest, path
    source = torch.load(declaration['source_policy'], map_location='cpu', weights_only=False)
    sys.path.insert(0, str(fitting.MAIN))
    from quadsmash.rl.episode_trace_v4 import unpack_public
    episodes = []
    for entry in fitting.read(fitting.TRAIN / 'trajectory_traces/manifest.json')['saved_episodes']:
        path = fitting.TRAIN / 'trajectory_traces' / entry['data_file']
        meta = fitting.read(path.with_suffix('.json'))
        with np.load(path) as data:
            public = torch.from_numpy(unpack_public(data['packed_public']))
            targets = torch.from_numpy(data['residual_actions'].copy())
        key = meta['env_id'], meta['episode_index']
        episodes.append(dict(key=key, TRAIN=meta['source_training_index'], public=public,
            targets=targets, complete=key in fitting.SUCCESS_KEYS))
    successes = [episode for episode in episodes if episode['complete']]
    assert len(successes) == 5 and sum(len(e['public']) for e in successes) == 4000
    args.output.mkdir(parents=True)
    fitting.atomic(args.output / 'PREDECLARED_EPISODE_CV.json', dict(
        created_at=datetime.now(timezone.utc).isoformat(), folds=[dict(key=e['key'], TRAIN=e['TRAIN']) for e in successes],
        training_configuration=fitting.CONFIG, CPU_fit_witness_declaration_sha256=fitting.sha(args.fitted_witness / 'PREDECLARED_CPU_FIT.json'),
        script_sha256=fitting.sha(__file__), epoch_selection='Largest epoch satisfying TRAIN anchor RMS; held-out labels scored only afterward',
        heldout_episode_features_and_targets_excluded_from_fitting_and_anchors=True,
        primary_comparison='Mean held-out episode raw18 residual-action MSE versus unmodified measured source',
        interpretation='Action-regression generalization diagnostic on already observed TRAIN episodes; no native performance claim',
        formal_cases_read=False, new_native_execution=False, formal_acceptance=False))
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    results = []
    for index, heldout in enumerate(successes):
        torch.manual_seed(fitting.CONFIG['seed'] + index)
        actor = torch.jit.load(io.BytesIO(source['residual_actor_jit']), map_location='cpu').eval()
        normalizer = {key: value.clone() for key, value in actor.state_dict().items() if key.startswith('0.')}
        training = [e for e in successes if e['key'] != heldout['key']]
        inputs, targets = torch.cat([e['public'] for e in training]), torch.cat([e['targets'] for e in training])
        anchors = torch.cat([e['public'] for e in episodes if e['key'] != heldout['key']])
        assert inputs.shape == (3200, 794) and all(e['key'] != heldout['key'] for e in training)
        with torch.no_grad():
            source_anchor = actor(anchors).clone()
            before = float(torch.square(actor(heldout['public']) - heldout['targets']).mean())
        optimizer = torch.optim.Adam(actor.parameters(), lr=fitting.CONFIG['learning_rate'])
        selected = None
        for epoch in range(1, fitting.CONFIG['epochs'] + 1):
            order = torch.randperm(len(inputs))
            for start in range(0, len(inputs), fitting.CONFIG['batch_size']):
                batch = order[start:start + fitting.CONFIG['batch_size']]
                anchor_batch = torch.randint(len(anchors), (len(batch),))
                optimizer.zero_grad(set_to_none=True)
                loss = torch.square(actor(inputs[batch]) - targets[batch]).mean()
                loss += fitting.CONFIG['source_anchor_weight'] * torch.square(actor(anchors[anchor_batch]) - source_anchor[anchor_batch]).mean()
                assert torch.isfinite(loss)
                loss.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.); optimizer.step()
            with torch.no_grad():rms = float(torch.square(actor(anchors) - source_anchor).mean(0).sqrt().max())
            assert all(torch.equal(actor.state_dict()[key], value) for key, value in normalizer.items())
            if rms <= fitting.CONFIG['max_anchor_residual_RMS']:
                selected = dict(epoch=epoch, state=copy.deepcopy(actor.state_dict()), anchor_RMS=rms)
        assert selected is not None
        actor.load_state_dict(selected['state'])
        with torch.no_grad():
            after = float(torch.square(actor(heldout['public']) - heldout['targets']).mean())
            train_mse = float(torch.square(actor(inputs) - targets).mean())
        path = args.output / f'heldout_TRAIN{heldout["TRAIN"]:03d}_actor_jit.pt'
        torch.jit.save(actor, str(path))
        results.append(dict(heldout_key=heldout['key'], heldout_TRAIN=heldout['TRAIN'],
            training_samples=len(inputs), heldout_samples=len(heldout['public']), anchor_samples=len(anchors),
            selected_epoch=selected['epoch'], max_TRAIN_anchor_RMS=selected['anchor_RMS'],
            source_heldout_MSE=before, fitted_heldout_MSE=after, fitted_training_MSE=train_mse,
            heldout_MSE_change=after-before, actor_sha256=fitting.sha(path),
            normalizer_unchanged=True, heldout_labels_used_for_epoch_selection=False))
        print(json.dumps(results[-1]), flush=True)
    for path, digest in declaration['source_hashes'].items():assert fitting.sha(path) == digest, path
    before = sum(r['source_heldout_MSE'] for r in results) / 5
    after = sum(r['fitted_heldout_MSE'] for r in results) / 5
    fitting.atomic(args.output / 'ACTUAL_EPISODE_CV_CHECK.json', dict(
        completed_at=datetime.now(timezone.utc).isoformat(), folds=results,
        mean_source_heldout_MSE=before, mean_fitted_heldout_MSE=after,
        heldout_mean_improved=after < before, individual_episodes_improved=sum(r['heldout_MSE_change'] < 0 for r in results),
        source_witness='Measured075 stage8 CPU witness; not the final selected source',
        scope='Entire-episode separation for action regression. These are known TRAIN cases, not a formal or native score.',
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False))
    print(json.dumps(dict(mean_source_heldout_MSE=before, mean_fitted_heldout_MSE=after,
        heldout_mean_improved=after < before)), flush=True)


if __name__ == '__main__':
    main()
