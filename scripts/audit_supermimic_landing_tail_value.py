"""Check the same-policy critic throughout measured post-landing stability maintenance."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.audit_supermimic_recovery_critic import ranking
from quadsmash.rl.supermimic_evidence_binding_v1 import complete_native_rtg
from quadsmash.rl.supermimic_horizon_learner_v2 import optimizer_step
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from quadsmash.rl.supermimic_v1 import MLP


def relative_order_loss(predicted, measured):
    """Compare different episodes at the same landing offset, in native value units."""
    i, j = torch.triu_indices(len(measured), len(measured), offset=1)
    order = torch.sign(measured[i] - measured[j])
    keep = order != 0
    scale = measured.std(unbiased=False).clamp_min(1.)
    margin = order * (predicted[i] - predicted[j]) / scale
    return scale.square() * F.softplus(-margin[keep]).mean() if keep.any() else predicted.sum() * 0


def audit(run, frozen, checkpoint, snapshots, diagnostic_weights=None):
    bank = torch.load(frozen/'native_critic_labels.pt', weights_only=True, map_location='cpu')
    archive_path = frozen/'complete_episodes.pt'
    with archive_path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    archive = torch.load(archive_path, weights_only=True, map_location='cpu')
    tail = torch.load(snapshots, weights_only=True, map_location='cpu')
    ck = torch.load(checkpoint, mmap=True, weights_only=False, map_location='cpu')
    actor = tensor_digest(ck['actor'].items())
    records = json.loads((frozen/'episodes_summary.json').read_text())
    receipt = archive['receipt']
    if (len({digest, tail['source_sha256'], bank['source_sha256']}) != 1
            or len({actor, receipt['actor_sha256'], bank['actor_sha256'], tail['actor_sha256'],
                    tensor_digest(bank['label_actor_state'].items())}) != 1
            or archive['policy_id'] != bank['policy_id'] or tail['reproduced_sparse_inputs'] <= 0):
        raise ValueError('Critic, labels, raw archive and reconstructed inputs must identify one frozen policy')
    records_by_lane = {r['env_id']: r for r in records}
    if (len(records_by_lane) != len(receipt['episode_ids']) or any(
            r['episode_index'] != receipt['episode_ids'][lane] or r['actor_sha256'] != actor
            or r['policy_execution'] != bank['policy_execution'] or r['policy_id'] != bank['policy_id']
            for lane, r in records_by_lane.items())):
        raise ValueError('Terminal label or execution identity differs')
    held = torch.tensor(json.loads((run/'config.json').read_text())['wm_holdout_lanes'])
    data = archive['tensors']
    gamma, = {r['native_return_gamma'] for r in records}
    rtg, complete = complete_native_rtg(data, gamma)
    def labels(identities):
        lane, ep, tick = torch.tensor(identities).T
        if not complete[tick,lane].all() or not torch.equal(data['episode'][tick,lane], ep):
            raise ValueError('Incomplete or mismatched episode target')
        return lane, tick, rtg[tick,lane].float()

    lanes, ticks, y = labels(tail['identity'])
    general_lanes, _, gy = labels(bank['identity'])
    torch.testing.assert_close(gy, bank['rtg'], atol=0, rtol=0)
    offsets = ticks - torch.tensor([tail['landing_by_lane'][lane] for lane in lanes.tolist()])
    qualified = torch.tensor([records_by_lane[l]['qualified_airborne_string_contact'] for l in lanes.tolist()])
    if not qualified.all() or len(set(tail['identity'])) != len(lanes):
        raise ValueError('Unexpected landing sample')
    gated = tail['recovery'].bool()
    valid = torch.isin(lanes, held) & gated
    gheld = torch.isin(general_lanes, held)
    train = ~torch.isin(lanes, held) & gated
    groups = [torch.where(train & (offsets == off))[0] for off in tail['offsets']]
    if any(len(ids)<2 for ids in groups):
        raise ValueError('Insufficient distinct training lanes for within-phase ranking')
    success = torch.tensor([records_by_lane[l]['recovery_success'] for l in lanes.tolist()], dtype=torch.bool)
    x = tail['inputs']
    gx = bank['inputs']
    gen = torch.Generator().manual_seed(221121)
    gtrain = torch.where(~gheld)[0]
    batches = [gtrain[torch.randint(len(gtrain),(128,),generator=gen)] for _ in range(8)]
    general_report, phase_report = {}, {}
    eligible = sum(l in held.tolist() for l in tail['landing_by_lane'])
    common = set(lanes[valid].tolist())
    for off in tail['offsets']:
        common.intersection_update(lanes[valid & (offsets==off)].tolist())
    cohort = torch.isin(lanes, torch.tensor(sorted(common),dtype=torch.long))
    for name in ('deployed', 'late_rtg', 'late_rtg_and_order'):
        model = MLP(x.shape[-1], (512,256,128), 1)
        model.load_state_dict(ck['critic'])
        if name != 'deployed':
            opt = torch.optim.Adam(model.parameters())
            opt.load_state_dict(copy.deepcopy(ck['critic_optimizer']))
            for batch in batches:
                phase_loss = []
                for ids in groups:
                    pred = model(x[ids]).flatten()
                    loss = (pred-y[ids]).square().mean()
                    if name == 'late_rtg_and_order':
                        loss = loss + .5*relative_order_loss(pred, y[ids])
                    phase_loss.append(loss)
                loss = .5*(model(gx[batch]).flatten()-gy[batch]).square().mean()+.5*torch.stack(phase_loss).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                optimizer_step(opt,model.parameters())
        if diagnostic_weights is not None and name == 'late_rtg_and_order':
            package = dict(critic=model.state_dict(), base_critic_sha256=tensor_digest(ck['critic'].items()),
                           actor_sha256=actor, archive_sha256=digest, held_lanes=sorted(held.tolist()),
                           scope='landing_tail_order_diagnostic', deployable=False)
            with diagnostic_weights.open('xb') as stream:
                torch.save(package, stream)
        with torch.no_grad():
            pred = model(x).flatten()
            general = torch.cat([model(t).flatten() for t in gx.split(128)])
        general_report[name] = {}
        for head in ('approach','recovery'):
            mask = gheld & bank['head_masks'][head].bool()
            general_report[name][head] = dict(rows=int(mask.sum()),mae=float((general[mask]-gy[mask]).abs().mean()) if mask.any() else None)
        for offset in tail['offsets']:
            mask = valid & (offsets==offset)
            row = phase_report.setdefault(offset, dict(eligible=eligible, observed_cases=int(mask.sum()), training=int((train & (offsets==offset)).sum()),
                                                       common_cases=len(common), recovered=int(success[mask].sum())))
            for suffix, selected in [('observed',mask),('common',mask & cohort)]:
                metrics = row.setdefault(suffix,{})
                metrics['native_rtg_auc'] = ranking(y[selected], success[selected])
                metrics[name] = dict(mae=float((pred[selected]-y[selected]).abs().mean()) if selected.any() else None,
                                     recovery_auc=ranking(pred[selected],success[selected]))
    return dict(actor_sha256=actor, critic_sha256=tensor_digest(ck['critic'].items()), archive_sha256=digest,
                source_policy=bank['policy_id'], phases=phase_report, general=general_report,
                paired_batch_seed=221121, offline_critic_steps_per_candidate=len(batches), actor_updates=0,
                scope='Frozen on-policy complete native returns. Fixed held lanes. Common cohort survives and stays gated at every offset.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('run','frozen','checkpoint','snapshots','output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--diagnostic-weights',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    result=audit(args.run,args.frozen,args.checkpoint,args.snapshots,args.diagnostic_weights)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
