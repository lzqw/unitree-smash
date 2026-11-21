"""Compare recorded native leg interventions with Body WM command derivatives near landing."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_action_identification_dataset_v1 import BRANCHES, _load_branch, _same_contract
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest
from scripts.evaluate_supermimic_fresh_wm_panel import digest


def load_world(run, checkpoint):
    ck = torch.load(checkpoint, weights_only=False, map_location='cpu', mmap=True)
    argv = json.loads((run/'startup.json').read_text())['argv']
    def option(name):
        return Path(argv[argv.index('--'+name)+1])
    meta = ck['metadata']['physical_imagination']
    paths = meta['checkpoint_paths']
    actor, adapter, template, launch, contract = load_physical_loop(
        archive=option('archive'), safety_mapping=option('safety-mapping'),
        physical_checkpoint=paths['physical'], ball_checkpoint=paths['ball'], task_checkpoint=paths['task'],
        action_ball_checkpoint=paths.get('action_ball'), execution_root=meta['g07_execution_root'], device='cpu')
    if adapter.manifest['checkpoint_sha256'] != meta['checkpoint_sha256']:
        raise ValueError('Source physical model identity changed')
    adapter.load_state_dict(ck['world'], strict=True)
    adapter.eval().requires_grad_(False)
    return ck, actor, adapter, template, launch, contract


def response(pred, real, noise, mask):
    norm = real.norm(dim=-1)
    selected = mask & (norm > torch.maximum(torch.full_like(norm, 1e-7), 3*noise.norm(dim=-1)))
    out = dict(eligible=int(mask.sum()), above_repeat_noise=int(selected.sum()))
    if mask.any():
        out['absolute_component_rmse'] = float((pred[mask]-real[mask]).square().mean().sqrt())
        out['zero_component_rmse'] = float(real[mask].square().mean().sqrt())
    if selected.any():
        real, pred = real[selected].double(), pred[selected].double()
        gain = pred.norm(dim=-1)/real.norm(dim=-1)
        cosine = torch.nn.functional.cosine_similarity(real, pred, dim=-1, eps=1e-15)
        out.update(gain_median=float(gain.quantile(.5)), cosine_median=float(cosine.quantile(.5)),
                   cosine_p10=float(cosine.quantile(.1)), zero_predictions=int((gain==0).sum()))
    return out


def audit(collection, run, checkpoint):
    status = json.loads((collection/'status.json').read_text())
    if status['status'] != 'COLLECTION_FINISHED' or len(status['completed']) != 4:
        raise ValueError('Expected four complete native branches')
    inputs = {}
    for row in status['completed']:
        name = row['branch']
        folder = collection/row.get('spec', '')
        if row['exit_code'] != 0 or digest(folder/name/'native_probe.pt') != row['probe_sha256']:
            raise ValueError('Changed paired probe')
        inputs[name] = _load_branch(folder, name)
    if set(inputs) != set(BRANCHES):
        raise ValueError('Incomplete native control pairing')
    _same_contract(inputs)
    base = inputs['nominal_a']
    native_actor = tensor_digest(base['actor_state'].items())
    if any(tensor_digest(v['actor_state'].items()) != native_actor for v in inputs.values()):
        raise ValueError('Native policy differs across branches')
    ck, _, adapter, _, launch, contract = load_world(run, checkpoint)
    if (base['spec']['head'] != 'recovery' or base['task_contract'] != contract['task_contract']
            or base['policy_execution'] != ck['metadata']['policy_execution']):
        raise ValueError('Recovery task or execution contract differs')
    if 'action_tick' in base['spec']:
        declaration = collection.parent/'specifications/manifest.json'
        registration = json.loads(declaration.read_text())
        sfile, = declaration.parent.glob('spec_*.json')
        if (registration['probe_actor_sha256'] != native_actor or registration['seed'] != base['seed']
                or registration['probe_checkpoint_sha256'] != json.loads((collection/'startup.json').read_text())['source_sha256']
                or registration['spec_sha256'] != digest(sfile) or json.loads(sfile.read_text()) != base['spec']):
            raise ValueError('Intervention differed from the registered frozen policy or command window')
    before = base['raw']['start']['before']
    phy = before['physical']
    ids = torch.tensor(base['spec']['lanes'])
    held = torch.isin(ids, torch.tensor(ck['config']['wm_holdout_lanes']))
    eligible = held & before['task']['released'][ids].bool() & before['task']['had_contact'][ids].bool()
    seqs = {}
    for k, v in inputs.items():
        raw = v['raw']
        seqs[k] = raw.get('step_sequence', [dict(next_state=raw['next_state'], final_joint_targets=raw['final_joint_targets'],
                              final_action=raw['final_normalized'], foot_contact=raw['next_feet'])])
    horizon = base['spec']['tick']-base['spec'].get('action_tick', base['spec']['tick'])+1
    if any(len(s) != horizon or any('done' in t and t['done'][ids].any() for t in s) for s in seqs.values()):
        raise ValueError('Incomplete common command window or reset')
    def stack(name, key):
        return torch.stack([step[key][ids] for step in seqs[name]])
    target = {n: stack(n,'final_joint_targets') for n in inputs}
    hist = {n: torch.cat([phy['final_action'][ids][None], stack(n,'final_action')[:-1]]) for n in inputs}
    def rollout(alpha):
        x = encode_physical(phy)[ids]
        feet = phy['foot_contact'][ids].float()
        present = phy['ball_present'][ids]
        stream = []
        for i in range(horizon):
            x, feet, present = ball_transition(adapter.physical_model, adapter.ball_model, x,
                torch.lerp(target['minus'][i], target['plus'][i], alpha),
                torch.lerp(hist['minus'][i], hist['plus'][i], alpha), feet, present,
                phy['elapsed_s'][ids]+i*contract['task_contract']['control_dt_s'], launch[ids],
                action_model=adapter.action_ball_model)
            stream.append(x)
        return torch.stack(stream)
    with torch.no_grad():
        diff = rollout(torch.tensor(1.))-rollout(torch.tensor(0.))
    _, jvp = torch.autograd.functional.jvp(rollout, torch.tensor(.5), torch.tensor(1.))
    truth = stack('plus','next_state')-stack('minus','next_state')
    noise = stack('nominal_b','next_state')-stack('nominal_a','next_state')
    feet = torch.stack([stack(n,'foot_contact') for n in BRANCHES]).bool()
    switch = (feet != feet[:1]).any(0).any(-1).int().cumsum(0).bool()
    sections = {'root_velocity_mps':slice(7,10),'root_angular_rad_s':slice(10,13),'leg_position_rad':slice(13,25),
                'leg_velocity_rad_s':slice(31,43),'root_height_m':slice(2,3)}
    scores = {}
    for h in (1,2,4,8,16,32,64):
        if h>horizon:
            continue
        t=h-1
        actuated=(target['plus'][:h]-target['minus'][:h]).abs().amax((0,2))>0
        subsets={'all_released_held':eligible & actuated, 'same_foot_flags':eligible & actuated & ~switch[t],
                 'different_foot_flags':eligible & actuated & switch[t]}
        scores[h]={group:{k:{m:response(result[t,:,ss],truth[t,:,ss],noise[t,:,ss],mask)
                        for m,result in [('finite_difference',diff),('midpoint_derivative',jvp)]}
                        for k,ss in sections.items()} for group,mask in subsets.items()}
    return dict(checkpoint_sha256=digest(checkpoint), world_sha256=tensor_digest(ck['world'].items()),
                model_policy_sha256=tensor_digest(ck['actor'].items()), probe_actor_sha256=native_actor,
                native_status_sha256=digest(collection/'status.json'), prefix_sha256=base['prefix_sha256'],
                selected=len(ids), held=int(held.sum()), released_held=int(eligible.sum()), horizon=horizon,
                released_held_ids=ids[eligible].tolist(), scores=scores, actor_updates=0, receipt=False,
                scope='Recorded joint command paths at one native prefix. Foot flag agreement is measured at control boundaries.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('collection','run','checkpoint','output'):
        parser.add_argument('--'+key,required=True,type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    result=audit(args.collection,args.run,args.checkpoint)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
