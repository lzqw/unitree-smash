"""Register a held native leg-response window in the direction of the measured Recovery policy update."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch
from quadsmash.rl.supermimic_native_pair_capture_v1 import validate_spec
from quadsmash.rl.supermimic_physical_loop_loading_v1 import RECOVERY_NAMES, tensor_digest
from scripts.audit_supermimic_landing_command_response import load_world
from scripts.evaluate_supermimic_fresh_wm_panel import digest


@torch.no_grad()
def declare(args):
    read=lambda path: json.loads(path.read_text())
    raw=_load_branch(args.previous, 'nominal_a')
    receipt=read(args.previous/'nominal_a/native_probe_manifest.json')
    if receipt['sha256'] != digest(args.previous/'nominal_a/native_probe.pt') or receipt['spec'] != raw['spec']:
        raise ValueError('Changed source of the nominal observation')
    ck, actor, _, _, _, contract=load_world(args.run,args.checkpoint)
    seed=raw['seed']
    if raw['task_contract'] != contract['task_contract'] or raw['policy_execution'] != ck['metadata']['policy_execution']:
        raise ValueError('Nominal observation used a different task or action contract')
    rows=list(map(json.loads,(args.evaluation/'episodes.jsonl').read_text().splitlines()))
    evaluated=read(args.evaluation/'metrics.json')
    behavior=tensor_digest(ck['actor'].items())
    argv=read(args.evaluation/'startup.json')['argv']
    if (len(rows)!=128 or {r['env_id'] for r in rows}!=set(range(128))
            or read(args.evaluation/'task_contract.json')!=raw['task_contract']
            or evaluated['policy_execution']!=raw['policy_execution'] or evaluated['actor_state_sha256']!=behavior
            or not evaluated['actor_parameters_unchanged'] or evaluated['control_ticks']!=800
            or int(argv[argv.index('--seed')+1])!=seed):
        raise ValueError('Different launch cases, task or execution')
    if digest(Path(argv[argv.index('--checkpoint')+1]))!=evaluated['evaluated_checkpoint_sha256']:
        raise ValueError('Evaluated checkpoint changed')
    start=raw['raw']['start']
    obs=start['observation']
    if isinstance(obs,dict):
        obs=obs['policy']
    rally=start['before']['task']
    gate=(rally['had_contact'] & rally['released']).reshape(-1)
    actors=[]
    requests=[]
    for path in (args.before_recovery,args.after_recovery):
        params=torch.load(path,map_location='cpu',weights_only=False,mmap=True)['actor']
        actors.append(params)
        actor.load_state_dict(params,strict=True)
        requests.append(actor.forward_with_recovery_gate(obs,gate,sample=False)[0])
    changed={k for k,v in actors[0].items() if not torch.equal(v,actors[1][k])}
    if not changed or not changed<=set(RECOVERY_NAMES):
        raise ValueError('Expected a Recovery-only change')
    diff=requests[1]-requests[0]
    tick=raw['spec']['tick']
    dt=raw['task_contract']['control_dt_s']
    window=16
    held=set(ck['config']['wm_holdout_lanes'])
    lanes=sorted(r['env_id'] for r in rows if r['env_id'] in held and r['release_time_s'] is not None
        and r['release_time_s']<=tick*dt and r['qualified_airborne_string_contact'] and gate[r['env_id']]
        and diff[r['env_id']].norm()>0 and r['terminal_control_tick']>tick+window)
    if not lanes:
        raise ValueError('No eligible post-release lane with a measured Recovery update')
    spec=validate_spec(dict(schema='native-paired-prefix-batch-spec-v2',head='recovery',action_tick=tick,tick=tick+window-1,
        action_hold_ticks=4,epsilon_request=0.01,lanes=lanes,request_directions=torch.nn.functional.normalize(diff[lanes]).tolist(),
        capture_substeps=False,data_role='development_action_response'))
    args.output.mkdir(parents=True,exist_ok=False)
    spec_path=args.output/f'spec_00_tick{spec["tick"]:03}.json'
    spec_path.write_text(json.dumps(spec,indent=2)+'\n')
    manifest=dict(spec_sha256=digest(spec_path),seed=seed,old_nominal_sha256=receipt['sha256'],
        probe_checkpoint_sha256=digest(args.checkpoint),probe_actor_sha256=behavior,source_actor_sha256=tensor_digest(raw['actor_state'].items()),
        recovery_actors=[tensor_digest(a.items()) for a in actors],changed_tensors=sorted(changed),lanes=lanes,
        reference_request_norm=diff[lanes].norm(dim=1).tolist(),evaluation_episodes_sha256=digest(args.evaluation/'episodes.jsonl'),
        scope='Directions of R8 to R16 evaluated on archived nominal observations, normalized and held four ticks. Actual interventions run under the specified frozen behavior. No policy training or promotion.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('previous','run','checkpoint','evaluation','before-recovery','after-recovery','output'):
        parser.add_argument('--'+key,required=True,type=Path)
    torch.set_num_threads(2)
    declare(parser.parse_args())
