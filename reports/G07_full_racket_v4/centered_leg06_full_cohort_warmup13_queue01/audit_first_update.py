"""Independent CPU audit of the actual first warmup13 native update."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TRAIN = ROOT/'reports/G07_full_racket_v4/residual_ppo_centered_leg06_full_cohort_warmup7_n2048_probe13_01'
SOURCE = ROOT/'reports/G07_full_racket_v4/residual_ppo_centered_placement075_n2048_probe8_01/policy.pt'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE/'ROOT_FIRST_UPDATE_INDEPENDENT_CHECK.json'; assert not output.exists()
    for pid,needle in [(1777456,'full_cohort_warmup13_queue01/run.py'),
                       (1777458,'run_gpu_job.py'),(1777471,'train_full_racket_residual_ppo_v4.py')]:
        proc = Path('/proc')/str(pid)
        assert (proc/'stat').read_text().split(') ',1)[1].split()[0] not in ['Z','X']
        assert needle in (proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    state = torch.load(TRAIN/'ppo/model_0.pt',map_location='cpu',weights_only=False)
    assert state['iter'] == 1 and state['native']['total_timesteps'] == 262144
    assert not state['native']['canonical_restart']
    model = state['model_state_dict']
    source = torch.load(SOURCE,map_location='cpu',weights_only=False)
    actor = torch.jit.load(io.BytesIO(source['residual_actor_jit']),map_location='cpu').eval().state_dict()
    weights = {'actor.'+k[2:]:v.clone() for k,v in actor.items() if k.startswith('1.')}
    last = [k for k,v in weights.items() if k.endswith('.weight') and v.ndim == 2 and v.shape[0] == 18]
    assert len(last) == 1
    for name in [last[0],last[0][:-len('weight')]+'bias']:
        weights[name][:12] *= .5
    assert len(weights) == 8 and all(torch.equal(model[k],v) for k,v in weights.items())
    assert int(model['actor_obs_normalizer.count']) == int(actor['0.count'])+262144 == 12845056
    assert int(model['critic_obs_normalizer.count']) == 262144
    assert torch.allclose(model['log_std'].exp(),torch.tensor([.05]*12+[.1]*6),atol=1e-7,rtol=0)
    optimizer = state['optimizer_state_dict']
    names = [k for k in model if k == 'log_std' or k.startswith(('actor.','critic.'))]
    ids = [index for group in optimizer['param_groups'] for index in group['params']]
    assert len(names) == len(ids) == 17 and len(optimizer['state']) == 8
    learned = {name:int(optimizer['state'][index]['step']) for name,index in zip(names,ids) if index in optimizer['state']}
    assert len(learned) == 8 and all(name.startswith('critic.') and step == 20 for name,step in learned.items())
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_native_live=1777471,
        actual_completed_iteration=1,actual_control_env_steps=262144,
        all8_mapped_actor_weight_tensors_unchanged=True,fresh_log_std_unchanged=True,
        only8_critic_Adam_states_each20=True,actor_RMS_count=12845056,critic_RMS_count=262144,
        actual_source_actor_RMS_count=12582912,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),TRAIN/'ppo/model_0.pt',SOURCE,TRAIN/'INPUTS.json']},
        previous_goal_turn_classification='progress: independently measured completed leg06 TRAIN128, delivered newly reviewed actual video and started declared native warmup13 after real prior exit',
        scope='Actual first critic-only native update; not policy improvement or frozen performance.',
        new_native_execution=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
