"""Recheck preserved real update11 evidence without relabeling a live job."""
import hashlib,importlib.util,json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from quadsmash.evaluation_v4 import independent_episode

def main():
    q=ROOT/'reports/G07_full_racket_v4/centered_placement075_resume32_queue02'
    spec=importlib.util.spec_from_file_location('preserved075_resume',q/'run.py');queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
    read=queue.core.read;sha=queue.core.sha;proof=read(HERE/'ACTUAL_UPDATE11_CHECK.json')
    source_path=queue.SOURCE/'checkpoint.pt';diag_path=queue.TRAIN/'ppo/model_10.pt'
    assert sha(source_path)==proof['source_checkpoint_sha256'] and sha(diag_path)==proof['diagnostic11_sha256']
    source=torch.load(source_path,map_location='cpu',weights_only=False);native=torch.load(diag_path,map_location='cpu',weights_only=False)
    assert source['iter']==8 and native['iter']==11 and not native['native']['canonical_restart']
    assert native['native']['total_timesteps']==2883584 and native['native']['total_timesteps']-source['native']['total_timesteps']==786432
    before,after=source['model_state_dict'],native['model_state_dict']
    assert all(torch.isfinite(v).all()for v in after.values()if torch.is_tensor(v))
    for key,expected in [('actor_obs_normalizer.count',13369344),('critic_obs_normalizer.count',2883584)]:
        assert int(after[key])==expected and int(after[key])-int(before[key])==786432
    a=source['optimizer_state_dict']['state'];b=native['optimizer_state_dict']['state'];assert set(a)==set(b)and len(b)==17
    assert all(float(b[k]['step'])-float(a[k]['step'])==60 for k in a)
    assert all(torch.isfinite(v).all()for state in b.values()for v in state.values()if torch.is_tensor(v))
    for prefix in ['actor.','critic.']:assert any(not torch.equal(before[k],after[k])for k in before if k.startswith(prefix))
    snapshot=HERE/'UPDATE11_BOUNDARY_RAW_EPISODES_SNAPSHOT.json';assert sha(snapshot)==proof['raw_snapshot_sha256']
    rows=read(snapshot);assert len(rows)==len({v['env_id']for v in rows})==1251
    assert queue.verify_episode_chains(source,native,rows,2048,allow_empty=True)==proof['actual_episode_chains']
    protocol=read(ROOT/'configs/g07_full_racket_eval_v4_r3.json');protocol['_root']=str(ROOT)
    checked=[independent_episode(v,protocol)for v in rows]
    counts=dict(completed_episodes=len(rows),air=sum(v['airborne_contact']for v in checked),conditional_recovery_count=sum(v['conditional_recovery']for v in checked),complete=sum(v['complete']for v in checked),numerical_failures=sum(v['numerical_failures']for v in rows))
    progress=read(HERE/'UPDATE11_PROGRESS_SNAPSHOT.json');assert progress['update']==11
    assert counts==proof['independent_closed_prefix_counts'] and all(progress[k]==v for k,v in counts.items())
    assert all(v['ground_init']and v['source_training_index']==v['env_id']%128 for v in rows)
    print(json.dumps(dict(preserved_update11_recomputed=True,counts=counts,source_inclusive_episode_ID_chains_exact=True,all17_Adam_states_advanced60=True,frozen_policy_performance=False,formal_acceptance=False,goal_complete=False)))
if __name__=='__main__':main()
