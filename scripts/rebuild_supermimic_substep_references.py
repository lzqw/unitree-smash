"""Reconstruct archived success targets at the measured contact substep for offline verification."""
import argparse
import json
from pathlib import Path

import torch
from quadsmash.rl.supermimic_archive_storage_v1 import read_archive_rows
from quadsmash.rl.supermimic_impact_timing_v1 import hermite_position_velocity
from quadsmash.rl.supermimic_native_pair_capture_v1 import tree_digest
from scripts.evaluate_supermimic_contact_rollout import sha


def reconstruct(row, lane, rid, dt):
    before, after = row['before'], row['after']
    if before['event_state'][lane, 1] or not after['event_state'][lane, 1]:
        raise ValueError('Expected the first qualified string control interval')
    stream = row['task_sequence']['streams']['rally.advance']
    t = stream['tensors']
    substeps = after['auxiliary']['timing']['_last_physics_frame'] - before['auxiliary']['timing']['_last_physics_frame']
    if substeps != stream['count'] or substeps != 160:
        raise ValueError('Incomplete native control interval')
    # The rally consumes court-local positions. Recover the world origin from the same tick.
    origin = after['physical']['ball_position_w'][lane] - t['position'][-1,lane]
    torch.testing.assert_close(before['physical']['ball_position_w'][lane] - t['previous_position'][0,lane], origin, rtol=0., atol=2e-5)
    eligible = (t['racket_touching'][:,lane] & t['string_contact'][:,lane] & t['airborne'][:,lane] & t['external_in_play'][:,lane])
    event = torch.where(eligible)[0]
    if not len(event):
        raise ValueError('No native airborne string contact in the archived interval')
    i = int(event[0])
    fraction = (i+1)/substeps
    first = t['time'][i,lane] == after['task']['contact_time'][lane]
    if not first:
        return None
    a = after['auxiliary']['environment']
    def body(s, k):
        return s[k][lane:lane+1,rid]
    center,_ = hermite_position_velocity(body(before,'body_position_w'),body(before,'body_linear_velocity_w'),
        body(after,'body_position_w'),body(after,'body_linear_velocity_w'),t['time'].new_tensor([fraction*dt]),dt)
    sample = torch.cat((center[0]-origin-t['position'][i,lane], a['first_tool_velocity'][lane],
                        a['first_tool_normal'][lane], body(after,'body_angular_velocity_w')[0]))
    if not torch.isfinite(sample).all() or abs(sample[6:9].norm()-1) > 2e-4:
        raise ValueError('Nonfinite or unnormalised target')
    return sample, float(before['physical']['elapsed_s'][lane])+fraction*dt, i+1


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for k in ('bank','output'):
        cli.add_argument('--'+k,type=Path,required=True)
    cli.add_argument('--refresh',nargs='+',type=Path,required=True,help='Frozen refresh folders in their source training runs')
    a=cli.parse_args()
    torch.set_num_threads(2)
    read=lambda f:json.loads(f.read_text())
    bank=torch.load(a.bank,weights_only=True,map_location='cpu')
    result=[]
    for folder in a.refresh:
        frozen=read(folder/'manifest.json')
        source=sha(folder/'complete_episodes.pt')
        run=folder.parent
        archive=run/'native_transitions'
        man=read(archive/'manifest.json')
        rid=man['body_names'].index('Racket')
        terminal_records = read(folder/'episodes_summary.json')
        labels = {(r['env_id'], r['episode_index']): r for r in terminal_records}
        if (not frozen['actor_unchanged'] or frozen['complete_lanes'] != man['num_envs']
                or len(labels) != len(terminal_records) or len(labels) != frozen['complete_lanes']):
            raise ValueError('Incomplete or mutable source policy')
        pending={}
        for e in bank['entries']:
            if e['mode']=='approach' and e['policy_id']==frozen['policy_id']:
                if e['source_sha256']!=source or e['score_schema']!='historical-independent-complete5-v1':
                    raise ValueError('Success reference or native source identity changed')
                key=(e['lane_id'],e['episode_id'])
                label = labels.get(key, {})
                if (label.get('ball_side_complete_except_recovery') is not True
                        or label.get('actor_sha256') != frozen['actor_sha256']
                        or label.get('scoring_schema') != e['score_schema']
                        or label.get('policy_id') != e['policy_id']
                        or e['lane_id'] in bank['excluded_lanes']):
                    raise ValueError('Reference differs from independently scored training episode')
                if key in pending:raise ValueError('Duplicate native reference')
                pending[key]=e
        if not pending:raise ValueError('No success references for supplied source')
        # Compact frozen episodes give the exact event tick for each lane. Native files are
        # visited once, retaining only labels whose original rollout identity matches.
        for chunk in man['chunks']:
            if not pending:break
            with read_archive_rows(archive,chunk,scratch=run) as rows:
                for row in rows:
                    if row.get('policy_id')!=frozen['policy_id'] or row.get('collection_mode')!='frozen_combination':
                        continue
                    b,y=row['before'],row['after']
                    for lane in torch.where((b['event_state'][:,1]<.5)&(y['event_state'][:,1]>.5))[0].tolist():
                        key=(lane,int(b['episode_index'][lane]))
                        if key not in pending:continue
                        e=pending.pop(key)
                        torch.testing.assert_close(y['physical']['elapsed_s'][lane],torch.tensor(e['anchor_time']),rtol=0,atol=1e-5)
                        data = reconstruct(row,lane,rid,man['control_dt_s'])
                        info=dict(policy_id=e['policy_id'],lane=lane,episode=key[1],trajectory_digest=e['digest'],chunk_sha256=chunk['sha256'])
                        if data is None:
                            result.append(dict(info,updated=False,reason='First racket latch and qualified string substep differ'))
                            continue
                        label,when,substep=data
                        old=e['toi_racket']
                        result.append(dict(info,updated=True,substep=substep,old_time_s=e['anchor_time'],new_time_s=when,
                                    target_change=(label-old).tolist()))
                        e['toi_racket']=label
                        e['toi_time']=when
                        e['toi_source']='native string substep, Hermite racket center and control-end angular velocity'
                        e['trajectory_digest']=e['digest']
                        e['digest']=tree_digest((e['trajectory_digest'],label,when,e['toi_source']))
        if pending:raise ValueError('Missing matching native rows: '+str(list(pending)))
    a.output.mkdir(parents=True,exist_ok=False)
    torch.save(bank,a.output/'bank.pt')
    report=dict(source_sha256=sha(a.bank),source_bank_identity=tree_digest(torch.load(a.bank,map_location='cpu',weights_only=True)),
                bank_sha256=sha(a.output/'bank.pt'),bank_identity=tree_digest(bank),cases=result,actor_updates=0,
                scope='Offline reference candidate. True substep ball, first-contact normal and linear velocity; interpolated racket center and control-end angular velocity.')
    (a.output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
