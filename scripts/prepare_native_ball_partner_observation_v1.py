"""Repeat the complete TRAIN87 controller with shuttle-inclusive telemetry."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source-plan','output','followup','run']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    out=a.output.resolve();follow=a.followup.resolve();run=a.run.resolve()
    if any(path.exists() for path in [out,follow,run]):raise ValueError('Fresh output paths required')
    plan=json.loads(a.source_plan.read_text())
    previous=Path(plan['output']);job=json.loads((previous/'job.json').read_text())
    if job['status']!='SUCCEEDED' or job['exit_code']!=0:raise ValueError('Prior native job must be terminal')
    source=Path(plan['command'][plan['command'].index('-u')+1])
    if sha(source)!=plan['entry_sha256']:raise ValueError('Changed executed controller entry')
    for path,digest in plan['source_sha256'].items():
        if sha(root/path)!=digest:raise ValueError('Changed pinned controller input '+path)
    text=source.read_text();attach='        finished = torch.zeros(count, dtype=torch.bool, device=env.device)'
    cleanup="    finally:\n        if 'head_observer' in locals():"
    if text.count(attach)!=1 or text.count(cleanup)!=1:raise ValueError('Unexpected observer lifecycle')
    text=text.replace(attach,'        from scripts.native_ball_partner_observation_v1 import NativeBallPartnerObservationV1\n'
        '        ball_partner_observer=NativeBallPartnerObservationV1(env,args.output)\n'+attach)
    text=text.replace(cleanup,"    finally:\n        if 'ball_partner_observer' in locals():\n"
        "            ball_partner_observer.save()\n            ball_partner_observer.detach()\n"
        "        if 'head_observer' in locals():")
    ast.parse(text);out.mkdir(parents=True);follow.mkdir(parents=True)
    entry=out/'evaluate_ball_partners.py';entry.write_text(text)
    pins=dict(plan['source_sha256'])
    for path in [a.source_plan,source,entry,Path(__file__),root/'scripts/native_ball_partner_observation_v1.py',
                 root/'scripts/audit_native_ball_partner_observation_v1.py',previous/'job.json',
                 previous/'fixed_policy_traces/manifest.json']:
        pins[str(path.resolve())]=sha(path)
    command=list(plan['command']);command[command.index('-u')+1]=str(entry)
    command=[str(run) if x==str(previous) else x for x in command]
    post=[[str(run) if x==str(previous) else str(follow/Path(x).name)
           if str(x).startswith(str(a.source_plan.resolve().parent)+'/') else x for x in row]
          for row in plan['post_commands']]
    post.insert(0,['bash','scripts/python.sh','scripts/audit_native_ball_partner_observation_v1.py',
        '--run',str(run),'--baseline',str(previous),'--output',str(follow/'ball_partner_audit.json')])
    prepared=dict(plan)
    prepared.update(command=command,post_commands=post,output=str(run),entry_sha256=sha(entry),
        source_sha256=pins,prior_native_job=str(previous/'job.json'),previous_process_ids=[],
        automatic_promotion=False,valid_expert_labels=False,
        scope='Exact TRAIN87 feedback repeat with passive robot/shuttle partner forces; original strict isolation remains required',
        controls_changed=False,collision_filters_changed=False,observer_passivity_unverified=True)
    (follow/'plan.json').write_text(json.dumps(prepared,indent=2)+'\n')
    (out/'preparation.json').write_text(json.dumps(dict(entry=str(entry),entry_sha256=sha(entry),
        plan=str(follow/'plan.json'),sensors=33,partners=3713,frames=1603,source_sha256=pins,
        native_execution=False,observer_passivity_unverified=True),indent=2)+'\n')
    print(json.dumps(dict(entry=str(entry),plan=str(follow/'plan.json'),native_execution=False)))


if __name__=='__main__':main()
