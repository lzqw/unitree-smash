"""Prepare a fixed-controller repeat with expanded partner telemetry."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if out.exists():raise ValueError('Fresh preparation required')
    planpath=root/'reports/landing_native_replanned_suffix_followup_20261108/plan.json'
    plan=json.loads(planpath.read_text());source=Path(plan['command'][plan['command'].index('-u')+1])
    if sha(source)!=plan['entry_sha256']:raise ValueError('Changed executed source')
    text=source.read_text();attach='        finished = torch.zeros(count, dtype=torch.bool, device=env.device)'
    cleanup="    finally:\n        if 'head_observer' in locals():"
    if text.count(attach)!=1 or text.count(cleanup)!=1:raise ValueError('Changed lifecycle')
    text=text.replace(attach,'        from scripts.native_all_partner_observation_v1 import NativeAllPartnerObservationV1\n'
        '        all_partner_observer=NativeAllPartnerObservationV1(env,args.output)\n'+attach)
    text=text.replace(cleanup,"    finally:\n        if 'all_partner_observer' in locals():\n"
        "            all_partner_observer.save()\n            all_partner_observer.detach()\n"
        "        if 'head_observer' in locals():")
    ast.parse(text);out.mkdir(parents=True);entry=out/'evaluate_all_partners.py';entry.write_text(text)
    files=[planpath,source,Path(__file__),root/'scripts/native_all_partner_observation_v1.py',
        root/'scripts/audit_native_all_partner_observation_v1.py']
    result=dict(entry=str(entry),entry_sha256=sha(entry),source_sha256={str(f.resolve()):sha(f) for f in files},
        source_controller_run=plan['output'],controls_changed=False,collision_filters_changed=False,
        expected_sensors=84,expected_partner_bodies=3584,expected_frames=962,native_execution=False,
        observer_passivity_unverified=True,valid_expert_labels=False)
    (out/'preparation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
