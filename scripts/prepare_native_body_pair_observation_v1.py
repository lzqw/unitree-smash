"""Add a passive pair observer to the unchanged completed early controller."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def prepare(root, output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists():raise ValueError('Fresh preparation required')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    previous=root/'reports/landing_native_early_support_followup_20261108_v2/plan.json'
    plan=json.loads(previous.read_text())
    source=root/'reports/landing_native_early_support_preparation_20261108_v2/evaluate_early_support.py'
    if sha(source)!=plan['entry_sha256']:raise ValueError('Completed controller source changed')
    text=source.read_text()
    attach='        finished = torch.zeros(count, dtype=torch.bool, device=env.device)'
    cleanup="    finally:\n        if 'head_observer' in locals():"
    if text.count(attach)!=1 or text.count(cleanup)!=1:raise ValueError('Changed lifecycle')
    text=text.replace(attach,
        '        from scripts.native_body_pair_observation_v1 import NativeBodyPairObservationV1\n'
        '        body_pair_observer=NativeBodyPairObservationV1(env,args.output)\n'+attach)
    text=text.replace(cleanup,
        "    finally:\n        if 'body_pair_observer' in locals():\n"
        "            body_pair_observer.save()\n            body_pair_observer.detach()\n"
        "        if 'head_observer' in locals():")
    ast.parse(text);output.mkdir(parents=True)
    entry=output/'evaluate_body_pairs.py';entry.write_text(text)
    files=[previous,source,Path(__file__),root/'scripts/native_body_pair_observation_v1.py',root/'scripts/audit_native_body_pairs_v1.py']
    result=dict(entry=str(entry),entry_sha256=sha(entry),
        source_sha256={str(f.resolve()):sha(f) for f in files},
        scope='Exact completed early-controller replay with additional pair-force observations.',
        controls_changed=False,collision_filters_changed=False,scoring_changed=False,
        native_execution=False,view_creation_passivity_unverified=True)
    (output/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.root,a.output)))
