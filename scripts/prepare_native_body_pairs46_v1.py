"""Repeat the completed tracking controller with TRAIN3/46 pair observations."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError('Fresh preparation required')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    original = root/'reports/landing_native_reference_tracking_followup_20261108/plan.json'
    plan = json.loads(original.read_text())
    prep = root/'reports/landing_native_reference_tracking_preparation_20261108'
    source = prep/'evaluate_early_support.py'
    if sha(source) != plan['entry_sha256']: raise ValueError('Completed entry changed')
    text = source.read_text()
    attach = '        finished = torch.zeros(count, dtype=torch.bool, device=env.device)'
    cleanup = "    finally:\n        if 'head_observer' in locals():"
    if text.count(attach)!=1 or text.count(cleanup)!=1: raise ValueError('Changed lifecycle')
    text = text.replace(attach,
        '        from scripts.native_body_pair46_observation_v1 import NativeBodyPair46ObservationV1\n'
        '        body_pair_observer=NativeBodyPair46ObservationV1(env,args.output)\n'+attach)
    text = text.replace(cleanup,
        "    finally:\n        if 'body_pair_observer' in locals():\n"
        "            body_pair_observer.save()\n            body_pair_observer.detach()\n"
        "        if 'head_observer' in locals():")
    ast.parse(text); output.mkdir(parents=True)
    entry = output/'evaluate_body_pairs46.py'; entry.write_text(text)
    files = [original, source, Path(__file__).resolve(),
        root/'scripts/native_body_pair46_observation_v1.py', root/'scripts/audit_native_body_pairs46_v1.py']
    result = dict(entry=str(entry), entry_sha256=sha(entry),
        source_sha256={str(f):sha(f) for f in files},
        paired_controller_run=str(root/'reports/native_reference_tracking_all128_20261108_01'),
        controls_changed=False, collision_filters_changed=False, scoring_changed=False,
        native_execution=False, view_creation_passivity_unverified=True,
        windows_s=[[3.55,3.85],[4.74,4.94]], selected_lanes=[3,46])
    (output/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(prepare(a.root,a.output)))
