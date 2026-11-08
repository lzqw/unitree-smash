"""Add contact-partner telemetry to the completed actual-state suffix run.

The native command arguments and controller are preserved. A paired full
trace audit is required to establish observer passivity before attribution.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Fresh preparation required')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    previous = root / 'reports/landing_native_replanned_suffix_followup_20261108/plan.json'
    plan = json.loads(previous.read_text())
    command = plan['command']
    source = Path(command[command.index('-u') + 1])
    if sha(source) != plan['entry_sha256']:
        raise ValueError('Completed native entry changed')
    text = source.read_text()
    attach = '        finished = torch.zeros(count, dtype=torch.bool, device=env.device)'
    cleanup = "    finally:\n        if 'head_observer' in locals():"
    if text.count(attach) != 1 or text.count(cleanup) != 1:
        raise ValueError('Changed observer lifecycle')
    text = text.replace(attach,
        '        from scripts.native_suffix_pair_observation_v1 import NativeSuffixPairObservationV1\n'
        '        body_pair_observer=NativeSuffixPairObservationV1(env,args.output)\n' + attach)
    text = text.replace(cleanup,
        "    finally:\n        if 'body_pair_observer' in locals():\n"
        "            body_pair_observer.save()\n            body_pair_observer.detach()\n"
        "        if 'head_observer' in locals():")
    ast.parse(text)
    output.mkdir(parents=True)
    entry = output / 'evaluate_suffix_pairs.py'
    entry.write_text(text)
    files = [previous, source, Path(__file__).resolve(),
        root / 'scripts/native_suffix_pair_observation_v1.py',
        root / 'scripts/native_body_pair_observation_v1.py',
        root / 'scripts/audit_native_suffix_pairs_v1.py']
    result = dict(entry=str(entry), entry_sha256=sha(entry),
        source_sha256={str(f): sha(f) for f in files},
        paired_controller_run=str(root / 'reports/native_replanned_suffix_all128_20261108_01'),
        controls_changed=False, collision_filters_changed=False, scoring_changed=False,
        native_execution=False, view_creation_passivity_unverified=True,
        windows_s=[[5.6, 5.75]], sampling_interval_s=.0000625, selected_lanes=[3, 125])
    (output / 'preparation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.output)))
