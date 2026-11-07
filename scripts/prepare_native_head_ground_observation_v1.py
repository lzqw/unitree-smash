"""Prepare a passive head-contact observation on the verified student-shadow entry."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError('Fresh preparation required')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source = root/'reports/landing_native_terminal_shadow_preparation_20261108/evaluate_terminal_shadow.py'
    if sha(source) != '7925de00527d2fbbcbfb765cc689cc59972bfc3f45e895ee0a0bc0fc98b0a529':
        raise ValueError('Changed verified shadow entry')
    text = source.read_text()
    attach = '        finished = torch.zeros(count, dtype=torch.bool, device=env.device)'
    cleanup = "    finally:\n        if 'native_replay' in locals():"
    if text.count(attach) != 1 or text.count(cleanup) != 1:
        raise ValueError('Changed lifecycle sites')
    insert = ('        from scripts.native_head_ground_observation_v1 import NativeHeadGroundObservationV1\n'
              '        head_observer = NativeHeadGroundObservationV1(env, args.output)\n')
    text = text.replace(attach, insert+attach).replace(cleanup,
        "    finally:\n        if 'head_observer' in locals():\n"
        "            head_observer.save()\n            head_observer.detach()\n"
        "        if 'native_replay' in locals():")
    ast.parse(text); output.mkdir(parents=True)
    entry = output/'evaluate_head_ground_shadow.py'; entry.write_text(text)
    files = [source, Path(__file__), root/'scripts/native_head_ground_observation_v1.py']
    result = dict(entry=str(entry), entry_sha256=sha(entry),
        source_sha256={str(path.resolve()): sha(path) for path in files},
        native_execution=False, scoring_changed=False, controls_changed=False,
        new_contact_view=True, note='View creation passivity must be checked against the recorded baseline.')
    (output/'preparation.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(prepare(args.root, args.output)))
