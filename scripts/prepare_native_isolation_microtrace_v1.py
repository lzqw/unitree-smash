"""Prepare paired passive-instrumentation entries without changing controllers."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

SOURCES = {
    'terminal': ('reports/landing_native_terminal_support_preparation_20261108/evaluate_terminal_support.py',
                 '7331853475fad92d7cfce0d657b003b4ac8f998316279102aa9e6d9128b325ce'),
    'shadow': ('reports/landing_native_terminal_shadow_preparation_20261108/evaluate_terminal_shadow.py',
               '7925de00527d2fbbcbfb765cc689cc59972bfc3f45e895ee0a0bc0fc98b0a529')}


def prepare(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('Fresh preparation directory required')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    entries = {}
    for name, (relative, digest) in SOURCES.items():
        source = root/relative
        if sha(source) != digest:
            raise ValueError('Changed source '+str(source))
        original = source.read_text()
        attach = "        finished = torch.zeros(count, dtype=torch.bool, device=env.device)"
        cleanup = "    finally:\n        if 'native_replay' in locals():"
        if original.count(attach) != 1 or original.count(cleanup) != 1:
            raise ValueError('Changed lifecycle integration points')
        insertion = (
            '        from scripts.native_isolation_microtrace_v1 import NativeIsolationMicrotraceV1\n'
            '        isolation_observer = NativeIsolationMicrotraceV1(env, args.output,\n'
            '            lanes=(3,125), first_frame=89600, last_frame=92000)\n')
        text = original.replace(attach, insertion+attach)
        text = text.replace(cleanup,
            "    finally:\n        if 'isolation_observer' in locals():\n"
            "            isolation_observer.save()\n            isolation_observer.detach()\n"
            "        if 'native_replay' in locals():")
        ast.parse(text)
        entries[name] = (text, source, digest)
    output.mkdir(parents=True)
    result = dict(native_execution=False, queued=False, controls_changed=False,
        physics_changed=False, source_sha256={str(Path(__file__).resolve()):sha(__file__),
        str((root/'scripts/native_isolation_microtrace_v1.py').resolve()):sha(root/'scripts/native_isolation_microtrace_v1.py')},
        window=dict(first_frame=89600,last_frame=92000,lanes=[3,125]), entries={})
    for name, (text, source, digest) in entries.items():
        path = output/('evaluate_'+name+'_isolation.py')
        path.write_text(text)
        result['entries'][name] = dict(path=str(path.resolve()), sha256=sha(path),
            source=str(source.resolve()), source_sha256=digest)
    (output/'preparation.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); print(json.dumps(prepare(a.root,a.output)))
