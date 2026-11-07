"""Generate the storage-protected evaluator from the pinned frictional entry."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

SOURCE_SHA256 = '7b464ec5d2fb7b3ac19b7b4e509020521882d2a31adc521245653d3d8f2d67b4'
REPLACEMENTS = [['def main():\n    args.output.mkdir(parents=True, exist_ok=True)', 'storage_reserve = None\n\ndef main():\n    global storage_reserve\n    args.output.mkdir(parents=True, exist_ok=True)\n    from scripts.native_output_reserve_v1 import NativeOutputReserve\n    storage_reserve = NativeOutputReserve(args.output)'], ['            for tick in range(requested):', '            for tick in range(requested):\n                storage_reserve.check()'], ['        contact_observer.save()', '        storage_reserve.release()\n        contact_observer.save()'], ['finally:\n    app.close(skip_cleanup=True)', 'finally:\n    if storage_reserve is not None:\n        storage_reserve.release()\n    app.close(skip_cleanup=True)']]

def prepare(source,output):
    source=Path(source);output=Path(output)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(source)!=SOURCE_SHA256 or output.exists():raise ValueError('Pinned source and fresh output required')
    text=source.read_text()
    for before,after in REPLACEMENTS:
        if text.count(before)!=1:raise ValueError('Native entry contract changed')
        text=text.replace(before,after)
    ast.parse(text);output.mkdir(parents=True)
    entry=output/'evaluate_frictional_storage.py';entry.write_text(text)
    record=dict(source=str(source.resolve()),source_sha256=sha(source),entry=str(entry.resolve()),entry_sha256=sha(entry),native_execution=False)
    (output/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output)))
