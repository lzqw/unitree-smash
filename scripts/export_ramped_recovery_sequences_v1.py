"""Normalize a ramped continuation report and independently recheck its blocks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.export_continued_recovery_sequences_v1 import main as export_blocks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['result','original-bundle','geometry','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();source=json.loads(a.result.read_text())
    if a.output.exists():raise ValueError('Preserve previous export')
    for name,digest in source['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed ramp source '+name)
    normalized=a.output.with_suffix('.input.json')
    if normalized.exists():raise ValueError('Preserve normalized evidence')
    records=[]
    for r in source['records']:
        records.append(dict(case=r['case'],after=r['prefix'],commands=r['prefix_commands'],
            verified_continuation_s=r['verified_continuation_s'],
            continuation=[dict(after=b['after'],commands=b.get('commands')) for b in r['blocks'] if 'after' in b]))
    hashes=dict(source['source_sha256'])
    for path in [a.result,Path(__file__)]:hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    normalized.write_text(json.dumps(dict(scope=__doc__,native_execution=False,valid_expert_labels=False,
        source_report=str(a.result.resolve()),records=records,source_sha256=hashes),indent=2)+'\n')
    sys.argv=[sys.argv[0],'--result',str(normalized),'--original-bundle',str(a.original_bundle),
        '--geometry',str(a.geometry),'--output',str(a.output)]
    export_blocks()

if __name__=='__main__':main()
