#!/usr/bin/env python3
"""Verify selected indexed raw chunks, preserving unknown policy/teacher provenance.

No training admission, model loading or imitation. Every source remains immutable.
"""
import argparse,hashlib,json,sqlite3
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def verify(database,archive,output):
    archive=Path(archive).resolve();db=sqlite3.connect(database)
    row=db.execute('SELECT manifest_sha256,manifest_json FROM archives WHERE path=?',(str(archive),)).fetchone()
    if row is None:raise ValueError('Archive not indexed')
    manifest=archive/'manifest.json'
    if sha(manifest)!=row[0]:raise ValueError('Manifest changed since inventory')
    value=json.loads(row[1]);chunks=[];failure=[]
    for item in value['chunks']:
        path=(archive/item['file']).resolve()
        if archive not in path.parents:raise ValueError('Chunk escapes archive directory')
        actual=sha(path) if path.is_file() else None
        good=actual==item.get('sha256') and actual is not None
        chunks.append(dict(file=item['file'],sha256=actual,expected_sha256=item.get('sha256'),passed=good))
        if not good:failure.append(item['file'])
    passed=bool(chunks) and not failure
    db.execute('UPDATE archives SET hash_status=? WHERE path=?',('chunks-sha256-verified' if passed else 'chunks-invalid',str(archive)));db.commit();db.close()
    result=dict(archive=str(archive),manifest_sha256=row[0],chunks=chunks,passed=passed,
        source_policy_sha256=value.get('source_sha256'),actor_state_sha256=value.get('actor_state_sha256'),
        policy_execution=value.get('policy_execution'),task_contract=value.get('task_contract'),
        admitted_to_training=False,scope='byte integrity only; complete episode identity, teacher provenance, split and reference compatibility still required')
    path=Path(output)
    with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--database',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=verify(a.database,a.archive,a.output);print(json.dumps(dict(passed=r['passed'],chunks=len(r['chunks']),admitted_to_training=False)))
    if not r['passed']:raise SystemExit(1)
