#!/usr/bin/env python3
"""Read-only source inventory. SQLite stores evidence/indexes, never invented trajectories.

No pickle loading, policy updates, environment steps or automatic reference-bank admission.
Unknown contracts, aggregate metrics and missing raw chunks remain explicitly ineligible.
"""
import argparse, hashlib, json, os, sqlite3
from pathlib import Path

EXCLUDE={'.git','.cache','site-packages','node_modules','.local','__pycache__','env_isaaclab','envs','.codex','.config','.vscode','.npm','.rustup','.cargo'}
FLAGS=('qualified_airborne_string_contact','ball_side_complete_except_recovery','recovery_success','complete5')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def rows(value):
    if isinstance(value,list):
        # Numerical state vectors are not containers of episode records.
        for item in value:
            if isinstance(item,dict) or (isinstance(item,list) and item and isinstance(item[0],dict)):
                yield from rows(item)
    elif isinstance(value,dict):
        if any(k in value for k in FLAGS) and any(k in value for k in ('env_id','case_id','episode_id','lane_id')):
            yield value
        else:
            for v in value.values():
                if isinstance(v,(list,dict)):yield from rows(v)

def build(roots,output,*,resume=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=resume)
    if resume and (output/'inventory.json').exists():raise ValueError('Completed inventory is immutable; create a new directory')
    db=sqlite3.connect(output/'success_trajectories.sqlite3')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY,sha256 TEXT,size INTEGER,kind TEXT,error TEXT);
    CREATE TABLE IF NOT EXISTS episodes(id INTEGER PRIMARY KEY,evidence_sha256 TEXT UNIQUE,case_id TEXT,episode_id TEXT,
      qualified INTEGER,ball_side INTEGER,recovery INTEGER,complete5 INTEGER,scoring_schema TEXT,record_json TEXT);
    CREATE TABLE IF NOT EXISTS occurrences(episode INTEGER,path TEXT,archive TEXT,identity_known INTEGER,raw_available INTEGER,
      eligibility TEXT,PRIMARY KEY(episode,path));
    CREATE TABLE IF NOT EXISTS archives(path TEXT PRIMARY KEY,manifest_sha256 TEXT,manifest_json TEXT,chunk_count INTEGER,
      missing_chunks INTEGER,hash_status TEXT);
    CREATE VIEW IF NOT EXISTS successes AS SELECT * FROM episodes WHERE qualified=1 OR ball_side=1 OR recovery=1 OR complete5=1;
    ''')
    errors=[];seen=set();count=0
    for root in roots:
        def onerror(error):errors.append(str(error))
        for directory,dirs,names in os.walk(root,followlinks=False,onerror=onerror):
            kept=[]
            for d in dirs:
                if d in EXCLUDE:continue
                try:
                    child=Path(directory,d)
                    if not child.is_symlink() and child.resolve()!=output.resolve():kept.append(d)
                except OSError as error:errors.append(str(child)+': '+str(error))
            dirs[:]=kept
            for name in names:
                p=Path(directory,name)
                # All JSON/JSONL evidence, plus inventory opaque trajectory files.
                if p.suffix not in ('.json','.jsonl','.pt','.npz','.h5','.hdf5'):continue
                try:
                    if p.is_symlink():continue
                    stat=p.stat();key=(stat.st_dev,stat.st_ino)
                    if key in seen:continue
                    seen.add(key);count+=1
                    if count%1000==0:
                        db.commit();print('INVENTORY_FILES='+str(count),flush=True)
                    if resume:
                        old=db.execute('SELECT sha256,size FROM files WHERE path=?',(str(p),)).fetchone()
                        if old is not None:
                            if old[1]!=stat.st_size or (old[0] is not None and old[0]!=digest(p)):
                                errors.append(str(p)+': source changed since partial inventory; retained old evidence, needs separate reindex')
                            continue
                    if p.suffix not in ('.json','.jsonl'):
                        db.execute('INSERT OR IGNORE INTO files VALUES(?,?,?,?,?)',(str(p),None,stat.st_size,'opaque-not-unpickled',None));continue
                    if stat.st_size>128*1024*1024:
                        db.execute('INSERT OR IGNORE INTO files VALUES(?,?,?,?,?)',(str(p),None,stat.st_size,'oversized-unparsed','metadata size >128MiB'));continue
                    # Inventory other JSON files without recursively parsing unrelated configurations.
                    if not any(token in name.lower() for token in ('episode','metric','eval','rollout','trajectory','manifest','result','trace','native_return')):
                        db.execute('INSERT OR IGNORE INTO files VALUES(?,?,?,?,?)',(str(p),None,stat.st_size,'other-json-unparsed',None));continue
                    sha=digest(p);raw=p.read_text()
                    objects=[json.loads(line) for line in raw.splitlines() if line.strip()] if p.suffix=='.jsonl' else [json.loads(raw)]
                    db.execute('INSERT OR IGNORE INTO files VALUES(?,?,?,?,?)',(str(p),sha,stat.st_size,'json-evidence',None))
                    for value in objects:
                        if name=='manifest.json' and isinstance(value,dict) and isinstance(value.get('chunks'),list):
                            chunks=value['chunks'];missing=sum(not (p.parent/c['file']).is_file() for c in chunks if isinstance(c,dict) and 'file'in c)
                            db.execute('INSERT OR REPLACE INTO archives VALUES(?,?,?,?,?,?)',(str(p.parent),sha,json.dumps(value),len(chunks),missing,'not-rehashed; source digests retained'))
                        for row in rows(value):
                            # Full physical arrays remain at source. The database is an index,
                            # not a second multi-gigabyte transition replay.
                            metadata={k:v for k,v in row.items() if k not in ('physical','observation','action','state','next_state','executor','history','tensors')}
                            canonical=json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=True)
                            h=hashlib.sha256(canonical.encode()).hexdigest()
                            values=[int(row[k]) if type(row.get(k))is bool else None for k in FLAGS]
                            db.execute('INSERT OR IGNORE INTO episodes(evidence_sha256,case_id,episode_id,qualified,ball_side,recovery,complete5,scoring_schema,record_json) VALUES(?,?,?,?,?,?,?,?,?)',
                              (h,str(row.get('case_id',row.get('env_id',row.get('lane_id')))),str(row.get('episode_id',row.get('episode_index','unknown'))),*values,row.get('scoring_schema'),canonical))
                            idx=db.execute('SELECT id FROM episodes WHERE evidence_sha256=?',(h,)).fetchone()[0]
                            archive=p.parent/'physical_transitions'
                            db.execute('INSERT OR IGNORE INTO occurrences VALUES(?,?,?,?,?,?)',(idx,str(p),str(archive) if (archive/'manifest.json').is_file() else None,0,0,'not-admitted'))
                except (OSError,ValueError,KeyError,TypeError) as e:
                    errors.append(str(p)+': '+str(e))
                    db.execute('INSERT OR IGNORE INTO files VALUES(?,?,?,?,?)',(str(p),None,None,'unreadable',str(e)))
    # Admission is deliberately separate from discovery: matching actor/execution/asset
    # contracts and chunk/episode validation are required by reference-bank tooling.
    for path,manifest,missing in db.execute('SELECT path,manifest_json,missing_chunks FROM archives').fetchall():
        m=json.loads(manifest);known=bool(m.get('actor_state_sha256') and m.get('policy_execution') and m.get('task_contract'))
        available=not missing and bool(m.get('chunks'))
        db.execute('UPDATE occurrences SET identity_known=?,raw_available=?,eligibility=? WHERE archive=?',
                   (int(known),int(available),'requires-chunk-hash-episode-and-contract-validation' if known and available else 'missing-identity-or-raw-data',path))
    db.execute("CREATE VIEW IF NOT EXISTS verified_score_successes AS SELECT * FROM successes WHERE scoring_schema='historical-independent-complete5-v1'")
    db.commit()
    totals={k:db.execute('SELECT COUNT(*) FROM episodes WHERE '+column+'=1').fetchone()[0] for k,column in zip(FLAGS,('qualified','ball_side','recovery','complete5'))}
    report=dict(roots=[str(x) for x in roots],exclusions=sorted(EXCLUDE),scanned_files=count,episode_records=db.execute('SELECT COUNT(*) FROM episodes').fetchone()[0],
        success_evidence_records=db.execute('SELECT COUNT(*) FROM successes').fetchone()[0],label_counts=totals,
        independently_scored_success_records=db.execute('SELECT COUNT(*) FROM verified_score_successes').fetchone()[0],
        linked_success_records=db.execute('SELECT COUNT(DISTINCT e.id) FROM successes e JOIN occurrences o ON e.id=o.episode WHERE o.raw_available=1').fetchone()[0],
        archives=db.execute('SELECT COUNT(*) FROM archives').fetchone()[0],errors=errors,
        database_scope='evidence inventory; duplicate identical records collapse; distinct policies/episodes remain separate; not unique physical trajectories',
        admitted_to_training=0,reason='No automatic admission of unknown, MPC/teacher, unverified hashes, incomplete or incompatible contracts; opaque files inventoried only')
    db.close();report['database_sha256']=digest(output/'success_trajectories.sqlite3')
    (output/'inventory.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-incomplete',action='store_true')
    a=p.parse_args();r=build(a.roots,a.output,resume=a.resume_incomplete);print(json.dumps({k:v for k,v in r.items() if k!='errors'},indent=2));print('scan_errors='+str(len(r['errors'])))
