"""Freeze model-feasible recovery references; never label them native experts."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['bank','search','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--additional-search',type=Path,action='append',default=[])
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if a.output.exists():raise ValueError('Fresh immutable reference snapshot required')
    bank=read(a.bank);lookup={r['case']:r for r in bank['queries']};records=[];rejected=[];pins={str(a.bank.resolve()):sha(a.bank)}
    # Source case IDs are provenance only. Runtime matching uses q/v and the
    # previous physical command, never lane numbers or episode outcomes.
    candidates={}
    for directory in [a.search,*a.additional_search]:
        for path in sorted(directory.glob('case_*.json')):
            result=read(path);fine=result.get('fine');pins[str(path.resolve())]=sha(path)
            if not fine or not fine['horizon_model_feasible'] or len(fine['trajectory'])!=60:
                rejected.append(result['case']);continue
            old=candidates.get(result['case'])
            if old is None or fine['rank']<old[1]['rank']:candidates[result['case']]=(path,fine)
    for case,(path,fine) in sorted(candidates.items()):
        records.append(dict(source_case=case,source_key=sha(path),query=lookup[case]['query'],
            trajectory=fine['trajectory'],terminal=fine['terminal'],native_success=False))
    rejected=sorted(set(rejected)-set(candidates))
    if not records:raise ValueError('No verified model horizon available')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(schema='model-recovery-reference-library-v1',time_utc=datetime.now(timezone.utc).isoformat(),
        references=records,unaccepted_or_unfinished_cases=rejected,source_sha256=pins,
        training=False,valid_expert_labels=False,native_success=False),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(model_references=len(records),source_cases=[r['source_case'] for r in records],native_success=False)))


if __name__=='__main__':main()
