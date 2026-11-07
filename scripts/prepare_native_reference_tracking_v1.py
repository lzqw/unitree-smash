"""Prepare a separately versioned native test of the model tracking candidate."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_native_early_support_v1 import prepare


def prepare_tracking(root,output,bundle_output,base_bundle,model_result):
    root,output,bundle_output,base_bundle,model_result=map(lambda x:Path(x).resolve(),
        (root,output,bundle_output,base_bundle,model_result))
    if output.exists() or bundle_output.exists():raise ValueError('Fresh preparation and bundle required')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    candidate=json.loads(model_result.read_text());base_manifest=base_bundle/'manifest.json'
    manifest=json.loads(base_manifest.read_text())
    if (not candidate['nominal_preserved'] or candidate['nominal_maximum_difference']!=0.
            or candidate['compute_expired_probes'] or candidate['accepted_model_probes']<=1
            or candidate['native_execution'] or candidate['valid_expert_labels']):
        raise ValueError('Completed paired model evidence with preserved nominal required')
    for sources in [candidate['source_sha256'],manifest['source_sha256']]:
        for f,h in sources.items():
            if sha(f)!=h:raise ValueError('Changed source '+f)
    checker=root/'scripts/check_early_support_dispatch_v1.py'
    if sha(checker)!='6eaba089be0665208c3ce90d65fd3a4d14ec50c568e2c036d01e792bc2694ad8':
        raise ValueError('Changed interface verification template')
    if sha(root/'scripts/prepare_native_early_support_v1.py')!='1c54f429dbe8914b24a83be560d3b8a93dd92944e993b4497ec14e5283a02e40':
        raise ValueError('Changed native preparation template')
    result=prepare(root,output)
    entry=Path(result['entry']);audit=Path(result['audit'])
    def replace_once(text,before,after):
        if text.count(before)!=1:raise ValueError('Changed template contract '+before)
        return text.replace(before,after)
    text=replace_once(entry.read_text(),
        'from scripts.native_early_support_tail_v1 import NativeEarlySupportTailV1 as NativeRecoverySequenceReplay',
        'from scripts.native_reference_tracking_v1 import NativeReferenceTrackingV1 as NativeRecoverySequenceReplay')
    ast.parse(text);entry.write_text(text)
    checktext=replace_once(checker.read_text(),'ROOT=Path(__file__).resolve().parents[1]',f'ROOT=Path({str(root)!r})')
    checktext=replace_once(checktext,'from scripts.native_early_support_tail_v1 import NativeEarlySupportTailV1',
        'from scripts.native_reference_tracking_v1 import NativeReferenceTrackingV1 as NativeEarlySupportTailV1')
    ast.parse(checktext);check_path=output/'check_reference_tracking.py';check_path.write_text(checktext)
    audittext=replace_once(audit.read_text(),'from scripts.check_early_support_dispatch_v1 import verify_feedback_dispatch',
        'from check_reference_tracking import verify_feedback_dispatch')
    audittext=replace_once(audittext,"files += [Path(__file__),ROOT/'scripts/check_early_support_dispatch_v1.py',",
        f"files += [Path(__file__),Path({str(check_path)!r}),ROOT/'scripts/native_reference_tracking_v1.py',\n"
        "        ROOT/'scripts/landing_reference_tracking_v1.py',ROOT/'scripts/check_early_support_dispatch_v1.py',")
    ast.parse(audittext);audit.write_text(audittext)
    archive=base_bundle/manifest['bundle_name']
    if sha(archive)!=manifest['bundle_sha256']:raise ValueError('Changed immutable nominal archive')
    bundle_output.mkdir(parents=True);shutil.copyfile(archive,bundle_output/manifest['bundle_name'])
    manifest=copy.deepcopy(manifest)
    manifest['reference_tracking_definition']=dict(gains=candidate['configuration'],model_result=str(model_result),
        model_result_sha256=sha(model_result),model_probes_passed=candidate['accepted_model_probes'],
        model_probes_total=candidate['probe_count'],native_execution=False)
    manifest['source_sha256'].update(candidate['source_sha256'])
    added=[Path(__file__),root/'scripts/native_reference_tracking_v1.py',root/'scripts/landing_reference_tracking_v1.py',
        checker,base_manifest,model_result]
    for f in added:manifest['source_sha256'][str(f)]=sha(f)
    (bundle_output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result.update(entry_sha256=sha(entry),audit_sha256=sha(audit),checker=str(check_path),checker_sha256=sha(check_path),
        bundle=str(bundle_output),bundle_manifest_sha256=sha(bundle_output/'manifest.json'),
        feedback_kind='full_reference_tracking_then_original_attitude_feedback',nominal_archive_unchanged=True)
    result['source_sha256'].update({str(f):sha(f) for f in added})
    (output/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    for key in ['output','bundle-output','base-bundle','model-result']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare_tracking(a.root,a.output,a.bundle_output,a.base_bundle,a.model_result)))
