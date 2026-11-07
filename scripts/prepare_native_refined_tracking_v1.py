"""Export a refined model trajectory without changing its fixed feedback target."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_native_reference_tracking_v1 import prepare_tracking


def prepare_refined(root, output, bundle_output, base_bundle, search_path):
    root,output,bundle_output,base_bundle,search_path = map(lambda p:Path(p).resolve(),
        (root,output,bundle_output,base_bundle,search_path))
    sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    search = json.loads(search_path.read_text())
    if (not search['all_thirteen_completed'] or not search['original_five_retained']
            or not search['model_candidate_improved'] or search['native_execution'] or search['valid_expert_labels']):
        raise ValueError('Completed preserved model improvement required')
    for p,h in search['source_sha256'].items():
        if sha(p)!=h:raise ValueError('Changed source '+p)
    nominal = next(r['result'] for r in search['final_checks'] if r['probe']['name']=='nominal')
    if not nominal['accepted'] or len(nominal['trajectory'])!=520:
        raise ValueError('Complete feasible nominal prediction required')
    source = Path(search['configuration']['source'])
    result = prepare_tracking(root,output,bundle_output,base_bundle,source)
    manifest_path = bundle_output/'manifest.json';manifest=json.loads(manifest_path.read_text())
    archive=bundle_output/manifest['bundle_name'];key='case_003'
    with np.load(archive,allow_pickle=False) as data:
        arrays={k:data[k].copy() for k in data.files}
    qkey=key+'_tracking_reference_q';vkey=key+'_tracking_reference_v'
    arrays[qkey]=arrays[key+'_predicted_q'][:-1].copy()
    arrays[vkey]=arrays[key+'_predicted_v'][:-1].copy()
    trajectory=nominal['trajectory']
    arrays[key+'_commands_rad']=np.asarray([r['target'] for r in trajectory])
    for suffix,field in [('q','q'),('v','v'),('contacts','contacts')]:
        name=key+'_predicted_'+suffix
        arrays[name]=np.vstack([arrays[name][0],[r[field] for r in trajectory]])
    with archive.open('wb') as stream:np.savez_compressed(stream,**arrays)
    manifest['bundle_sha256']=sha(archive)
    manifest['scope']='Refined early full-body trajectory with separately frozen tracking references; native execution unverified.'
    manifest['feedback_definition']['fixed_early_knots_rad']=search['best_knots_rad'][:3]
    manifest['feedback_definition']['tail_knots_rad']=search['best_knots_rad'][3:]
    manifest['reference_tracking_definition'].update(
        model_result=str(search_path),model_result_sha256=sha(search_path),
        model_probes_passed=search['accepted_model_probes'],model_probes_total=13,
        fixed_reference_model_source=str(source),fixed_reference_model_sha256=sha(source),
        state_reference_contract='explicit_frozen_tracking_reference_v1',q_key=qkey,v_key=vkey)
    manifest['source_sha256'].update(search['source_sha256'])
    added=[Path(__file__),root/'scripts/native_reference_tracking_v2.py',search_path]
    manifest['source_sha256'].update({str(p):sha(p) for p in added})
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    def replace_once(text,before,after):
        if text.count(before)!=1:raise ValueError('Changed generated template '+before)
        return text.replace(before,after)
    for label in ['entry','checker']:
        p=Path(result[label]);text=replace_once(p.read_text(),
            'from scripts.native_reference_tracking_v1 import NativeReferenceTrackingV1',
            'from scripts.native_reference_tracking_v2 import NativeReferenceTrackingV2')
        ast.parse(text);p.write_text(text);result[label+'_sha256']=sha(p)
    audit=Path(result['audit']);text=replace_once(audit.read_text(),
        "ROOT/'scripts/native_reference_tracking_v1.py',",
        "ROOT/'scripts/native_reference_tracking_v1.py',ROOT/'scripts/native_reference_tracking_v2.py',")
    ast.parse(text);audit.write_text(text)
    result.update(audit_sha256=sha(audit),bundle_manifest_sha256=sha(manifest_path),
        nominal_archive_unchanged=False,original_query_unchanged=True,
        explicit_tracking_reference=True,model_search=str(search_path),
        model_probes_passed=search['accepted_model_probes'],native_execution=False)
    result['source_sha256'].update({str(p):sha(p) for p in added})
    (output/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    for key in ['output','bundle-output','base-bundle','search']:parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();print(json.dumps(prepare_refined(a.root,a.output,a.bundle_output,a.base_bundle,a.search)))
