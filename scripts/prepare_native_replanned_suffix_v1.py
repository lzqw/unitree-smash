"""Export a replay-verified suffix while preserving its actual native prefix."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_native_early_support_v1 import prepare
from scripts.check_constrained_recovery_suffix_v1 import source_context


def validate(search):
    if (search.get('full_suffix_verified') is not True or search.get('native_execution') is not False
            or search.get('valid_expert_labels') is not False):
        raise ValueError('Independently verified complete model suffix required')
    tick=search['configuration']['tick'];length=800-tick
    best,replay=search['best'],search['independent_replay']
    for row in [best,replay]:
        r=row['result']
        if (r['substeps']!=160 or r['accepted'] is not True or r['failure'] is not None
                or r['completed_model_steps']!=length or r['required_model_steps']!=length
                or r['control_tick']!=tick or r['original_episode_end_tick']!=800
                or r['terminal_violations'] or r['trailing_model_window_s']<.8
                or len(r['trajectory'])!=length):
            raise ValueError('Incomplete constrained suffix')
        if [r['control_tick'] for r in r['trajectory']]!=list(range(tick,800)):
            raise ValueError('Suffix reference clock changed')
        for k,q in [('initial_q','q'),('initial_v','v'),('initial_contacts','contacts'),('initial_previous_command','target')]:
            if not np.array_equal(r[k],search['query'][q]):raise ValueError('Suffix query mismatch '+k)
    if not np.array_equal(best['corrections_rad'],replay['corrections_rad']):raise ValueError('Replay controls changed')
    arrays={}
    for key,width in [('q',25),('v',24),('contacts',4),('target',18)]:
        x=np.asarray([r[key] for r in best['result']['trajectory']])
        y=np.asarray([r[key] for r in replay['result']['trajectory']])
        if x.shape!=(length,width) or not np.isfinite(x).all() or not np.array_equal(x,y):
            raise ValueError('Malformed or unreproduced suffix '+key)
        arrays[key]=x
    if not np.array_equal(arrays['target'][0],search['query']['target']):
        raise ValueError('First suffix request does not preserve actual command history')
    return tick,arrays


def prepare_suffix(root,output,bundle_output,base_bundle,search_path):
    root,output,bundle_output,base_bundle,search_path=[Path(p).resolve() for p in
        [root,output,bundle_output,base_bundle,search_path]]
    if output.exists() or bundle_output.exists():raise ValueError('Fresh preparation and bundle required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    search=read(search_path);tick,trajectory=validate(search)
    manifest=read(base_bundle/'manifest.json');archive=base_bundle/manifest['bundle_name']
    native=Path(search['configuration']['native_run']).resolve();meta=read(native/'sequence_native_result.json')
    if sha(archive)!=manifest['bundle_sha256'] or meta['bundle_sha256']!=manifest['bundle_sha256']:
        raise ValueError('Recorded prefix used another controller bundle')
    for sources in [search['source_sha256'],manifest['source_sha256']]:
        for path,h in sources.items():
            if sha(path)!=h:raise ValueError('Changed source '+path)
    _,knots,_,_,context=source_context(search['configuration']['source'])
    if not np.array_equal(knots,search['original_knots_rad']) or context[0]!=manifest['configuration']:
        raise ValueError('Controller configuration/reference changed')
    with np.load(archive,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    offset=tick-280;key='case_003'
    arrays[key+'_commands_rad'][offset:]=trajectory['target']
    for field in ['q','v','contacts']:
        # The suffix prediction is explicitly re-anchored at its observed
        # query, not represented as an uninterrupted nominal prediction.
        arrays[key+'_predicted_'+field][offset:]=np.vstack([search['query'][field],trajectory[field]])
        arrays['suffix_query_'+field]=np.asarray(search['query'][field])
        arrays['suffix_origin_'+field]=np.asarray(context[1]['initial_'+field])
    arrays['suffix_query_previous']=np.asarray(search['query']['target'])
    arrays['suffix_origin_previous']=np.asarray(context[1]['actual_previous_command'])
    arrays['suffix_tracking_reference_q']=context[4];arrays['suffix_tracking_reference_v']=context[5]
    manifest=copy.deepcopy(manifest)
    manifest['replanned_suffix_definition']=dict(contract='preserved_native_prefix_actual_query_suffix_v1',
        control_tick=tick,native_prefix_run=str(native),prefix_bundle=str(base_bundle),
        prefix_bundle_sha256=sha(base_bundle/'manifest.json'),model_search=str(search_path),
        model_search_sha256=sha(search_path),original_knots_rad=search['original_knots_rad'],
        corrections_rad=search['best']['corrections_rad'],tracking_gains=context[6],
        model_independent_replay=True,online_mpc=False,valid_expert_labels=False)
    manifest['scope']='Preserved native prefix and actual-query conditioned full suffix; native result unverified.'
    manifest['prediction_contract']='Two prediction segments separated by a recorded actual-state query; never a native state reset.'
    added=[search_path,base_bundle/'manifest.json',Path(__file__).resolve(),
        root/'scripts/native_replanned_suffix_v1.py',root/'scripts/check_native_replanned_suffix_v1.py',
        root/'scripts/solve_recorded_recovery_suffix_v1.py',root/'scripts/constrained_recovery_suffix_v1.py']
    manifest['source_sha256'].update(search['source_sha256'])
    manifest['source_sha256'].update({str(p):sha(p) for p in added})
    bundle_output.mkdir(parents=True)
    with (bundle_output/manifest['bundle_name']).open('wb') as f:np.savez_compressed(f,**arrays)
    manifest['bundle_sha256']=sha(bundle_output/manifest['bundle_name'])
    (bundle_output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result=prepare(root,output)
    def replace(text,before,after):
        if text.count(before)!=1:raise ValueError('Changed preparation template '+before)
        return text.replace(before,after)
    entry=Path(result['entry']);text=replace(entry.read_text(),
        'from scripts.native_early_support_tail_v1 import NativeEarlySupportTailV1 as NativeRecoverySequenceReplay',
        'from scripts.native_replanned_suffix_v1 import NativeReplannedSuffixV1 as NativeRecoverySequenceReplay')
    ast.parse(text);entry.write_text(text)
    audit=Path(result['audit']);text=replace(audit.read_text(),
        'from scripts.check_early_support_dispatch_v1 import verify_feedback_dispatch',
        'from scripts.check_native_replanned_suffix_v1 import verify_feedback_dispatch')
    text=replace(text,"files += [Path(__file__),ROOT/'scripts/check_early_support_dispatch_v1.py',",
        "files += [Path(__file__),ROOT/'scripts/check_native_replanned_suffix_v1.py',ROOT/'scripts/native_replanned_suffix_v1.py',\n"
        "        ROOT/'scripts/check_early_support_dispatch_v1.py',")
    ast.parse(text);audit.write_text(text)
    result.update(entry_sha256=sha(entry),audit_sha256=sha(audit),
        checker=str(root/'scripts/check_native_replanned_suffix_v1.py'),
        checker_sha256=sha(root/'scripts/check_native_replanned_suffix_v1.py'),
        bundle=str(bundle_output),bundle_manifest_sha256=sha(bundle_output/'manifest.json'),
        preserved_prefix=True,suffix_query_required=True,model_search=str(search_path),
        native_execution=False,online_mpc=False,teacher_acceptance=False)
    result['source_sha256'].update(manifest['source_sha256'])
    (output/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    for key in ['output','bundle-output','base-bundle','search']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare_suffix(a.root,a.output,a.bundle_output,a.base_bundle,a.search)))
