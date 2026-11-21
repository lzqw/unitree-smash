"""Compare repeated training launches across two independently commanded arm directions."""
import argparse
import json
from pathlib import Path

import torch
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch,_same_contract
from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1
from quadsmash.rl.supermimic_native_pair_capture_v1 import BRANCHES,tree_digest
from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical
from quadsmash.rl.supermimic_racket_dynamics_v1 import racket_state
from scripts.evaluate_supermimic_contact_rollout import sha


def read_panel(root):
    read=lambda p:json.loads(p.read_text())
    data=read(root/'support.json')
    path=root/'raw/status.json'
    status=read(path)
    if (not data['training_only'] or data['contact_receipt'] or data['policy_training_admitted']
            or data['status_sha256']!=sha(path) or status['status']!='COLLECTION_FINISHED' or len(status['completed'])!=4):
        raise ValueError('Need completed, scored, training-only capture')
    runs={}
    for item in status['completed']:
        branch=item['branch']
        folder=root/'raw'/item['spec']
        if item['exit_code'] or item['probe_sha256']!=sha(folder/branch/'native_probe.pt'):
            raise ValueError('Capture changed')
        runs[branch]=_load_branch(folder,branch)
    if set(runs)!=set(BRANCHES):
        raise ValueError('Incomplete signed capture')
    _same_contract(runs)
    return runs,data


def verify_alternate_directions(original, tangent):
    direction_fields = {'request_directions', 'direction_seed', 'source_direction_seed', 'direction_design'}
    if {k: v for k, v in original.items() if k not in direction_fields} != {
            k: v for k, v in tangent.items() if k not in direction_fields}:
        raise ValueError('Changed case panel or intervention window')
    if (tangent.get('direction_design') != 'training_prefix_tangent_orthogonal'
            or original.get('direction_seed') is None
            or original['direction_seed'] != tangent.get('source_direction_seed')):
        raise ValueError('Unmatched source direction design')
    x, y = [torch.tensor(spec['request_directions'], dtype=torch.float64) for spec in (original, tangent)]
    if (x.shape != (len(original['lanes']), 18) or y.shape != x.shape
            or any(not torch.isfinite(d).all() or d[:, :12].count_nonzero()
                   or (d.norm(dim=1) - 1).abs().max() > 1e-5 for d in (x, y))
            or (x * y).sum(dim=1).abs().max() > 1e-5):
        raise ValueError('Expected distinct orthogonal unit arm directions for every lane')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('original','tangent','output'):
        p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    a,sa=read_panel(args.original)
    b,sb=read_panel(args.tangent)
    spec=a['nominal_a']['spec']
    other=b['nominal_a']['spec']
    ids=spec['lanes']
    for k in ('case_sha256','source_sha256','physics_fit_sha256'):
        if sa[k]!=sb[k]:
            raise ValueError('Launch, policy or physics fit changed')
    verify_alternate_directions(spec, other)
    for k in ('physics','task_contract','policy_execution','execution_code_sha256','seed','prefix_sha256','executor_before_sha256'):
        if a['nominal_a'][k]!=b['nominal_a'][k]:
            raise ValueError('Unmatched prefix '+k)
    for key in ('start','step_sequence'):
        if tree_digest(a['nominal_a']['raw'][key])!=tree_digest(b['nominal_a']['raw'][key]):
            raise ValueError('Nominal execution differs between directions')
    geom=FullBodyGeometryV1(a['nominal_a']['task_contract']['asset']['sha256'])
    x=encode_physical(a['nominal_a']['raw']['start']['before']['physical'])[ids]
    normal=racket_state(x,geom)[:,3:12].reshape(-1,3,3)[:,:,0]
    def differences(runs):
        def seq(branch):return runs[branch]['raw']['step_sequence']
        def commands(branch):
            return torch.stack([s['final_joint_targets'][ids] for s in seq(branch)],1)
        hold=spec['action_hold_ticks']
        center=[racket_state(seq(k)[hold-1]['next_state'][ids],geom)[:,:3] for k in ('minus','plus')]
        delta=center[1]-center[0]
        dn=(delta*normal).sum(1)
        return (commands('plus')-commands('minus')).flatten(1),dn,(delta-dn[:,None]*normal).norm(dim=1)
    u,dn,dt=differences(a)
    v,en,et=differences(b)
    valid=(u.norm(dim=1)>0)&(v.norm(dim=1)>0)
    cosine=torch.nn.functional.cosine_similarity(u,v,dim=1)
    case_rows=lambda support:{r['lane']:r for r in support['native_racket']['cases']}
    old,new=case_rows(sa),case_rows(sb)
    if (set(old) != set(ids) or set(new) != set(ids)
            or any(old[l]['case_id'] != new[l]['case_id'] for l in ids)):
        raise ValueError('Case identity changed across directions')
    result=[]
    for i,l in enumerate(ids):
        result.append(dict(lane=l,case_id=old[l]['case_id'],executed_command_cosine=float(cosine[i]) if valid[i] else None,
            initial_face_normal_motion_m=[float(dn[i]),float(en[i])],initial_face_tangent_motion_m=[float(dt[i]),float(et[i])],
            modes=[old[l]['mode'],new[l]['mode']],ball_response_mps=[r[l]['native_delta_mps'] for r in (old,new)]))
    stable=set(sa['independent_stable_with_response']);new_stable=set(sb['independent_stable_with_response'])
    report=dict(same_launch_actors_and_nominal_trajectories=True,distinct_launches=len(ids),direction_pairs=2*len(ids),
                stable_union=len(stable|new_stable),stable_intersection=len(stable&new_stable),
                signed_target_direction_cosine_median=float(cosine[valid].median()) if valid.any() else None,
                support_hashes=[sha(s/'support.json') for s in (args.original,args.tangent)],
                cases=result,contact_receipt=False)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)


if __name__=='__main__':main()
