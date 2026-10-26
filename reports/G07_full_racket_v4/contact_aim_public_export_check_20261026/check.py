"""Round-trip proposed aim policies and replay actual training base commands.

The proposed policy is explicitly unexecuted, exists only in memory, and is
never saved as a measured robot policy. Its embedded source is the measured
grid8 teacher. This checks the public factory later used by fixed assessment.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent/'unitree_smash'
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(MAIN))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.initialize_centered_policy_v4 import same_tree

module_path = ROOT/'quadsmash/control/full_racket_contact_aim_v4.py'
spec = importlib.util.spec_from_file_location('preserved_contact_aim_factory',module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE/'PLAN.json').exists()
    owner = MAIN/'reports/G07_full_racket_v4/body_residual48_after_grid16_queue01'
    source_path = Path(read(owner/'ADMISSION_DECISION.json')['source']['policy'])
    source = torch.load(source_path,map_location='cpu',weights_only=False)
    actual = ROOT/'reports/G07_full_racket_v4/body_residual48_actual_command_check_20261026'
    manifest_path = actual/'ACTUAL_MANIFEST_SNAPSHOT.json'
    manifest = read(manifest_path)
    trace_root = MAIN/'reports/G07_full_racket_v4/body_residual_leg15_fresh_n2048_updates48_01/trajectory_traces'
    streams = []
    dependencies = [Path(__file__),module_path,source_path,manifest_path]
    for entry in manifest['saved_episodes']:
        path = trace_root/entry['data_file']
        meta = read(path.with_suffix('.json'))
        assert sha(path) == meta['data_sha256'] == entry['data_sha256']
        with np.load(path) as arrays:
            public = unpack_public(arrays['packed_public'])
            base = arrays['base_actions'].copy()
        assert hashlib.sha256(public.tobytes()).hexdigest() == meta['full_public794_sha256']
        streams.append(dict(public=public,base=base))
        dependencies.extend([path,path.with_suffix('.json')])
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p):sha(p) for p in dependencies},
        original_actual_episodes=len(streams),selected_not_random=True,
        tested_aims_uv_m=[[0,.1],[0,.06]],public_factory_tolerance=5e-4,
        pure_memory_unexecuted_proposals=True,new_native_execution=False,formal_acceptance=False)
    (HERE/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    count = len(streams)
    provenance = manifest['provenance']
    default,limits = [torch.tensor(provenance[k],dtype=torch.float32)
        for k in ('default_joint_position_rad','joint_limits_rad')]
    controllers = []
    for uv in plan['tested_aims_uv_m']:
        proposal = dict(schema=module.SCHEMA,source_public_policy=source,
            source_policy_sha256=sha(source_path),parameters=source['parameters'].clone(),
            asset_sha256=source['asset_sha256'],aim_uv_m=uv,
            native_execution=False,performance_inherited=False,formal_acceptance=False)
        buffer = io.BytesIO()
        torch.save(proposal,buffer)
        buffer.seek(0)
        restored = torch.load(buffer,map_location='cpu',weights_only=False)
        assert same_tree(restored,proposal) and not restored['native_execution']
        assert same_tree(restored['source_public_policy'],source)
        controllers.append(module.make_rally_controller(restored,count,'cpu'))
    maximum_reference = maximum_change = 0.
    changed_rows = total_rows = 0
    slew_excess = []
    previous = [None,None]
    previous_clocks = None
    with torch.inference_mode():
        for tick in range(max(len(s['public']) for s in streams)):
            keep = np.array([tick < len(s['public']) for s in streams])
            indices = [min(tick,len(s['public'])-1) for s in streams]
            public = torch.from_numpy(np.stack([s['public'][i] for s,i in zip(streams,indices)]))
            actual = np.stack([s['base'][i] for s,i in zip(streams,indices)])
            commands = [c.action({'policy':public},default,limits) for c in controllers]
            maximum_reference = max(maximum_reference,float(abs(commands[0].numpy()[keep]-actual[keep]).max()))
            delta = abs(commands[1].numpy()[keep]-commands[0].numpy()[keep])
            maximum_change = max(maximum_change,float(delta.max()))
            changed_rows += int((delta.max(1) > 1e-7).sum())
            total_rows += int(keep.sum())
            for i,command in enumerate(commands):
                assert torch.isfinite(command).all() and (command.abs() <= 1).all()
                target = limits[:,0]+.5*(command+1)*(limits[:,1]-limits[:,0])
                if previous_clocks is not None:
                    consecutive = keep & (public[:,775].numpy() > previous_clocks)
                    if consecutive.any():
                        slew_excess.append(float((abs(target.numpy()[consecutive,12:]-previous[i][consecutive,12:])-.05).max()))
                previous[i] = target.numpy().copy()
            previous_clocks = public[:,775].numpy().copy()
    assert maximum_reference <= plan['public_factory_tolerance']
    assert maximum_change > 0 and changed_rows > 0 and max(slew_excess) <= 2e-6
    assert same_tree(source,torch.load(source_path,map_location='cpu',weights_only=False))
    assert all(sha(Path(p)) == digest for p,digest in plan['source_sha256'].items())
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        actual_selected_episodes=count,actual_public_rows=total_rows,
        public_factory_baseline_CPU_vs_actual_base_max_action_error=maximum_reference,
        interior_aim_max_action_change=maximum_change,actual_input_rows_with_changed_commands=changed_rows,
        max_arm_target_slew_excess_rad=max(slew_excess),
        both_public_proposals_serialized_and_restored_exactly=True,
        embedded_measured_source_unchanged=True,unexecuted_proposal_flags_preserved=True,
        native_policy_files_written=False,new_native_execution=False,formal_acceptance=False,
        scope='Public policy export/factory and slew check on5197 previously recorded public inputs; no hit or performance inference.')
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
