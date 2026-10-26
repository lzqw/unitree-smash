"""Exercise all7 aim proposals over the actual trained third residual actor.

No physics or policy files are produced. The actor is exported only in memory
from real subsequently collected update36. This is interface/command evidence,
not a fixed-policy assessment or a new native execution.
"""
from datetime import datetime,timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ISOLATED = ROOT.parent/'unitree_smash_landing_targets'
REPORTS = ROOT/'reports/G07_full_racket_v4'
OWNER = REPORTS/'body_residual48_after_grid16_queue01'
TRAIN = REPORTS/'body_residual_leg15_fresh_n2048_updates48_01'
sys.path.insert(0,str(ROOT))
from quadsmash.rl.full_racket_body_residual_ppo_v4 import export_residual_policy
from quadsmash.control.full_racket_body_residual_v4 import make_rally_controller
from scripts.initialize_centered_policy_v4 import same_tree


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    assert not (HERE/'PLAN.json').exists()
    factory_path = ISOLATED/'quadsmash/control/full_racket_contact_aim_v4.py'
    factory = module('held_aim_factory_actual_body36',factory_path)
    helper = module('held_actual_body_actor_loader',OWNER/'export_intermediate.py')
    selected = read(OWNER/'ADMISSION_DECISION.json')['source']
    base_path = Path(selected['policy'])
    stream_path = Path(selected['evaluation'])/'public_and_actions_first2.npz'
    diagnostic_path = TRAIN/'ppo/model_35.pt'
    configuration_path = ROOT/'configs/g07_full_racket_body_residual_leg15_v4.json'
    assert sha(diagnostic_path) == read(OWNER/'ROOT_ACTUAL_UPDATE36_CHECK.json')['diagnostic_sha256']
    assert read(TRAIN/'progress.json')['update'] > 36
    with np.load(stream_path) as data:
        public,valid = [data[key].copy() for key in ['observations','valid_first_episode']]
    assert public.shape[1:] == (2,794) and valid.shape == public.shape[:2]
    provenance = read(TRAIN/'trajectory_traces/manifest.json')['provenance']
    (HERE/'ACTUAL_JOINT_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    sources = [Path(__file__),factory_path,OWNER/'export_intermediate.py',base_path,stream_path,
        diagnostic_path,configuration_path,HERE/'ACTUAL_JOINT_PROVENANCE.json']
    held = read(OWNER/'ADMISSION_DECISION.json')['source_hashes']
    assert all(sha(ROOT/name) == digest for name,digest in held.items())
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(path):sha(path) for path in sources},actual_diagnostic_iteration=36,
        actual_valid_public_rows=int(valid.sum()),tested_aims_uv_m=factory.aim_candidates(),
        tolerance=dict(actor_export=2e-5,matched_reference=0.,arm_target_slew_excess_rad=2e-6),
        policy_files_written=False,new_native_execution=False,formal_cases_read=False,formal_acceptance=False)
    (HERE/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    state = torch.load(diagnostic_path,map_location='cpu',weights_only=False)
    assert state['iter'] == 36 and not state['native']['canonical_restart']
    assert state['native']['total_timesteps'] == 9437184
    policy = helper.from_state(state,read(configuration_path))
    sample = torch.from_numpy(public[valid].copy())
    base = torch.load(base_path,map_location='cpu',weights_only=False)
    exported = export_residual_policy(base,policy,read(configuration_path)['joint_scale_rad'],sample,
        dict(native_execution=True,completed_updates=36,actual_control_env_steps=9437184,
            base_controller_mode='frozen_public_policy',
            source_diagnostic_sha256=sha(diagnostic_path),source_canonical_restart=False,
            CPU_export_of_actual_subsequently_collected_actor=True,frozen_policy_performance=False))
    assert same_tree(exported['base_controller'],base)
    eager = torch.jit.load(io.BytesIO(exported['residual_actor_jit']),map_location='cpu').eval()
    with torch.inference_mode():
        expected = policy.act_inference(TensorDict({'policy':sample},batch_size=[len(sample)]))
        export_error = float((eager(sample)-expected).abs().max())
    assert export_error <= plan['tolerance']['actor_export']
    original = make_rally_controller(exported,2,'cpu')
    controllers = []
    for uv in factory.aim_candidates():
        proposal = dict(schema=factory.SCHEMA,source_public_policy=exported,
            parameters=exported['parameters'].clone(),asset_sha256=exported['asset_sha256'],
            aim_uv_m=list(uv),native_execution=False,performance_inherited=False,formal_acceptance=False)
        buffer = io.BytesIO()
        torch.save(proposal,buffer)
        buffer.seek(0)
        restored = torch.load(buffer,map_location='cpu',weights_only=False)
        assert same_tree(restored,proposal) and not restored['native_execution']
        controller = factory.make_rally_controller(restored,2,'cpu')
        node,actors = controller.source,0
        while hasattr(node,'actor') or hasattr(node,'source'):
            if hasattr(node,'actor'):
                assert all(not parameter.requires_grad for parameter in node.actor.parameters())
                actors += 1
                node = node.base
            else:
                node = node.source
        assert actors == 3
        controllers.append(controller)
    default,limits = [torch.tensor(provenance[key],dtype=torch.float32) for key in
        ['default_joint_position_rad','joint_limits_rad']]
    reference_error = 0.
    changes,changed_rows = [0.]*7,[0]*7
    previous,previous_clocks,slew = [None]*7,None,[]
    with torch.inference_mode():
        for obs,keep in zip(public,valid):
            tensor = torch.from_numpy(obs)
            if previous_clocks is not None:
                reset = torch.from_numpy(np.where(obs[:,775] < previous_clocks-1e-5)[0])
                if len(reset):
                    for controller in [original,*controllers]:
                        controller.reset(reset)
            reference = original.action({'policy':tensor},default,limits).numpy()
            for index,controller in enumerate(controllers):
                command = controller.action({'policy':tensor},default,limits)
                assert command.shape == (2,18) and torch.isfinite(command).all() and (command.abs() <= 1).all()
                if keep.any():
                    delta = abs(command.numpy()[keep]-reference[keep])
                    changes[index] = max(changes[index],float(delta.max()))
                    changed_rows[index] += int((delta.max(1) > 1e-7).sum())
                    if index == 0:
                        reference_error = max(reference_error,float(delta.max()))
                target = (limits[:,0]+.5*(command+1)*(limits[:,1]-limits[:,0])).numpy()
                if previous_clocks is not None:
                    consecutive = keep & (obs[:,775] > previous_clocks) & (obs[:,775]-previous_clocks < .011)
                    if consecutive.any():
                        slew.append(float((abs(target[consecutive,12:]-previous[index][consecutive,12:])-.05).max()))
                previous[index] = target.copy()
            previous_clocks = obs[:,775].copy()
    assert reference_error == 0. and all(value > 0 for value in changes[1:])
    assert max(slew) <= plan['tolerance']['arm_target_slew_excess_rad']
    assert same_tree(exported['base_controller'],base)
    assert same_tree(base,torch.load(base_path,map_location='cpu',weights_only=False))
    assert all(sha(Path(name)) == digest for name,digest in plan['source_sha256'].items())
    assert all(sha(ROOT/name) == digest for name,digest in held.items())
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        actual_diagnostic_iteration=36,actual_public_rows=int(valid.sum()),actor_export_max_error=export_error,
        matched_reference_max_action_error=reference_error,all7_commands_finite_and_bounded=True,
        all3_residual_actor_tiers_frozen_and_preserved=True,all7_unexecuted_proposals_round_tripped=True,
        max_arm_target_slew_excess_rad=max(slew),
        candidates=[dict(uv_m=uv,max_action_change=change,actual_rows_with_changed_commands=count)
            for uv,change,count in zip(factory.aim_candidates(),changes,changed_rows)],
        original407_active_sources_unchanged=True,policy_files_written=False,new_native_execution=False,
        scope='CPU proposal compatibility over actual update36 actor and previously measured real public inputs only; no fixed-policy score or native candidate outcome.',
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
