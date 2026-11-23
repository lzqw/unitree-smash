"""Probe recorded pre-contact states with a common frozen world-model checkpoint."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch

from quadsmash.physics.full_racket_v4 import load_racket, string_strips
from quadsmash.rl.supermimic_archive_storage_v1 import read_archive_rows
from quadsmash.rl.supermimic_compact_starts import compact_start
from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
from quadsmash.rl.supermimic_physical_learner_v1 import PhysicalHorizonTrainer
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest
from quadsmash.rl.supermimic_task_space_reference_v1 import TaskSpaceReferenceBankV1
from quadsmash.rl.supermimic_terminal_scoring import approach_ball_scoring_protocol
from quadsmash.rl.supermimic_timebase_v2 import Timebase
from quadsmash.rl.supermimic_toi_racket_objective import string_boundary
from run_supermimic_approach_v2 import dump
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_physical_dataset_v1 import decode_physical, encode_physical
from quadsmash.rl.supermimic_release_calibration import racket_release_boundary

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lead-ticks', type=int, choices=(3, 5, 20), default=20)
for flag in ('initial-policy', 'later-policy', 'initial-capture', 'later-capture', 'registration', 'output'):
    parser.add_argument('--'+flag, type=Path, required=True)
parser.add_argument('--world-base', choices=('initial','later'), default='later')
parser.add_argument('--body-candidate', type=Path)
args = parser.parse_args()
lead_ticks = args.lead_ticks
run = args.output.resolve().parent
registration = args.registration
checkpoints = [args.initial_policy, args.later_policy]
captures = [args.initial_capture, args.later_capture]


def read(path):
    return json.loads(path.read_text())


def load(path):
    return torch.load(path, map_location='cpu', mmap=True, weights_only=False)


saved = [load(p) for p in checkpoints]
base = 0 if args.world_base == 'initial' else 1
config = saved[base]
info = config['metadata']['physical_imagination']
paths = info['checkpoint_paths']
argv = read(registration)['command']
actor, adapter, template, launch, _ = load_physical_loop(
    archive=argv[argv.index('--archive')+1], safety_mapping=argv[argv.index('--safety-mapping')+1],
    physical_checkpoint=paths['physical'], ball_checkpoint=paths['ball'],
    task_checkpoint=paths['task'], action_ball_checkpoint=paths['action_ball'],
    execution_root=info['g07_execution_root'], device='cpu')
adapter.load_state_dict(config['world'])
candidate_sha = None
if args.body_candidate is not None:
    candidate = load(args.body_candidate)
    with checkpoints[base].open('rb') as stream:
        source_sha = hashlib.file_digest(stream,'sha256').hexdigest()
    if (candidate['schema'] != 'e1-v2-event-precontact-body-v1'
            or candidate['parent_checkpoint_sha256'] != source_sha
            or candidate['parent_world_sha256'] != tensor_digest(config['world'].items())
            or candidate['body_sha256'] != tensor_digest(candidate['model'].items())
            or set(candidate['train_lanes']) & set(config['config']['wm_holdout_lanes'])):
        raise ValueError('Body candidate does not match frozen diagnostic source and split')
    adapter.physical_model.body.load_state_dict(candidate['model'],strict=True)
    candidate_sha = candidate['body_sha256']
actor.configure_approach_gate(.25)
actor_hashes = [tensor_digest(checkpoint['actor'].items()) for checkpoint in saved]
terminal = [{r['case_id']:r for r in map(json.loads,(directory/'native_episodes.jsonl').read_text().splitlines())}
            for directory in captures]
failure_keys, control_keys = [], []
for key in sorted(terminal[0].keys() & terminal[1].keys()):
    a,b = (x[key] for x in terminal)
    if (any(r['arm_identification_episode'] or r['episode_index'] != 0 for r in (a,b))
            or not a['qualified_airborne_string_contact'] or b['contact_time_s'] is None):
        continue
    if b['native_contact']['frame_hit'] and not b['native_contact']['string_hit']:
        failure_keys.append(key)
    elif b['qualified_airborne_string_contact']:
        control_keys.append(key)
holdout = set(config['config']['wm_holdout_lanes'])
groups = {'flip': failure_keys, 'normal': control_keys[:len(failure_keys)], 'low_speed': [], 'marginal_success': []}
for k, r in terminal[1].items():
    v = r['outgoing_speed_mps']
    if (k in terminal[0] and r['env_id'] not in holdout and v is not None
            and not any(x[k]['arm_identification_episode'] or x[k]['episode_index'] != 0 or x[k]['contact_time_s'] is None
                        for x in terminal) and r['qualified_airborne_string_contact']):
        if v < 4.5:
            groups['low_speed'].append(k)
        elif v < 4.75 and r['ball_side_complete_except_recovery']:
            groups['marginal_success'].append(k)
groups['marginal_success'] = sorted(groups['marginal_success'], key=lambda k:terminal[1][k]['outgoing_speed_mps'])[:4]
key_groups = {}
for group, members in groups.items():
    for k in members:
        key_groups.setdefault(k, []).append(group)
keys = list(key_groups)
assert failure_keys
tick = {key: int(min(t[key]['contact_time_s'] for t in terminal) / .01)-lead_ticks for key in keys}
cases = {key: dict(case_id=key, cohorts=key_groups[key], pre_contact_tick=tick[key]) for key in keys}

def geometry(state):
    body = decode_physical(state)
    kin = adapter.physical_model.geometry
    pose = kin(body['joint_position'], body['joint_velocity'], body['root_position_w'],
               body['root_quaternion_w'], body['root_link_velocity_w'], body['root_angular_velocity_w'])
    return pose, kin.ids['Racket']


def trace_one(state, targets, impact=None):
    pose, racket = geometry(state)
    center, rotation = pose['position'][0,racket], pose['rotation'][0,racket]
    lever = state[0,49:52]-center
    point_v = pose['velocity'][0,racket] + torch.linalg.cross(pose['angular_velocity'][0,racket],lever)
    local = rotation.T @ lever
    rel_v = rotation.T @ (state[0,52:55]-point_v)
    incoming_normal = rotation[:,0] * torch.where(rel_v[0]>0,-1.,1.)
    inward,offset = trainer.string_boundary
    at_contact = impact is not None and bool(impact['active'][0]) and int(impact['object_id'][0])==0
    racket_part = int(impact['racket_part'][0]) if at_contact else None
    return dict(
        final_joint_targets=targets[0].tolist(),root_position=state[0,:3].tolist(),
        root_quaternion=state[0,3:7].tolist(),root_linear_velocity=state[0,7:10].tolist(),
        root_angular_velocity=state[0,10:13].tolist(),racket_center=center.tolist(),
        racket_normal=incoming_normal.tolist(),racket_contact_point_velocity=point_v.tolist(),
        relative_ball_position=local.tolist(),relative_ball_velocity=rel_v.tolist(),
        signed_string_margin_m=float((local[1:]@inward.T-offset).amin()),
        collision_time_within_tick_s=float(impact['time'][0]) if at_contact else None,
        collision_string_margin_m=float((impact['local'][0,1:]@inward.T-offset).amin()) if at_contact else None,
        contact_kind=('string' if 0<=racket_part<trainer.string_strips else 'frame_or_nonstring') if at_contact else None)


def summarize(trace, native, start):
    first = next((i for i,r in enumerate(trace) if r['contact_kind']),None)
    deltas = []
    for i,(p,r) in enumerate(zip(trace,native)):
        field_errors={name:math.dist(p[name],r[name]) for name in
                      ('final_joint_targets','root_position','root_linear_velocity','root_angular_velocity',
                       'racket_center','racket_contact_point_velocity','relative_ball_position','relative_ball_velocity')}
        field_errors['root_quaternion']=min(math.dist(p['root_quaternion'],r['root_quaternion']),
                                           math.dist(p['root_quaternion'],[-x for x in r['root_quaternion']]))
        field_errors['racket_normal']=math.dist(p['racket_normal'],r['racket_normal'])
        deltas.append(dict(tick=start+i,**field_errors,
                           signed_string_margin_m=abs(p['signed_string_margin_m']-r['signed_string_margin_m'])))
    levels={'final_joint_targets':.02,'racket_center':.01,'racket_normal':.10,
            'relative_ball_position':.01,'signed_string_margin_m':.01}
    first_errors={key:next((d['tick'] for d in deltas if d[key]>limit),None) for key,limit in levels.items()}
    return dict(first_threshold_crossings=first_errors,thresholds=levels,first_predicted_contact=(
        dict(tick=start+first,kind=trace[first]['contact_kind'],**{k:trace[first][k] for k in
           ('collision_time_within_tick_s','collision_string_margin_m')}) if first is not None else None),
        errors=deltas)


@torch.no_grad()
def forced(start, window, mode):
    state, c = start.physical, start.clock_goal_context
    feet,present = c['foot_contact'],c['ball_present']
    trace, release = [], None
    hit = False
    for row in window:
        if mode in ('C','D'):
            robot = row['state'][:,:49]
            state = torch.cat((robot,state[:,49:]),-1)
            feet = row['feet']
        if mode=='D' and not hit and not row['native_hit_before']:
            state = row['state']
            present = row['present']
        next_state,feet,present,details=ball_transition(adapter.physical_model,adapter.ball_model,state,
            row['targets'],row['previous'],feet,present,row['elapsed'],c['launch'],
            action_model=adapter.action_ball_model,return_details=True,
            native_robot_endpoint=row['next_state'][:,:49] if mode in ('C','D') else None)
        step=trace_one(state,row['targets'],details['impulse_info'])
        trace.append(step)
        hit |= step['contact_kind'] is not None
        pose, racket = geometry(next_state)
        if hit and release is None and bool(racket_release_boundary(next_state[:,49:],pose,racket,
                adapter.action_ball_model.impulse_prior.preview,separation_m=trainer.approach_separation_m)[0]):
            release=next_state[0,52:55].tolist()
        state=next_state
    return trace,release


for stage, directory, episodes, weights, sha in zip(('a32','a40'),captures,terminal,saved,actor_hashes,strict=True):
    archive = directory/'native_transitions'
    manifest = read(archive/'manifest.json')
    actor.load_state_dict(weights['actor'])
    trainer=None
    windows = {k:[] for k in keys}
    states = {}
    cursor = 0
    for chunk in manifest['chunks']:
        needed = [k for k in keys if tick[k]<cursor+chunk['ticks'] and tick[k]+40>cursor]
        if needed:
            with read_archive_rows(archive,chunk,scratch=run) as rows:
                for key in needed:
                    lane=episodes[key]['env_id']
                    for t in range(max(tick[key],cursor),min(tick[key]+40,cursor+len(rows))):
                        row=rows[t-cursor]
                        assert row['actor_state_sha256']==sha and episodes[key]['behavior_at_terminal']['actor_sha256']==sha
                        b,a = row['before'],row['after']
                        assert int(b['elapsed_tick'][lane])==t and int(b['episode_index'][lane])==0
                        before,after=(x['physical'] for x in (b,a))
                        lane_slice=slice(lane,lane+1)
                        windows[key].append(dict(state=encode_physical(before)[lane_slice].clone(),
                            next_state=encode_physical(after)[lane_slice].clone(),
                            targets=row['final_joint_targets'][lane_slice].clone(),previous=before['final_action'][lane_slice].clone(),
                            feet=before['foot_contact'][lane_slice].float().clone(),present=before['ball_present'][lane_slice].clone(),
                            elapsed=before['elapsed_s'][lane_slice].clone(),native_hit_before=bool(b['task']['had_contact'][lane])))
                        if t==tick[key]:
                            assert not bool(b['task']['had_contact'][lane])
                            mask=torch.zeros(len(row['observation']),dtype=torch.bool)
                            mask[lane]=True
                            record,sched=compact_start(row,mask,launch)
                            states[key]=PhysicalImaginationStateV1.from_record(record,executor_template=template,
                                launch=sched,device='cpu',array_plan_observations=True)
        cursor+=chunk['ticks']
    for key in keys:
        state=states[key]; native=episodes[key]; window=windows[key]
        assert len(window)==40
        if trainer is None:
            trainer = PhysicalHorizonTrainer(actor,adapter,state,
                Timebase(**read(Path('configs/supermimic_task_horizon_v2.json'))),
                block_name='approach',reference_bank=TaskSpaceReferenceBankV1.from_payload(config['reference_bank']),
                tracking_weight=1.,approach_objective='toi_racket_only')
            trainer.approach_v2 = True
            trainer.approach_ball_protocol = approach_ball_scoring_protocol()
            trainer.prioritize_approach_flight = True
            trainer.approach_relative_face_speed = True
            trainer.approach_separation_m = read(Path('configs/shuttle_physics_v2.json'))['separation_hysteresis_m']
            trainer.research_court = read(Path('configs/court_net_v4.json'))
            trainer.landing_and_first_contact_objectives = True
            racket=Path(info['g07_execution_root'])/'configs/full_racket_v4_r3.json'
            trainer.string_strips = len(string_strips(load_racket(racket)))
            trainer.string_boundary = string_boundary(racket,next(actor.parameters()))
            trainer.approach_speed_calibration = config['approach_speed_calibration']
            trainer.reference.configure_approach_gate(.25)
        trace_a=[]
        release_a=[]
        def hook(module, inputs, output):
            trace_a.append(trace_one(inputs[0].physical,output[3]['final_joint_targets'],output[3]['impulse_info']))
            if release_a or not any(row['contact_kind'] for row in trace_a):
                return
            pose,racket = geometry(output[0].physical)
            ball = output[3]['outgoing_ball']
            if bool(racket_release_boundary(ball,pose,racket,adapter.action_ball_model.impulse_prior.preview,
                                            separation_m=trainer.approach_separation_m)[0]):
                release_a.append(ball[0,3:].tolist())
        handle=adapter.register_forward_hook(hook)
        try:
            with torch.no_grad():
                out=trainer._rollout_objective(state,collect_critic_data=False)
        finally:
            handle.remove()
        native_trace=[trace_one(row['state'],row['targets']) for row in window]
        route_a=dict(command_source='Actor + G07 closed loop',trace=trace_a,**summarize(trace_a,native_trace,tick[key]),
                     resolved_string_release=bool(out['ball_scored'][0]),
                     predicted_release_velocity_mps=release_a[0] if release_a else None)
        paths={'A':route_a}
        for mode,label in (
            ('B','recorded final joint targets + previous final normalized action'),
            ('C','recorded final commands and consecutive native robot states'),
            ('D','recorded final commands, native robot states and native incoming shuttle until first contact')):
            trace,release=forced(state,window[:len(trace_a)],mode)
            paths[mode]=dict(command_source=label,trace=trace,**summarize(trace,native_trace,tick[key]),
                             predicted_release_velocity_mps=release)
        cases[key][stage]=dict(native=dict(kind=native['native_contact']['contact_type'],
             contact_time_s=native['contact_time_s'],signed_margin_m=native['native_contact'].get('signed_string_margin_m'),
             outgoing_speed_mps=native['outgoing_speed_mps'],trace=native_trace),paths=paths)
        print(stage,key,{k:(v['first_predicted_contact'] or {}).get('kind') for k,v in paths.items()},flush=True)

dump(args.output,dict(world_model_checkpoint=str(checkpoints[base]),body_candidate_sha256=candidate_sha,actor_checkpoints=list(map(str,checkpoints)),
    lead_ticks=lead_ticks,cohorts=groups,scope='Read-only frozen diagnostics on native archived trajectories',
    cases=list(cases.values())))
