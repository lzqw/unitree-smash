"""Freeze one learned-phase complete episode and two explicit warmup failures."""
from datetime import datetime,timezone
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
MAIN=ROOT.parent/'unitree_smash'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(MAIN))
from quadsmash.evaluation_v4 import independent_episode,FAULTS
from quadsmash.rl.episode_trace_v4 import unpack_public
from quadsmash.control.full_racket_fixed_actor_body_v4 import make_rally_controller
from quadsmash.control.full_racket_residual_v4 import ResidualActionMixerV4


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert Path.cwd()==MAIN, 'Existing frozen Go2 actor paths require the actual main working directory'
    assert not(HERE/'PLAN.json').exists()
    run=MAIN/'reports/G07_full_racket_v4/body_residual_leg15_fresh_n2048_updates48_01'
    owner=MAIN/'reports/G07_full_racket_v4/body_residual48_after_grid16_queue01'
    admission=read(owner/'ADMISSION_DECISION.json')
    assert all(sha(MAIN/n)==s for n,s in admission['source_hashes'].items())
    assert 'train_full_racket_body_residual_ppo_v4.py' in Path('/proc/1818808/cmdline').read_bytes().decode()
    source_path=Path(admission['source']['policy'])
    protocol_path=MAIN/'configs/g07_full_racket_eval_v4_r3.json'
    protocol=read(protocol_path);protocol['_root']=str(MAIN)
    for name in ['configs/court_net_v4.json','configs/g07_full_racket_eval_v4_r3.json','quadsmash/evaluation_v4.py']:
        assert sha(ROOT/name)==sha(MAIN/name),name
    stems=['lane1824_episode000003','lane0023_episode000000','lane0058_episode000000']
    sources=[Path(__file__),HERE.parent/'body48_replay_walk_phase_check_20261026/RESULT.json',source_path,protocol_path,MAIN/'configs/court_net_v4.json',
        MAIN/'configs/evaluation_protocol_v2.json',owner/'ACTUAL_UPDATE7_CHECK.json',owner/'ACTUAL_UPDATE8_CHECK.json']
    sources += [run/name for name in ['geometry.json','geometry.npz','asset_dependencies.json','INPUTS.json',
        'initial_ground_states.json','source_hashes.json','source_snapshot.zip','dirty.patch']]
    sources += [run/'trajectory_traces'/(stem+ext) for stem in stems for ext in ['.json','.npz']]
    manifest_snapshot=read(run/'trajectory_traces/manifest.json')
    (HERE/'ACTUAL_TRACE_MANIFEST_SNAPSHOT.json').write_text(json.dumps(manifest_snapshot,indent=2)+'\n')
    sources.append(HERE/'ACTUAL_TRACE_MANIFEST_SNAPSHOT.json')
    plan=dict(created_at=datetime.now(timezone.utc).isoformat(),source_sha256={str(p):sha(p) for p in sources},
        selected_stems=stems,selection='Posthoc one learned-phase complete and two early warmup failure modes; not a random or matched-policy sample.',
        canonical_episode_and_full_public_checks=True,base_plus_actual_residual_replay_tolerance=5e-4,
        new_native_execution=False,formal_acceptance=False)
    (HERE/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    snapshot=HERE/'snapshot';snapshot.mkdir(exist_ok=False)
    destination=snapshot/run.name;destination.mkdir()
    hashes={}
    for path in sources:
        if path.parent==run:
            shutil.copy2(path,destination/path.name)
            assert sha(path)==sha(destination/path.name)
            hashes[str(path)]=sha(path)
    provenance=manifest_snapshot['provenance']
    default,limits=[torch.tensor(provenance[k],dtype=torch.float32) for k in
        ['default_joint_position_rad','joint_limits_rad']]
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    cases=[];checks=[]
    for index,stem in enumerate(stems):
        path=run/'trajectory_traces'/(stem+'.npz');meta_path=path.with_suffix('.json');meta=read(meta_path)
        row=meta['raw_episode'];checked=independent_episode(row,protocol)
        assert checked['complete']==(index==0) and row['numerical_failures']==0 and row['ground_init']
        assert sha(path)==meta['data_sha256']
        with np.load(path) as data:
            arrays={k:data[k].copy() for k in data.files}
        public=unpack_public(arrays['packed_public'])
        assert hashlib.sha256(public.tobytes()).hexdigest()==meta['full_public794_sha256']
        clock=arrays['episode_clock_s'];iterations=arrays['learner_iteration']
        assert abs(clock[0])<1e-6 and np.max(abs(np.diff(clock)-.01))<2e-6
        if index==0:
            assert int(iterations.min())>=8 and row['airborne_string_hit'] and not any(row[k] for k in FAULTS)
        else:
            assert int(iterations.max())<=7
        ids=np.arange(0,len(clock),2)
        np.savez_compressed(snapshot/f'case{index}.npz',link_poses=arrays['link_poses'][ids],
            ball_position_m=arrays['ball_position_m'][ids],episode_clock_s=clock[ids],
            valid_first_episode=np.ones(len(ids),dtype=bool),valid_selected_episode=np.ones(len(ids),dtype=bool),
            native_episode_index=np.full(len(ids),meta['episode_index'],dtype=np.int64),
            learner_iteration=iterations[ids])
        boundary=int(np.flatnonzero(iterations!=iterations[0])[0])
        global_start=int(iterations[boundary])*128-boundary
        assert np.array_equal((global_start+np.arange(len(iterations)))//128,iterations)
        assert np.max(abs(arrays['link_poses'][0,0,:3]-[0.,0.,.323]))<1e-5
        assert np.max(abs(arrays['link_poses'][0,0,3:]-[1.,0.,0.,0.]))<1e-5
        controller=make_rally_controller(source,1,'cpu');mixer=ResidualActionMixerV4(1,'cpu',[.15]*18)
        # Lane reset clears lane history but preserves the actual global50Hz hold phase.
        controller.reference_controller.teacher.walk.tick=global_start
        base_error=total_error=0.
        with torch.inference_mode():
            for obs,residual,actual_base,actual in zip(public,arrays['residual_actions'],arrays['base_actions'],arrays['native_actions']):
                obs=torch.from_numpy(obs[None]);residual=torch.from_numpy(residual[None])
                base=controller.action({'policy':obs},default,limits)
                predicted=mixer.action(base,residual,obs,default,limits)
                base_error=max(base_error,float(abs(base.numpy()[0]-actual_base).max()))
                total_error=max(total_error,float(abs(predicted.numpy()[0]-actual).max()))
        assert max(base_error,total_error)<=plan['base_plus_actual_residual_replay_tolerance']
        for p in [path,meta_path]:
            shutil.copy2(p,destination/p.name);hashes[str(p)]=sha(p)
        events=meta['actual_contact_events']
        event=next((e for e in events if e['string_contact']),events[0] if events else None)
        contact_iteration=None
        if event is not None:
            j=int(np.searchsorted(clock,event['time_s'],side='right')-1)
            contact_iteration=int(iterations[j])
        if index==0:
            title='学习阶段｜同回合完整扣杀成立'
            subtitle='TRAIN32｜实际训练回合｜已开始actor学习，仍含随机探索'
            status='拍线击球、攻击出球、越网、原目标首落与稳定恢复全部成立'
            limitation='出球8.15米/秒；首落[3.93,-0.42]米；连续稳定2.00秒'
        elif index==1:
            title='预热期失败｜拍框误击'
            subtitle='TRAIN23首回合｜同轮早期记录，与成功例处于不同训练阶段'
            status='拍框误击，未满足拍线击球要求；完整任务失败'
            limitation='出球9.42米/秒，但没有有效越网；稳定恢复4.04秒'
        else:
            title='预热期失败｜漏球并身体触地'
            subtitle='TRAIN58首回合｜同轮早期记录，与成功例处于不同训练阶段'
            status='未接触球拍，身体触地终止；完整任务失败'
            limitation='终止约2.72秒；没有合格恢复，也没有有效出球记录'
        cases.append(dict(id=index,run=run.name,source_type='actual_saved_changing_policy_training_trace',
            env_id=meta['env_id'],episode_index=meta['episode_index'],source_training_index=meta['source_training_index'],
            row=row,contact=event,title=title,subtitle=subtitle,terminal_status=status,limitation=limitation,
            selected_source_frames=ids.tolist(),source_sampling_hz=100,display_sampling_hz=50,
            actor_iterations_encountered=meta['actor_iterations_encountered'],actual_start_global_control_tick=global_start,
            contact_collection_iteration_zero_based=contact_iteration,whole_source_job_complete=False,
            source_trace_data_sha256=sha(path),source_trace_metadata_sha256=sha(meta_path)))
        checks.append(dict(env_id=meta['env_id'],episode_index=meta['episode_index'],
            outcome=checked,actual_public_rows=len(public),base_action_max_error=base_error,
            full_action_max_error=total_error,contact_collection_iteration_zero_based=contact_iteration,
            min_collection_iteration=int(iterations.min()),max_collection_iteration=int(iterations.max()),actual_start_global_control_tick=global_start,actual_ground_root_pose_checked=True))
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),cases=cases,
        scope='One actual learned-phase complete training episode plus two explicitly earlier warmup failures; no frozen-policy comparison or population score.',
        opening_title='新48更新课程｜学习阶段出现完整回合',
        opening_lines=['第一例：actor已开始学习，实际完成击球、越网、落点和恢复。',
            '另保留同轮预热期的拍框误击、漏球摔倒两类失败。',
            '三例均含随机探索、经过后选；新策略固定128评估尚未完成。'],
        closing_title='一个实际完整回合｜仍需固定策略评估',
        closing_lines=['完整回合跨越多次权重更新，不能据此计算固定成功率。',
            '原2×2米目标、攻击速度和恢复要求全部保留。',
            '训练继续；原正式A/B各200及50/70/85仍未通过。'],
        evaluation_footer='实际训练样本｜1个学习阶段成功＋2个预热期失败｜非固定评估',
        original_source_hashes=hashes,formal_cases_read=False,formal_acceptance=False,goal_complete=False,
        renderer_compatibility_mask='valid_first_episode means selected-trace validity for this renderer only. Actual native episode3 remains3 in case metadata and every native_episode_index sample.',
        caveats=['Actual100Hz body/ball records shown at every second recorded frame; no interpolated or generated motion.',
            'One successful learned-phase episode was selected after inspection. Two early warmup failures are explicitly from a different policy phase, not matched-policy controls.',
            'The successful episode uses actual zero-based collection iterations11..17; actor learning began after collection iteration7. It includes sampled exploration.',
            'The producer job is still live. No new frozen-policy success rate or formal A/B result is inferred.',
            'Exact contact and terminal states come from actual native metadata; first-landing markers use exact event coordinates and adjacent50Hz body poses.'])
    (snapshot/'cases.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    receipt=dict(checked_at=datetime.now(timezone.utc).isoformat(),selected_episode_checks=checks,
        source_actor_checkpoint_sha256=sha(source_path),all_original407_inputs_unchanged=True,
        successful_episode_fully_after_actor_learning_started=True,new_native_execution=False,formal_acceptance=False)
    assert all(sha(MAIN/n)==s for n,s in admission['source_hashes'].items())
    assert all(sha(Path(p))==digest for p,digest in plan['source_sha256'].items())
    (HERE/'NATIVE_EPISODE_CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
