"""Freeze two actual saved TRAIN traces for an explicitly exploratory video.

Both lanes were registered before training. Their selection for this short
diagnostic video occurs after seeing their failure modes, and is marked so.
This never rewrites native outcomes or pretends a changing policy is frozen.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from quadsmash.evaluation_v4 import independent_episode
from quadsmash.rl.episode_trace_v4 import unpack_public


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def same(a,b):return json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)


def main():
    reports=ROOT/'reports/G07_full_racket_v4'
    run=reports/'residual_ppo_centered_base8_n2048_probe8_02';directory=run/'trajectory_traces'
    output=reports/'video_snapshot_centered_training_capture_20261026_v2'
    recorded=json.loads((directory/'manifest.json').read_text());assert recorded['enabled']and recorded['native_execution']
    parent_plan=reports/'centered_residual_pilot8_queue02/PREDECLARED_PLAN.json'
    assert set([107,8])<=set(recorded['selection_settings']['first_episode_lanes'])
    raw=json.loads((run/'episodes.partial.json').read_text());indexed={(r['env_id'],r['episode_index']):r for r in raw}
    for lane in [107,8]:
        if (lane,0)not in indexed:
            raise RuntimeError('Wait for the running producer to publish its next completed-update raw rows: '+str(lane))
    output.mkdir(exist_ok=False)
    protocol=json.loads((ROOT/'configs/g07_full_racket_eval_v4_r3.json').read_text());protocol['_root']=str(ROOT)
    destination=output/run.name;destination.mkdir();hashes={}
    for name in ['geometry.json','geometry.npz','asset_dependencies.json','INPUTS.json','initial_ground_states.json','source_hashes.json','source_snapshot.zip','dirty.patch']:
        source=run/name;assert source.exists();shutil.copy2(source,destination/name)
        assert sha(source)==sha(destination/name);hashes[str(source.relative_to(ROOT))]=sha(source)
    cases=[]
    for index,lane in enumerate([107,8]):
        entry=next(e for e in recorded['saved_episodes']if e['env_id']==lane and e['episode_index']==0)
        source=directory/entry['data_file'];meta_path=source.with_suffix('.json');meta=json.loads(meta_path.read_text());row=meta['raw_episode']
        assert sha(source)==entry['data_sha256']==meta['data_sha256']
        assert same(row,indexed[lane,0])and row['ground_init']
        outcome=independent_episode(row,protocol);assert not outcome['complete']and not outcome['airborne_contact']
        assert not meta['actual_contact_events']and not row['had_contact']
        with np.load(source)as data:
            public=unpack_public(data['packed_public']);assert hashlib.sha256(public.tobytes()).hexdigest()==entry['full_public794_sha256']
            ids=np.arange(0,len(public),2);clock=data['episode_clock_s'][ids]
            assert np.max(abs(np.diff(clock)-.02))<2e-6 and abs(clock[0])<1e-6
            np.savez_compressed(output/f'case{index}.npz',link_poses=data['link_poses'][ids],
                ball_position_m=data['ball_position_m'][ids],episode_clock_s=clock,
                valid_first_episode=np.ones(len(ids),dtype=bool))
        for path in [source,meta_path]:
            shutil.copy2(path,destination/path.name);hashes[str(path.relative_to(ROOT))]=sha(path)
        recovered=row['jump']['stable_recovery']
        title='起跳后稳定恢复｜本回合漏球'if recovered else'身体触地终止｜本回合漏球'
        limitation=f'实测稳定窗口{row["jump"]["recovery_s"]:.2f}秒；'+('首次落点在己方，未触球'if row['ground_seen']else'终止前未观察到首次落点')
        cases.append(dict(id=index,run=run.name,source_type='actual_saved_changing_policy_training_trace',
            env_id=lane,episode_index=0,source_training_index=lane,
            title=title,subtitle=f'预登记TRAIN{lane}首回合｜实际训练探索动作｜含完整机械臂和球拍',
            terminal_status='任务失败｜'+title,limitation=limitation,row=row,contact=None,
            selected_source_frames=ids.tolist(),source_sampling_hz=100,display_sampling_hz=50,
            actor_iterations_encountered=entry['actor_iterations_encountered'],whole_source_job_complete=False,
            scope='One complete saved native ground-start episode across changing weights/normalization and exploration; no frozen actor performance.'))
    hashes[str(parent_plan.relative_to(ROOT))]=sha(parent_plan);hashes[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Actual initial exploratory episodes of the centered learner. Two failure modes selected after inspecting saved traces; not blind sampling or a frozen policy demonstration.',
        cases=cases,opening_title='新增量方法｜首批真实训练片段',
        opening_lines=['同一完整球拍模型；两段均从地面实际执行。','展示起跳恢复但漏球，以及身体触地终止。','这些是含探索动作的训练回合，冻结评测尚待。'],
        closing_title='两例均失败｜继续训练与冻结评测',
        closing_lines=['稳定恢复与实际击球分别核对，不拼接成成功。','选择发生在观察轨迹之后，属于诊断展示。','完整目标仍须同一冻结策略通过原A/B各200回合。'],
        evaluation_footer='实际训练探索轨迹｜两例均失败｜无新冻结或正式成绩',
        original_source_hashes=hashes,formal_cases_read=False,formal_acceptance=False,goal_complete=False,
        caveats=['Actual100Hz full-body poses/actions reduced to every second recorded frame for50Hz; no interpolated or generated motion.',
            'Both lanes were registered before training; their selection for this short two-case diagnostic occurs after inspecting outcomes.',
            'Actor iteration indices denote actual weights used during collection. TRAIN8 uses iterations0..2 before the first actor update; TRAIN107 spans0..6.',
            'The producer job is still running; the later8-update frozen actor performance is not shown or inherited.',
            'Exact terminal-state poses and exact first-landing events come from saved native metadata, not reconstructed success motion.'])
    (output/'cases.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(snapshot=str(output),cases=[c['env_id']for c in cases],display_frames=[len(c['selected_source_frames'])for c in cases],formal_acceptance=False)))


if __name__=='__main__':main()
