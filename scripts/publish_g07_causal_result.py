#!/usr/bin/env python3
"""Recompute both formal panels, publish evidence only if they actually pass.

No GPU work is started. Failed aggregates remain available for inspection and
never replace root evidence. Run timestamps come from the two completed jobs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.verify_goal import verify


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_bytes(data)
    os.replace(temporary, path)


def save_json(path, value):
    atomic(path, (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--policy-lock', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--panel-output', type=Path, action='append', required=True)
    parser.add_argument('--aggregate-output', type=Path, required=True)
    args = parser.parse_args()
    command = [sys.executable, str(ROOT/'scripts/aggregate_g07_hybrid_evaluation.py'),
               '--protocol', str(args.protocol), '--policy-lock', str(args.policy_lock),
               '--output', str(args.aggregate_output)]
    for path in args.panel_output:
        command += ['--panel-output', str(path)]
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode:
        return result.returncode
    metrics = read(args.aggregate_output/'metrics.json')
    provenance = read(args.aggregate_output/'eval_manifest.json')
    lock, protocol = read(args.policy_lock), read(args.protocol)
    if not metrics.get('thresholds_met') or digest(args.checkpoint)!=lock['policy_sha256']:
        raise ValueError('Formal metrics or frozen checkpoint do not qualify')
    for path, sha in lock['source_hashes'].items():
        if digest(ROOT/path)!=sha:
            raise ValueError('Locked implementation changed: '+path)
    report = ROOT/'reports/G07'
    policy = ROOT/'artifacts/G07/policy.pt'
    if policy.exists() or (report/'evidence.json').exists():
        raise ValueError('Refuse to overwrite an existing formal policy/evidence')
    ledger = read(ROOT/'status/GPU_LEDGER.json')
    window = 'g07-net-feasibility-20261023'
    runs = [r for r in ledger['runs'] if r.get('budget_window')==window]
    if any(r['status']=='RUNNING' for r in runs):
        raise ValueError('Finish the active job before publishing')
    hours = sum(r['charged_gpu_hours'] for r in runs)
    allocation = ledger['budget_windows'][window]
    if any(j.get('budget_window')!=window or datetime.fromisoformat(j['started_at'])<datetime.fromisoformat(allocation['started_at']) for j in provenance['jobs']):
        raise ValueError('Formal jobs do not belong to this authorization')
    if hours>10+1e-9 or any(datetime.fromisoformat(j['finished_at'])>datetime.fromisoformat(allocation['deadline']) for j in provenance['jobs']):
        raise ValueError('Resource authorization was exceeded')
    history = report/'history_before_causal_formal_pass_v1'
    if history.exists():
        raise ValueError('Historical archive already exists; inspect before retrying')
    history.mkdir()
    previous = {}
    for name in ['REPORT.md','commands.log','episodes.csv','episodes.json','metrics.json','eval_manifest.json']:
        path = report/name
        if path.exists():
            previous[name] = digest(path)
            shutil.copy2(path, history/name)
    save_json(history/'archive_manifest.json', dict(kind='Unmodified prior G07 root files, including earlier failures',sha256=previous))
    for name in ['metrics.json','episodes.csv','episodes.json','eval_manifest.json']:
        atomic(report/name, (args.aggregate_output/name).read_bytes())
    atomic(policy, args.checkpoint.read_bytes())
    rows = read(args.aggregate_output/'episodes.json')
    lines = []
    for panel in metrics['panel_metrics']:
        batch = [r for r in rows if r['frozen_panel_id']==panel['panel']]
        air = sum(r['airborne_contact'] for r in batch)
        recovery = sum(r['airborne_contact'] and r['recovery_success'] and not r['robot_net_contact'] for r in batch)
        full = sum(r['complete_smash'] and not r['robot_net_contact'] and r['ground_init'] for r in batch)
        lines.append(f"| {panel['panel']} | {full}/{len(batch)} ({100*panel['complete_smash_success_rate']:.2f}%) | {air}/{len(batch)} | {recovery}/{air} ({100*panel['recovery_given_airborne_contact']:.2f}%) | {sum(r['robot_net_contact'] for r in batch)} | {panel['numerical_failures']} |")
    body = f'''# G07：固定协议400回合结果

同一冻结策略、源码和协议完成两组各200个地面起始回合，原数值门槛全部满足。随后执行结构/依赖/哈希校验，结果见 `formal_verification.json`；只有校验通过才将状态设为PASSED。

| 面板 | 完整扣杀 | 空中触球 | 触球条件恢复 | 机器人碰网 | 数值故障 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

验收使用两组较差比例：完整 `{metrics['complete_smash_success_rate']:.6f}`（要求≥.5）、空中触球 `{metrics['airborne_contact_rate']:.6f}`（≥.7）、条件恢复 `{metrics['recovery_given_airborne_contact']:.6f}`（≥.85）；地面起始比例 `{metrics['ground_init_fraction']}`，数值故障合计 `{metrics['numerical_failures']}`，总回合 `{metrics['sample_count']}`。碰网和所有其他失败始终保留在分母，不以筛选成功视频替代统计。

完整成功要求同一回合依次满足：实际腾空有效拍面碰撞、出球速度≥8m/s且vx>0/vz≤−1m/s、真实越网、首次落入x[2,4]/y[−1,1]目标区，以及原连续.8秒稳定恢复；机器人碰网回合不计完整或条件恢复成功。空中触球率本身不等同于有效拍面率，完整事件链另行要求有效拍面。

冻结策略SHA256：`{lock['policy_sha256']}`，文件 `artifacts/G07/policy.pt` 与评测权重原字节一致。协议：`{args.protocol}`，SHA `{digest(args.protocol)}`；源码锁：`{args.policy_lock}`，源码Git `{lock['git_sha']}`。A/B原始数据、初始化状态、首回合完整记录、固定案例位姿、job和source快照均保留在 `{args.panel_output[0]}` 与 `{args.panel_output[1]}`。聚合来源在 `{args.aggregate_output}`。

执行系统是冻结学习Go2行走/站立网络、实测数据拟合的12段PWA全身动作、搜索/学习适配参数、基于当前公开观测的球模型预测，以及有界pitch/角速度/前向速度反馈与机械臂残差。教师前缀与教师动作函数调用0；没有回合中传送、外加悬空力或出球速度覆盖。实际采用何种可选控制字段以冻结权重和源锁为准，开发中的其他分支不混入这份成绩。

资产为标准PiPER+Go2+球拍，20.646991kg，USD SHA `bfb831e4afc8a1a705bb0fb6f001c25eeaa401822978267a0ee5e151007511f3`。原关节范围、腿23.5Nm/30rad/s与臂100Nm/5rad/s、PD均保留。物理步62.5微秒、控制.01秒、回合8.5秒。网体v3中心1.524m/侧1.55m、宽2m/厚.02m、32段抛物线顶，仍是ground-to-top实体研究障碍；长准备来球可能从数十米初始位置发射。这是明确版本化的仿真研究场景，未完成现实球网/气动/硬件标定。固定400属于G07开发验收，G10多种子holdout与方法优势研究仍未执行。

扫掠审计仍使用原几何、查询函数与1e-6容差，最大计算4096次；未收敛仍为硬数值失败。旧256次的失败不回写。CPU/CUDA10查询复核见 `sweep_compute_cpu02`、`sweep_compute_gpu01`。协议v3与原侧高网v2保留完全相同的面板文件/哈希、物理和验收门槛。

实际400回合运行：UTC `{provenance['run']['started_at']}` 至 `{provenance['run']['finished_at']}`；GPU费用 `{provenance['run']['gpu_hours']:.6f}` 小时，交互计数 `{provenance['run']['env_steps']}`。本轮独立窗口UTC `{allocation['started_at']}` 至 `{allocation['deadline']}`，截至两组完成已结算 `{hours:.6f}` GPU-hours（上限10）。所有失败尝试、查询、训练和评测均在GPU_LEDGER中，历史用量保留。

先前根报告和失败指标原样归档在 [history_before_causal_formal_pass_v1](history_before_causal_formal_pass_v1/REPORT.md)。错误网坐标的教师53/64、学习54/64声明继续撤回；矩形恒高网单一来球32/32只是历史课程。G00–G06原证据保持原字节。新增场景假设见 [ASSET_ASSUMPTIONS.md](ASSET_ASSUMPTIONS.md)，不要改动G01签名的assets/assumptions.md。

[源码与全部报告分支](https://github.com/lzqw/unitree-smash/tree/g04-guided-learning-20261022) · [权重/视频/交接发布页](https://github.com/lzqw/unitree-smash/releases/tag/g07-progress-20261023)。视频及各历史附件均按来源范围标记，数值验收以本表和原始400回合为准。
'''
    atomic(report/'REPORT.md', body.encode())
    commands = ['# Actual child commands; all GPU work was launched through run_gpu_job.py.']
    commands += [f"# {r['started_at']} {r['status']} {r['charged_gpu_hours']:.9f} GPUh\n{shlex.join(r['command'])}" for r in runs if r.get('command')]
    atomic(report/'commands.log', ('\n'.join(commands)+'\n').encode())
    artifacts = [*read(ROOT/'configs/acceptance.json')['goals']['G07']['required_artifacts'],
                 'reports/G07/metrics.json','reports/G07/episodes.json',str(args.protocol),str(args.policy_lock),
                 str(args.aggregate_output/'metrics.json'),str(args.aggregate_output/'eval_manifest.json')]
    for directory in args.panel_output:
        artifacts += [str(directory/name) for name in ['episodes.json','episodes.csv','metrics.json','eval_manifest.json','job.json','source_hashes.json','completed.json']]
    definition = read(ROOT/'configs/acceptance.json')['goals']['G07']
    evidence = dict(schema_version=1,goal_id='G07',task_id=definition['task_id'],evidence_kind='observed',
                    contract_sha256=digest(ROOT/'configs/acceptance.json'),metrics=metrics,
                    run=dict(command='\n'.join(shlex.join(j['command']) for j in provenance['jobs']),
                             git_commit=lock['git_sha'],**provenance['run']),
                    artifacts=[dict(path=name,sha256=digest(ROOT/name)) for name in dict.fromkeys(artifacts)])
    save_json(report/'evidence.json', evidence)
    checked = verify(ROOT,'G07')
    save_json(report/'formal_verification.json',checked)
    if checked['status']!='PASSED':
        print(json.dumps(checked)); return 2
    status = read(ROOT/'status/STATUS.json')
    status['goals']['G07'].update(state='PASSED',evidence='reports/G07/evidence.json',active_gpu_jobs=[],
        policy_sha256=lock['policy_sha256'],method='learned walking/PWA/standing hybrid with causal prediction and bounded feedback',
        notes=f"Both frozen200-case panels passed original thresholds; full minimum{metrics['complete_smash_success_rate']:.6f}, recovery minimum{metrics['recovery_given_airborne_contact']:.6f}, numerical0. G10 research/real-hardware validation not performed.")
    save_json(ROOT/'status/STATUS.json',status)
    print(json.dumps(dict(status='PASSED',checkpoint_sha256=lock['policy_sha256'],metrics=metrics,window_gpu_hours=hours)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
