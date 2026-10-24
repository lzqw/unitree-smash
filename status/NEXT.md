# 完整球拍、真实球场与合法跳跃扣杀 v4-r3

当前分支g07-full-racket-court-v4-20261024。授权55e175f0...延续：仅GPU0，无累计GPUh、工作窗或日历截止；所有作业仍通过run_gpu_job，保留单作业有限超时/交互上限、单GPU任务保护和真实费用。旧授权窗口与历史费用不变。不要重新询问旧截止。

当前资产assets/quadsmash_full_racket_v4_r3.usd，总质量20.709991kg，完整球拍675mm、法兰至拍线中心约452mm，球拍+夹具153g。13.4m球场、6.1m宽/0.76m悬空网，中央1.524m、侧1.55m。来源、假设、规则映射与限制在reports/G07_full_racket_v4/MODEL_RULES.json。旧G00–G06通过仅适用于旧短拍板，不能继承。

必要修复与证据：
- 初始安全包围盒漏掉purpose=guide，产生FLT_MAX空范围；改为实际碰撞顶点/原始几何尺寸，未关闭身体越网检查。
- r1凸网格拍线面实际漏碰（contact_fixtures07/08，NF=1）；r2使用同一32gon内62个非空薄盒片，占面积97.8505%，不扩大有效区域。旧资产和失败均保留。
- r2夹具从法兰z=-5mm开始，与官方link5.STL（最高z=-1mm）相交，原生测到持续21–36kN自碰力。r3将夹具法向深度45->38mm，开始位置+2mm；球拍/握持位置/452mm力臂/65g夹具质量/原执行器不变，重新计算COM/惯量。没有删除任何机器人碰撞对。
- reset_self_contact_r3_01成功：直立持拍地面2秒，实际自碰力0N、最低根高0.23424m、最大重力XY0.12169；原生工具COM/惯量匹配。t=0旧传感器缓存单独保存，不作新姿态接触，后续真实步仍全部检查。
- contact_fixtures_r3_01成功：62.5us原生真实拍线中心正例、边角/拍杆/夹具反例、网下/网上、网体/柱接触；NF=0。仅是非地面固定根接触工装，不是策略成功。
-22项针对性CPU测试通过，含源STL夹具间隙、真实盒片边界、质量惯量、连续扫掠和规则正反例。

评测：configs/g07_full_racket_eval_v4_r3.json已冻结；A/B各200和训练128案例与最初冻结完全相同，没有因结果删除来球或降低阈值。真实半场x=6..7.8m、初速7.47..11.93m/s，明确2秒回合前准备。协议r1/r2保留；r3记录必要夹具/初始持拍修正及此前无效r2搜索。正式训练须用r3，最终正式400仍未运行。

已实际搜索：reach_cem_r3_01，8代×64候选×6秒=307200控制步，GPU0 245.730464秒/0.068258462GPUh，保存每代population、resume.pt和best_candidate.pt。512候选中15次原任务跳跃恢复；最佳综合候选在2.81s虚拟击球点[1.2456,-0.0293,1.5743]m、实际点速[-0.2949,0.1987,0.0409]m/s，腾空0.446s、稳定恢复1.626s，无身体落地或碰网。最高位候选与最佳恢复候选不同；这不是实际触球或扣杀。最后一代63/64目标时间附近腾空、4/64恢复、14/64碰网。

已启动真实球开发探针（不是正式400）：scripts/probe_full_racket_rally_v4.py用该学习候选、当前公开794D观测估计来球、校正动作时机和臂位，执行真实球开发试验。只取冻结训练库预定A/B案例，不用正式400挑策略。开始前核对GPU账本无RUNNING/PID活跃，不能重复启动。命令通过run_gpu_job：max-seconds1200，env-steps-cap3200，output reports/G07_full_racket_v4/rally_probe_r3_01；子命令python scripts/probe_full_racket_rally_v4.py --checkpoint reports/G07_full_racket_v4/reach_cem_r3_01/best_candidate.pt --count4 --output同上 --headless。物理62.5us，控制10ms，8秒，保留真实失败、公开数据和原始几何。该脚本是开发探针，不是正式评估器；正式初始扰动和400锁定评估仍需实现/运行。之后继续联合优化、BC/DAgger或RL及独立评测，不在无球课程结束。

已交付38秒进度视频（包含此前r1机械测试与明确标注的旧短拍板对照，不是r3扣杀）：https://github.com/lzqw/unitree-smash/releases/download/g07-progress-20261023/Current_effect_v4_20261024.mp4 。SHA256与GitHub大小核对见video/upload_receipt.json。

当前活跃原生作业：rally_probe_r3_01，exec session22673，guard PID 1309544 / child PID 1309557，started_at 2026-10-24T02:53:35.653414+00:00。已用/proc存活检查确认运行；优先write_stdin轮询同一session或核对同一PID/command/progress.json/job.json，不因观察超时重复启动。每100控制步保存partial公开观测/动作/几何轨迹；只在completed.json+终态job.json后作为完整开发结果。
