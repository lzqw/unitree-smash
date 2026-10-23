# 实时继续入口（2026-10-23 近网策略学习）

- `eval_piecewise_curriculum01` 正在独立原生进程评测 `fit_piecewise_curriculum03/policy.pt`：从数据学习的14段仿射反馈策略，全程自主输出动作，尚无本次实测结果。场景root x=1.0近网课程，来球固定，不能当正式G07。
- `train_dynamic_curriculum01` 已在第4批回合后主动保存停止：前三轮无越网；第4轮62/64空中触球、越网/落点0、恢复23/64。最佳可用权重为bc_03.pt/checkpoint.pt。账本FAILED因请求6轮但主动只做4轮；不是程序崩溃，详见stop_reason.json。
- `fit_piecewise_curriculum01/02` 为不准确CPU试拟合；修正饱和动作的删失回归后03的独立回合验证最大动作误差约9.27e-6。这是CPU拟合结果，不是仿真成绩。
- 第二轮步态20代完成，稳定前进仅约0.10m；换为全身前跳路线。PWA评测结束后运行 `search_g07_dynamic_reach.py --root-x 0 --initial reports/G07/dynamic_reach_near_net02/selected_contact_seed.pt --forward-seed --global-fraction .25 --num-envs 128 --generations 32`，必须通过现有GPU0预算守卫。
- GitHub代码报告提交591a5d2；近网教师、数据、资产和视频发布于 https://github.com/lzqw/unitree-smash/releases/tag/g07-progress-20261023 。本段之后的新学习代码尚待下一次提交。
- 独立授权窗口、预算和旧成果均保持。继续至完整目标达标或2026-10-24 02:04:13 Asia/Taipei/新增10GPUh先到；不要在课程阶段停止。

---

# 当前继续入口（2026-10-23 有效越网课程已独立复现）

- 窗口 `g07-net-feasibility-20261023`：Asia/Taipei 2026-10-23 16:04:13 至 2026-10-24 02:04:13，独立新增10 GPU-hours，GPU0；历史账本保留。
- 已完成 `dynamic_strike_refine01` 8代搜索，冻结教师 `best_candidate.pt`。独立 `dynamic_strike_replay01`：64/64腾空有效拍面接触、向下高速出球并实际越网、首次目标落点；53/64恢复/完整，数值0。root x=1.0近网课程，单一来球，尚非自主神经策略/正式成绩。
- 视频和实测连杆/球轨迹在 `reports/G07/dynamic_strike_replay01`。
- 正在运行 `reports/G07/loaded_walk_bridge02`：原始root x=0、完整负载、无球步态课程；20代×128环境×6秒，必须轮询现有作业，不能重复启动。
- 下一步：验证并改进真实移动衔接、恢复，随后学习全程观测闭环策略并扩展来球/目标。不能以近网教师53/64替代原G07双固定面板至少400回合的验收。
- 所有GPU命令必须经 `scripts/run_gpu_job.py --budget-window g07-net-feasibility-20261023`，检查实时账本与固定截止时间。不得重新授权计时。

---

# G07 resumed with new independent GPU0 authorization

Current authoritative authorization: g07-net-feasibility-20261023; T0 2026-10-23T16:04:13.742590+08:00; deadline 2026-10-24T02:04:13.742590+08:00 (Asia/Taipei); 10 additional GPU-hours. Historical resource-block notes below are resolved. Do not reset this clock.

# G07 blocked: new GPU0 authorization required

Read the new objective at `/home/autodrive/.codex/attachments/1e7d228f-313f-4cad-b9b1-a68c87655170/pasted-text-1.txt` and `reports/G07/feasibility_v3_20261023/REPORT.md`.

G04–G06 stay PASSED. The latest G07 native actor remains `reports/G07/train_high_attack_all01/checkpoint.pt`: 400/400 actual airborne contacts, recovery 357/400, complete success 0. No new native rollout ran in this goal turn. New CPU analysis and code are preparation, not G07 completion.

Withdraw the old 1.535 m hardware-bound claim. Finite random FK/jump maxima and local 0.11 m blade half-height do not establish impossibility. The new pose projection and physical free-flight targets are in `reports/G07/feasibility_v3_20261023`. The current USD racket and 0.18 m link6 mounting offset match the assembly script but are unmeasured assumptions, not a confirmed real racket design.

The next native entrypoint is `scripts/search_g07_whole_body_v3.py`, using `configs/g07_feasible_search_launch_v3.json` and the opt-in `OrderedSmashEnv`. All 18 joint trajectories and timing are searchable. Actual ordered net/first landing dominates the ranking; bounces, side bypass, rim contacts, recontact and numerical collision failures cannot produce full success. This path is syntax-checked and covered by CPU event/geometry regressions, but has NOT run in Isaac yet. First run a small native batch once a valid new authorization exists; then use the result to improve search, capture repeatable legal demonstrations, train a target/observation-conditioned policy and do the original minimum-panel 400-episode evaluation. Do not stop at a single demonstration or substitute the development seed for formal target/launch coverage.

The original 10-hour GPU0 window started at 2026-10-22T14:08:14.991569+00:00 and expired at 2026-10-23T00:08:14.991569+00:00. Family charges are about 3.182811 GPU-hours. Continuation keys cannot reset this clock or the original 10 GPU-hour family cap. `scripts/run_gpu_job.py` now enforces that, and an actual expired-window preflight was rejected before launching/creating a run. GPU ledger is unchanged in this turn and no project GPU job is live.

An asynchronous question requested a fresh GPU0 wall-clock window and GPU-hour limit. Read `status/PENDING_G07_RESOURCE_AUTHORIZATION.json` and any newer user reply before launching. Do not create a new window merely because the old one has unused GPU-hours. New goal explicitly forbids extending expired windows. The initial two turns made CPU progress. The third turn revalidated the same resource blocker, no authorization reply, no RUNNING ledger entry and no live project process; the thread goal is now BLOCKED, not complete. Resume only after a new explicit GPU0 work window and GPU-hours cap.

CPU tests: 85 passed, `reports/G07/feasibility_v3_20261023/cpu_tests.xml`. Historical NEXT content is preserved in `status/history/NEXT_before_G07_v3_20261023.md`.

Latest CPU follow-up: `reports/G07/search_objective_regression_v3/REPORT.md`. Coupled search loss now includes full flight and first-landing violations; higher upward net margin cannot outrank a legal downward flight. Final recontact invalidity is respected by ranking. Outgoing position/velocity/time share the same force frame, and release requires geometric separation. Actual ordered-net candidates retain complete replay parameters even before recovery succeeds. 89 CPU checks passed; native path remains unrun, authorization pending. Second actual goal turn made real code progress. Third actual turn confirmed the same unresolved authorization condition; mark blocked per the audit threshold.
