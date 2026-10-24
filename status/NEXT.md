当前唯一GPU0作业（2026-10-24T12:14:58.901280+00:00核查）：ball_attack_recovery_cem17，guard1416543 / child1416555；启动源码f1deb6ab826084af33ad970ef9afa4b35e12c676。4代×64候选×TRAIN[1,65]，有限7200秒/409600控制环境步；无累计GPU小时/日历截止，仅GPU0，无额外队列。不手改GPU_LEDGER/RUNS、preexisting RESOURCE_STOP及无关MPS，不因观察超时重开作业。

正确输入attack_recovery_ready02_1214，policy SHA ea7c3c5fe64ff283a5fa7d9b1d0a058386addaac78564c41a29ae5658fdf2838，bank SHA 04523cbbffb86b0fe6d549ac8bd014db7deb6f7a5ed19293dd84e2649bb5d1f1。ready01_1208明确未上GPU/已被替代，SOURCE.py与REJECTED.json保留。新参考严格保留CEM16实际最佳攻击候选17全部32参数。60bank=2个此前真实空中拍线+合格恢复身体来源(CEM02g0cand6/CEM03g1cand16)×5混合比例(.2/.4/.6/.8/1)×6恢复局部时机(1.30/1.35/1.40/1.45/1.50/1.55)。仅11身体坐标与p3改变，所有20个挥拍/拍面/速度/拦截坐标在bank中与实际攻击参考相同；真实资产质量惯量/关节位置速度力矩/球网球拍/规则/成功门槛均不改。CPU报告明确“原算法1.55搜索范围”为false，而新声明范围true；这不是降低恢复成功或硬件门槛。

新运行flag --retain-contact-recovery --recovery-search-min1.30。选择契约actual-contact-recovery-chain-v2：实际完整成功始终优先，其次同回合合法空中拍线+原合格恢复，再是更多接触。默认旧排序不变，不能把不同回合接触与恢复相加。历史7个完整世代数据重排证明旧排序曾丢掉3次已合格的触球恢复链；正式/native标签从不改写。101项CPU测试通过，120路287步新指令有限有界；仅瞬时/反事实指令检查，非新动力学成功。实际generation_start已确认新contract/min1.30/60bank/inputSHA/formal_panels_readfalse，下一步核对首100/200步平衡及300步真实出球/同回合恢复。

CEM16完整第0代128执行：37接触/26拍框/11空中拍线/1攻击达标/2合格恢复(均拍框)/6有效拍线/5任务有效越网/0场内首落/0目标/0拍线+恢复/0完整/NF0。36触网、0提前越网；89身体触地终止。最佳候选17在TRAIN65合法空中拍线，出球[7.887247,-1.488526,-1.651805]m/s，速度8.194683，实际越网余量.074476米，但3秒身体触地、原生身体接地力116.54N、恢复失败，首落点未观察到。同候选TRAIN1为拍框误击后合格恢复3.919898秒，没有拍线或合法回球。两回合不能合成一次完整成功。见attack_return_gen0_candidate17/MILESTONE.json及原始完整逐回合数据。

CEM16guard1413106/child1413122均已退出，真实FAILED/exit0/缺完成receipt保留；第1代26步中断，空终端事件/64×32人口/26步公开流/13×128全身姿态已核对有限，见INTERRUPTED_NEAR_INTERCEPT_LOGGING_VERIFIED_1215.json。实际(800+26)×128=105728控制环境步，.300986815680GPU小时，原声明cap409600不改。完整失败和CEM17新程序/权重/原生prefix需增量上传。

已上传并核对：101876566字节CEM15完整失败+实际398步中断/CEM16源码与首100步；37384992字节CEM16实际300步和攻击跨网里程碑。receipt分别measured_failed_near_intercept_upload_receipt_1202.json/attack_partial300_upload_receipt_1210.json。完整CEM16候选权重与全代数据保留本地待下一包。RECOVERY_HANDOVER_TIMING_1213.json表明攻击回合恢复约2.89764秒开始、2.99764秒结束，身体触地3.0秒；提前恢复仅待验证算法假设。

当前视频仍是CEM12/CEM07的34.72秒对照：https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_latest_20261024_1048.mp4 。CEM16两回合新视频正在CPU llvmpipe生成：video_snapshot_20261024_1216，case0env35攻击后身体失败，case1env34拍框后恢复，独立回合/均完整失败；不包含CEM17。

目标不变：同回合地面自主起跳、己方合法空中拍线、实测出球≥8且vx>0/vz≤−1、真实越网首落原目标、原合格跳跃恢复≥.8秒、身体触地0且无规则/数值故障。不改jump_metrics消除1ms再次离地。找到可行链后扩展TRAIN128/原扰动并冻结策略，运行A200+B200，以较差面板完整≥.50/空中接触≥.70/条件恢复≥.85/NF0验收。完整扣杀仍0，正式400未运行；还需成功/失败视频、旧短拍板量化对照与全部可复现交付。继续。
