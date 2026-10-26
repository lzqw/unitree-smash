# 十来源条件leg06流程：CPU执行边界已验证

本流程扩展既有七来源程序，保留原文件，新增075的16/24/32冻结来源并等待其真实比较进程1744903退出。当前1734658和1744903均实核运行，故程序拒绝准备或执行任何候选GPU作业。

实际CPU检查已通过：真实旧8/40及0758来源的均值迁移、normalizer计数、critic/Adam新建；拒绝把075原训练或严格续训输入误标成新动作单位阶段；候选工作区不能创建自己的GPU账本任务。CPU准入情景还验证：即使旧观察进程及账本清空，真实仍存活的中间比较进程也独立阻止启动。该情景不是原生结果。

未来准入要求：原训练/最终40/视频READY，selected-trace审计SHA通过；中间比较COMPLETED，全部实际源政策与原TRAIN128的原生成功收据匹配，seed724201/noise=none，失败结果不删除。十来源排序及就绪时跳过新修订的规则见相邻candidate目录CONDITIONAL_PLAN.json。main集成只允许先前声明的trainer和新初始化helper两个运行时变更，并对中间比较的全部284份输入检查其余字节保持。

新阶段原生时必须验证真实首更新：2048地面、源actor RMS M+262144/critic262144、mapped actor在前两次critic warmup保留、fresh std .05/.1、8个critic Adam各20步。最终8应为2097152步、actor/logstd Adam120、critic Adam160、actor实际改变、canonical地面检查点、全部2048回合ID链与完整选定轨迹审计。随后原全128和实际六例视频，不自动长训或上传。

已运行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 /home/autodrive/Isaac/env_isaaclab/bin/python reports/G07_full_racket_v4/centered_leg06_after075_intermediates_pilot8_queue01/check.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 /home/autodrive/Isaac/env_isaaclab/bin/python reports/G07_full_racket_v4/centered_leg06_after075_intermediates_candidate_20261026/prepare.py
```

目前仅CPU准备。未选择来源、未main集成、未新训练或冻结评测，正式A/B及全目标均未通过。
