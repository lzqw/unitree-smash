# 当前策略效果视频

`G04_autonomous_jump_demo.mp4`：1280×720，50 fps，13 秒，4 秒正常速度 + 8 秒半速回放 + 1 秒末帧。

使用已通过 G04 的冻结检查点 cd9119ad85d4ac45a76d9c0014058e4edfbcc4ded47317893e55419781c8d980，从地面确定性执行，预先指定面板 B 的场景 0。本次单环境演示成功：起跳后 COM 上升 0.116626 m，连续腾空 0.220000 s，连续恢复 1.819984 s，无非足端触地。单环境渲染演示不是重新进行的双面板验收。

本机原生 RTX 渲染启动崩溃，失败日志保留在相邻 policy_video_20261022 目录。随后以原 Isaac Lab / PhysX 物理和策略采集 50 Hz 真实连杆位姿，再用 CPU llvmpipe 离线栅格化 USD 几何。没有插值、手工编排运动或状态传送；展示用材质/灯光/地面网格不参与物理。球拍保持当前资产中的简化拍面几何。

源码：scripts/record_g04_policy.py、scripts/render_g04_video.py。离线依赖 moderngl==5.12.0、glcontext==3.0.0，安装在 /tmp/unitree_video_deps，没有修改 Isaac 环境。渲染命令：

```bash
PYTHONPATH=/tmp/unitree_video_deps /home/autodrive/Isaac/env_isaaclab/bin/python scripts/render_g04_video.py --input reports/G04/policy_video_trajectory_20261022
```

两次 GPU 作业共保守计入 800 交互、0.007463746 GPU h；累计 3.598004408/8 GPU h、108147008/200000000 交互。按照此前授权临时挂起了 7 个 SAFEFlow 训练进程，两个作业退出后均恢复；没有终止或重启这些训练，也未修改它们的代码和检查点。录制完成后无项目 GPU 作业运行。
