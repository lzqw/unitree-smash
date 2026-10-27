# 完整输入树、机器人资产与新版教师原128例

本包保留实际完成的新版冷128，结果为0完整、4次空中拍线、0合格恢复、NF0。全部128行去掉新增关联字段后与原教师逐项一致，48305组共同有效50Hz姿态/球/时钟逐位一致；新评分没有改变动作或提高这组成绩。实际102400控制环境步、0.314017821 GPU小时。

包内包含全部931项固定输入、完整两层教师权重、原始128行和动作/轨迹、实际执行源码快照、原精度准入的12项哈希依赖及两次真实探针回执、10个实际USD层、原始PiPER URDF、walking策略和原TRAIN128案例。无需再从1.21GB历史训练包补齐这些文件。安装好的Isaac/Kit、Torch、RSL-RL等软件仍是外部依赖，版本参考恢复后的`reports/G00/environment.lock.json`。OmniPBR.mdl是Isaac/Kit内置视觉着色器；独立USD依赖发现得到10层、非USD外部资产0，只有此内置着色器未在独立USD环境解析。

ZIP使用按SHA存储的objects，MANIFEST记录原始绝对路径、字节、SHA及链接目标，同内容只存一份。`restore_inputs.py`会恢复到一个全新的前缀目录并验证每个文件，拒绝覆盖已有目录；源链接在前缀内指向对应恢复文件，不读取运行中的工程。

```bash
python restore_inputs.py --archive G07_contact_recovery_v5_cold128_complete_inputs_and_assets_20261027.zip --destination /tmp/g07_recovery_restored
python check_fresh_cpu.py --restored-root /tmp/g07_recovery_restored --output /tmp/g07_recovery_fresh_cpu.json
```

第二条在新恢复树中重新计算原128结果、核验原精度证据/所有固定输入/10个USD源层，并从实际1600条公共输入复算完整教师动作。检查器拒绝通过Python读取原`/home/autodrive/LZQW`运行目录。实际执行是否完成以包外FRESH_CPU_CHECK.json为准；这不是新的原生仿真，也不是CPU物理替代。

原生运行仍有绝对USD引用及准入路径：应在干净环境恢复原`/home/autodrive/LZQW/...`与`/home/autodrive/Isaac/unitree_model/...`命名空间，或者显式审计路径迁移后再运行。不要覆盖现有工程或复用运行中账本。安装好锁定软件后，克隆相应Git分支、恢复输入树，从原`job.json`取得命令，用全新输出目录及独立GPU0 guard/账本记录本次资源。旧运行记录仅作证据，不能当作新的资源记录或继承成功。此包没有声称已在新解压环境重跑Native。

运行代码位于恢复树的`unitree_smash_contact_recovery`，Main guard的快照和`execution_source_snapshot.zip`作用不同。教师权重仍为原SHA204ff76及原协议元数据；新版评估使用显式contact-flight-recovery-v1和协议SHA bfe6f2，二者均在INPUTS保存，没有重标历史权重或改写旧成绩。

另附四次实际触球速度重建及9个挥拍候选的CPU程序/数据。四次拍头前向速度均为负；瞬时关节速度候选与原源动作输入的复算不能证明新碰撞、可达轨迹、越网、落点或恢复。它们没有在Native运行，不能当作新教师成绩。

正在运行的新15更新训练没有打入此完成包。后续仍需新训练、冻结评估、正式A/B各200及原50%/70%/85%门槛和真实视频。当前G07未达标。
