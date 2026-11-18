# 持续修复与原生数据补充，Actor长训尚未启动

新用户目标授权继续修正，已解除此前追加模型修订的等待。目标保持原128环境、256次实际Actor更新、R8→A8交替、0.64s想象和0.16s模型窗口；物理、动作、奖励、评分不变。实际HEAD 0e417225f2b75c5978e6f72856d0f423d79a31c8，保留用户dirty和历史文件。

## 已完成GPU模型修复

1. 旧球门控显式解冻，在原96/32划分完成固定256次模型更新，退出0，418.21秒，Torch峰值326990336字节。Actor/Critic更新0。接触前球速RMSE5.1509→1.1234m/s；接触中1.7301m/s，接触后0.3282m/s，后者仍不满足旧模型不退化检查。4个原生留出接触案例仍3个负方向余弦。因此保存为未准入模型。
2. 只读检查350个训练事件边沿发现1177次布尔记忆翻转，真实上升字段平均预测仅0.0111。任务模型改为布尔logit BCE、连续字段保留原归一化误差；未改变参数结构和前向预测语义。加入真实回合终止/complete5采样池，防止唯一终止帧被800tick均匀采样淹没。沿用同一数据、归一化与独立划分，从兼容任务模型初始化，明确使用新Adam，固定512次GPU模型更新，退出0，8.52秒。实际本轮512次、累计1024次，不冒充Actor更新。

新任务模型留出触球TP从2/19到16/19，FP0→4；拍框TP2/11→5/11、FP0→3；越网TP0/14→1/14；恢复和身体触地仍未达到0.5诊断阈值检出。这只是模型事件诊断，不是原生成功率。时序邻域检查未发现只是错一两tick可解释的漏检。

## 为什么补真实数据

球响应模型在5个训练触球样本上方向余弦0.748～0.996，但留出样本失败，不能继续只刷训练loss。已依据固定96训练lane的原生触球边沿数量，预先选定tick271和267，各含10和6个训练接触边沿；不使用留出结果选择时刻。两个时刻分别执行原128环境四分支同前缀重放，原请求joint12±0.02、原执行限幅，全共2160控制tick／276480环境转移，优化器更新0，无自动重试。这是训练数据补充，不是候选正式评价。

持久作业supermimic-native-contact-coverage-20261118.service；实际路径见results.json。采集过程中冻结原生相关源码，等待完成后独立核对完整起点、控制器记忆、最终命令与分支哈希。

准备了多时刻／Recovery与Approach原生配对监督接口：每个数据源必须分别通过四分支检查，Actor、物理执行合同及96/32划分严格一致；重复源和混用保留集均拒绝。尚未运行这个新增多源拟合，也未声称通过。

## 检查与证据

本次分别运行19项球相关、8项任务状态、13项多源合同CPU测试，均退出0。它们不是一次新完整仓库测试，不与历史309项合并统计。语法检查和git diff --check通过。results.json含模型完整SHA256、实际更新数、分项验证和运行路径；actual.diff和源文件快照可复核。

历史Recovery采集series.json中的output/predeclared曾残留早期Approach模板文字；旧文件不改。实际四分支命令、spec.tick302、shutdown303tick、原始PT及哈希共同证明执行的是Recovery实验。新补充采集使用新建manifest，逐项写入实际tick和路径，避免复制该陈旧描述。

目前Actor更新0，没有新的长训TensorBoard或性能成绩。原定长训目标保持active，仍需击球/恢复动力学、事件与Critic准入以及真实学习闭环；不将单项模型改善等同于全部通过。

## 本轮续接核验

完整SuperMimic CPU集合在任务模型修复后实际329 passed、0 failed、0 skipped，退出0；日志cpu_suite_after_task_repair.log。随后多源足端标签合同23 passed，增加分支足端误差统计后相关集合26 passed，退出0；日志paired_contact_metrics_cpu.log。这些是不同检查的实际数量，不相加为完整测试总数。

任务记忆的一步teacher-forced布尔翻转概率不能直接代表连续想象门控。新增独立诊断只使用真实记录的物理转移，任务记忆连续自行推进64步：在32个留出lane的30个触球案例中，提前1步或16步起算，Recovery门控首次激活30/30时刻一致；提前32步则25/30一致，合计13个lane-step门控不一致。这不是完整物理WM闭环，也不是Actor更新。task_boolean_repair_01/contact_rollout_*保留逐例证据。

Recovery t302的当前模型足端诊断：32个留出lane18个实际足端接触，名义分支漏检3、误检1；其中两个首次触地lane共2个实际接触都被漏检。0.5只用于报告二分类，不改变原生碰撞定义。新增拟合选项显式解冻原足端接触头，使用原WM中的二元交叉熵监督；真实四足标签必须是128×4有限二值，并与原生文件SHA匹配。修改未进入正在运行的原生采集源码。

已预先固定一次多源GPU模型修复：同一96/32划分，t272与Recovery t302加新t271/t267的真实正负扰动，等权汇总各数据源；256次模型Adam、128 batch、16步、2e-4。Actor/Critic更新0。起点为已保存gate修复模型，不是从失败策略续训。固定源码、输入模型和数据哈希；等待当前采集完整结束且退出0后才占用GPU。失败不自动重试，拟合完成不自动生成准入凭证。该作业是模型修复，不冒充原定256次Actor学习。

持久队列supermimic-multisource-repair-20261118.service，计划和日志路径见results.json。最终仍需按原门槛检查留出响应、接触前中后误差和完整学习闭环，未启动Actor长训，未产生新正式成绩。

## 事件时序连续推进审计（只读）

新增audit_task_event_rollouts.py，在同32个留出lane的真实物理转移上，让任务记忆和一次性事件锁存自行推进，分别提前1/16/32tick起算，最长64tick；遇终止保留终止步后停止，不跨reset。未来任务标签仅用于事后比对。物理预测器不参与，Actor/模型优化器均0次。

提前16tick时：19个合法触球事件时刻全部对齐；9个恢复事件中8个提前、1个窗口内未检出；16个身体事件中11个提前、4个延后、1个对齐。提前32tick时，19个触球事件8个未检出，恢复8个提前/1个未检出，身体14个提前/2个延后。说明单步损失和Recovery门控检查不能代表完整事件奖励时序可信。这里只使用0.5进行诊断，不把它改为原生评分阈值，也不据此宣称物理模型失败或策略成功。窗口围绕事件选取且有重叠，计数不是独立总体假阳性率。

现有任务拟合使用16步自回归任务记忆，但事件hazard监督以真实未支付标记掩码计算；本次审计显示连续软锁存仍有提前累积的风险。尚未增加新损失或启动额外任务模型拟合，先保存证据，等待已排定物理模型修复完成后统一检查。

另外已用真实原生文件独立复核t272 Approach与t302 Recovery四分支来源、身份和足端标签；二者完整合同兼容。新t271/t267仍待采集完成，不能提前宣布全部多源检查通过。

## 新触球数据第一批已完成

预先选定t271的四分支均退出0，同前缀两次名义重放误差0。原128环境、96/32划分及策略执行身份与t272一致。训练lane新增10个真实触球动作响应，留出lane新增2个；正负分支边界接触模式一致，最终命令差约0.016rad，无饱和。训练球速响应幅值0.00769～0.20957m/s，留出两例0.13677与0.00105m/s；幅值小的例仍保留，不能把非零直接认定为可信光滑梯度。完整证据见native_contact_coverage_g7h9mqsy/t271/verification.json和measured_support_audit.json。

t267四分支仍由原服务串行采集；多源拟合队列继续等待第二批完整验证。没有重启第一批，没有修改原生进程源码，没有新增Actor更新。

## 进一步定位：拟合与想象的任务记忆反馈不一致

此前裸任务网络连续推进没有包含实际想象中的schema.decode_surrogate→schema.encode，因而不能直接代表实际入口。实际入口对float缺失字段按validity>0.5屏蔽，bool和counter有限值重新编码为有效；离线task拟合及fit_world_window却直接反馈numeric和sigmoid(valid_logits)。这是两处源码可定位的训练/使用接口差异，不是新物理失败。

只读检查加入与实际入口一致的有效性和时钟/命令/几何回写（仍使用真实物理转移隔离任务误差）后，提前16tick的19次合格触球事件仅7次时刻对齐、12次未检出。逐字段排查中，仅有效性编码往返就降至2次对齐/17次未检出。恢复事件提前累积问题仍存在。不得再用此前裸网络19/19宣称真实想象事件已验证。所有结果保留于task_boolean_repair_01/adapter_feedback_event_rollouts.json、feedback_*_audit.json。

以128个真实状态直接比较诊断反馈与实际schema编码往返，值和输入梯度最大差均0；拟合原反馈与实际使用数值最大差2.5638976（出球速度字段），有效性最大差0.1385072。初次诊断助手的梯度检查因原位切片写入失败，已改为函数式拼接并保留错误记录；该助手未用于原生采集或任何优化器。

准备了三文件最小补丁pending_feedback_fix/proposed.patch：增加与既有decode/encode完全等价的feedback_values，并让离线task拟合和在线WM拟合使用它。隔离候选9项CPU合同通过，含不同batch/dtype、未知字段、布尔/计数器、输入梯度和维度错误。补丁尚未应用生产源码，等待当前原生采集及已固定哈希的GPU拟合结束；不能把隔离测试写成主训练修复已上线。其他确定性几何/时钟回写差异单独保留，不声称此最小补丁全部解决。

## 冻结源码留存与采集进度

已保存frozen_execution_and_fit_sources_20261118.tar.gz，254个实际运行源文件，包含外部G07控制器、当前工作树模块以及未跟踪的新模块。所有内容先对照运行manifest哈希，再从压缩包重新逐项回验；SHA256为18c3c0d73d0b99c97190b5aac0b53097aed387b5600254fd7654b5debceba6eb。该包不含权重或资产，后二者另有固定哈希，不能误认为完整环境镜像。

t267两次名义分支已退出0，正在minus分支；加上t271四分支，当前已完成6/8个采集分支。原生服务和模型等待队列仍live，未重启，Actor更新0。

## 球预测分项与输出目录保护

现有gate修复模型的30个留出触球后一步：真实球速度相对自由飞行残差为0；legacy分支单独误差0.0165087m/s，加上action-ball后误差0.328157m/s，action修正自身RMS0.330746m/s。平均软几何支持仍0.107628，旧racket模式概率仅0.006327。主要多余响应位于新增action分支，不能归因于真实球在分离后仍受拍头冲量。分项幅值不是完整因果误差分解；未改forward、未测试替代网络、未运行新优化器。原始逐例输出在ball_gate_repair_candidate_01/component_audit.json。

长训shell启动器原mkdir -p允许复用已有RUN_ROOT，可能使tee覆盖旧日志。已改为原子创建全新根目录，已有目录退出15；包括resume也要求新输出目录。bash -n通过，临时真实已有目录检查确认旧log不变、未创建TensorBoard、未启动Isaac。当前原生和排队拟合不使用此shell，二者冻结源码哈希再次逐项一致。

最新原生进度7/8分支退出0，最后t267 plus运行。仍未启动Actor长训。

## 任务反馈合同修复与有限GPU验证（实际已完成）

训练原先直接递推soft validity，与实际schema decode/encode不一致；现离线及在线WM拟合共用精确反馈合同，128真实状态的数值及输入梯度对照最大误差均为0。15项相关测试及随后346项SuperMimic CPU测试通过，退出0；存在1条既有FutureWarning。

单次GPU任务模型拟合512步，Actor/Critic均0，耗时9.2701秒。task.pt SHA256：`5a1d47461bd79dab9f93ac1560408cf5913ec9d9c1ee66c51222196dcf382ba9`。在真实物理输入及实际adapter反馈的64步诊断中，从触球前16步开始：19例触球时刻准确数7→19；前32步开始4→18。恢复9例仍8提前、1漏报；身体接触、越网亦未完成可信验收。不能将接口改善写成真实任务成功率提升。完整数值见task_feedback_result.json。

## 多来源响应模型复核（未通过放行）

前一轮256模型更新完成退出0；7个留出触球响应4个同方向、3个反方向。触球后速度RMSE0.349395m/s，高于原模型0.026637m/s。Recovery角速度响应余弦中位数0.4393→0.9718，但仍2个反方向，不能宣布所有恢复梯度可信。所有旧产物保留。

## 当前球响应候选修正（GPU拟合中）

新增独立schema `racket-mode-conditioned-action-sweep-v2`：共享球模型当前预测的拍面接触概率控制额外动作响应；原V1计算及旧checkpoint身份不变，不输入未来原生标签，不修改物理或评分。27项相关测试及351项SuperMimic CPU测试通过，退出0。新的固定256模型步作业为supermimic-racket-mode-repair-20261118.service；输出见racket_mode_repair_location.json。尚无最终结果，Actor仍0，未启动256次Actor长训。

## 恢复事件的独立诊断：尚未解决

真实物理+真实任务记忆输入下，28个训练恢复事件前16步平均hazard约0.08065，9个留出事件约0.08530；真正发生时只有0.10171/0.10763。另一方面，事件当步的恢复成功boolean预测约0.000325/0.000320，原生标签为1。说明提前发奖不只由递推状态漂移导致，独立hazard与状态翻转监督均不充分。不能用降低0.5诊断阈值、放宽评分或继续Actor更新掩盖。完整逐时刻数据在task_feedback_repair_d6l7l46z/recovery_hazard_components.json；本诊断未更新任何参数。

## 拍面模式条件化候选GPU结果（已结束，未放行）

固定256次模型更新完成，退出0，耗时489.762秒；Actor/Critic更新均0。接触前/当步/后球速RMSE分别0.778371/1.957224/0.019203m/s。接触后已优于原模型0.026637m/s，但触球时方向错误仍2/30，7个留出原生动作响应仍3个反方向；因此未放行，不晋升候选、不启动Actor长训。

仅用于误差归因的诊断：把球响应输入中的机器人终点换成记录真实终点后，原3个反方向仍然为负；对应机器人关节位置响应方向余弦均大于0.994。说明这些反向响应不能只归因于机器人预测，球响应映射仍不可信。这种真实未来终点只用于诊断，绝未送入正式想象、Actor或执行路径。详情见endpoint_sensitivity_attribution.json。

Checkpoint SHA256：
- world_sha256: `1f095bab19023b985d6a6dd6c058924f804cc05c2c000ea5913afc71d5e0a751`
- action_ball_sha256: `93d413dceb53c45890c44ae5a89c82aa3e8cda0c5cb2499fc0bbbcd0270e6e44`
- legacy_ball_sha256: `12f4f667d3ae5c4c57d5ca9ee53580859ccc433c3a09c56905f258436a404566`

本轮没有新的原生任务成绩。所有失败模型、原始轨迹和历史策略保留。当前系统仍需解决球响应方向及恢复/身体事件的预测一致性，不能以CPU测试或GPU拟合完成替代这些准入证据。

## 绝对布尔状态候选：有限GPU拟合与留出诊断

V1 logit-delta使False转True需克服约−9.21的先验。本轮显式新增V2任务schema，布尔值采用绝对logit，连续状态仍为原残差；旧V1保留不变。仅布尔输出行按96个训练lane原生频率重新初始化，其他权重和归一化严格保留，独立新Adam；未改变原生事件或评分。512次GPU任务模型更新完成退出0，8.8491秒，Actor/Critic均0；checkpoint SHA256 `036663181317db47ff1f891c67070ffa6e9af039f7c78d5c4034aecc82c68821`。

32步提前起点、实际adapter反馈的64步诊断中，19例触球时刻均准确；14例越网中6准确、7提前、1晚；9例恢复中8提前、1漏报，故不宣布事件时序验收通过。模型累计计数2048包含历史共享权重训练，V2实际拟合512，不是2048次V2或Actor更新。

新增测试曾错误假设输出Linear索引，保留两份失败日志；查明MLP含LayerNorm后按实际末层标识核对，未放宽断言。修正后354项CPU测试通过。

## 碰撞时刻诊断与V3候选

同一32-lane开发留出集、7个已测触球动作响应中，3个方向反向样本原生触球延迟约3.873ms、0.250ms、0.812ms；模型此前使用10ms末端状态。当前相对运动加模型预测终点的二次近似，在21个训练/7个留出接触样本上的首次几何交叉时间平均绝对误差约52.2/61.5微秒，均找到交叉。该检查只说明几何时间近似，不能视为PhysX子步轨迹复现或梯度方向通过。

V3候选以估算碰撞时刻插值机器人状态：首区间选择离散、区间内根和Hermite状态连续可微，未发现交叉时显式报告最近节点fallback，不能称为真实碰撞。所有真实未来时刻/终点只作监督或诊断。16项相关测试及359项CPU测试通过，128真实起点前向/命令梯度有限。

已启动一个固定256模型步GPU拟合作业supermimic-impact-time-response-20261118.service，输出impact_time_response_ua3p6h6_；尚未完成，模型方向待复核。仍没有Actor更新、没有新的正式任务成绩。

## GPU作业运行中发现的验证反馈合同差异

`supermimic_wm_validation_v1.validate_windows`仍直接反馈numeric与soft validity；实际想象及现有拟合使用schema decode→encode等价反馈。对含无效float和有效boolean的连续4步测试，旧验证器在第二步出现数值/有效位差异，严格相等断言失败；候选复用`memory_schema.feedback_values`后通过。未改变任何评分或误差门槛。

证据保存在`pending_validation_feedback/`：旧源码、候选源码、完整失败堆栈和退出码。为保证impact_time_response_ua3p6h6_作业来源不变，暂未应用到生产模块；运行期间源码哈希检查无变化。这是验证路径合同修复，不构成新的原生数据或性能成绩。

## V3碰撞时刻候选已完成（未放行Actor）

systemd退出0，源码哈希无运行中变化；固定256模型更新，672.182秒，Actor/Critic仍0。接触前/当步/后球速RMSE为1.335411/1.954303/0.039156m/s；接触当步速度方向错误3/30，原对照2/30；7个留出配对动作响应4正3负。因此接触方向和接触后非退化检查失败，未生成准入凭证，未启动大规模Actor训练。

输出world SHA256 `cba565bf47b1066d0b5cd71939d9a349207a25d5e278f5bd358a926a7514392d`；action-ball `c8e84fd8af2d8ac45b5053b71da6fe80427cec02672539708ffc4627d0cd1955`；ball `7202edee6671b55a99357b252eb3b06da8122d3032e7bb99c30084cbd5c0b9e7`。旧产物全部保留。

作业终止后才应用验证器decode→encode反馈修复；新增64步CPU回归，23项相关测试通过，退出0。此修复仅统一schema反馈，尚不证明adapter中确定性时钟、命令历史和几何缓存覆盖与记录动作验证完全一致，后续须分别核对。

用户最新要求尽快启动大规模训练并提高显存利用率。目标仍为256实际Actor更新、128原生环境、0.64秒想象/0.16秒WM；本次显存低来自模型拟合小网络，不是正在执行该Actor预算。不得为提高显存占用跳过已证实的物理响应方向失败。

## 拍框响应断点定位与紧凑原生子步采集

上一轮为实质进展：完成固定GPU模型拟合、保留失败方向证据并修复验证器schema反馈；并非训练已经启动。本轮只读对照使用原几何/恢复系数，先比较单次法向冲量，再加入原摩擦系数0.08及均匀球切向有效质量。两种诊断均未进入Actor/WM前向；它们忽略关节化有效质量和多接触求解，不能冒充PhysX或可信新模型。

加入摩擦的只读对照在7个开发留出接触响应中6个方向同向，剩余拍框案例仍反向。所有案例全部保留，不按编号切换模型。7个首次接触起点的真实球角速度经原`shuttle_state_w`检查均为0；其他lane有非零自转，因此不能把55维状态中的缺项普遍填0。数据驱动V3仍为4正3负，不能用该诊断代替其准入。

当前时刻FK对比真实body-link字段：四个采集时刻、原128lane的拍头位置最大误差1.51e-6m、线速度最大7.69e-6m/s、角速度最大1.23e-5rad/s。排除明显关节/坐标映射错误，但尚未证明步内轨迹准确。详见候选运行目录下`current_racket_fk_diagnostic.json`、`impact_response_law_diagnostic.json`与`impact_friction_response_law_diagnostic.json`。

新增可选`NativeSubstepTraceV1`：只在预先指定的一个控制tick记录160个真实物理子步及起点，包含球完整13维状态、拍头位姿/速度、关节位置/速度/力矩、接触力与接触点有效性。GPU固定缓冲、不在子步复制对象树、不增加task/history/reward调用；无接触点保留NaN及valid标记。原始物理方法每次调用一次并在结束后恢复。35项CPU合同测试通过，语法验证通过；不是原生运行验收。

已启动一次原128环境、seed882072、tick272、原请求扰动±0.02的四分支GPU原生采集，新增的仅为读取诊断。输出`native_substep_contact_8zmiqosh`，systemd服务`supermimic-native-substep-contact-20261118.service`，Restart=no、总3600秒上限、单分支900秒上限。第一条名义轨迹先与既有名义输出严格核对；新名义重复、正负分支继续遵守同前缀验证。每分支273tick，共1092tick；不是800tick正式评分，也不是Actor更新。启动时Isaac环境已成功构造，实际采集尚未完成。

后续等待期间已准备`analyze_native_substep_pairs.py`：仅在四分支完整退出后读取同前缀数据，检查161帧连续性、首末状态与控制转移一致性、名义重复逐张量完全一致（未定义接触点显式保留），逐lane报告关节/球速度响应和真实接触边沿。接触阈值读取原物理配置，合力方向不被视为唯一接触法向；不改变任何模型、Actor或验收门槛。当前采集仍在第一名义分支，200tick进度已落盘，未完成配对验证。

## 子步采集：首名义分支已完成

首分支273tick完成，退出0，耗时416.668秒。新增记录与旧名义分支的next_state、final_joint_targets、final_normalized、request_before/request_after全部最大绝对差0，161帧与控制转移首末状态一致。记录器的只读性在这个名义轨迹上得到原生证据支持；四分支重复及动作响应尚未完成。

该控制步9个原生拍面/拍框接触lane各出现1个有力的62.5微秒子步，接触后的球自转瞬时峰值可超过100rad/s；不能用10ms端点中近100rad/s状态推断整个接触过程。拍框诊断案例127在第146个物理子步发生接触，真实合力/球速/自转/平均接触点已保存。该编号只作报告定位，未进入策略条件。详情`native_substep_contact_8zmiqosh/nominal_contact_substeps.json`及`instrumentation_parity.json`。

第二名义重复分支仍由同一个已启动服务顺序运行。没有新增Actor/WM/Critic更新，没有新的完整任务return或成功率。

## 子步机械响应归因与可微惯量原型（2026-11-18 08:36）

两条名义分支均退出0，原四分支作业继续负向/正向采集；未热改scripts/quadsmash源码。GPU此时3842/16303MiB、约48%利用率，对应128环境原生诊断，不是Actor训练。Actor/Critic更新仍0。

真实碰撞导致拍头法向表面速度变化约0.44–1.44m/s；使用碰前速度当作碰后不变表面会误判有效恢复系数。用真实碰后表面速度只读反推，多数案例恢复系数约0.717–0.743，与原0.72相近。不修改物理系数。

`audit_articulated_collision_response.py`用原资产质量/惯量、J M^-1 J^T与真实测得冲量做机械归因；9个接触的拍头响应方向余弦0.922–0.999，速度增量误差0.075–0.626m/s。该计算使用了真实接触力和点作为事后诊断，不能作为想象输入；未包含隐式PD、motor armature或地面约束，不能称作精确PhysX模型。

`audit_articulated_impact_law.py`将原球质量、恢复系数和摩擦与整机惯量结合；已观测冲击时刻下，拍线案例碰后球速误差约0.093–0.585m/s，部分拍框案例仍0.80–1.44m/s。它使用观测时刻/几何的oracle，因此只是归因证据。各原始结果保存于`native_substep_contact_8zmiqosh/nominal_*response.json`及`nominal_articulated_impact.json`。

新增隔离原型`articulated_candidate/inertia_candidate.py`仅在报告目录，尚未加入训练图。使用Torch批量计算原资产整机质量矩阵及接触点有效惯量，不调用MPC/控制器。3个CPU测试通过（1.80s，退出0）：对照现有Pinocchio惯量/点响应、状态梯度有限差分、恢复系数及Coulomb冲量约束。生产模型未替换。

将该原型用于原7个开发留出配对响应的只读因果计算，仍为6个正向、1个反向，且部分幅值明显不准。不能因惯量归因改善就宣称动作响应准入通过；不把单案例条件加入模型，不更改原阈值。本轮尚未产生新的模型拟合、Actor更新或正式成绩。当前剩余重点是拍框复合几何/接触模式响应，并等待同前缀正负子步证据。

## 新定位：拍线与拍框的材质合同不同，不能统一用0.08

静态USD审计确认原c2264f...资产拍线位于`/World/Racket/Collider/strip_*`，拍框为其兄弟`Frame_*`。原生`BallEnv._setup_scene`只将`ShuttleContactMaterial`（摩擦0.08）绑定到Shuttle和`Racket/Collider`；它不覆盖拍框。安装中的Isaac Lab `SimulationCfg.physics_material`默认静/动摩擦0.5，平均组合；`SimulationContext`明确将该材质用于没有指定材质的shape。当前源码无改写该默认值的任务设置。因此静态调用链推导拍框-球有效摩擦为(0.5+0.08)/2=0.29，而拍线-球是0.08。**尚未读取运行时PhysX材质张量，不能将静态推导写成已完成原生核验。**

证据：`audit_asset_material_bindings.py`只用USD库读原资产，不启动Kit/PhysX；结果`native_substep_contact_8zmiqosh/static_asset_materials.json`。模型中统一0.08的近似不能代表完整原生材质合同。不改原资产、材质绑定或真实物理。

按原绑定路径和已安装默认值（非搜索/拟合系数）进行只读冲击律重算：拍框诊断41误差由1.44降到0.39m/s，127由0.80降到0.13m/s，75由0.50降到0.19m/s。该结果仍使用观测冲击时刻，是oracle归因；因果端点预测对127方向仍错误，不能放行Actor。分别保存`nominal_authored_material_impact.json`及V3目录`impact_authored_material_response_law_diagnostic.json`。

还需核验真正的接触点/摩擦冲量；现有ContactSensor均值点、合力不足以还原多点摩擦求解。安装版API有只读`get_material_properties`、`get_contact_data`和`get_friction_data`。后两者共享count/start内部缓存，若增加诊断必须分别立即复制，不能让后一调用覆盖前一索引。当前已启动四分支源代码不变，未热加探针。

## 四分支原生子步核验完成，材质差异获运行时确认

`native_substep_contact_8zmiqosh`四分支均退出0，单次耗时416.668/412.625/418.165/409.066秒。273×4=1092控制tick，原128环境，共139776条原生转移。完整来源哈希核对未变化，systemd MainPID=0、inactive。`paired_substep_audit.json`确认161帧完整、名义重复所有字段逐元素一致（NaN按未知处理）、控制转移首末和实际限幅命令一致。仍是诊断，不是完整8秒评价或任何Actor学习成绩。

随后执行一次原128环境、原资产/seed/解释器的只读材质核验，**0控制tick，0 Actor/WM/Critic更新**，退出0。没有使用2/8/74环境替代原面板，没有录视频或修改材质。

输出：`/home/autodrive/LZQW/experiments/supermimic_admission256_20261118_s5lbit6k/native_material_readonly_20261118_0845`，包括启动命令/提交、`native_material_tensors.pt`、`material_contract.json`、生命周期及退出码。读取真实PhysX张量，机器人[128,132,3]包含[0.08,0.08,0.72]及[0.5,0.5,0]；球为[0.08,0.08,0.72]，128lane完全一致。实时USD显示`Collider/strip_30`绑定ShuttleContactMaterial，Frame_00及Clamp未指定材质、使用场景默认0.5。平均摩擦组合因此为拍线0.08、拍框0.29；恢复系数max仍0.72。

**这是模型统一摩擦假设与原生合同的具体差异，不是更改真实环境的授权或理由。** 下一修正必须按真实部件材质建模，并保持原环境、评分及物理参数；不能把oracle误差下降当作因果响应通过。惯量原型仍只在报告目录，未替换生产模型。Actor/Critic实际更新仍0，256 Actor预算尚未启动。


## V4 材质与惯量特征修复：GPU作业进行中

记录时间 2026-11-18T09:05:03.833362+08:00。新增独立schema `native-material-articulated-impact-action-v4`；以真实128环境读取的逐部件材质、固定资产惯量和当前预测碰撞状态构造24维因果特征。新增零初始化机械特征残差头，显式V3迁移严格保留旧参数与初始输出。没有更改原生摩擦、碰撞、动作、奖励或评分，也没有将未知球自转填零。该模型仍是学习型近似，未宣称等于PhysX。

GPU作业 `/home/autodrive/LZQW/experiments/supermimic_admission256_20261118_s5lbit6k/material_inertia_response_5a7ivx69`，systemd `supermimic-material-inertia-response-20261118.service`。固定256模型步、512样本、16步窗口、lr=2e-4；采样比例等比例扩大，原生128与Actor64步窗口不变。当前实际模型更新 181，Actor/Critic更新均0；峰值Torch分配 1718309376字节，最近16步中位耗时3.7228秒。尚无最终验证，不能写成已启动256次Actor长训。

运行前相关CPU集合49 passed；另一个证据合同集合28 passed，集合重叠，不累计成77。运行期间准备了隔离补丁：模型代码与真实执行源码适用范围区分12 passed；WM批量同比例配额11 passed；父adapter递归resume仍检查不可变材质/惯量4 passed。补丁仅在`evidence_scope_staged`，生产源码尚未覆盖，避免热修改作业。

原始同前缀物理数据重新核验并生成`collected_response_inventory.json`：t272/t271/t267留出接触4/2/1例，另含t302 Recovery32留出lane。所有反向响应与未知项保留，未按结果挑选样本。


## V4最终结果与工程补丁接入

固定256次WM修复已完成，systemd退出0，耗时975.05秒，Torch峰值1,718,309,376字节；源文件1310个逐项哈希未变。Actor/Critic更新均0。接触前/当步/后速度RMSE 0.912287/1.960487/0.023493 m/s；当步前向符号错误4/30，历史对照2/30。7个留出配对响应4正3负；Recovery角速度方向30/32为正，中位余弦0.98084。未准入，不生成大规模训练完成标记或新性能成绩。

真实下一机器人状态仅用于独立oracle归因：替换预测机器人端点之后，3个反方向击球案例仍反方向，关节响应方向余弦约0.99。因此主要断点位于球响应映射，而不是Actor参数未连通或关节目标设备错误。oracle不得输入生产想象或策略。

作业退出、源文件哈希核对后才接入工程补丁：保留完整原始代码哈希，允许明确的预测模块变化后重算当前原生响应；Actor/G07/mixer/限速/未知源码变化仍拒绝，缺少当前响应检查仍拒绝。父adapter递归恢复也强制核对材质和惯量缓冲区。长训WM batch新增可配置512，来源与事件配额按128单位同比例放大，记录实际配额、恢复配置和正确统计分母；原生仍128，H64/H16不变。67项合同测试及42项相关集成测试分别通过，退出0，未声称完整仓库或PhysX验收。

输出哈希：world `b5c50f07ab5330c3080e3d984c590909d34377040b8c4a9e480810edc0afc747`；action-ball `68881ff36306910eb8feb7d4115cba3ff6c0364794d4be16550f786bf72a4109`；legacy ball `e45ed823301a081705c98c1ff0bd2ebb64c241cf422fa0bd1b966d04ca3d48a7`。全部保留作诊断。

下一步核查已有不同策略原生接触数据覆盖及实际命令身份，避免反复只拟合相同π0轨迹。旧数据只能在已证明物理合同兼容时用于动力学监督；缺失策略身份不能伪装成当前策略RTG或完整想象起点。当前没有Actor长训、没有新的TensorBoard训练曲线。


## GPU批量吞吐只读测量

同一V4模型、原16步窗口，GPU正反向但optimizer=0，前后全部权重哈希一致。两个重复，以下为第二次。各批次用途仅吞吐测量，不比较loss选参数，不作为原生成绩。

| batch | 时间s | Torch峰值GB | 监督位置/s |
|---:|---:|---:|---:|
| 128 | 2.3010 | 0.486 | 890.1 |
| 512 | 2.3933 | 1.686 | 3422.9 |
| 1024 | 2.3637 | 3.267 | 6931.5 |
| 2048 | 2.5134 | 6.439 | 13037.3 |

该测量不包含Isaac、task model和64步Actor。当前长训默认512，不根据单独模型峰值宣称2048在完整训练中一定安全；也必须考虑有效独立窗口与episode重复上限，不能用重复数据灌满显存。

## 历史真实数据与固定GPU修复队列

原972tick归档已验证124416条真实转移，命令解码最大差0，16步有效窗口120546；历史policy version缺失明确保留未知，仅用于物理监督，不作为当前策略RTG或完整想象起点。第二份1632tick归档仍在逐块验证。旧归档含完整控制器对象树，CPU解包是当前耗时点；紧凑包完成后可直接复用。

已排定同一V4模型的固定256次WM更新，batch1024、H16、lr2e-4；来源配额为初始化50%和两份经验证历史各25%。77/77可匹配训练触球记录的实际命令与初始化不同，提供了新增物理状态和动作监督；没有按案例编号改策略，没有扩大时域或改变原生物理。所有源均排除固定32个开发留出lane，但原始WM历史拟合独立性仍未知，不宣称独立泛化。

后台队列在CPU校验退出0且真实completed.json存在后启动GPU拟合；失败不重试。新增只读复核服务在拟合结束后核对源代码未变、原生配对PT哈希和固定验证集合，再生成review.json，不会自动生成通过凭证或把WM更新冒充Actor训练。

另外已在报告目录隔离准备证据校验器来源分类补丁：校验器本身变化不等于原生执行变化，但必须对当前权重与当前校验器重算响应区域。真实执行/采集源码变化仍拒绝。5项CPU测试通过；为保护排队作业来源，补丁尚未应用到生产源码。事件索引6已核对原生实现，确实是frame_hit锁存，并非通用身体触球故障，无需修改该标签。

本次目标仍为256实际Actor更新；当前Actor/Critic均0，尚无新的候选原生成绩或长训TensorBoard。实时作业位置和PID见queued_pipeline_status时间戳记录。

## 固定历史多源模型修复最终结果

历史数据校验完成：source00 为 972 控制 tick / 124416 条转移 / 120546 个有效 16 步窗口；source01 为 1632 控制 tick / 208896 条转移 / 203136 个有效 16 步窗口；两源的最终命令解码最大差均为 0。两源策略版本缺失保持未知，仅允许进入动力学监督。

固定 GPU 修复实际完成 256 次模型优化（batch=1024、window=16、lr=2e-4），耗时 1000.935386 s，PyTorch 峰值 3525296128 bytes；Actor=0、Critic=0。初始化 50%/history0 25%/history1 25% 的采样配额在每个日志更新中实际记录。

固定留出复核结果：接触前速度 RMSE 0.94969094 m/s，接触当步 1.64610159 m/s，接触后 0.04665850 m/s；与旧模型相比接触速度 RMSE 改善、接触前不退化，但接触方向错误率未改善，接触后不满足不退化条件。7 个原生配对响应中 5 个方向为正、2 个为负。状态为 `PREDICTION_GATE_FAILED`，没有生成 Actor 准入凭证，也没有启动 Actor 训练。

这次失败不是进程崩溃，而是预先声明的动作响应物理准入门槛未通过。当前不能为了尽快占满显存而绕过该门槛；否则会把两个方向错误的球响应梯度用于 Approach BPTT。原始模型、修复模型、历史压缩包和所有失败证据均保留。

## 修复模型动作响应归因

对最终历史多源候选进行了 GPU 只读端点归因，未执行优化器。7 个留出响应中，机器人预测端点相对真实端点的方向余弦均约 0.987–0.999；替换为记录的未来机器人端点后，拍球响应仍有负向/近零案例（TRAIN lane 7: -0.6229，lane 37: +0.0753，lane 5: +0.5399）。实际候选为 lane 7=-0.6867、lane 37=-0.4740、lane 5=+0.0334。说明当前主要误差在球拍接触/球响应映射，不能归因于 G07 机器人端点预测，不能放宽门槛或直接启动 Actor。

## 2026-11-18 frame-mode repair completion

The corrected GPU response fit was rerun through the formal `scripts/python.sh` entrypoint after the first direct Python launch failed with `ModuleNotFoundError: quadsmash` (zero updates in that failed run). The corrected run completed 256 bounded response-model updates with actor/critic updates equal to zero. It used the existing `other_residual` mode for native frame-contact labels; the final fit recorded 202 frame-mode, 310 racket-mode, 64 unknown and 960 supervised rows in its last batch. Peak GPU allocation was 3,525,296,128 bytes.

The held-out review remains `PREDICTION_GATE_FAILED`: contact velocity RMSE improved (2.66863 to 1.02325), contact forward-sign errors improved (6 to 1), and pre-contact RMSE did not worsen relative to the unchanged legacy reference (1.39564 to 1.58504 is compared against the run's declared criterion and is recorded as passing there), but post-contact velocity RMSE worsened from 0.05235 to 0.07137. Native paired action response remained 5 positive and 2 negative directions out of 7. The candidate is not admitted; no actor training or formal evaluation was started.

## 2026-11-18 12:08 final action-only V5 fit

The final action-only frame-aware candidate completed exactly 256 GPU response-model optimizer updates. It froze the physical response and legacy ball checkpoints; no Actor or Critic update was executed. Peak PyTorch allocation was 3,098,397,184 bytes and median update time was 3.21996 s. Artifacts:

- `frame_aware_action_only_115433/fit/world.pt` SHA256 `2361616befd3b5ef386654b67eb52ab923858a0b6d1511f18dec205b75a8d58d`
- `frame_aware_action_only_115433/fit/action_ball_candidate.pt` SHA256 `f291d9952611a5df4500bc75550523e4fc9eda386c51ceb9ffb21ef2ab17e9b`
- `frame_aware_action_only_115433/fit/legacy_ball.pt` is reused from the recorded historical path and is not overwritten.

The fixed multisource review returned `PREDICTION_GATE_FAILED`: contact offsets were 6/7 positive and 1/7 negative; the comparison against the declared legacy reference reports contact sign error not improved and post-contact velocity not non-degraded. The model's own held-out metrics were: offset -1 velocity RMSE 1.292868 (sign error 0.0333), offset 0 RMSE 1.634326 (sign error 0.1000), and offset +1 RMSE 0.041852 (sign error 0). `native_admission=false`; no current `supermimic-contact-validation-v1` receipt exists for these exact model weights and the action-ball candidate therefore cannot pass the large-profile admission contract.

Large 256-Actor training was intentionally not started. Starting it now would require bypassing the existing contact-validation gate and would make the resulting BPTT updates scientifically untraceable. The failed candidate, review JSON, logs and all earlier candidates remain preserved.

## 2026-11-18 V6 analytic-impact-prior follow-up

A V6 candidate was implemented and fit once with the fixed 256-update budget. It adds a fixed causal articulated-material impulse prior computed from the predicted impact state; the learned V5 head remains a residual. Syntax and the 12 targeted action-ball tests passed before launch. The fit completed with Actor/Critic updates equal to zero.

Run: `/home/autodrive/LZQW/experiments/supermimic_admission256_20261118_s5lbit6k/frame_aware_analytic_prior_v6_124942`

- world SHA256: `ad8dfe9cd34ea27b25270e437413bc0b2b723fe794281e5e9883e9474360dfec`
- action-ball SHA256: `d4ea4378d9ac53ee907d32bef68c061fa0374e38ee9f7fe110f4b34837051362`
- native response directions: 4 positive / 3 negative of 7
- review status: `PREDICTION_GATE_FAILED`
- model updates: 256; Actor/Critic updates: 0

The analytic prior therefore does not provide a valid admission path. Large training remains stopped because the exact model weights still lack a passing native contact-validation receipt. No threshold, episode denominator, or execution contract was relaxed.
