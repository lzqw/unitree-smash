# 隔离的可微惯量核对

原型只在报告目录，未接入当前运行的原生采集、WM或Actor。它使用原资产的所有刚体质量、COM和惯量，通过全身Jacobian构造M及接触点J M^-1 J^T。无MPC或控制命令输出。

CPU原型检查3 passed，退出0；对照Pinocchio是数学实现检查，不是PhysX质量认证。忽略隐式PD、motor armature、接触中的地面约束。单次球碰撞还忽略多个同时接触点的耦合求解。

原生拍头速度响应方向与该机械归因基本一致，但单次碰后球速在部分拍框案例仍有较大误差。7个开发留出配对方向仍6正1负，不得升级为可训练模型或生成通过凭证。未来真实接触时刻、力、接触点、碰后拍头速度只用于诊断。

`audit_articulated_impact_law.py`使用原物理系数进行冲击时刻oracle检查；`audit_impact_articulated_response_law.py`使用当前状态及已有机器人预测端点进行因果单步诊断。二者必须区分。
