# 21/128 首次非足部触地诊断（2026-11-02）

来源：`transfer_blend04_continuous8_lowppo_frozen128_01/first_nonfoot_contact_details.json`。这是冻结 21/128 策略的原生接触遥测，不改变正式评分。

- 首次非足部触地事件：68
- FL_thigh：57
- FR_thigh：7
- FL_hip：4

按记录四元数（wxyz）计算，FL_thigh 事件的滚转中位数约为 -0.626 rad，FR_thigh 事件约为 +0.641 rad；这表明多数失败是机身滚转导致低侧前大腿先触地。下一轮只加强恢复滚转反馈到 -0.8，其余 PPO、教师、任务阈值和物理条件保持不变。该假设需要新的冻结评估验证。
