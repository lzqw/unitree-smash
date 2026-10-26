# 新body教师的第三残差桥接：仅CPU准备

新公共schema full-racket-body-residual-ppo-v4保留实测body4和两层已有残差actor及归一化，再添加全新零输出残差。它在1600条实际100Hz输入上与body4指令误差为0；新的actor和critic归一化计数0，Adam为空，初始末层输出0，探索std腿.2/臂.1、尺度均.15，对应请求物理噪声.03/.015rad。7个文件与隔离已测源码逐字节核对见INTEGRATION_FILE_CHECK。

这是后续可能方法的CPU兼容性准备，没有启动原生训练，也没有继承或声称性能提高。当前真正运行的是另立的body_strike_grid16_queue01固定参数网格。若之后启用该PPO，需另行明确PLAN、源选择、实际地面初始化和原GPU0 guard；不能把旧0758初始化静默套在body4上。
