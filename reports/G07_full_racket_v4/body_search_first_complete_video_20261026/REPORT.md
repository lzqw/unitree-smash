# 搜索中的完整回合与失败视频

[观看40.74秒视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_training_body_search_complete_and_failures_20261026.mp4)。实际来自已完成的body搜索第1代候选6，搜索后选取TRAIN68完整、TRAIN23落点失败、TRAIN107挥空与非足端触地失败。完整Go2、标准PiPER、拍柄/拍杆/拍头、球网和目标区均使用真实50Hz全身记录及精确接触/终止事件渲染；没有生成或插值动作。

候选6内部128例为23空中、12条件恢复、1完整；其随后独立冷原128为41空中、29恢复、0完整，不能把本片当作稳定成功或冷评估通过。原生env836/791/875与EP3完整保留，没有把实际回合ID伪改为EP0。

2037帧全解码，主代理实际查看8张事件图与8张解码帧，未逐帧观看全片。上传3419799字节，SHA5b1a98fa87de5b3237666ebdc95d68b835f0fe0532acb6d87478d99b6b670821，新下载与服务端digest均一致。SPEC与适配器在隔离分支保存，原native代码和现有renderer未修改。
