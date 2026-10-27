# 正常比例的完整扣杀成功视频

[观看／下载约15秒视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Smash_success_TRAIN29_normal_ratio_20261027.mp4)

这是上一仿真布局中真实完成的 TRAIN29 回合，使用完整球拍 R3 物理模型。当前原点0新版尚无完整扣杀成功，不能把本片作为新版成绩。全部画面均明确显示历史来源；本次仅重新制作相机、字幕和播放片段，没有执行新的原生仿真。

视频先保留完整8秒回合，再播放触球四倍慢放、真实接触姿态和恢复定格。原生记录为合法空中拍线接触、出球9.07 m/s、向下3.29 m/s、真实越网、首次落点 [3.893, −0.726] m 命中原目标区，4.064秒合格恢复，无该回合违规或数值故障。来源冻结策略在原全128例中为3完整、60空中、26条件恢复；不是正式A/B成绩。

场景保留米制几何、机器人与球拍网格原尺寸，采用固定正常透视和方形像素，无横纵拉伸。画面包含1米标尺、完整机器人/机械臂/675毫米球拍、1.524米中央网高、实际首落点及同步全场俯视。球网以原刚性悬挂网体的线框方式绘制，视觉简化不修改碰撞物理。

741帧已完整解码。全部400帧原生身体姿态、球位置和时钟与原始记录逐位核对；最终文件与原渲染逐帧像素相同，只补充显式1:1像素比例。相机同深度水平与竖直米制投影均为181.512731像素/米。实际解码的起跳与恢复画面已人工检查。

最终文件：1280×720、50 fps、H.264、14.82秒、1,103,678字节。SHA256：`bec509f2ba5266a6983983a35ccd47a9b94c7cb98bf09e6718210290c001b406`。

来源策略 SHA256：`a7953fb80d0ea647cbd6d525d8ddbf8a0a0986f1bcecfbe8831db2c1384473a3`。原始数据目录为主项目的 `reports/G07_full_racket_v4/contact_aim7_best_fixed_train128_01`；完整来源数据见[已上传的冻结128例与教师搜索证据包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_contact_aim7_completed_search_cold_TRAIN128_origin16_20261027.zip)。

`render_success.py`、`verify_video.py`、`RENDER_RECEIPT.json`、`FINAL_VIDEO.json`、`DECODE_AND_SCALE_CHECK.json`、`VISUAL_REVIEW.json` 分别保留制作、验证及来源证据。`UPLOAD_RECEIPT.json`记录GitHub文件的新下载整文件SHA核验。

当前GPU0的新15更新训练已结束，独立原128例评测继续由现有队列执行。该训练738个闭合回合为8空中、1合格恢复、0完整、数值故障0；这不是冻结评测结果。下一步由同一队列完成独立评测，依据新布局实际结果继续适配；本片不改变正式门槛或G07未完成状态。
