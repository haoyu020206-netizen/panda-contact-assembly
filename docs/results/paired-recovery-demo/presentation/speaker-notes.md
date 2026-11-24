# Franka Panda 配对演示讲稿

选一套语言讲即可。句首时间是视频进度，画面右上角显示仿真时间。视频静音，留给你现场讲。

## 中文发言

**0:00–0:08 · 先看结果**

这是我在 MuJoCo 做的 Panda 插孔对照。同一扰动下，左边成功，右边超时。

**0:08–0:16 · 交代对照**

两边用同一个检查点，位姿估计都有延迟和丢帧。右边加了带补偿的螺旋搜索。

**0:16–0:34 · 指一下橙色虚线，再指绿色深度曲线**

橙色虚线是触发点，仿真时间十点五六秒。看深度曲线，左边继续推进，在十一点八九秒完成插入。右边则停在孔口附近。

**0:34–0:43 · 指一下右侧结果**

十八秒时，右边超时。两边按仿真步记录的峰值力都是五十八牛。

**0:43–0:54 · 从单例回到实验**

各跑六十回合，关闭搜索成功四十三次，补偿搜索三十一次。下一步验证推进停滞时再触发。

## English talk track

**0:00–0:08 · Show the outcome**

I used MuJoCo to compare two recovery decisions for Panda peg insertion. The left run succeeds. The right run times out.

**0:08–0:16 · Explain the setup**

Both use the same checkpoint, three millimeters of radial clearance, and delayed, lossy pose estimates. The right run adds compensated spiral search.

**0:16–0:34 · Point to the orange marker and mint depth curve**

The orange line marks the trigger at 10.56 seconds of simulation time. From there, watch the insertion depth. The peg on the left keeps advancing and completes insertion at 11.89 seconds. The one on the right stays near the socket opening.

**0:34–0:43 · Point to the timeout**

The right run times out at 18 seconds. Both trials recorded a 58-newton peak at the physics step rate.

**0:43–0:54 · Connect the case to the benchmark**

In 60 trials per policy, search off succeeded 43 times and compensated search 31. Next, I’d test triggering only after insertion stalls.

## 观众追问时

- **为什么力图里看不到 58 N？** 图上是按 50 Hz 控制频率画出的力采样。58.0 N 是每个仿真物理步都检查后得到的峰值，可能出现在两次控制采样之间。
- **这个案例能证明搜索触发得太早吗？** 它说明了同一扰动下，加上搜索会怎样改变这一次的轨迹和结果。是否属于过早触发，要看触发前是否还在稳定推进，并在其他扰动上复验。
- **60 回合独立吗？** 每种策略有 60 个评估回合，来自三个训练检查点。种子安排使初始扰动有复用，每种条件共 22 组不同的扰动向量。
