# Franka Panda Contact Assembly

[![CI](https://github.com/haoyu020206-netizen/panda-contact-assembly/actions/workflows/ci.yml/badge.svg)](https://github.com/haoyu020206-netizen/panda-contact-assembly/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-4.x-00A6D6.svg)](https://mujoco.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个面向控制/机器人岗位作品集的可复现实验项目：在 MuJoCo 中实现 7-DoF Panda 风格机械臂的关节轨迹控制、笛卡尔阻抗控制与 peg-in-hole 接触装配，并量化扰动下的表现。

当前是考试前 MVP。范围有意限制在 Python + MuJoCo，不包含 ROS2、视觉、强化学习或真机部署。

GitHub Actions 会在标准 Linux runner 上执行 lint、单元测试、关节/笛卡尔/插孔实验和完整鲁棒性批次，并保存可下载的 CSV、JSON 与 PNG 结果，确保仓库不是“只能看、不能跑”的演示代码。

## 已实现

- 自带无网格依赖的 7-DoF Panda 风格 MJCF，可立即验证控制代码；
- 五次多项式关节轨迹与带偏置力补偿的 PD 力矩控制；
- 末端 Jacobian、位置/姿态误差、笛卡尔阻抗与零空间姿态控制；
- 方孔插入任务、接触力读取和成功判据；
- 孔位、摩擦和工具质量扰动的批量实验；
- CSV 时序数据、JSON 摘要和可选 PNG 曲线；
- 单元测试和命令行入口。

> `src/panda_assembly/assets/panda_mvp.xml` 是为控制链路开发准备的轻量化 Panda 风格模型，并非 Franka 官方惯量/碰撞模型。完成 MVP 后，应将它替换为 MuJoCo Menagerie 的官方 Franka Panda 模型，再重新整定参数并报告模型来源。

## 环境

推荐 Python 3.11–3.14。一般情况下，Windows PowerShell 可在项目内创建环境：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

如果本机只有更新的 Python，可先尝试安装；若 MuJoCo 尚未提供对应 wheel，再安装 Python 3.12。

如果出现“应用程序控制策略已阻止此文件”的 NumPy DLL 错误，这通常来自受管理的 Python 执行环境，而不是项目路径。请改用正常安装的 Python 3.12，或在允许原生扩展的 WSL2/开发容器中运行；无需移动 D 盘上的源码和实验输出。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

## 运行

先做无界面验收：

```powershell
python -m panda_assembly joint --output runs\joint_smoke
python -m panda_assembly cartesian --output runs\cartesian_smoke
python -m panda_assembly peg --output runs\peg_smoke
python -m panda_assembly batch --output runs\robustness_mvp
```

观察单次实验：

```powershell
python -m panda_assembly peg --viewer
```

每次运行生成：

```text
runs/<experiment>/
├── summary.json       # 成功率相关指标、RMS/最大误差、最大接触力、完成时间
├── timeseries.csv     # 可复核的逐步记录
└── metrics.png        # 误差与接触力曲线（安装 matplotlib 时）
```

批量实验额外生成 `batch_summary.csv` 与 `aggregate.json`。

## 指标定义

- `success`：peg tip 的横向误差不超过配置的 clearance，且插入深度达到阈值；
- `rms_position_error_m`：控制周期内末端位置误差的均方根；
- `rms_joint_error_rad`：关节实验中 7 轴误差向量范数的均方根；
- `rms_orientation_error_rad`：笛卡尔/插孔实验中的末端姿态误差均方根；
- `max_contact_force_n`：peg 所有接触点法向/切向合力范数之和的最大值；
- `completion_time_s`：首次同时满足横向误差和插入深度阈值的仿真时间；
- `final_xy_error_m`、`final_insertion_depth_m`：最终装配几何误差。

配置集中在 `configs/mvp.toml`。调整刚度时同步检查阻尼、力矩饱和、峰值接触力与仿真稳定性，不要只追求成功率。

## 项目结构

```text
configs/mvp.toml                 # 控制器、任务与输出配置
src/panda_assembly/
├── assets/panda_mvp.xml         # 轻量化 7-DoF 模型与插孔工装
├── controllers.py               # 关节 PD / 笛卡尔阻抗
├── experiments.py               # joint / cartesian / peg / batch
├── math_utils.py                # 姿态误差与阻尼零空间投影
├── metrics.py                   # 指标、CSV、JSON、曲线
├── model.py                     # 模型加载与接口校验
└── trajectories.py              # 五次多项式轨迹
tests/                            # 配置、数学、轨迹与模型测试
```

## 五周里程碑

1. 环境、模型编译、关节轨迹 smoke test；
2. 对比不同 `Kp/Kd` 的跟踪误差与力矩饱和；
3. 验证 Jacobian 与笛卡尔阻抗，记录位置/姿态阶跃响应；
4. 插孔接触、成功判据、孔位/摩擦/质量扰动；
5. 固化实验种子和配置，补结果表、README 图与两分钟演示视频。

## 后续升级顺序

1. 替换为 MuJoCo Menagerie 的官方 Panda 模型并建立参数基线；
2. 增加位置控制 vs. 阻抗控制的同条件对比；
3. 固定随机种子，扩展多次 Monte Carlo 实验和置信区间；
4. 考试后再接 ROS2/`ros2_control`、MoveIt2 与 residual RL。

## 简历表述

> Developed and evaluated Cartesian impedance control for contact-rich manipulation with a 7-DoF Franka Panda in MuJoCo, including automated robustness testing under pose, mass, and friction variations.

