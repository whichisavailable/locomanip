## Overview

**robot_lab** is a RL extension library for robots, based on IsaacLab. It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

## Go2Arm Task

`go2arm` 是一个四足机器人背载机械臂的 loco-manipulation 任务，目标是在移动过程中完成末端位姿跟踪。

当前工作主要复现文章 *Learning Whole-Body Loco-Manipulation for Omni-Directional Task Space Pose Tracking with a Wheeled-Quadrupedal-Manipulator* (RAL)。原文面向轮足机器人，主要贡献集中在三类 reward shaping 设计。本项目沿用其奖励形式，并将任务迁移到四足版本。

目前观察到的结果是：在未显式加入机身命令的情况下，再叠加四足机器人的步态约束，训练效果不理想，教师策略也尚未稳定收敛。因此，这份 README 先以 `flat` 任务为主，`rough` 版本仅作为补充保留。

### Robot Setup

- 机器人主体：Unitree Go2 四足机器人
- 机械臂：Piper 6 自由度机械臂
- 安装方式：机械臂通过背部固定安装位 `arm_mount` 装到 Go2 机体上，URDF 里的链路顺序是 `arm_mount -> link1 -> link2 -> link3 -> link4 -> link5 -> link6`
- 末端执行器：`link6`

### Default Joint State

当前任务使用的默认初始关节位来自本地 Go2Arm 资产配置，默认值如下：

```text
四足关节
FL_hip_joint   = 0.0
FL_thigh_joint = 0.8
FL_calf_joint  = -1.5
FR_hip_joint   = 0.0
FR_thigh_joint = 0.8
FR_calf_joint  = -1.5
RL_hip_joint   = 0.0
RL_thigh_joint = 0.8
RL_calf_joint  = -1.5
RR_hip_joint   = 0.0
RR_thigh_joint = 0.8
RR_calf_joint  = -1.5

机械臂关节
joint1 = 0.0
joint2 = 0.314
joint3 = -0.2967
joint4 = 0.0
joint5 = 0.0
joint6 = 0.0
```

其中，机械臂默认姿态会尽量避免把 `joint2` 和 `joint3` 放在单侧关节极限附近，以减少训练初期随机重置时被卡住的风险。

### Local Version

我本地使用的版本是：

- Isaac Sim 4.5
- Isaac Lab 2.2.1

### Training

`go2arm` 任务已经注册了两个环境：

- `RobotLab-Isaac-Flat-Go2Arm-v0`
- `RobotLab-Isaac-Rough-Go2Arm-v0`

当前主要推荐先跑 `flat` 版本：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py ^
  --task RobotLab-Isaac-Flat-Go2Arm-v0 ^
  --num_envs 4096 ^
  --headless
```

常用补充参数：

- `--video`：训练时录制视频
- `--max_iterations`：覆盖默认训练轮数
- `--seed`：设置随机种子

训练输出会保存在 `logs/rsl_rl/<experiment_name>` 下，`flat` 版本默认是 `unitree_go2arm_teacher_flat`。

如果你后面要切到 `rough`，把 `--task` 换成 `RobotLab-Isaac-Rough-Go2Arm-v0` 就可以。

### Play

训练完成后，可以用 `play.py` 读取最近一次训练的 checkpoint：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py ^
  --task RobotLab-Isaac-Flat-Go2Arm-v0 ^
  --num_envs 1
```

如果要指定固定 checkpoint，可以额外传入 `--checkpoint`；如果要直接加载发布的预训练权重，可以使用 `--use_pretrained_checkpoint`。

Go2Arm 的 `play` 脚本还支持两个常用调试参数：

- `--go2arm_ee_pos X_B Y_B Z_W`：固定末端目标位置
- `--go2arm_ee_rpy ROLL_B PITCH_B YAW_B`：固定末端目标姿态
- `--go2arm_trace_actions`：打印动作和关节状态调试信息

### Note

- 这份说明主要面向 `flat` 任务；`rough` 版本是后续补充，不是当前主线
- 当前任务请使用 URDF 版本，不要依赖 USD 文件
- USD 文件无法单独在足端添加 contact sensor，所以 go2arm 现在的精确足端接触逻辑不能直接靠 USD 资产来做

## Citation

This repository is a modified version of `robot_lab`.
