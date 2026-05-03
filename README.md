## Overview

**robot_lab** is a RL extension library for robots, based on IsaacLab. It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

## Go2Arm Task

`go2arm` 是一个四足机器人背载机械臂的 loco-manipulation 任务，目标是在移动过程中完成末端位姿跟踪。

当前工作主要复现文章 *Learning Whole-Body Loco-Manipulation for Omni-Directional Task Space Pose Tracking with a Wheeled-Quadrupedal-Manipulator* (RAL)。原文面向轮足机器人，主要贡献集中在三类 reward shaping 设计。本项目沿用其奖励形式，并将任务迁移到四足版本。

目前观察到的结果是：由于命令仅为末端执行器位姿，未显式加入机身命令，再叠加四足机器人的步态约束，训练效果不理想。原定通过蒸馏训练学生策略，但是目前教师策略的效果不佳，因此还未添加完整域随机化、蒸馏适配。

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

本地使用的版本是：

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

- `--agent=rsl_rl_with_symmetry_cfg_entry_point`：左右对称增强（不使用镜像损失）
- `--num_envs`：覆盖默认训练并行环境数
- `--seed`：设置随机种子

训练输出会保存在 `logs/rsl_rl/<experiment_name>` 下，`flat` 版本默认是 `unitree_go2arm_teacher_flat`。

### Play

训练完成后，可以用 `play.py` 读取最近一次训练的 checkpoint：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py ^
  --task RobotLab-Isaac-Flat-Go2Arm-v0 ^
  --num_envs 4
```

如果要指定固定 checkpoint，可以额外传入 `--checkpoint`。

Go2Arm 的 `play` 脚本还支持两个常用调试参数：

- `--go2arm_ee_pos X_B Y_B Z_W`：固定末端目标位置
- `--go2arm_ee_rpy ROLL_B PITCH_B YAW_B`：固定末端目标姿态
- `--go2arm_trace_actions`：打印动作和关节状态调试信息

### Note

- 这份说明主要面向 `flat` 任务；`rough` 版本是后续补充，不是当前主线
- 当前任务请使用 URDF 版本，不要依赖 USD 文件
- USD 文件无法单独在足端添加 contact sensor，所以 go2arm 现在的精确足端接触逻辑不能直接靠 USD 资产来做

### Go2Arm Task Design Details

这部分主要展示当前这一版本go2arm任务的具体任务细节。

主要文件:

- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/rough_env_cfg.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/cus_velocity_env_cfg.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/commands.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/curriculums.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/observations.py`

值得注意的是，`cus_velocity_env_cfg.py`的主要作用是配置好接口，具体参数设计以`rough_env_cfg.py`为准。

#### 动作设计

`go2arm` 当前使用 **18 维、以默认姿态为中心的 delta joint-position action**：

```text
target_joint_pos = default_joint_pos + clipped_delta_action
```

关节顺序固定为：

```text
FL_hip_joint, FL_thigh_joint, FL_calf_joint,
FR_hip_joint, FR_thigh_joint, FR_calf_joint,
RL_hip_joint, RL_thigh_joint, RL_calf_joint,
RR_hip_joint, RR_thigh_joint, RR_calf_joint,
joint1, joint2, joint3, joint4, joint5, joint6
```

设计要点：

- 零动作表示保持在默认关节姿态附近，而不是把所有关节直接打到 `0 rad`。
- clip 施加在 **delta** 上，而不是施加在最终关节目标上。
- `scale=1.0`，因此策略输出可直接解释为关节偏移量，单位是弧度。
- 前 `500` 个 iteration 会强制把机械臂 6 个关节的 delta 固定为 `0.0`，因此早期阶段本质上是 locomotion-only。

`rough` 配置把 delta 范围放宽（极限的80%），例如：

- 腿部 `hip` 关节约为 `[-0.83776, 0.83776]`
- 前腿 `thigh` 关节约为 `[-1.86465, 2.18455]`
- 后腿 `thigh` 关节约为 `[-0.81745, 3.23175]`
- `calf` 关节约为 `[-1.03421, 0.47375]`
- 注意动作clip 80%指的是两侧各留10%的关节极限裕度，而非以默认关节姿态为中心

#### 观测设计

当前任务将观测分成三组：

- `policy`：给 actor 使用的核心观测
- `privileged`：给 teacher / privileged 分支使用的特权观测
- `critic_extra`：给 critic 额外使用的观测

##### 核心观测（policy）

核心观测总维度为 **92D**，按如下顺序拼接：

| 项 | 维度 | 含义 |
| --- | ---: | --- |
| `joint_pos` | 18 | 腿部和机械臂全部关节的相对位置 |
| `joint_vel` | 18 | 腿部和机械臂全部关节的相对速度 |
| `ee_current_pose` | 12 | `link6` 在 `base_link` 坐标系下的当前位姿：`3D 位置 + 9D 旋转矩阵` |
| `actions` | 18 | 上一步实际执行的 effective action |
| `ee_pose_command` | 12 | 当前末端目标命令 |
| `base_lin_vel` | 3 | 基座线速度 |
| `base_ang_vel` | 3 | 基座角速度 |
| `projected_gravity` | 3 | 重力在 base frame 下的投影 |
| `reference_tracking_error` | 1 | 随时间衰减的参考跟踪误差 |
| `gait_phase` | 4 | 四条腿的 trot 相位正弦信号 |

几个关键点：

- 当前 EE 位姿和目标 EE 位姿都在 **base frame** 下表达。
- `actions` 读取的是经过早期机械臂冻结后的 **effective action**，因此观测和真实执行行为保持一致。
- `gait_phase` 使用 `cycle_time = 0.5` 和相位偏置 `(0.0, 0.5, 0.5, 0.0)` 生成，即trot步态。

##### 特权观测（privileged）

特权观测主要提供地形、接触、外力和扰动信息，包括：

- `base_height`，1D
- `foot_heights`，4D
- `feet_contact_forces`，展平后为 12D
- `static_friction`，4D
- `base_external_wrench`，6D
- `base_external_push_velocity`，6D
- `base_mass_disturbance`，1D
- `ee_mass_disturbance`，1D
- `ee_external_wrench`，6D
- `ee_velocity_b`，6D
- `feet_planar_velocities_w`，8D
- `observation_delay`，1D

特权信息会经过MLP输出32维特征向量，与本体感知拼接后作为动作网络输入。后面蒸馏操作将会使用历史本体感知信息经过MLP输出32维特征向量。

##### Critic 额外观测

- `cumulative_tracking_error`，1D

#### 命令设计

当前任务只保留一个命令项：`ee_pose`，它表示 `link6` 的固定末端位姿目标。

- 只在 reset 时重采样一次，之后整个 episode 内保持不变。（保持世界坐标系下命令不变）
- 共12D：`3D target position + 9D target rotation matrix`。
- 主要在 **base frame** 中对 `x/y` 和姿态进行采样。
- `z` 单独在 **world frame** 中采样，这样更容易围绕地面高度组织 staged curriculum。
- 机身附近有一个 reject cuboid，用于过滤明显不合理或不可达的目标。
- 如果多次采样仍失败，实现会回退到采样范围中心点，以保证命令始终有效。

teacher 的基础命令空间会在 `rough_env_cfg.py` 中被改写成一个分阶段的前向 reaching curriculum：

- `x` 固定在 `[0.40, 1.60]`
- `y` 固定为 `0.0`
- locomotion warmup 阶段把 world `z` 固定在 `0.7126649548`（机械臂默认构型正运动学求解得到的高度与loco阶段机身默认高度相加）
- locomotion warmup 阶段把 ee orientation 固定在 `0,1.5008926535,0`（末端执行器默认角度）

命令项内部还会维护一组误差状态：

- `position_tracking_error`
- `orientation_tracking_error`
- `tracking_error`
- `reference_tracking_error = max(initial_tracking_error - v * t, 0)`
- `cumulative_tracking_error`

这些量会被观测、reward gating 和 debug logging 复用。

#### 奖励设计

这个任务不是简单地把 locomotion reward 和 manipulation reward 直接相加，而是使用一个 **门控总奖励**：

```text
total_reward =
    (1 - D) * mani_total
  + D * loco_total
  + basic_total
  + workspace_position_reward
```

workspace奖励项是由于机器人倾向于走到任务正下方再举手而增加的期望机身位置奖励。

门控项为：

```text
D = sigmoid((5 / gating_l) * (reference_tracking_error - gating_mu))
```

当前 rough 配置中的门控参数为：

- `gating_mu = 0.65`
- `gating_l = 0.45`

直观理解：

- 当 reaching 误差较大时，总奖励更偏向 manipulation
- 当 reaching 误差较小时，总奖励更偏向 locomotion 稳定性

##### Manipulation 奖励

manipulation 分支形式为：

```text
mani_total =
    mani_regularization * (
        1
      + enhanced_position_tracking
      + raw_position_tracking * enhanced_orientation_tracking
    )
  + potential_reward
  - cumulative_error_penalty
```

当前 manipulation regularization 包含：

- 支撑期间的 `support_roll` 稳定性
- 支撑足滑动惩罚
- 支撑足离地惩罚
- 非足端接触惩罚
- 随目标高度变化的 pitch regularization
- base 最低高度 regularization
- 机械臂姿态偏离默认位形惩罚
- 机械臂关节接近 joint limit 的安全惩罚
- 左右支撑对称性
- 足端在 XY 平面中的支撑范围 regularization

代表性的 rough 权重包括：

- `support_feet_slide_weight = 0.10`
- `support_foot_air_weight = 0.16`
- `support_non_foot_contact_weight = 0.20`
- `target_height_pitch_weight = 0.08`
- `min_base_height_weight = 0.10`
- `posture_deviation_weight = 0.035`
- `joint_limit_safety_weight = 0.10`

##### Locomotion 奖励

locomotion 分支形式为：

```text
loco_total =
    loco_regularization * (1 + locomotion_tracking)
  - moving_arm_default_deviation_penalty
  - moving_arm_joint_velocity_penalty
```

这个分支不是由一个独立的 base-velocity command 驱动的，而是通过 EE tracking error 与 reaching 过程耦合。

当前 locomotion regularization 包含：

- base 高度
- base 的 roll / pitch
- base 的 roll / pitch 角速度
- base 的竖直速度
- base 的侧向漂移
- 腿部姿态偏离
- touchdown 左右对称性
- touchdown 足端 y 向间距
- 对角足对称性
- soft trot 接触 regularization

代表性的 rough 权重包括：

- `base_height_weight = 0.08`
- `base_roll_weight = 0.16`
- `base_pitch_weight = 0.14`
- `base_lateral_vel_weight = 0.12`
- `diagonal_foot_symmetry_weight = 0.20`
- `feet_contact_soft_trot_weight = 0.8`
- `loco_arm_swing_weight = 0.15`
- `loco_arm_dynamic_weight = 0.01`

其中 `feet_contact_soft_trot` 非常关键。它不是一个硬编码 gait controller，而是利用 phase、contact、足端高度和足端速度构造出的软 gait-shaping 因子。

##### Basic 奖励

basic 分支是线性叠加的工程稳定项，主要包括：

- 存活奖励
- 非成功终止惩罚
- 碰撞惩罚
- 一阶动作平滑惩罚
- 二阶动作平滑惩罚
- 关节力矩平方惩罚
- 关节功率惩罚

代表性的 rough 配置值包括：

- `basic_is_alive_weight = 0.2`
- `basic_termination_penalty_weight = -2.0`
- `basic_collision_weight = -5.0`
- `basic_action_smoothness_first_weight = -0.005`
- `basic_action_smoothness_second_weight = -0.0015`
- `basic_joint_torque_sq_weight = -1.6e-3`
- `basic_joint_power_weight = -1.32e-2`

此外还额外加入了一个工作空间位置 shaping 项：

- `workspace_position_weight = 0.5`
- `workspace_position_x_min = 0.30`
- `workspace_position_x_max = 0.50`
- `workspace_position_y_weight = 1.0`
- `workspace_position_std = 0.1`

#### 课程设计

`go2arm` 通过 `go2arm_reaching_stages` 实现了一个比较明确的 staged curriculum。

##### Stage 0: `0 ~ 500` iterations

- locomotion-only warmup
- 机械臂 delta action 固定为 `0`
- 前向目标范围保持不变
- `world z = 0.7126649548`
- `pitch = 1.5008926535`
- reset 扰动保持较小

##### Stage 1: `500 ~ 1000`

- 解锁机械臂动作
- 将 `world z` 从默认高度逐步扩展到 `[0.45, 0.75]`
- 将姿态范围扩展到：
  - `roll  [-0.35, 0.35]`
  - `pitch [-0.35, 0.35]`
  - `yaw   [-1.20, 1.20]`

##### Stage 2: `1000 ~ 3000`

- 保持主 forward-reaching 分布
- 引入 secondary 的 low-z 样本和 tertiary 的 high-z 样本
- 初始 low/high 采样概率为 `0.08 / 0.08`
- 后续逐渐增大到 `0.20 / 0.20`
- 随着 low-z 样本增多，会同步放宽 base-height termination

##### Stage 3: `3000+`

- 去掉 secondary 和 tertiary 的混合采样
- 切换到完整的 world-z 范围 `[0.10, 1.10]`
- 进一步增大 reset 扰动
- 使用完整的实用目标姿态范围

这个 curriculum 不只是修改命令空间，还会联动调节：

- reset 时的关节位置和速度扰动
- reset root 的 `x/y/yaw` 扰动
- workspace shaping 强度
- base-height termination 阈值

#### Events 与随机化

当前 events 主要分成 startup、reset 和 interval 三类。

##### Startup event

- `scale_arm_mass_validation`
  - 只在 startup 执行一次
  - 作用对象是 `arm_mount` 和 `link1~link6`
  - 当前主要用于代码注释中提到的验证版质量覆盖

##### Reset-time events

- `randomize_reset_joints`
  - 全部 18 个关节在 reset 时加入扰动
  - rough 配置会先把扰动收紧，后续再由 curriculum 逐步放大

- `randomize_reset_base`
  - 在 reset 时对 base 的 `x/y` 做小范围扰动
  - rough 配置基本把 reset yaw 固定在 0 附近
  - reset 时线速度和角速度保持为 0

- `randomize_apply_external_force_torque_base`
  - 给 `base_link` 施加一个 episode 内持续存在的外 wrench

- `randomize_apply_external_force_torque_ee`
  - 给 `link6` 施加一个 episode 内持续存在的外力

##### Interval event

- `randomize_push_robot`
  - 每 `6~8s` 触发一次
  - 通过直接扰动 root velocity 的方式实现瞬时 push

##### 当前关闭的随机化

完整 domain randomization 还没补完，关闭了以下 randomization：

- `randomize_rigid_body_material`
- `randomize_rigid_body_mass_base`
- `randomize_rigid_body_mass_ee`
- `randomize_com_positions`

观测延时默认为 0 s。

#### 接触建模与终止条件

四个足端被视为唯一合法的支撑 body，其余非足端 body 的接触一律视为非法。

实现上主要包括：

- 一个全局 `contact_forces` sensor，覆盖足端和相关非足端 body（如calf和base）
- 四个专用的 filtered foot contact sensor，分别挂在：
  - `FL_foot_contact`
  - `FR_foot_contact`
  - `RL_foot_contact`
  - `RR_foot_contact`
- 所有非足端 body 都通过 `GO2ARM_NON_FOOT_BODY_REGEX` 归为非法接触集合

这会同时影响 reward 和 termination。

主要 termination 包括：

- `time_out`
- `terrain_out_of_bounds`
- `non_foot_contact_termination`
- `base_orientation_termination`
- `base_height_termination`
- `joint_position_termination`
- `joint_velocity_termination`
- `joint_torque_termination`
- `task_success`

其中一些阈值在rough中覆盖：

- `base_orientation_termination`
  - soft roll/pitch limit `0.30`
  - hard roll/pitch limit `0.40`
  - `3` consecutive steps

- `base_height_termination`
  - soft minimum height `0.29`
  - hard minimum height `0.25`
  - `3` consecutive steps

- `joint_position_termination`
  - soft max violation `0.15`
  - hard max violation `0.30`

- `joint_torque_termination`
  - uses `soft_max_ratio = 0.5`

- `task_success`
  - requires low base linear velocity
  - requires low base angular velocity
  - requires low arm joint velocity
  - requires small EE tracking error
  - requires `3` consecutive successful steps

#### Flat 与 Rough 的区别

`flat` 是从同一套 `go2arm` rough 任务结构继承出来的，主要区别在于 terrain：

- `flat` 会把 terrain 切换成 plane
- rough-terrain progression 被关闭
- 动作、观测、命令、奖励、课程和终止逻辑整体保持一致


### Go2Arm 调整日志

为便于持续记录实现过程中的问题、排查、修改和阶段性结论，新增了一份独立日志：

- `go2arm_tuning_log.md`

记录内容包括：

- 训练或 `play` 中观察到的异常现象
- 对问题来源的假设与排查路径
- 对 reward / observation / command / curriculum / event / termination 等的具体修改
- 修改前后的对比结果
- 暂时无效但值得保留的失败尝试


## Citation

This repository is a modified version of `robot_lab`.
