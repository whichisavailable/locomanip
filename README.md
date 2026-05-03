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

This section documents the current implementation of the `go2arm` task in code, with emphasis on actions, observations, commands, rewards, curriculum, events, and terminations.

Primary implementation files:

- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/rough_env_cfg.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/cus_velocity_env_cfg.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/commands.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/curriculums.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/observations.py`

#### Actions

`go2arm` uses an 18D default-centered delta joint-position action:

```text
target_joint_pos = default_joint_pos + clipped_delta_action
```

Joint order:

```text
FL_hip_joint, FL_thigh_joint, FL_calf_joint,
FR_hip_joint, FR_thigh_joint, FR_calf_joint,
RL_hip_joint, RL_thigh_joint, RL_calf_joint,
RR_hip_joint, RR_thigh_joint, RR_calf_joint,
joint1, joint2, joint3, joint4, joint5, joint6
```

Design points:

- Zero action means staying near the robot default pose, not sending every joint to `0 rad`.
- Clipping is applied to the **delta**, not the final joint target.
- `scale=1.0`, so the policy output is interpreted directly as a joint offset in radians.
- For the first `500` iterations, arm deltas are forcibly fixed to `0.0`, so the early stage is effectively locomotion-only.

The rough config further expands the allowed delta ranges. In particular:

- leg hip joints: about `[-0.83776, 0.83776]`
- front thigh joints: about `[-1.86465, 2.18455]`
- rear thigh joints: about `[-0.81745, 3.23175]`
- calf joints: about `[-1.03421, 0.47375]`
- arm joints use wider task-specific safe ranges instead of a tiny symmetric clip

#### Observations

The task uses three observation groups:

- `policy`: actor-facing core observations
- `privileged`: teacher / privileged observations
- `critic_extra`: extra critic-only observations

##### Core policy observations

The core observation vector is **92D** and is concatenated in this order:

| Term | Dim | Meaning |
| --- | ---: | --- |
| `joint_pos` | 18 | relative joint positions for all leg and arm joints |
| `joint_vel` | 18 | relative joint velocities |
| `ee_current_pose` | 12 | current `link6` pose in `base_link` frame: `3D position + 9D rotation matrix` |
| `actions` | 18 | previously executed effective action |
| `ee_pose_command` | 12 | current end-effector target command |
| `base_lin_vel` | 3 | base linear velocity |
| `base_ang_vel` | 3 | base angular velocity |
| `projected_gravity` | 3 | gravity projected into base frame |
| `reference_tracking_error` | 1 | decaying reference tracking error |
| `gait_phase` | 4 | four trot phase sine signals |

Important details:

- Both the current EE pose and the target EE pose are represented in the base frame.
- The action observation uses the **effective** action after early-stage arm freezing, so observations stay aligned with executed behavior.
- `gait_phase` is generated with `cycle_time = 0.5` and phase offsets `(0.0, 0.5, 0.5, 0.0)`.

##### Privileged observations

These are mainly terrain, contact, disturbance, and external-force signals:

- `base_height`
- `foot_heights`
- `feet_contact_forces` (flattened 12D)
- `static_friction`
- `base_external_wrench`
- `base_external_push_velocity`
- `base_mass_disturbance`
- `ee_mass_disturbance`
- `ee_external_wrench`
- `ee_velocity_b`
- `feet_planar_velocities_w`
- `observation_delay`

##### Critic extra observation

- `cumulative_tracking_error`

#### Commands

The task currently keeps only one command term: `ee_pose`.

This is a fixed end-effector pose command for `link6`, with several important semantics:

- The command is resampled only at reset and then held fixed for the entire episode.
- The command exposed to the policy is 12D: `3D target position + 9D target rotation matrix`.
- Sampling is primarily done in the **base frame** for `x/y` and orientation.
- `z` can be sampled separately in the **world frame**, which makes staged reaching curriculum easier to define around ground height.
- A reject cuboid near the robot body removes obviously bad or unreachable targets.
- If repeated sampling fails, the implementation falls back to the center of the range so the command always remains valid.

The teacher command config starts from a generic base-frame workspace, but `rough_env_cfg.py` overrides it into a staged forward-reaching curriculum:

- `x` fixed to `[0.40, 1.60]`
- `y` fixed to `0.0`
- locomotion warmup keeps world `z` fixed at `0.7126649548`
- locomotion warmup keeps pitch fixed at `1.5008926535`

The command term also maintains internal error state:

- `position_tracking_error`
- `orientation_tracking_error`
- `tracking_error`
- `reference_tracking_error = max(initial_tracking_error - v * t, 0)`
- `cumulative_tracking_error`

These are reused by observations, reward gating, and debug logging.

#### Rewards

The task does not simply sum locomotion and manipulation rewards. It uses a gated total reward:

```text
total_reward =
    (1 - D) * mani_total
  + D * loco_total
  + basic_total
  + workspace_position_reward
```

The gate is:

```text
D = sigmoid((5 / gating_l) * (reference_tracking_error - gating_mu))
```

Current rough-task gate parameters:

- `gating_mu = 0.65`
- `gating_l = 0.45`

Interpretation:

- large reaching error -> reward leans more toward manipulation
- smaller reaching error -> reward leans more toward locomotion stability

##### Manipulation reward

The manipulation branch is:

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

Current rough config emphasizes:

- end-effector position tracking
- end-effector orientation tracking
- potential-based progress
- cumulative tracking error penalty
- support stability and anti-cheating regularization

The manipulation regularization currently includes:

- support roll stability
- support foot slide penalty
- support foot air penalty
- non-foot contact penalty
- target-height-conditioned pitch regularization
- minimum base height regularization
- arm posture deviation penalty
- arm joint-limit safety penalty
- left-right support symmetry
- support foot XY workspace regularization

Representative rough-task weights:

- `support_feet_slide_weight = 0.10`
- `support_foot_air_weight = 0.16`
- `support_non_foot_contact_weight = 0.20`
- `target_height_pitch_weight = 0.08`
- `min_base_height_weight = 0.10`
- `posture_deviation_weight = 0.035`
- `joint_limit_safety_weight = 0.10`

##### Locomotion reward

The locomotion branch is:

```text
loco_total =
    loco_regularization * (1 + locomotion_tracking)
  - moving_arm_default_deviation_penalty
  - moving_arm_joint_velocity_penalty
```

This branch is not driven by a separate base-velocity command. Instead, it is coupled to the reaching process through the EE tracking errors.

The locomotion regularization includes:

- base height
- base roll and pitch
- base roll / pitch angular velocity
- base vertical velocity
- base lateral drift
- leg posture deviation
- touchdown left-right symmetry
- touchdown foot y-distance
- diagonal foot symmetry
- soft trot contact regularization

Representative rough-task weights:

- `base_height_weight = 0.08`
- `base_roll_weight = 0.16`
- `base_pitch_weight = 0.14`
- `base_lateral_vel_weight = 0.12`
- `diagonal_foot_symmetry_weight = 0.20`
- `feet_contact_soft_trot_weight = 0.8`
- `loco_arm_swing_weight = 0.15`
- `loco_arm_dynamic_weight = 0.01`

The `feet_contact_soft_trot` term is especially important. It acts as a soft gait-shaping factor using phase, contact, foot height, and foot velocity, rather than a hard gait controller.

##### Basic reward

The basic branch is a linear engineering-stability term:

- alive reward
- non-success termination penalty
- collision penalty
- first-order action smoothness penalty
- second-order action smoothness penalty
- joint torque squared penalty
- joint power penalty

Representative rough-task values:

- `basic_is_alive_weight = 0.2`
- `basic_termination_penalty_weight = -2.0`
- `basic_collision_weight = -5.0`
- `basic_action_smoothness_first_weight = -0.005`
- `basic_action_smoothness_second_weight = -0.0015`
- `basic_joint_torque_sq_weight = -1.6e-3`
- `basic_joint_power_weight = -1.32e-2`

There is also an extra workspace-position shaping term with:

- `workspace_position_weight = 0.5`
- `workspace_position_x_min = 0.30`
- `workspace_position_x_max = 0.50`
- `workspace_position_y_weight = 1.0`
- `workspace_position_std = 0.1`

#### Curriculum

`go2arm` uses a clear staged curriculum through `go2arm_reaching_stages`.

##### Stage 0: `0 ~ 500` iterations

- locomotion-only warmup
- arm delta action fixed to `0`
- forward target range stays fixed
- `world z = 0.7126649548`
- `pitch = 1.5008926535`
- reset noise kept small

##### Stage 1: `500 ~ 1000`

- unlock arm motion
- expand `world z` from the default height toward `[0.45, 0.75]`
- expand orientation range toward:
  - `roll  [-0.35, 0.35]`
  - `pitch [-0.35, 0.35]`
  - `yaw   [-1.20, 1.20]`

##### Stage 2: `1000 ~ 3000`

- keep the main forward-reaching distribution
- introduce secondary low-z and tertiary high-z samples
- initial low/high sample probabilities are `0.08 / 0.08`
- they later increase toward `0.20 / 0.20`
- base-height termination is relaxed as low-z reaching becomes more common

##### Stage 3: `3000+`

- remove secondary and tertiary mixture sampling
- switch to the full world-z range `[0.10, 1.10]`
- further widen reset perturbations
- use the full practical target orientation range

The curriculum does more than change commands. It also co-schedules:

- reset joint position and velocity noise
- reset root `x/y/yaw` perturbations
- workspace shaping strength
- base-height termination thresholds

#### Events and Randomization

Events are organized into startup, reset, and interval categories.

##### Startup event

- `scale_arm_mass_validation`
  - applied once at startup
  - targets `arm_mount` and `link1~link6`
  - currently acts as the validation-side mass override described in the code comments

##### Reset-time events

- `randomize_reset_joints`
  - all 18 joints receive reset perturbations
  - rough task narrows them to small ranges, then curriculum may widen them later

- `randomize_reset_base`
  - small `x/y` base perturbations
  - rough task keeps reset yaw effectively fixed at zero
  - reset linear and angular velocities remain zero

- `randomize_apply_external_force_torque_base`
  - samples a persistent external wrench on `base_link`

- `randomize_apply_external_force_torque_ee`
  - samples a persistent external force on `link6`

##### Interval event

- `randomize_push_robot`
  - every `6~8s`
  - injects a transient push by directly perturbing root velocity

##### Currently disabled randomization

The following are still disabled in the shared config:

- `randomize_rigid_body_material`
- `randomize_rigid_body_mass_base`
- `randomize_rigid_body_mass_ee`
- `randomize_com_positions`

That matches the current README note that full domain randomization is not finished yet.

#### Contact Modeling and Terminations

An important engineering rule in `go2arm` is that the four feet are treated as the only legal support bodies. All non-foot body contacts are considered illegal.

Implementation details:

- one global `contact_forces` sensor covers feet plus relevant non-foot bodies
- four dedicated filtered foot contact sensors are attached to:
  - `FL_foot_contact`
  - `FR_foot_contact`
  - `RL_foot_contact`
  - `RR_foot_contact`
- all non-foot bodies are grouped by `GO2ARM_NON_FOOT_BODY_REGEX`

This affects both rewards and termination conditions.

Main terminations:

- `time_out`
- `terrain_out_of_bounds`
- `non_foot_contact_termination`
- `base_orientation_termination`
- `base_height_termination`
- `joint_position_termination`
- `joint_velocity_termination`
- `joint_torque_termination`
- `task_success`

Rough-task overrides include:

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

#### Flat vs Rough

`flat` is inherited from the same `go2arm` rough task structure. The main difference is terrain:

- `flat` switches terrain to a plane
- rough-terrain progression is disabled
- the same action, observation, command, reward, curriculum, and termination logic is otherwise retained

So from an MDP-design perspective, the current `flat` and `rough` variants mostly differ in terrain complexity and terrain sensing, not in the core loco-manipulation task definition.

## Citation

This repository is a modified version of `robot_lab`.
