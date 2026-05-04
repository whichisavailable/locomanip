# Go2Arm 调整日志

本文档用于记录 `go2arm` 任务在实现过程中的问题、分析、修改和结果，方便后续对比和复盘。

## Version 1.0

### 状态

- `go2arm` 首次能够正常进入训练流程。
- 这一版的目标不是训练效果正确，而是让训练入口、配置入口、`play` 入口和旧版 Isaac Lab / `rsl_rl` 接口至少能接通。

### 主要问题（表现）

- `RobotLab-Isaac-Flat-Go2Arm-v0` 环境无法正常注册。
- 缺少依赖，例如 `pxr`。
- `go2arm` 在旧版 Isaac Sim / Isaac Lab 接口下存在兼容性问题：
  - `ObservationsCfg` 未定义。
  - `total_reward` 参数签名不一致。
  - `rsl_rl` 相关接口不一致。
- 必须先区分问题来自环境、reward、observation，还是 policy 入口。

### 解决思路（原因分析）

- 先建立可运行基线，再谈 reward、command 和 curriculum 的效果。
- 尽量只修改 `robot_lab` ，不修改 `IsaacLab` 本体，避免把干净内容污染。

### 具体改动

- 适配环境注册、旧接口兼容层和 `rsl_rl / IsaacLab` 版本桥接。
- 核对训练入口、回放入口和配置入口是否适配。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/*`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/*`
  - `scripts/reinforcement_learning/rsl_rl/play.py`

## Version 1.0 -> 1.1

### 状态

- 在训练和 `play` 基本能跑之后，进入 reward 结构层面的真实修改。
- 主要处理 `mani regularization` 的总形式，让它从负向惩罚转向可解释的正向调制项。

### 主要问题（表现）

- 原来的 `mani regularization` 更像一组负权重惩罚，与原文所表现的效果相悖。
- 日志里看到 reward 数值变化时，无法判断是在反映任务进步，还是只是在反映稳定性。
- 某个 reward 项在总奖励里到底是加法项、门控项还是正则项，日志输出不够清楚。

### 解决思路（原因分析）

- 先把 `mani regularization` 的实现效果从直观惩罚改成门控。
- 这是原论文中所没有注明的一个问题：**正则项必须是0-1之间的**。

### 具体改动

- 将 `mani regularization` 从负惩罚改为指数形式。
- 原典型负权重包括：
  - `support_roll = -0.1`
  - `support_feet_slide = -0.05`
  - `support_foot_air = -0.05`
  - `support_non_foot_contact = -0.10`
  - `posture_deviation = -0.02`
  - `joint_limit_safety = -0.05`
- 总形式从线性负惩罚转向 `0~1` 正则分数。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/rough_env_cfg.py`

## Version 1.1 -> 1.2

### 状态

- `mani regularization` 已经改成指数形式，但子项仍存在量纲不一致问题。
- 把重点放在每个子项归一化上

### 主要问题（表现）

- 某些子项因为原始数值范围大，容易主导总和。
- weight 同时承担“重要性”和“补偿量纲”的作用，导致后续调参不好解释。

### 解决思路（原因分析）

- 每个子项先本地归一化，再统一汇总。
- weight 只表达重要性，不再表达量纲补偿。

### 具体改动

- 引入子项归一化和裁剪：
  - `support_roll` 除以 `0.15^2`
  - `support_feet_slide` 除以 `0.1`
  - `support_foot_air` 裁到 `0~2`
  - `support_non_foot_contact` 的力幅值尺度改到 `30`
  - 总 penalty 裁到 `0~4`
  - `posture_deviation` 除以 `sqrt(6) * 0.4`
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`

## Version 1.2 -> 1.3

### 状态

- reward 结构调整后，需要检查具体 reward 参数是否真的接入总奖励。
- 主要修正 `support_non_foot_contact` 的实现链路。

### 主要问题（表现）

- 配置层已经有 `support_non_foot_contact` 相关参数，但它们是否真的被 `total_reward` 使用还有问题（日志输出一直为0）。
- 训练虽然在跑，但行为没有真正学起来，单看总 reward 已经不足以定位问题。

### 解决思路（原因分析）

- 先排除“配置里改了，实际总奖励没用上”的假调参。
- 后续日志诊断必须建立在 reward 参数真实生效的前提上。

### 具体改动

- 修正 `support_non_foot_contact_penalty` 的参数链路。
- 确认以下参数进入 `total_reward`：
  - `count_weight`
  - `force_weight`
  - `force_scale`
- 当时相关配置包括：
  - `count_weight = 1.0`
  - `force_weight = 0.5` 或 `1.0`
  - `force_scale = 10.0`
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/cus_velocity_env_cfg.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/rough_env_cfg.py`

### 后续复盘
- **复盘发现问题不在于非足接触奖励没有接好，而是力传感器根本没配置上，这个问题在修改很多版之后才被修复**

## Version 1.3 -> 1.4

### 状态

- `mani` 相关 reward 已经可运行，但累计 tracking error 的阶段语义不清。
- 这一版调整 `mani cumulative error` 的门控。

### 主要问题（表现）

- `cumulative tracking error` 更像固定权重项。
- 当前是 `loco` 还是 `mani` 阶段时，这个项的参与强度耦合不清。

### 解决思路（原因分析）

- 如果累计误差在 locomotion 主导阶段过强介入，会把命令误差和步态稳定阶段混在一起。
- 它应只在 manipulation 更相关的阶段明显生效。

### 具体改动

- 将 `mani cumulative error` 改为跟随 manipulation 门控，与 `1 - D` 绑定，不再始终固定强度参与。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
  - 与 `ee_pose` 命令缓存相关的实现

### 后续复盘
- **这一点在原文中确实指出了，但是实现时没注意到，在这一版进行了修复**

## Version 1.4 -> 1.5

### 状态

- `mani` 主 tracking 项仍然过于敏感。
- 这一版重设 position / orientation tracking 的尺度。

### 主要问题（表现）

- 早期配置中：
  - `mani_position_std = sqrt(0.0004) = 0.02`
  - `mani_orientation_std = sqrt(0.01) = 0.1`
- `std` 太小导致 tracking 曲线过陡，误差稍大就导致奖励过于稀疏。

### 解决思路（原因分析）

- 放缓 `r_pos` 和 `r_ori` 的敏感度。
- 让 tracking 曲线能更稳定地表达误差变化，而不是过早饱和。

### 具体改动

- 将位置和姿态 tracking 的归一化尺度统一到 `0.25`。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/rough_env_cfg.py`

## Version 1.5 -> 1.6

### 状态

- `basic reward` 开始被拆成可解释子项。
- 这一版是 basic 项权重和尺度的首轮体系化整理。

### 主要问题（表现）

- 不能只看总 reward 是否增长。
- 需要追问 basic 项是不是把策略稳定在错误行为上。

### 解决思路（原因分析）

- 把 basic 的各个子项单独解释，明确 alive、collision、action smoothness、joint torque / velocity 的作用。
- 先建立一个可对照的基础参数组。

### 具体改动

- 首轮整理 `basic reward` 关键参数：
  - `alive = 2`
  - `collision = -5`
  - `collision threshold = 1`
  - `collision force_scale = 20`
  - `action smoothness 1 = -0.003`
  - `action smoothness 2 = -0.001`
  - `joint velocity weight = 0.001`
  - `joint torque weight = 0.001`
  - `joint torque std = 40`
  - `joint velocity std = 4`
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/rough_env_cfg.py`

## Version 1.6 -> 1.7

### 状态

- basic 项中暴露出关节力矩量级异常。
- 这一版重写 torque / power 相关项。

### 主要问题（表现）

- 训练输出中观察到 `basic` 里的关节力矩项数值能到 `5800`。
- 原写法接近“平方和直接进 reward”，量级容易失控。
- 同时还保留了单独 `joint_velocity` 惩罚和依赖 `std` 的形式。

### 解决思路（原因分析）

- 将“力矩大”和“功率大”拆开解释。
- 避免一个量级异常的项混在 basic 里主导判断。

### 具体改动

- 重写关节力矩 / 功率项：
  - 去掉独立 `joint_velocity` 惩罚。
  - 不再使用 `std` 式或指数式表达。
  - 改成 `sum of squared torques` 和 `norm(torque * velocity)`。
- 移除相应 `std` 参数，并按新语义重命名。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`

### 后续复盘
- **这一版的力矩、功率还没有使用effort limits加权的形式，算是一种初步尝试**

## Version 1.7 -> 1.8

### 状态

- reward 分解开始依赖日志，但训练输出过杂。
- 这一版清理训练日志输出项。

### 主要问题（表现）

- 日志里混有目标位姿明细、`gating_mu / gating_l` 等与当前定位关系不大的项，而且是所有环境的平均，即使不在同一个episode阶段。
- 日志很多，但看不到真正想看的 reward

### 解决思路（原因分析）

- 训练输出应优先服务 reward 诊断。
- 内部固定参数没有必要每次都输出，只是之前debug阶段方便审查。

### 具体改动

- 清理训练输出：
  - 保留各 reward 项日志。
  - 删除目标位姿明细、`gating_mu / gating_l` 等与当前定位关系不大的输出。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/cus_velocity_env_cfg.py`

## Version 1.8 -> 2.0

### 状态

- 从 reward 局部修补转向课程几何基准重做。
- 这一大版本开始重新核对 reset、`play` 初始姿态和课程工作空间是否一致。

### 主要问题（表现）

- `stage1` 命令空间此前主要凭经验估计，没有真实初始末端位姿作为参照。
- `reward` 和日志虽然更可解释，但仍然存在“reward 在涨，行为不一定真的更对”的问题。

### 解决思路（原因分析）

- 如果 `stage1` 的命令基准本身不是从真实初始姿态出发，继续调 reward 不能很好地进行解耦，不确定是loco部分奖励还是mani部分奖励有问题导致最终效果不好
- 需要把 `play` 里看到的起始姿态、训练时实际 reset 的姿态、课程中的近端工作空间对齐。

### 具体改动

- 将课程、命令、奖励和 `play` 观察拉到同一个参照系里。
- `stage1` 命令空间开始依赖真实初始末端位姿，而不是经验估计。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/curriculums.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/*`
  - `scripts/reinforcement_learning/rsl_rl/play.py`

## Version 2.0 -> 2.1

### 状态

- `stage1` 几何基准明确后，继续把 reset、初始末端位姿和命令空间固定下来。
- 这一版是 `stage1` 几何基准的具体落地。

### 主要问题（表现）

- `tracking_error` 不明显下降。
- `episode_length` 很长，环境并未频繁终止。
- `mani_reward` 和 `basic_reward` 看起来正常，但目标能力没有明显提升。

### 解决思路（原因分析）

- 这说明 reward shaping 和真实任务进步之间可能脱节。
- 也可能意味着 `stage1` 太容易，或者策略在错误方向上形成局部最优。
- 先固定 `stage1`，避免课程多因素同时变化。

### 具体改动

- 固定 reset。
- 记录真实初始末端 `ee` 位置和姿态。
- 用真实初始位姿重新设计 `stage1 command` 和 reset 扰动范围。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/curriculums.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/*`

## Version 2.1 -> 2.2

### 状态

- 跟踪误差中的权重没有调整，忽略了量纲带来的数值大小差异，导致跟踪误差不能很好反映真实状态。
- 这一版确定跟踪误差的具体加权。

### 主要问题（表现）

- `tracking_error` 很小，但是实际跟踪效果不佳。
- 奖励看起来正常，但目标能力没有明显提升。

### 解决思路（原因分析）

- 将0.05m的位置跟踪误差和0.1rad的姿态跟踪误差大概放在同一量级


### 具体改动

- 总跟踪误差计算中将姿态跟踪误差的权重调整为0.1，位置跟踪误差权重保持为1
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/cus_velocity_env_cfg.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/rough_env_cfg.py`

## Version 2.2 -> 3.0

### 状态

- 第二次明显大改，核心从课程/几何问题切换到接触语义。
- `3.0` 成为后续步态、非法接触和 termination 分析的统一接触基线。

### 主要问题（表现）

- 原始实现中，**由于 Isaac Sim 版本和 USD 转换限制，足端不是刚体，无法直接挂载接触传感器**
- 足端接触依赖 `calf + offset` 近似实现。
- 这种近似会把策略引向错误行为，例如爬行式前移、贴地拖行、膝关节触地。
- 日志、`play` 和实际行为的接触定义持续对不上。

### 解决思路（原因分析）

- 合法接触应收敛到明确的四足足端语义。
- `calf` 受力不应默认为可接受支撑，而应作为非法接触处理。
- reward、termination、observation 和 contact 诊断必须围绕同一套接触定义。

### 具体改动

- 改为明确使用 URDF，且不合并关节。
- 合法支撑从 `calf + offset` 近似收缩为四个足端。
- `calf` 不再被默认视为合法支撑。
- 重写接触相关 reward、termination 和诊断项的逻辑。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/terminations.py`
  - 接触 sensor 配置相关文件

## Version 3.0 -> 3.1

### 状态

- 接触逻辑修改之后，先修日志、终止和随机化语义的一致性。
- 这一版不是大重构，而是为后续对比实验补基础诊断。

### 主要问题（表现）

- `play`、训练日志和 reward 分解仍可能使用不同版本的指标。
- `task_success` 是否真正触发 termination 需要确认。
- 训练结果play发现机器人行走时机身高度过低，同时接近目标位姿之后**姿态明显不鲁棒，单脚着地，说明学到了某种平衡点，能够得到比较好的奖励，但是没有抵抗一点干扰的能力**


### 解决思路（原因分析）

- 先让日志、回放和 reward 对齐，再做更细的 reward 调参。
- 避免不同地方看到的是不同版本的训练指标。
- 需要增加外力随机化，避免机器人学到这种一点扰动就不稳定了的局部值。
- 机身期望高度从0.3m增加到0.4m。

### 具体改动

- 重新整理 `mani / loco / basic` 的日志输出。
- 检查 `task_success` 是否真正触发终止。
- 调整机身期望高度。
- 小幅调整部分 `mani regularization` 权重。
- 梳理外力随机化的真实语义。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/terminations.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/rough_env_cfg.py`

### 后续复盘

- **这里机身高度由于没有合并关节，因此是以base_link作为标准，相比于合并关节后的base要高 8.9 cm**。后面才注意到这个问题，并把机身参考高度增大到0.42m


## Version 3.1 -> 3.2

### 状态

- 第一次高频 `loco reward` 微调。
- 这一版重点判断哪些门控或压缩在压制 locomotion 行为。

### 主要问题（表现）

- `tracking` 驱动力不足。
- `loco` 中 `trot` 权重存在 `0.3` 次方压缩。
- 动作幅度可能被 clip 限制过紧。

### 解决思路（原因分析）

- 先不改 `loco reward` 主结构，优先调权重和动作幅度，保持跟原文奖励框架一致。
- **这里的loc trackingo奖励可能数值太小，导致四足机器人不能像轮足机器人那样被驱动到主动跟踪，反而更希望站在原地拿到存活奖励**
- 把“reward 数值看起来大”和“真正主导策略更新”区分开

### 具体改动

- 将 tracking 的 `exp` 项加权系数提到配置层。
- 腿部 `action clip` 提到 `0.4`。
- `is_alive` 降到 `0.5`。
- `trot` 的 `0.3` 次方压缩改回 `1.0`。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/rough_env_cfg.py`

## Version 3.2 -> 3.3

### 状态

- 调试重点转到日志可解释性。
- 这一版先让 `trot`、`support_factor` 和 regularization 的关系能被看清。

### 主要问题（表现）

- 日志难以分清显示的是原始值、加权值，还是经过门控后的值。
- `trot`、`support_factor` 和 regularization 之间的关系不直观。

### 解决思路（原因分析）

- 日志应成为 reward 可解释性工具，而不只是结果面板。
- 调试阶段先减少无关输出，明确每个 reward 项的层级。

### 具体改动

- 调试阶段先不输出 `mani` 奖励。
- 将 `soft trot` 输出改成更接近真实实现的形式。
- 区分：
  - `loco_regu_base_raw`
  - `loco_regu`
  - `trot`
- 输出不含 `support_factor` 的版本。
- 清理不再需要的中间缓存项。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/rough_env_cfg.py`

## Version 3.3 -> 3.4

### 状态

- 在不大改命令语义的前提下，局部验证 `loco` 是否被命令分布误导。
- 这一版强调最小改动面。

### 主要问题（表现）

- 需要区分 `loco` 学不会到底来自 reward 还是 command。
- 有些配置只是“看起来能动”，但不代表命令真的合理。

### 解决思路（原因分析）

- 用局部试验验证 command 分布，但避免让实验性改法污染主线。
- 通过 `play` 现象和日志共同判断命令语义是否合理。

### 具体改动

- 试探关闭动作限制、固定姿态命令、拉远 `x` 命令。
- 回退过强的实验性命令改法，只保留 `stage1` 局部修改。
- 将姿态误差在总 tracking error 中的权重接入 `rough`。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/rough_env_cfg.py`

## Version 3.4 -> 3.5

### 状态

- 转向 locomotion regularization 的细调。
- 目标从“能走”转向“走得自然”。

### 主要问题（表现）

- base 稳定项可能把策略锁死成僵硬解，走路姿态过于奇怪，不够自然。
- reward 问题从“要不要加这项”变成“这项会不会压过真正想学的行为”。

### 解决思路（原因分析）

- 降低过强的保守稳定约束。
- 让“自然步态”和“自然构型”进入主体设计。

### 具体改动

- 下调一批偏“压稳定”的项：
  - `base_pitch_weight`
  - `base_pitch_ang_vel_weight`
  - `base_z_vel_weight`
  - `base_height_weight`
- 调小 `loco_tracking_std`。
- 将 `support_factor` 等接口外提到 `rough`。
- 对机械臂动作幅度和默认构型惩罚做关节级区分。
- 在 `loco_regularization` 中加入腿关节偏离默认构型惩罚，并按 `hip / thigh / calf` 分权重。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/rough_env_cfg.py`

## Version 3.5 -> 3.6

### 状态

- 系统清点腿序、观测顺序、对称性和脚端几何约束。
- 这一版主要清理“语义不一致”和“死接口”。

### 主要问题（表现）

- 需要确认底层顺序和逻辑没有错位。
- `play` 动作维度、reward 腿索引、接触传感器身体顺序可能不一致。
- **训练效果发现机器人一只脚踩在中间，一只脚踩在外侧很远，回看奖励发现没有对落足点y轴进行约束**

### 解决思路（原因分析）

- 后续调参不能建立在错位索引上。
- 先确认动作、关节、传感器和 reward 使用的是同一套顺序。

### 具体改动

- 检查 reset 后默认站姿是否对称。
- 核对：
  - `joint` 名称顺序
  - action 输出维度到关节的映射
  - foot contact sensor 顺序
  - `feet slide / air time / non-foot contact` 的腿索引
- 将左右脚 `x` 距离不一致惩罚扩展为 `x+y`。
- 清理遗留的 `max_err` 和相关接口。
- 检查新增奖励项是否真的进入日志。

## Version 3.6 -> 3.7

### 状态

- 对随机化和训练配置边界进行区分，先训练好效果再域随机化。
- 这一版把“学不会能力”和“泛化增强”拆开。

### 主要问题（表现）

- 需要明确当前到底有哪些 domain randomization，哪些是真启用。
- 初始动作噪声和 `init_noise_std` 的影响还不清楚。

### 解决思路（原因分析）

- 优先级应是先调通 curriculum，再补随机化。
- 避免把泛化增强误当成学习能力不足的主因。

### 具体改动

- 梳理 `domain randomization` 项。
- 检查初始动作噪声和 `init_noise_std` 相关逻辑。

## Version 3.7 -> 4.0

### 状态

- 第三次明显大改，主题是重做课程学习语义。
- curriculum 从经验式范围切换到几何和任务语义驱动。

### 主要问题（表现）

- 现有 `stage1 / stage2 / stage3` 数值并不可信，很多只是临时写入。
- 接触语义统一后，主要矛盾转到课程本身是否在引导正确的任务序列。

### 解决思路（原因分析）

- 课程设计必须围绕 reset 默认姿态、机械臂工作空间、末端与机身距离和避撞风险来定义。
- 先学近距离 `mani`，再逐步加入高难 `z`，最后扩大 `xy` 范围。

### 具体改动

- 重做三阶段课程：
  - 第一阶段先学近距离 `mani`。
  - 第二阶段逐步加入需要 `pitch` 配合的高难 `z` 命令。
  - 第三阶段再扩大 `xy` 范围，进入“走过去再操作”。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/curriculums.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/rough_env_cfg.py`

## Version 4.0 -> 4.1

### 状态

- 在课程主框架之上补几何细节和采样参考系。
- 这一版继续把课程从“调范围”推进到“调几何语义”。

### 主要问题（表现）

- 高难 `z` 不仅有低位，还有高位。
- 低位任务需要覆盖地面物体（期望）。
- 当前命令空间中还存在断裂和空白区域，并没有覆盖满。

### 解决思路（原因分析）

- 高难 `z` 应拆成不同任务类别，而不是只用一个连续范围表示，**逐步增加课程中的高难度任务比例，驱动机器人学会利用机身pitch帮助manipulation**。
- `xy` 和 `z` 对参考系的需求不同，需要开始拆开处理，**xy应该在机身系下，便于排除明显不合理的命令，z应该在世界系下**。

### 具体改动

- 将高难 `z` 明确定义为低位贴地和高位上仰两类。
- 开始形成 `xy` 用 base-frame、`z` 用 world-z 的混合采样思路。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/curriculums.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/rough_env_cfg.py`

## Version 4.1 -> 4.2

### 状态

- 课程切换从粗略阶段标签收敛到 iteration 语义。
- 这一版加细阶段内部调度。

### 主要问题（表现）

- 按总 step 理解课程不直观，也不方便与实验迭代对应。
- 原 `stage2` 过于粗糙。

### 解决思路（原因分析）

- 用 iteration 作为主要调度轴。
- 阶段内部也应有渐进结构，而不是三段硬开关。

### 具体改动

- `stage1` 缩短。
- `stage2` 拆成更细的几个子过程：
  - 保持低比例高难命令。
  - 扩大 `z` 上下极限。
  - 增大高难命令比例。
- 保留最终范围适应期，不等插值跑到终点才开始学最难部分。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/curriculums.py`

## Version 4.2 -> 4.3

### 状态

- 现实校准阶段：本地代码版本与真实训练版本并不同步。
- 这一版重点不是改 reward，而是明确实验记录口径。

### 主要问题（表现）

- 同样的代码理解对应不上真实训练结果。
- 实际训练版本与本地版本在 `num_envs` 和课程切换 iteration 上不一致。

### 解决思路（原因分析）

- 之后的记录必须区分代码版本、训练环境规模和实际课程阈值。
- 否则同一个结论可能对应不同训练条件。

### 具体改动

- 后续记录明确区分：
  - `num_envs`
  - curriculum 阈值
  - 实际训练命令范围

## Version 4.3 -> 4.4

### 状态

- 异常行为分析阶段。
- 后续 `pitch` 校准、终止阈值和 `support_factor` 重写都从这里展开。

### 主要问题（表现）

- 近距离任务中机身下伏严重。
- **高位任务也先下伏再伸手**。
- 远 `x` 任务更像跳过去，而不是走过去。
- 出现 `fz = -2m`、`reward pitch = +0.55rad`、`gravity pitch = -0.55rad` 这类符号不一致现象。

### 解决思路（原因分析）

- 同时检查 `pitch` 定义是否一致，以及 `trot/support` 约束是否过粗。
- 将“危险下伏”从模糊现象转成可量化约束。

### 具体改动

- 修正 `target_height_conditioned_pitch_penalty` 的语义。
- 在 `play` 中直接输出 `pitch`。
- 姿态终止阈值设为更严格形式：
  - 软阈值 `0.4`
  - 硬阈值 `0.5`
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/terminations.py`
  - `scripts/reinforcement_learning/rsl_rl/play.py`

## Version 4.4 -> 4.5

### 状态

- `pitch` 校准后，继续细化 `trot` 支撑因子。
- 这一版把支撑从单一开关改成分层偏好。

### 主要问题（表现）

- 不希望用硬惩罚一刀切地打掉跳跃。
- 也不能把坏支撑和可接受支撑混成一类。

### 解决思路（原因分析）

- 允许短暂四脚支撑，但仍让真正对角支撑更占优。
- 将支撑模式分档，而不是二值化。

### 具体改动

- 将 `support_factor` 细化成三档：
  - `diag_double_factor = 1.0`
  - `all_four_factor = 0.9`
  - `bad_support_factor = 0.25`
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/rough_env_cfg.py`

## Version 4.5 -> 4.6

### 状态

- `trot` 支撑因子作用位置效果过于强大，导致策略根本学不会。

### 主要问题（表现）

- 由于trot在四足支撑时相当于直接把loco阶段奖励乘了0.25，再减去机械臂偏移惩罚，就导致loco阶段奖励特别小，这导致机器人偏向于任何时候都对角脚支撑
- 对角脚支撑无法保持稳定，机器人撑几步就会跌倒

### 解决思路（原因分析）

- 去掉`support_factor`，因为其作用形式不太好调，更关注使用soft_trot门控来调节步态

### 具体改动

- 直接去掉 `support_factor`
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/rough_env_cfg.py`

## Version 4.6 -> 4.7

### 状态

- 讨论是否改成先学 `loco` 再学 `loco + mani`。
- 这一版形成一个原则：不手动固定 gate 破坏原 soft switch。

### 主要问题（表现）

- 课程效果差，但直接固定 `gate=1` 会破坏论文里的门控机制。

### 解决思路（原因分析）

- 保留现有 soft switch 机制，优先调整命令分布和阶段范围。
- 不用人为固定 gate 来绕过问题。

### 具体改动

- 保留现有三段结构。
- 优先调整：
  - 各阶段 `xyz` 范围
  - `gate` 分布位置
  - near / far 占比

## Version 4.7 -> 4.8

### 状态

- 补日志语义，让课程分析落到 near / far 任务类别上。
- 这一版让 curriculum 诊断更直接。

### 主要问题（表现）

- `x_near` 的日志语义与实际课程观察不一致。
- 只看总平均无法判断 near 和 far 的行为差异。
- **机器人只能学会走到任务正下方抬手，没有机身工作位置的显示指令**。

### 解决思路（原因分析）

- 课程问题必须落到具体命令类别上。
- 日志应显式区分当前命令属于 near 还是 far。
- **在总奖励项中加入一个workspace位置的惩罚**，引导机器人将机身放在ee位置对应的舒适工作区内

### 具体改动

- 日志不再只输出模糊的 `x_near`。
- 改为输出：
  - 当前命令属于近端还是远端。
  - 该类别下的存活时长。
- 增加workspace_position_reward，惩罚机身离末端位置过近

## Version 4.8 -> 4.9

### 状态

- **先mani再loco+mani和先loco再loco+mani都试过了，发现都有一定问题**

### 主要问题（表现）

- 先mani再loco+mani会导致机器人没办法很好地进行移动，倾向于站在原地不动，还是mani的先验太强
- 先loco会导致机器人不会伸手，对于低处任务倾向于压低机身高度，但是这样会导致关节力矩过大，姿态不稳定，容易跌倒，还是loco的先验太强

### 解决思路（原因分析）

- **还是决定先loco再loco+mani，因为对于四足机器人，loco是比mani更难学的，如果先学mani再学loco先验太强了，loco不太好学到位（也是参考了很多其他的论文，大多都是先学loco）**

### 具体改动

- 课程固定为先学loco再学loco+mani

## Version 4.9 -> 4.10

### 状态

- 考虑使用左右对称增强，一方面便于训练出更对称、自然的姿态，另一方面加速训练（减小collection时长影响）

### 主要问题（表现）

- 左右对称增强没有实现

### 解决思路（原因分析）

- 参考对称增强的实现，将左右脚、关节、传感器等成对元素进行交换，保持其他不变，增强数据量
- 将逻辑与命令等对影响，不要没有改命令还继续增强，那就出现了逻辑错误。

### 具体改动

- 实现左右对称增强，但是没有加入镜像损失

## Version 4.10 -> 4.11

### 状态

- 排查任务异常终止的原因。
- 排查重点从“行为学不会”转向“termination 是否判错、放大错”。

### 主要问题（表现）

- **现在的主要问题不是能不能学会，而是刚开始就死了，平均episode length只有10步**
- `joint_torque_termination` 比例异常高。
- 并行环境规模放大后问题显著恶化。
- torque reward 惩罚没有明显变大，但 torque termination 明显变多。

### 解决思路（原因分析）

- 终止统计或连续违规判定逻辑本身可能有问题。
- 不能只看 reward，输出 `Term/*` 细分指标。

### 具体改动

- 观察重点转向：
  - `joint_torque_termination`
  - `done_count`
  - 持续违规终止逻辑
- 开始结合训练日志和实际回放排查 termination。
- 相关位置：
  - `scripts/reinforcement_learning/rsl_rl/play.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/terminations.py`

## Version 4.11 -> 4.12

### 状态

- 检查连续违规终止实现逻辑是否有误
- 终止逻辑基础实现逻辑本身被纳入排查范围

### 主要问题（表现）

- `_persistent_violation` 语义可疑。
- 如果连续违规逻辑有问题，不止 torque termination 会出问题。

### 解决思路（原因分析）

- 需要先确认连续违规判定是否正确。
- 其他连续终止也可能受同一问题影响。

### 具体改动

- 将 `_persistent_violation` 纳入重点排查。
- 对照各类连续 termination 是否存在同类异常。

### 后续复盘

- 发现连续步数的判断条件确实有问题，**当某一个环境由于连续步数超限终止后，新的环境中的连续步数没有清空**，导致一开始就又终止了。这个问题在4096并行环境中并不明显，但是在16384并行环境中就非常明显了，导致核对日志时才注意到这个问题

## Version 4.12 -> 5.0

### 状态

- 收束`play` 和训练日志的 debug 通道。
- 后续判断开始直接依赖 `play` 和训练日志暴露出的 termination 原因。
- 一次主要大改，这一版本重心已经从调整奖励、课程让策略学会变为了查找导致现在落地就重置的原因

### 主要问题（表现）

- 仅靠训练日志里的汇总指标，难以快速定位具体终止原因。
- 临时 debug 输出过多，影响诊断主线。

### 解决思路（原因分析）

- `play` 应直接输出终止原因，不能靠肉眼看，猜测终止原因（基座高度、关节力矩、非足接触等不好区分）。
- debug 输出需要清理一下，删除已经不需要的输出

### 具体改动

- `play` 终止时直接输出终止原因。
- 清理此前临时添加的大量 debug 输出。
- 继续核对当前终止条件是否过严。
- 相关位置：
  - `scripts/reinforcement_learning/rsl_rl/play.py`
  - 终止统计输出相关实现

## Version 5.0 -> 5.1

### 状态

- 明显的回归排查和小范围回退。
- 这一版开始区分“叠改版”和“单点试验版”。

### 主要问题（表现）

- 叠加多项修改后，策略几乎连站都学不会。
- 某些 reward、termination 和课程参数可能只在局部实现改了，没有进入实际 `rough` 训练配置。

### 解决思路（原因分析）

- 多项同时改动导致回归的风险已经显性化。
- 部分支撑门控增强值得优先回退。

### 具体改动

- 优先回退部分支撑门控增强。
- 检查 reward、termination 和课程参数是否真的接到 `rough`。
- 后续记录开始区分“叠改版”和“单点试验版”。

## Version 5.1 -> 5.2

### 状态

- 开始系统利用 `robot_lab/logs` 回溯是哪一版开始坏掉。
- 版本日志开始具备“回归取证”功能。

### 主要问题（表现）

- `done_count` 和 termination 行为在若干轮实验后开始明显异常。
- `done_count` 降不下来时，无法只靠 reward 判断问题层级。

### 解决思路（原因分析）

- 不能只看参数，还要对照 reward、termination、curriculum、reset 的实现变化。
- reward 正常但 termination 异常时，应优先怀疑终止和统计链路。

### 具体改动

- 对照不同训练 run 的日志。
- 回溯 reward、termination、课程和 reset 的实现变化。
- 继续追踪 `done_count` 与各 termination 曲线。

## Version 5.2 -> 5.3

### 状态

- 重审支撑对称奖励是否真的在发训练信号。
- 这一版识别出“reward 存在但长期不发有效梯度”的问题。

### 主要问题（表现）

- 某些支撑对称项长期为零。
- 不清楚这是设计如此、条件太苛刻，还是该项根本没真正参与训练。

### 解决思路（原因分析）

- 如果必须四脚全接触才给信号，这类 reward 很可能长期不起作用。
- 需要把硬条件改为更连续的 soft gate。

### 具体改动

- 将“全接触才有信号”的硬条件改成接触比例 soft gate。
- 重新核对 symmetry augmentation 是否真的启用和记录。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`

## Version 5.3 -> 5.4

### 状态

- 将 termination、reward、curriculum、reset 放到同一张回归表中对照。
- 记录方式从“改了什么”扩展到“拿哪些 run 做了对照”。

### 主要问题（表现）

- 单看代码 diff 已经不足以解释训练退化。
- 不同 run 的课程阈值、reset、命令范围和 action 限制都可能不同。

### 解决思路（原因分析）

- 用回归表统一记录训练条件和结果。
- 避免把不同训练条件下的现象混成一个结论。

### 具体改动

- 对照不同 run 的：
  - curriculum 阈值
  - reset 抖动
  - `ee_pose` 范围
  - action `delta_clip`
  - support symmetry 权重和 std
  - `done_count` 与各 termination 曲线

### 最终问题定位

- **导致训练完全崩了的原因是之前由于训练效果不好而增大了workspace惩罚的权重，这一项是负的惩罚，数值太大改过了存活奖励，导致机器人倾向于早死，避免一直受到惩罚**
- 因此将workspace也修改为指数形式惩罚，数值大小还是正的

## Version 5.4 -> 5.5

### 状态

- 回到默认站姿几何、机身高度与 termination 语义的匹配问题。
- 几何可行性和终止语义开始一起校准。

### 主要问题（表现）

- 默认站姿可能本身就与当前高度阈值和终止条件错位。
- 前面**核对了很长时间的终止等配置，发现初始版本确实没有现在这一版合理**（按effort limits加权等）

### 解决思路（原因分析）

- 如果默认形态和阈值本身不相容，再怎么训练也会持续撞 termination。
- 将终止、命令、reset等数值通过与urdf数据进行对照，确认其适配当前默认站姿和机身高度

### 具体改动

- 检查默认站姿、机身高度和 termination 阈值是否匹配。

## Version 5.5 -> 5.6

### 状态

- **由于loco阶段固定机械臂，但是后面训练一段时间才开始学操纵，因此直觉上需要在加入mani时重置机械臂动作分布和学习率（当前固定机械臂操作实现仅仅是通过将动作clip到0）**
- 训练器侧的阶段性重置和策略噪声控制进入主线。
- 版本日志不再只记录环境配置，也开始记录训练脚本机制。

### 主要问题（表现）

- 环境之外的 trainer/runtime 逻辑也在影响训练行为。

### 解决思路（原因分析）

- 需要检查 PPO 和训练脚本中的阶段逻辑是否与环境课程一致。

### 具体改动

- 重点关注：
  - `mani phase reset hook`
  - 指定 iteration 的重置逻辑
  - PPO 配置里与 `go2arm` 阶段相关的行为

## Version 5.6 -> 5.7

### 状态

- 检查 teacher / privileged 结构和课程实现是否一致。
- 排查层级从 reward / termination 微调提升到训练架构一致性。

### 主要问题（表现）

- 课程、观测和特权结构可能只是表面接入，没有真正形成一致的训练架构。

### 解决思路（原因分析）

- 需要区分“模块存在”和“模块真的在学”。
- 课程、观测和 privileged encoder 必须共同匹配当前任务设计。

### 具体改动

- 核对：
  - `curriculum` 是否真正按预期接入。
  - `legacy_actor_critic` 与 privileged encoder 是否匹配当前观测设计。
  - `flat` 与 `rough` 的相关结构是否应共用或分开。

## Version 5.7 -> 5.8

### 状态

- 将 `gate_d`、小腿接触、mani 落脚点约束和课程衔接问题拆细。
- 这一版把“门控可能没调好”的直觉落成可复现实验。

### 主要问题（表现）

- 近处任务脚落点太近。
- 远处任务坏姿态终止多。
- 小腿接触没有稳定计入非法接触。
- 所有任务都不伸手。
- 近处任务活得久，但仍然原地踏步。

### 解决思路（原因分析）

- 问题核心很可能在 `loco / mani` 软切换门控 `gate_d` 和偏 `loco` 的前期课程。
- 目标不是推翻论文方案，而是尽量把论文里的 soft switch 机制调顺。

### 具体改动

- 在 `mani regularization` 中加入落脚点 `xy` 范围约束，参考系为 `base-frame`。
- 放松姿态终止到更宽区间，并抬高 `loco` 阶段部分 `roll / pitch` 约束。
- 将小腿加入接触感知。
- 将 `gate_d` sigmoid 参数调整为：
  - `l = 0.45`
  - `mu = 0.65`
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/terminations.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/rough_env_cfg.py`

## Version 5.8 -> 5.9

### 状态

- 继续细化课程衔接。
- 核心是让 `mani` 更早、更明确参与，而不破坏论文主思路。

### 主要问题（表现）

- 纯 `loco` 与纯 `mani` 之间缺少可学习的过渡段。
- 原来的“中远距离先学 loco，近距离再学 mani”过于粗糙。

### 解决思路（原因分析）

- 不优先新增“手臂离 base 越远越好”的新 reward。
- 优先从课程和门控调整入手。
- 不对 `mu` 和 `l` 再做课程化调度。

### 具体改动

- 缩短纯 `loco` 阶段。
- `workspace std` 直接固定，不再单独做课程。
- `x` 从一开始覆盖全距离，不再单独做 `x` curriculum。
- 加入 `loco + micromani` 过渡段。
- 过渡段重点逐步放开 `z` 和姿态范围。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/curriculums.py`

## Version 5.9 -> 6.0

### 状态

- 从门控和课程微调升级为训练目标与阶段切换重组。
- `micromani` 过渡段正式接入课程主线。

### 主要问题（表现）

- 只调局部参数已经不足以处理“纯走”“轻操纵过渡”“明确操纵”的目标链。
- 失败终止没有显式进入总奖励语义。

### 解决思路（原因分析）

- 需要把训练目标链拆得更清楚。
- 总奖励也要能区分成功结束和失败结束。

### 具体改动

- `micromani` 过渡段正式提出并接入课程主线。
- 失败终止开始被显式写进总奖励语义。
- `loco` 阶段不再通过固定 gate 人为隔离。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/curriculums.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/rough_env_cfg.py`

## Version 6.0 -> 6.1

### 状态

- 怀疑是否是urdf导致机器人本身工作空间无法够到地面

### 主要问题（表现）

- 调整了很多版本，但是机器人对于低z任务还是不会伸手。

### 解决思路（原因分析）

- 通过运动学和urdf具体数据计算机械臂是否能够够到地面，以及此时机身高度、倾斜pitch角度是多少

### 具体改动

- 发现当前urdf下确实无法够到地面，甚至x=0.25m（机身系）z=0.1m（世界系）的位置都很难够到
- 在urdf中将肩关节安装位置向前调整了10cm


## Version 6.1 -> 6.2

### 状态

- 非成功终止惩罚加入总奖励。
- 总奖励开始显式区分“成功结束”和“失败结束”。

### 主要问题（表现）

- 仅靠存活长度和局部 reward 增长，无法区分“真的接近成功”和“只是没立刻倒”。

### 解决思路（原因分析）

- 失败终止必须在总奖励里有明确代价。
- 成功结束和失败结束不能只靠 episode length 间接区分。

### 具体改动

- 除成功终止外，其余终止统一加 `-2` 惩罚。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`

## Version 6.2 -> 6.3

### 状态

- 取消 `loco` 阶段固定 gate，并开始关注策略网络结构。
- 排查从任务设计扩展到训练架构问题。

### 主要问题（表现）

- `loco` 阶段存在固定 gate 的思路或实现痕迹。
- 当前最后一层直接同时输出 `leg` 和 `arm` 动作，导致进入 `loco + mani` 后遗忘已学到的 `loco`。
- privileged encoder 结构虽然存在，但需要确认是否真的有梯度流过。

### 解决思路（原因分析）

- soft switch 的语义必须前后一致，不能训练早期一套、后期另一套。
- 如果 reward 和课程都改过但行为仍偏单一，问题也可能出在动作头耦合过强。

### 具体改动

- 取消 `loco` 阶段固定 gate，恢复正常 gate 计算。
- 训练网络结构方向：
  - 前端共享编码。
  - 后端 `leg / arm` 分头输出。
  - 共享部分指 privileged 与普通观测合并后的前几层，而不是 privileged encoder 本身。
- 相关位置：
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
  - curriculum / gate 相关配置
  - trainer / PPO 配置与 actor-critic 结构相关实现
- **这一版同时注意到arm头动作固定为0在ppo训练时带来的错误梯度以及KL散度计算问题，纯loco课程阶段只会考虑leg动作，同时输入的上一帧动作修改为实际作用的动作（arm为0）**，关掉之前设计的重置arm噪声和学习率机制


## Version 6.3版本总结

经过这么多次迭代和调整，发现训练效果还是不够好。近处、远处较高z的任务能够存活很长时间，但是还是存在很大问题
- 1. 工作位置问题。通过使用奖励鼓励的形式虽然可以一定程度上降低走到正下方，但是还是没有明显的动力让机器人把机身放在一个合适的工作位置，导致机器人对于较高的任务还是走的比较近。
- 2. 低z任务实现效果很差。一是机械臂伸手的驱动不够，不太会伸手，二是机器人很难保持存活，容易因各种原因终止。
- 3. 步态问题，虽然加入了对称约束、足端落点约束和腿部关节偏离默认构型约束等，但是毕竟是奖励驱动的，并非强制约束，导致学习出来的足端落点还是比较奇怪。loco阶段还算正常，但是mani的支撑很差。

后续调整思路：
- 毕竟原论文是面向轮足的，四足的机械结构和运动方式与轮足有很大不同，可能通过这种奖励融合的方式没办法很好地学会loco-manipulation任务。
- 命令只包含了末端执行器的位置姿态，并没有对基座的显示命令。想通过这种端到端的形式学习过于困难。
- 后续参考其他针对四足机器人的论文，将奖励融合、增强走位一种奖励塑形的方法。




