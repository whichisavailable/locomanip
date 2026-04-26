# `robot_lab` 学习阶段总复盘 03172114

这份文档不是新增一套体系，而是把目前已经积累下来的几条学习主线整合成一份更适合长期回看的总复盘。

它回答的不是“某一个文件里写了什么”，而是：

1. 到目前为止，已经真正看懂了什么
2. 这些理解之间如何连成一条主线
3. 现在应该把 `robot_lab` 当成什么来学
4. 今天新增的内容到底推进了哪些判断
5. 下一步最值得继续看的是什么

---

## 1. 当前最重要的总判断

到目前为止，最值得固定下来的判断是：

- `robot_lab` 不是一堆零散脚本，而是一套已经跑通的 IsaacLab 训练模板工程
- 你的长期目标不再是“把每个 manager 都追到底”，而是“知道如何在这套模板上改任务、改机器人、改命令、改奖励、接自己的算法”
- 从这个目标看，`robot_lab` 已经足够好，后面大多数工作都应当建立在“复用它，再做局部改造”上，而不是试图重搭整套系统

也就是说，现在最合适的视角已经不是：

- 我要不要把 IsaacLab 内部每个细节都吃透

而是：

- 我已经有了一套可运行主链，后面重点是学会在哪些边界上做修改

这个判断会直接影响后续学习重心：

- 任务与环境层：学会改任务定义
- 算法接入层：学会理解并替换 runner / trainer
- 资产层：学会加自己的机器人

---

## 2. 到目前为止，已经走通的主链

如果把目前已经建立的理解压缩成一条主链，可以写成：

`task 名 -> gym.register(...) -> env_cfg / agent_cfg 入口 -> train.py -> gym.make(...) -> env -> runner / trainer`

其中关键文档是：

- 学习入口总图：[`learning_path_03142344.md`](./learning_path_03142344.md)
- 训练入口主流程：[`train_py_flow_03142344.md`](./train_py_flow_03142344.md)
- `task -> cfg` 主链复盘：[`task_to_cfg_chain_03162353.md`](./task_to_cfg_chain_03162353.md)
- `agent_cfg -> runner` 复盘：[`agent_cfg_to_runner_chain_03170044.md`](./agent_cfg_to_runner_chain_03170044.md)

### 2.1 `task` 是查表入口，不是配置对象本身

当你在命令行里传 `--task=...` 时，系统并不是直接拿到一个现成环境对象。

它先去任务注册表里查：

- 这个 task 对应哪个 `env_cfg_entry_point`
- 这个 task 对应哪个 `rsl_rl_cfg_entry_point`
- 这个 task 对应哪个 `cusrl_cfg_entry_point`

例如 Unitree A1 的任务注册在：

- [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/__init__.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/__init__.py)

这里最重要的不是语法，而是判断：

- task 名只是索引键
- 真正的配置对象要通过 entry point 再去加载

### 2.2 `gym.register(...)` 做的是任务路由，不是训练逻辑

在 A1 注册里：

- `env_cfg_entry_point` 指向环境配置
- `rsl_rl_cfg_entry_point` 指向 RSL-RL 算法配置
- `cusrl_cfg_entry_point` 指向 CusRL 算法配置

这说明一件很关键的事：

- 同一个任务可以挂多套算法入口
- 任务系统和算法系统本来就是解耦的

### 2.3 `hydra_task_config(...)` 的职责是装配配置对象

你已经明确了这一点：

- CLI 负责本次运行输入
- Hydra 负责装配结构化配置对象
- `main(...)` 负责真正创建环境和训练器

### 2.4 `train.py` 是装配入口，不是算法实现本体

对于 RSL-RL 训练来说，`train.py` 的核心职责是：

1. 解析命令行
2. 启动 Isaac Sim
3. 通过 Hydra 得到 `env_cfg` 和 `agent_cfg`
4. 根据本次运行参数覆盖少量配置
5. 调用 `gym.make(...)` 创建环境
6. 把环境包装成训练器需要的接口
7. 创建 runner
8. 调用训练循环

因此：

- `train.py` 决定训练怎么被“接起来”
- 它不负责 PPO 数学细节

---

## 3. 你已经建立起来的 `env_cfg` 理解

你已经不再只是知道 `env_cfg` 里“有很多块”，而是已经形成了以下判断。

### 3.1 `env_cfg` 是任务世界的结构化定义

通用速度任务配置主要看：

- [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)

它不是在“罗列参数”，而是在定义一个完整任务世界：

- `scene`
- `commands`
- `observations`
- `actions`
- `rewards`
- `terminations`
- `events`
- `curriculum`

### 3.2 阅读 `env_cfg` 时，最有效的是三层框架

你已经把这套框架用顺了：

1. 先判断某一行 cfg 属于哪一类
2. 再判断它在 env 生命周期的哪个阶段落地
3. 再判断是哪个 manager 在运行时消费它

这套框架已经用代表性例子走通了三类：

- reward
- event
- observation

### 3.3 `robot_lab` 的 `env_cfg` 是按 IsaacLab 协议填任务说明书

你已经明确：

- IsaacLab 提供 `ManagerBasedRLEnv` 和一系列 managers
- `robot_lab` 是在这些协议上填自己的任务内容

因此：

- 如果只是改“这个任务用哪些项”，优先改 `env_cfg`
- 如果要改“某一项具体怎么计算”，才去改对应实现文件

---

## 4. 今天真正推进出来的新理解

今天的推进，不是又扫了一遍配置，而是把几个你后面一定会频繁用到的判断钉牢了。

这些内容分别是：

1. `mdp` 到底是什么
2. command 和 reward 的分工
3. 新奖励如何增加
4. A1 当前任务的真正目标是什么
5. 新机器人应该从哪里接入
6. `rsl_rl` 在 `robot_lab` 里到底用了哪些东西

---

## 5. `mdp` 的真正含义：先把它当成“任务工具箱”

在速度任务配置里，关键导入在：

- [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)

这一行本质上是在做：

- 把 `robot_lab.tasks.manager_based.locomotion.velocity.mdp` 这个模块起别名叫 `mdp`

因此，在 `velocity_env_cfg.py` 里看到的：

- `mdp.JointPositionActionCfg`
- `mdp.base_lin_vel`
- `mdp.track_lin_vel_xy_exp`
- `mdp.UniformThresholdVelocityCommandCfg`

最直接的意义都是：

- 去 `mdp` 这个模块里拿一个现成函数或配置类来用

### 5.1 为什么这个理解重要

因为它直接帮你区分了两层：

- 配置层：在 `env_cfg` 里决定“用哪个函数、给什么参数、权重是多少”
- 实现层：在 `mdp/*.py` 里决定“这个函数到底怎么计算”

这是一条非常实用的判断线：

- 只改配置组合，去 `rough_env_cfg.py` 或 `velocity_env_cfg.py`
- 要新造观测、奖励、命令逻辑，去 `velocity/mdp/*.py`

### 5.2 `mdp` 是一个聚合模块

它的聚合入口在：

- [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/__init__.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/__init__.py)

它把三类来源汇总到一起：

1. IsaacLab 通用 `mdp`
2. IsaacLab locomotion 任务的 `mdp`
3. `robot_lab` 自己的 `commands.py`、`rewards.py`、`observations.py`、`events.py`、`curriculums.py`

---

## 6. command 和 reward 的分工：目标不由 reward 单独定义

你已经明确了：

- `command` 定义系统要机器人做什么
- `reward` 定义机器人做得像不像这个目标

也就是说：

- 目标不是单纯由奖励项决定的
- reward 不是“凭空定义任务”
- reward 通常是在鼓励策略去完成 command 所表达的目标

### 6.1 A1 当前任务的真正目标是什么

对 A1 速度任务来说，当前默认目标不是：

- 到某个位置
- 走一条固定几何轨迹

而是：

- 跟踪系统给出的底盘速度命令

命令主要定义在：

- [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)

对应的跟踪奖励在：

- [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py)

因此，当前 A1 任务更准确的描述是：

- 训练一个带命令输入的通用速度跟踪策略

### 6.2 为什么基础 command 是一个范围

基础 command 不是固定常数，而是一个范围，其目的不是让速度“毫无规律地乱跳”，而是：

- 训练时从范围里分段采样不同命令
- 让策略学会在一族命令上都能工作

因此更准确的理解是：

- 训练时命令是随机采样的
- 策略学到的是一个更普适的跟踪能力

### 6.3 如果要走 5m 半径圆圈，是不是只改 reward

不是。

今天已经形成了非常实用的判断：

- 如果想让机器人执行一个新目标，第一优先应当考虑改 command
- reward 负责鼓励它跟上这个 command
- 只有当目标是更严格的几何轨迹约束时，才需要再加专门的轨迹 reward

---

## 7. 自定义奖励的最小工作流已经明确了

如果你要定义自己的奖励，流程非常清楚：

1. 先写实现：奖励到底怎么计算
2. 再写配置：要不要启用、权重多少、参数多少

### 7.1 实现层

通常放在：

- [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py)

### 7.2 配置层

通常接在：

- 通用任务里：[`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)
- 或具体机器人任务里：例如 [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/rough_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/rough_env_cfg.py)

因此这条判断已经可以固定下来：

- 奖励怎么算，写在 `mdp/rewards.py`
- 奖励是否启用、权重和参数，写在 `env_cfg`

---

## 8. 新机器人如何接入：现在应当形成的高层地图

当前最应该固定下来的不是某个字段细节，而是分工地图。

### 8.1 机器人“是什么”放在哪

机器人资产相关内容主要放在：

- [`../source/robot_lab/robot_lab/assets`](../source/robot_lab/robot_lab/assets)

例如 Unitree 资产配置在：

- [`../source/robot_lab/robot_lab/assets/unitree.py`](../source/robot_lab/robot_lab/assets/unitree.py)

这里负责的是：

- 资产路径
- 刚体参数
- articulation 参数
- 初始姿态
- 执行器参数

### 8.2 任务如何使用这个机器人

这层通常放在机器人自己的 `rough_env_cfg.py` / `flat_env_cfg.py`。

例如 A1 的 rough 配置在：

- [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/rough_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/rough_env_cfg.py)

这里负责：

- `self.scene.robot = ...`
- 哪些关节参与动作
- 哪些关节参与观测
- reward / events 的局部调整

### 8.3 训练参数放在哪

通常在机器人自己的 agent 配置里：

- [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/agents/rsl_rl_ppo_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/agents/rsl_rl_ppo_cfg.py)

### 8.4 任务怎么变成可训练入口

在对应机器人的注册文件里：

- [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/__init__.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/__init__.py)

### 8.5 现在最值得记住的一句话

新增机器人时，最实用的四层分工是：

- `assets/*.py` 解决“机器人本体是什么”
- `rough_env_cfg.py` 解决“这个任务怎么用这个机器人”
- `agents/*.py` 解决“怎么训练这个任务”
- `__init__.py` 解决“怎么把它注册成可用 task”

---

## 9. URDF 和 meshes：当前阶段最够用的理解

最实用的理解是：

- URDF：机器人的结构说明书
- meshes：机器人的几何外形文件

更具体地说：

- URDF 负责定义 link、joint、惯量、关节限制、几何引用关系
- mesh 负责提供 visual 或 collision 几何

在 `robot_lab` 的训练配置语境里，你通常不会直接在训练脚本中操作 mesh，而是：

1. 资产配置指向 URDF
2. URDF 再去引用 meshes
3. IsaacLab / Isaac Sim 用这些信息生成可模拟机器人

---

## 10. 算法层的判断已经开始转向“接口边界”

你已经开始问：

- 如果参考 `rsl_rl` 的写法与接口，只改里面的算法实现呢

这说明问题已经自然从“怎么跑起来”转到了“怎么控制算法层”。

当前最该固定的结论是：

- `robot_lab` 对 `rsl_rl` 的主要依赖，其实更多在接口和配置协议，而不是 PPO 内核细节本身

### 10.1 当前 `rsl_rl` 训练链依赖哪些东西

主要依赖四类内容：

1. 训练和播放脚本里的 runner 类接口
2. IsaacLab 提供的 `rsl_rl` 配置 schema
3. task 注册里的 `rsl_rl_cfg_entry_point`
4. `play.py` 额外依赖的推理与导出接口

这意味着：

- `robot_lab` 真正绑定的并不是“PPO 公式必须这样写”
- 而是“训练器长成什么接口、配置对象长成什么 schema”

### 10.2 为什么这个判断重要

因为它意味着你未来有两条路线：

#### 路线 A：继续保留 `rsl_rl` 风格接口，只替换算法内核

这种做法的意思是：

- 让新的 runner 继续兼容现有构造方式
- 继续接收类似 `agent_cfg.to_dict()` 的配置
- 继续暴露 `learn()`、`load()`、`get_inference_policy()` 这类方法

#### 路线 B：直接新增自己的一套 trainer 入口

这其实已经被 `CusRL` 这条链路证明可行。

对应文件：

- [`../scripts/reinforcement_learning/cusrl/train.py`](../scripts/reinforcement_learning/cusrl/train.py)

它说明：

- 同一个 `robot_lab` 任务系统
- 完全可以同时对接另一套算法训练器

### 10.3 当前阶段最合适的算法学习深度

现在应当开始学习 `rsl_rl`，但重点不是全仓库精读，而是：

1. `robot_lab/train.py -> RslRlVecEnvWrapper -> OnPolicyRunner(...)`
2. `runner.learn()` 的主循环组织
3. PPO 更新骨架和主要数据流

---

## 11. 今天学习内容对应的“判断升级”

如果把今天新增的理解压缩成几条判断升级，可以写成下面这样。

### 11.1 关于任务目标

旧理解：

- 奖励很重要

新理解：

- 任务目标首先由 command 定义
- reward 主要负责鼓励策略学会完成这个目标

### 11.2 关于改任务

旧理解：

- 改任务大概就是改配置

新理解：

- 改任务时要先区分改 command、改 reward、改 observation、改 action
- 其中 command 决定目标，reward 决定优化偏好

### 11.3 关于 `mdp`

旧理解：

- `mdp` 看起来像个抽象概念

新理解：

- 在这个项目里，先把它当成任务工具箱最实用
- `env_cfg` 负责选用哪个工具
- `mdp/*.py` 负责实现这些工具

### 11.4 关于新增奖励

旧理解：

- 奖励可能就是在配置里加一项

新理解：

- 自定义奖励是“实现层 + 配置层”两步
- 实现写在 `mdp/rewards.py`
- 配置写在 `env_cfg`

### 11.5 关于新增机器人

旧理解：

- 似乎要改很多地方

新理解：

- 可以按资产层、任务适配层、训练参数层、任务注册层四层来拆
- 大部分时候是以现有机器人为模板做局部替换

### 11.6 关于算法改造

旧理解：

- 改算法似乎就得大动整个工程

新理解：

- 任务系统和算法系统本来就有接口边界
- `robot_lab` 对 `rsl_rl` 更强的绑定在接口和配置协议，而不在 PPO 数学本身
- 因此“保留接口、替换算法实现”是现实可行的

---

## 12. 到目前为止，你真正已经掌握的“够用工程图”

如果以后回看，只想记住一张够用工程图，可以直接记下面这张。

### 12.1 任务层

- task 名在注册表里查到 `env_cfg` 和 `agent_cfg`
- 同一个 task 可以挂多套算法入口

### 12.2 环境层

- `env_cfg` 定义任务世界
- `commands` 定义目标
- `observations` 定义策略能看到什么
- `actions` 定义策略能控制什么
- `rewards` 定义优化方向

### 12.3 实现层

- `mdp/*.py` 提供观测、奖励、命令等实现函数和类
- `env_cfg` 负责把这些实现拼成一个任务

### 12.4 资产层

- `assets/*.py` 定义机器人物理与资产参数
- 机器人专属 `rough_env_cfg.py` / `flat_env_cfg.py` 完成任务适配

### 12.5 训练层

- `train.py` 负责装配
- `agent_cfg` 定义训练器参数
- `runner / trainer` 负责真正执行训练循环

### 12.6 扩展层

- 改任务：优先改 `env_cfg`
- 改奖励实现：改 `mdp/rewards.py`
- 加机器人：改 `assets + robot env cfg + 注册`
- 改算法：优先考虑在 runner / trainer 接口边界替换

---

## 13. 现在最适合你的后续学习顺序

### 13.1 第一优先：把 `robot_lab` 当成训练模板来学会“怎么改任务”

这一条后面最值得继续的，不是继续细追 manager 内部，而是：

- 如何修改 command
- 如何修改 observation
- 如何修改 reward
- 如何修改 action
- 如何为自己的机器人建立任务适配

### 13.2 第二优先：开始学习 `rsl_rl` 的接口与主循环

重点不是一次吃透所有细节，而是：

- env 是如何被 wrapper 喂给 runner 的
- `agent_cfg.to_dict()` 进入 runner 后，哪些配置控制训练循环
- PPO 更新循环大概怎样组织

### 13.3 第三优先：形成“如何接自己算法”的最小方案

后续最有价值的问题会逐步变成：

- 如果保留现有 task 和 env，我最少要改哪些地方才能接自己的算法
- 我是继续沿用 `rsl_rl` 风格接口，还是仿照 `cusrl` 单独开入口

---

## 14. 当前阶段最值得记住的几句话

1. `robot_lab` 已经是一套很好的 IsaacLab 训练模板，后面主要应该学会在它基础上改，而不是重搭。
2. task 名只是索引入口，真正训练靠的是注册表找到 `env_cfg` 和 `agent_cfg`。
3. `env_cfg` 定义任务世界，`agent_cfg` 定义训练器行为，二者不要混。
4. `mdp` 在这个项目里先把它当成任务工具箱最实用。
5. command 定义目标，reward 负责鼓励策略学会目标，不要把任务目标全部归结为 reward。
6. 新奖励遵循“先写实现，再写配置”的两层工作流。
7. 新机器人遵循“资产层、任务适配层、训练参数层、注册层”的四层分工。
8. 对你来说，下一步学习 `rsl_rl` 的重点是接口、主循环和改造边界，而不是先死磕所有 PPO 细节。

---

## 15. 今天新增内容的回看入口

如果以后想只回看今天真正新学到的部分，可以优先重看下面这些文件：

- `mdp` 聚合入口：[`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/__init__.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/__init__.py)
- 自定义命令实现：[`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/commands.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/commands.py)
- 自定义奖励实现：[`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py)
- 通用速度任务配置：[`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)
- A1 具体任务适配：[`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/rough_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/rough_env_cfg.py)
- A1 任务注册：[`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/__init__.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/__init__.py)
- A1 的 RSL-RL 配置：[`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/agents/rsl_rl_ppo_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/agents/rsl_rl_ppo_cfg.py)
- RSL-RL 训练入口：[`../scripts/reinforcement_learning/rsl_rl/train.py`](../scripts/reinforcement_learning/rsl_rl/train.py)
- CusRL 训练入口：[`../scripts/reinforcement_learning/cusrl/train.py`](../scripts/reinforcement_learning/cusrl/train.py)

---

## 16. 配套文档导航

这份总复盘建议和下面几份一起看：

- 主路径入口：[`learning_path_03142344.md`](./learning_path_03142344.md)
- 训练入口流程：[`train_py_flow_03142344.md`](./train_py_flow_03142344.md)
- 训练入口术语：[`train_py_terms_03142344.md`](./train_py_terms_03142344.md)
- 文件概览：[`file_overview_03142344.md`](./file_overview_03142344.md)
- `task -> cfg` 主链：[`task_to_cfg_chain_03162353.md`](./task_to_cfg_chain_03162353.md)
- `agent_cfg -> runner` 主链：[`agent_cfg_to_runner_chain_03170044.md`](./agent_cfg_to_runner_chain_03170044.md)
- Python 基础术语补充：[`python_basic_terms.md`](./python_basic_terms.md)
