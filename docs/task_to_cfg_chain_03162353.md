# `task` 到配置对象，再到 Isaac Lab 运行时消费的复盘 03162353

这份文档是在 [`task_to_cfg_chain_03152113.md`](./task_to_cfg_chain_03152113.md) 基础上的续写版本。

它记录今天新补上的这部分理解：

1. `robot_lab` 里的 `env_cfg` 并不是自说自话，而是会被 Isaac Lab 的 `ManagerBasedRLEnv` 和各类 manager 在运行时消费
2. 学习这套系统时，可以分成三层：
   - 第一层：知道某一行 cfg 属于哪一类
   - 第二层：知道它在 env 生命周期的哪个阶段落地
   - 第三层：知道 manager 是怎么具体消费它的
3. 已经用 reward / event / observation 三类代表项走通了“配置 -> env 时机 -> manager 消费”的链路
4. 顺手澄清了两个训练设计问题：
   - 为什么 observation 要加噪声和 clip
   - 为什么 actor 和 critic 的 observation 可以不同

## 快速跳转

- 上一版复盘：[`task_to_cfg_chain_03152113.md`](./task_to_cfg_chain_03152113.md)
- 学习路径入口：[`learning_path_03142344.md`](./learning_path_03142344.md)
- 训练入口：[`../scripts/reinforcement_learning/rsl_rl/train.py`](../scripts/reinforcement_learning/rsl_rl/train.py)
- 通用环境配置：[`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)
- A1 粗糙地形环境配置：[`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/rough_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/rough_env_cfg.py)
- Isaac Lab 通用环境骨架：[`../../IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py`](../../IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py)
- Isaac Lab RL 环境骨架：[`../../IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`](../../IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)
- Isaac Lab term 配置协议：[`../../IsaacLab/source/isaaclab/isaaclab/managers/manager_term_cfg.py`](../../IsaacLab/source/isaaclab/isaaclab/managers/manager_term_cfg.py)
- Isaac Lab reward manager：[`../../IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py`](../../IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py)
- Isaac Lab event manager：[`../../IsaacLab/source/isaaclab/isaaclab/managers/event_manager.py`](../../IsaacLab/source/isaaclab/isaaclab/managers/event_manager.py)
- Isaac Lab observation manager：[`../../IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py`](../../IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py)

---

## 1. 先把 `robot_lab` 和 Isaac Lab 的分工再钉牢一次

到今天为止，最重要的结构理解可以压缩成这句话：

- Isaac Lab 提供框架层的 env、manager 和 cfg 协议
- `robot_lab` 负责在这些协议上填写具体任务内容

所以：

- Isaac Lab 不是在替代 `robot_lab`
- `robot_lab` 也不是在重复写一遍 Isaac Lab

两者是上下层关系。

更具体地说：

- Isaac Lab 提供：
  - `ManagerBasedEnv`
  - `ManagerBasedRLEnv`
  - `ObservationManager`
  - `RewardManager`
  - `EventManager`
  - `TerminationManager`
  - `ObservationTermCfg / RewardTermCfg / EventTermCfg / ...`

- `robot_lab` 负责提供：
  - 具体任务名
  - 具体环境配置
  - 具体算法配置
  - 具体奖励项参数、观测项参数、事件项参数

所以 `robot_lab` 里的 cfg，本质上是在“按 Isaac Lab 规定的格式填写任务说明书”。

---

## 2. 现在这套学习可以分成三层

今天把这件事梳理得更清楚了。

### 第一层：归属

看到一行 cfg，先知道它属于哪一类：

- `rewards`
- `events`
- `observations`
- `actions`
- `terminations`
- `curriculum`

这一层回答的是：

“这行配置归哪个模块管？”

### 第二层：在 env 里的落地时机

也就是这类配置在环境生命周期的哪个阶段触发。

这一层主要看：

- `ManagerBasedEnv`
- `ManagerBasedRLEnv`

这一层回答的是：

- 它在 `startup / step / reset / interval` 的哪一段发生
- 它是在 reward 前还是后
- 它是在 reset 前还是 reset 后

### 第三层：在 manager 里的具体消费

这一层主要看各个 manager：

- `RewardManager`
- `EventManager`
- `ObservationManager`
- `TerminationManager`

这一层回答的是：

- manager 具体怎么读 `func`
- 怎么读 `params`
- 怎么处理 `weight / mode / noise / clip / scale`
- 最后实际执行了什么

今天的判断是：

- 你已经知道第一层
- 当前最合适的顺序是先补第二层，再补第三层

---

## 3. 为什么叫 `ManagerBasedRLEnv`

今天通过阅读 Isaac Lab 源码，已经能把这个名字解释清楚。

关键文件是：

- [`../../IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py`](../../IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py)
- [`../../IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`](../../IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)

这里的核心意思不是“有很多配置组”，而是：

- 环境本身不直接手写所有逻辑
- 它把 observation、action、reward、event、termination、curriculum 等职责交给不同 manager
- 自己负责按环境生命周期组织这些 manager 的调用顺序

所以：

- `env` 是总调度者
- `manager` 是分工模块
- `cfg` 是这些模块的输入说明书

---

## 4. 第二层：这些配置在 env 生命周期里的落地顺序

今天重点看了 `ManagerBasedEnv` 和 `ManagerBasedRLEnv` 的运行时顺序。

### 创建阶段

环境创建时，大致顺序是：

1. 校验 cfg
2. 创建 simulation context
3. 创建 `scene`
4. 创建 `event_manager`
5. 如果有 `prestartup` 事件，就先执行
6. 启动模拟器
7. `load_managers()`
8. 初始化 observation buffer 等

也就是说：

- `scene` 最早落地
- `events.prestartup` 比其他 manager 更早

### manager 装配阶段

通用环境层先装：

- `EventManager`
- `RecorderManager`
- `ActionManager`
- `ObservationManager`

RL 环境层再补：

- `CommandManager`
- `TerminationManager`
- `RewardManager`
- `CurriculumManager`

同时，如果有 `startup` 事件，也会在 manager 都准备好后执行一次。

### 每次 `step()` 的顺序

今天把 RL 环境主循环顺序理顺了，顺序是：

1. `action_manager.process_action`
2. 物理循环里 `action_manager.apply_action`
3. 更新 step 计数
4. `termination_manager.compute`
5. `reward_manager.compute`
6. 如有需要，进入 `_reset_idx()`
7. `command_manager.compute`
8. `event_manager.apply(mode="interval")`
9. `observation_manager.compute`

其中最值得记住的几点是：

- `termination` 在 `reward` 之前
- `reward` 是每步都算
- `observation` 在 step 最后统一生成
- `event` 会按 mode 分散在不同生命周期节点触发

### reset 阶段

一旦某些 env 结束，进入 reset 流程时，大致顺序是：

1. `curriculum_manager.compute`
2. `scene.reset`
3. `event_manager.apply(mode="reset")`
4. 各 manager 自己 `reset`
5. 清 episode 长度 buffer

这意味着：

- `curriculum` 主要在 reset 点更新
- `events.reset` 是 reset 生命周期的一部分
- reset 不是单纯“清零”，而是一套完整流程

---

## 5. 第三层代表例子一：reward 是怎么被消费的

今天追的代表项是：

```python
self.rewards.track_lin_vel_xy_exp.weight = 3.0
```

它在：

- [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/rough_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/rough_env_cfg.py)

### 它在做什么

它不是创建新 reward，而是在修改基类里已经存在的 reward term。

基类里这个 term 已经定义好了：

- `func`
- `params`
- 默认 `weight=0.0`

子类把它改成 `3.0`，意思是：

- 当前任务启用这项奖励
- 并赋予较高正权重

### 它在 env 的哪个阶段生效

它在每次 `step()` 的 reward 计算阶段生效。

也就是：

- 先算 termination
- 再算 reward

### manager 怎么消费它

`RewardManager.compute(dt)` 会遍历所有 active reward term，并执行：

```python
value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
```

所以这条 reward 配置最终的实际含义是：

- 调用 reward 函数
- 把环境对象和参数传进去
- 乘权重
- 乘时间步长
- 累加到总 reward

### 顺手澄清：为什么 `compute()` 里只判断 `weight == 0`，不判断 `None`

因为这是两个不同阶段的处理：

- `term_cfg is None`
  - 在 `_prepare_terms()` 阶段直接跳过，不进入 active term 列表
- `weight == 0`
  - 进入 active term 列表，但在 `compute()` 阶段跳过实际计算

所以：

- `None` 更像“这项不参加当前任务”
- `0` 更像“这项保留着，但当前不贡献 reward”

---

## 6. 第三层代表例子二：event 是怎么被消费的

今天追的代表项是：

```python
self.events.randomize_reset_base.params = {...}
```

它在：

- [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/rough_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/rough_env_cfg.py)

### 它在做什么

这也不是创建新 event，而是在修改基类里已有的 `randomize_reset_base` 事件项。

基类里这个 event 已经定义了：

- `func=mdp.reset_root_state_uniform`
- `mode="reset"`
- 默认参数

子类做的是：

- 把 root 重置时的位置、姿态、速度范围改成更适合 A1 的范围

### 它在 env 的哪个阶段生效

因为它的 `mode="reset"`，所以它不是每步都跑，而是在 `_reset_idx()` 阶段触发。

也就是说：

- env 结束
- 进入 reset 流程
- `scene.reset`
- `event_manager.apply(mode="reset", ...)`

### manager 怎么消费它

`EventManager.apply(mode=...)` 会根据 mode 找到对应事件项，并执行对应 `func(env, **params)`。

所以这里最重要的不是某个函数细节，而是形成这种判断：

- `mode="reset"` -> reset 阶段执行
- `mode="interval"` -> step 末尾按时间执行
- `mode="startup"` -> manager 装好后执行一次
- `mode="prestartup"` -> 更早执行一次

---

## 7. 第三层代表例子三：observation 是怎么被消费的

今天追的代表项是：

```python
self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
```

以及：

```python
self.observations.policy.height_scan = None
```

### 它们分别代表两种常见操作

- 第一种：改 observation term 的参数
- 第二种：直接禁用某个 observation term

### 它们在做什么

基类里 `joint_pos` 这个 observation term 默认会对更泛化的一组关节生效。

子类把它改成：

- 只使用 A1 明确列出的 12 个关节
- 并按这组关节顺序组织 observation

而把 `height_scan = None`，则表示：

- 这个 observation term 在模板里存在
- 但当前 A1 `policy` observation 不启用它

### 它在 env 的哪个阶段生效

observation 的主时机是在每次 `step()` 最后统一计算。

之所以放在最后，是因为：

- 要先完成动作施加
- 再完成物理仿真
- 再完成 termination / reward
- 如果需要 reset，就先 reset
- 最后生成“下一轮策略要看到的 observation”

### manager 怎么消费它

`ObservationManager.compute()` 会读取每个 observation term 的：

- `func`
- `params`
- `noise`
- `clip`
- `scale`

然后：

1. 调用 `func(env, **params)`
2. 加噪声
3. clip
4. scale
5. 按 group 组织输出

所以 observation 配置最终决定的不是“说明文档里的字段”，而是：

- actor / critic 最终看到哪些量
- observation 向量的维度
- observation 向量的顺序
- 输入分布的处理方式

---

## 8. 为什么 observation 要加噪声和 clip

今天顺手把这件事也讲清楚了。

### observation 加噪声的主要目的

1. 模拟真实传感器不完美
2. 避免策略过拟合到“过于干净的仿真观测”
3. 提高策略对小扰动的鲁棒性

所以 observation noise 更像一种：

- sim-to-real 手段
- 训练正则化手段

### observation clip 的主要目的

1. 限制偶发异常值
2. 稳定网络输入分布
3. 避免极端 observation 破坏训练稳定性

所以可以把 observation 处理顺序理解成：

- 先算原始 observation
- 再加噪声
- 再 clip
- 再 scale

---

## 9. 为什么 actor 和 critic 的 observation 可以不同

今天也顺手澄清了这一点。

理论教材里最基础的 actor-critic 往往写成：

- actor 用同一个状态
- critic 也用同一个状态

但在工程上，这并不是必须的。

### actor 和 critic 的职责不同

- actor
  - 面向部署
  - 最好只看真实执行时可获得的信息

- critic
  - 面向训练
  - 可以看更多信息，以便更准确地估计 value

这就是常见的 asymmetric actor-critic 设计。

### 这个项目里已经体现了这一点

在 [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py) 里，observation 本来就分成：

- `policy`
- `critic`

而且它们默认还有一个明显差别：

- `policy` 默认启用 corruption
- `critic` 默认不启用 corruption

这说明：

- actor 输入更贴近真实部署条件
- critic 输入更偏向帮助训练稳定

所以“actor 和 critic 的 observation 不同”不是违反理论，而是机器人 RL 里的常见工程设计。

---

## 10. 到今天为止，应该形成的判断标准

如果你现在看到 `env_cfg` 里的某一行配置，应该至少能做到：

### 第一问

它属于哪一类？

### 第二问

它在 env 生命周期的哪个阶段生效？

### 第三问

它最终大概会被哪个 manager 消费？

如果这三问都能答出来，说明你对 `env_cfg` 的“运行时消费”已经不是停留在表面了。

---

## 11. 今天结束时，已经完成到哪一步

到今天为止，这条学习链已经完成了这些阶段：

1. 搞清楚 `task` 是怎么找到 `env_cfg` 和 `agent_cfg` 入口的
2. 搞清楚 `env_cfg` 和 `agent_cfg` 各自负责什么
3. 搞清楚 `env_cfg` 里的配置组在 `ManagerBasedRLEnv` 生命周期里的落点
4. 用 reward / event / observation 三类代表项走通了“配置 -> env 时机 -> manager 消费”链路
5. 搞清楚了 observation 噪声/clip 和 actor-critic 观测差异背后的工程原因

所以当前真正还没继续展开的主线，只剩下：

**`runner_cfg` 在训练器里是怎么被消费的。**

---

## 12. 下一步最应该做什么

下一步已经可以自然切到 `agent_cfg / runner_cfg` 这一侧。

最适合的继续问题是：

**`UnitreeA1RoughPPORunnerCfg` 里的 `runner / policy / algorithm` 三层配置，进入 `OnPolicyRunner` 之后分别在哪里生效？**

换句话说，接下来应该顺着：

- [`../scripts/reinforcement_learning/rsl_rl/train.py`](../scripts/reinforcement_learning/rsl_rl/train.py) 里创建 `OnPolicyRunner(...)`
- `agent_cfg.to_dict()`

这条线继续往下看。

这一步要回答的核心不是“PPO 理论公式”，而是：

- 配置对象是怎么变成训练循环的
- actor / critic 网络是怎么按 cfg 建出来的
- PPO 超参数是怎么被 runner 和 algorithm 读进去的
