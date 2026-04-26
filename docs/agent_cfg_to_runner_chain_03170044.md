# `agent_cfg` 到 `OnPolicyRunner` 的复盘 03170044

这份文档继续承接 [`task_to_cfg_chain_03162353.md`](./task_to_cfg_chain_03162353.md)。

这一版只做一件事：
顺着 [`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py) 里创建 `OnPolicyRunner(...)` 的位置，继续往下看 `agent_cfg` 这一侧，回答 `UnitreeA1RoughPPORunnerCfg` 里的三层配置在训练时分别落到哪里：

1. `runner`
2. `policy`
3. `algorithm`

---

## 快速跳转

- 上一篇复盘：[`task_to_cfg_chain_03162353.md`](./task_to_cfg_chain_03162353.md)
- 学习路径入口：[`learning_path_03142344.md`](./learning_path_03142344.md)
- `robot_lab` 训练入口：[`../scripts/reinforcement_learning/rsl_rl/train.py`](../scripts/reinforcement_learning/rsl_rl/train.py)
- A1 的 PPO 配置：[`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/agents/rsl_rl_ppo_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/agents/rsl_rl_ppo_cfg.py)
- Isaac Lab 的 RSL-RL 配置协议：[`../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py)
- Isaac Lab 的 RSL-RL 环境包装器：[`../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py)
- Isaac Lab 上游 `rsl_rl/train.py`：[`../../IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py`](../../IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py)
- Isaac Lab 兼容层：[`../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/utils.py`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/utils.py)

---

## 关键定位

- `robot_lab` 训练入口导入任务包：[`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py) 中的 `import robot_lab.tasks`
- `robot_lab` 训练主函数：[`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py)
- `robot_lab` 创建 `OnPolicyRunner(...)`：[`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py)
- `robot_lab` 调用 `runner.learn(...)`：[`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py)
- A1 的 runner 配置类：[`rsl_rl_ppo_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/agents/rsl_rl_ppo_cfg.py)
- A1 的 `policy` 配置定义：[`rsl_rl_ppo_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/agents/rsl_rl_ppo_cfg.py)
- A1 的 `algorithm` 配置定义：[`rsl_rl_ppo_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/agents/rsl_rl_ppo_cfg.py)
- Isaac Lab 的 `RslRlPpoAlgorithmCfg`：[`rl_cfg.py`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py)
- Isaac Lab 的 `RslRlBaseRunnerCfg`：[`rl_cfg.py`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py)
- Isaac Lab 的 `RslRlOnPolicyRunnerCfg`：[`rl_cfg.py`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py)
- Isaac Lab 的 `RslRlPpoActorCriticCfg`：[`rl_cfg.py`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py)
- Isaac Lab 的 `RslRlVecEnvWrapper`：[`vecenv_wrapper.py`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py)
- Isaac Lab 的 `RslRlVecEnvWrapper.reset()`：[`vecenv_wrapper.py`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py)
- Isaac Lab 的 `RslRlVecEnvWrapper.get_observations()`：[`vecenv_wrapper.py`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py)
- Isaac Lab 的 `RslRlVecEnvWrapper.step()`：[`vecenv_wrapper.py`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py)
- Isaac Lab 上游的 `handle_deprecated_rsl_rl_cfg(...)` 调用点：[`../../IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py`](../../IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py)
- Isaac Lab 的兼容函数 `handle_deprecated_rsl_rl_cfg(...)`：[`utils.py`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/utils.py)

---

## 0. 先把两个 `rsl_rl` 层次分开

这里很容易混淆，先钉牢：

### IsaacLab 仓库里有的，是适配层

也就是这些本地能直接点开的文件：

- [`../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py)
- [`../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py)
- [`../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/utils.py`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/utils.py)
- [`../../IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py`](../../IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py)

这一层负责的是：

- 定义 `agent_cfg` 的协议
- 把 Isaac Lab env 包装成 RSL-RL 认识的 `VecEnv`
- 处理不同 `rsl-rl` 版本之间的配置兼容

### `train.py` 最终导入的核心实现，来自外部 `rsl-rl-lib`

例如 [`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py) 里的：

- `from rsl_rl.runners import OnPolicyRunner`
- `from rsl_rl.runners import DistillationRunner`

这说明真正执行 PPO iteration 的核心 runner / algorithm，不是在 `isaaclab_rl/rsl_rl` 这个适配层里定义的，而是在安装到环境里的 `rsl-rl-lib` 包里。

所以后面看链路时要始终分两层：

1. 本地 IsaacLab 层：负责“怎么把配置和 env 接到 rsl-rl 上”
2. 外部 rsl-rl 层：负责“真正怎么跑 PPO 训练循环”

---

## 1. 先给总链路

这一段主链可以先压缩成一句话：

`task` 找到 `agent_cfg`，`train.py` 把它装配完后交给 `OnPolicyRunner`，而 `env` 通过 `RslRlVecEnvWrapper` 把 `observation / reward / done` 喂进训练循环。

如果展开成更接近真实执行顺序的版本，就是：

1. 任务注册阶段，把某个 task 名映射到 `env_cfg` 和 `rsl_rl_cfg_entry_point`
2. Hydra 把 `agent_cfg` 配置对象实例化出来
3. [`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py) 先用命令行参数覆盖 `agent_cfg` 的少数字段
4. `gym.make(...)` 创建 env
5. `RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)` 把 Isaac Lab env 适配成 RSL-RL 认识的 `VecEnv`
6. `OnPolicyRunner(env, agent_cfg.to_dict(), ...)` 接收训练配置字典
7. `runner.learn(num_learning_iterations=agent_cfg.max_iterations, ...)` 进入外层训练循环

这里最重要的理解是：

- `agent_cfg` 是配置对象，不是训练器本身
- `OnPolicyRunner` 才是运行时对象
- `runner.learn(...)` 才是真正开始执行训练循环的地方

---

## 2. 先把 `UnitreeA1RoughPPORunnerCfg` 三层拆开

在 [`rsl_rl_ppo_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/agents/rsl_rl_ppo_cfg.py) 里，A1 的配置长这样：

```python
class UnitreeA1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "unitree_a1_rough"
    policy = RslRlPpoActorCriticCfg(...)
    algorithm = RslRlPpoAlgorithmCfg(...)
```

这几行正好对应三层：

### 第一层：`runner` 层

- `num_steps_per_env`
- `max_iterations`
- `save_interval`
- `experiment_name`
- 以及继承自 [`RslRlBaseRunnerCfg`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py) 的 `device`、`resume`、`load_checkpoint`、`logger` 等

这一层决定的是：
训练循环怎么组织，而不是网络怎么长、PPO 怎么更新。

### 第二层：`policy` 层

- `actor_hidden_dims`
- `critic_hidden_dims`
- `activation`
- `init_noise_std`
- `actor_obs_normalization`
- `critic_obs_normalization`

这一层决定的是：
actor / critic 网络结构，以及策略输出分布的初始形式。

### 第三层：`algorithm` 层

- `clip_param`
- `num_learning_epochs`
- `num_mini_batches`
- `learning_rate`
- `schedule`
- `gamma`
- `lam`
- `entropy_coef`
- `desired_kl`
- `max_grad_norm`
- `value_loss_coef`
- `use_clipped_value_loss`

这一层决定的是：
PPO 收到 rollout 之后，具体怎么做一次更新。

---

## 3. `runner` 层配置落在哪

这部分最适合按“哪一层循环在读它”来理解。

### 3.1 `max_iterations`

这个字段在 `robot_lab` 当前训练脚本里是直接可见的。

在 [`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py) 里：

- 第 133-135 行，命令行 `--max_iterations` 会覆盖配置对象里的 `agent_cfg.max_iterations`
- 第 253 行，`runner.learn(num_learning_iterations=agent_cfg.max_iterations, ...)`

所以：

- 读取者：[`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py)
- 控制层级：最外层训练迭代次数
- 作用：决定整个训练一共跑多少次 `learn` 里的大循环

这点和 `env_cfg` 不一样。
`max_iterations` 不是 env 生命周期里的参数，而是训练器外层循环的参数。

### 3.2 `num_steps_per_env`

这个字段没有在 `robot_lab/train.py` 里被单独读出来。
它是随着 `agent_cfg.to_dict()` 一起传进 `OnPolicyRunner(...)` 的。

从 [`RslRlBaseRunnerCfg`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py) 的定义看，第 239-240 行明确写的是：

- `num_steps_per_env`: 每次 update 前，每个环境要采多少步

所以这项控制的是：

- 读取者：`OnPolicyRunner` 内部
- 控制层级：单次 policy update 之前的 rollout 收集长度
- 更具体地说：不是控制“总共训练多少轮”，而是控制“每一轮先采多少步数据再更新”

这里要特别区分：

- `max_iterations` 决定大循环跑多少次
- `num_steps_per_env` 决定每次大循环里先采多长 rollout

### 3.3 `save_interval`

这个字段也没有在 `robot_lab/train.py` 里单独展开；
它同样是通过 `agent_cfg.to_dict()` 交给 runner。

从 [`RslRlBaseRunnerCfg`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py) 第 284-285 行的定义可以直接知道：

- `save_interval`: 两次保存 checkpoint 之间隔多少个 iteration

所以这项控制的是：

- 读取者：`OnPolicyRunner` 内部
- 控制层级：外层训练循环中的 checkpoint 保存节奏

### 3.4 这一层的核心图

可以把 `runner` 层先记成：

- `max_iterations`：大循环跑多少轮
- `num_steps_per_env`：每轮先采多少步
- `save_interval`：隔多少轮存一次模型

---

## 4. `policy` 层配置落在哪

这部分不要先背“PPO 网络理论”，先记住它在工程里做的事：

`policy` 层负责告诉训练器，actor 和 critic 这两个网络该怎么建。

在 A1 配置里：

```python
policy = RslRlPpoActorCriticCfg(
    init_noise_std=1.0,
    actor_obs_normalization=False,
    critic_obs_normalization=False,
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation="elu",
)
```

可以按三问来拆：

### 第一问：这几行属于哪一类

它们属于“网络结构和策略分布配置”。

### 第二问：它在训练生命周期哪个阶段落地

它不是在 env step 时才生效，而是在 runner 初始化、构建 policy 网络时生效。

### 第三问：最后由谁消费

在 `robot_lab` 当前脚本里，是 `OnPolicyRunner(env, agent_cfg.to_dict(), ...)` 接收到这部分字典。

从 Isaac Lab 上游兼容层 [`utils.py`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/utils.py) 可以进一步看出，对于 `rsl-rl >= 4.0.0`：

- 第 91-103 行：会把旧的 `policy.actor_hidden_dims` 等字段迁移成 `actor` 模型配置
- 第 117-136 行：会把 `policy.critic_hidden_dims` 等字段迁移成 `critic` 模型配置
- 第 192-198 行：对于 `rsl-rl >= 5.0.0`，还会把旧的 `init_noise_std` 等字段迁移成新的 `distribution_cfg`

这说明 `policy` 这一层在语义上实际控制的是：

- actor MLP 的层数和每层宽度
- critic MLP 的层数和每层宽度
- 激活函数
- 策略输出噪声/分布初始化
- actor / critic 输入是否做 observation normalization

### 4.1 `actor_hidden_dims` / `critic_hidden_dims`

这两个最直观。

例如：

- `[512, 256, 128]`

可以直接理解成：

- 输入层之后接 3 层隐藏层
- 宽度依次是 512、256、128

所以它们决定的不是“训练多久”，而是“网络长什么样”。

### 4.2 `activation`

这个字段决定隐藏层之间用什么激活函数。

在 A1 里是：

- `elu`

也就是说 actor / critic 的 MLP 不是纯线性堆叠，而是在每层之间插入 `ELU`。

### 4.3 `init_noise_std`

这个字段和策略输出分布有关。

对连续动作 PPO，actor 通常不是只输出一个确定动作，而是输出一个动作分布。
`init_noise_std` 决定的就是这个分布初始有多“散”。

从学习角度看，这一项的作用是：

- 不是定义网络宽度
- 而是定义策略一开始的探索强度

---

## 5. `algorithm` 层配置落在哪

这部分最容易被一句话概括：

`algorithm` 决定的不是“采样什么”，而是“拿到 rollout 后怎么更新 PPO”。

在 [`RslRlPpoAlgorithmCfg`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py) 第 161-220 行里，这一层的语义已经写得很清楚。

按功能分，可以先这样记：

### 5.1 rollout 进入更新阶段后，要切几遍数据

- `num_learning_epochs`
- `num_mini_batches`

它们决定的是：

- 一轮 rollout 不是只过一次优化器
- 而是可以重复学习多轮，并拆成多个 mini-batch

### 5.2 PPO 的核心约束怎么设

- `clip_param`
- `desired_kl`

它们决定的是：

- 每次更新允许策略变多快
- 是否需要根据 KL 约束调整学习节奏

### 5.3 return / advantage 怎么算

- `gamma`
- `lam`

这两个就是折扣因子和 GAE 相关参数。

也就是说：

- `reward` 不是直接原样拿来做最终更新
- 它会先经过 return / advantage 计算
- 这里的计算规则由 `gamma` 和 `lam` 控制

### 5.4 loss 的配比怎么设

- `value_loss_coef`
- `entropy_coef`
- `use_clipped_value_loss`

这几项决定的是：

- value loss 占多大比重
- entropy bonus 占多大比重
- value 分支要不要也做 clipping

### 5.5 优化器怎么跑

- `learning_rate`
- `schedule`
- `max_grad_norm`
- `optimizer`

它们控制的是：

- 学习率大小
- 学习率是否按策略自适应
- 梯度裁剪阈值
- 用哪种优化器

---

## 6. env 的 `observation / reward / done` 是怎么进入 runner 的

这一段是把 `env_cfg` 主线和 `runner_cfg` 主线真正接起来的关键。

### 6.1 先看 env 是怎么被包装的

在 [`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py) 第 230-235 行：

1. 先 `env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)`
2. 再 `runner = OnPolicyRunner(env, agent_cfg.to_dict(), ...)`

所以 runner 拿到的并不是原始 Isaac Lab env，而是包装后的 `VecEnv`。

### 6.2 `observation` 怎么进去

在 [`vecenv_wrapper.py`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py)：

- 第 143-149 行，`get_observations()` 会调用 `observation_manager.compute()`，把 env 当前 observation 取出来
- 第 138-141 行，`reset()` 返回 `obs_dict`
- 第 156 行，`step()` 也返回新的 `obs_dict`

也就是说，runner 看到的 observation 来源就是：

- Isaac Lab env 内部 manager 已经算好的 observation group
- 然后由 wrapper 转成 RSL-RL 兼容格式

### 6.3 `reward` 怎么进去

同样在 [`vecenv_wrapper.py`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py)：

- 第 156 行，wrapper 直接接住 `env.step(actions)` 返回的 `rew`
- 第 164 行，把它原样作为 step 返回值交给 runner

所以 reward 的链路是：

`RewardManager` 在 env 内部算出 reward
-> `ManagerBasedRLEnv.step(...)` 返回 `rew`
-> `RslRlVecEnvWrapper.step(...)` 转交给 runner

### 6.4 `done` 怎么进去

还是看 [`vecenv_wrapper.py`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py)：

- 第 156 行拿到 `terminated` 和 `truncated`
- 第 158 行把二者合并成 `dones = (terminated | truncated)`

所以 runner 最终看到的 `done`，并不是只看 termination term，而是：

- `terminated`
- `truncated`

两者取并集后的结果。

如果任务不是 finite horizon：

- 第 161-162 行还会额外把 `truncated` 放进 `extras["time_outs"]`

这一步为什么存在？

因为对训练器来说：

- “因为失败/成功而结束”
- “因为时间截断而结束”

在 bootstrapping 处理上往往不能完全等同。

---

## 7. 一个很关键的分叉：`robot_lab` 当前脚本和上游 IsaacLab 不完全一样

这一点必须单独记住。

在 `robot_lab` 当前脚本 [`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py) 里：

- 没有调用 `handle_deprecated_rsl_rl_cfg(...)`

而在上游 [`IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py`](../../IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py) 第 124-125 行：

- 明确先执行了 `agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)`

这会带来一个很实际的理解差异：

### `robot_lab` 当前脚本的直接事实

`agent_cfg.to_dict()` 之前，没有在脚本里显式把旧式 `policy` 迁移成新式 `actor/critic`。

### 上游 IsaacLab 的标准做法

如果 `rsl-rl` 版本较新，就先把：

- `policy.actor_hidden_dims`
- `policy.critic_hidden_dims`
- `policy.init_noise_std`

这些旧字段迁移成新的：

- `actor`
- `critic`
- `distribution_cfg`

### 这意味着什么

对你做长期理解最重要的不是先判断“谁对谁错”，而是先把这两个层次分开：

1. `robot_lab` 当前脚本的实际源码链路
2. Isaac Lab 上游想兼容的新版本 `rsl-rl` 语义

以后如果你发现：

- 本地训练能跑
- 但上游文档在讲 `actor/critic`

不要立刻怀疑自己看错了。
更可能是因为你现在看的 `robot_lab` 脚本，比上游少了一层兼容迁移。

---

## 8. 这一轮最该记住的结论

把今天这条主线压成四句话，就是：

1. `max_iterations` 由 [`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py) 直接读取，控制最外层训练循环跑多少轮。
2. `num_steps_per_env` 和 `save_interval` 作为 runner 层配置随 `agent_cfg.to_dict()` 进入 `OnPolicyRunner`，分别控制每轮 rollout 长度和 checkpoint 节奏。
3. `policy` 层负责定义 actor / critic 网络结构和策略分布初始化，`algorithm` 层负责定义 PPO 更新规则。
4. env 算出的 `observation / reward / done` 先经过 [`RslRlVecEnvWrapper`](../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py) 再进入 runner。

---

## 9. 下一步最自然的继续点

这一篇把“字段属于哪一层、落在哪个阶段、由谁消费”先钉住了。

下一步最值得继续追的，不是再扫更多配置文件，而是继续往训练器内部压一层：

- `OnPolicyRunner` 在一轮 iteration 里到底按什么顺序调用 `env.step(...)`
- rollout buffer 是什么时候写入的
- PPO update 是什么时候开始的
- actor / critic 前向是在“采样阶段”还是“更新阶段”分别怎么被调用的

如果继续写下一篇，最自然的标题应该就是：

- `OnPolicyRunner` 的 iteration 内部循环
