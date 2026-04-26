# `task` 到配置对象，再到配置内容的复盘 03152113

这份文档是在 [`task_to_cfg_chain_03151855.md`](./task_to_cfg_chain_03151855.md) 基础上的同日增补版本。

它复盘今天已经弄清楚的两层内容：

1. 命令行里的 `task` 字符串，怎么变成 `main(env_cfg, agent_cfg)` 里的两个配置对象
2. 找到配置入口之后，`env_cfg` 和 `agent_cfg` 里面到底各装了什么

这份文档仍然只整理今天已经确认过的理解，不追求把整个项目一次讲完。

---

## 1. 先看今天最重要的总图

当前这条训练链可以先压缩成下面这条主线：

`task 字符串`
-> 在 `unitree_a1/__init__.py` 里通过 `gym.register(...)` 注册
-> 登记好 `env_cfg_entry_point` 和 `rsl_rl_cfg_entry_point`
-> `@hydra_task_config(args_cli.task, args_cli.agent)` 根据这些入口装配配置对象
-> 把 `env_cfg` 和 `agent_cfg` 传给 `main(...)`
-> `main(...)` 再创建环境、创建 runner、启动训练

然后今天新增弄清楚的一层是：

- `env_cfg` 负责定义训练世界
- `agent_cfg` 负责定义训练器怎么学

所以当前最值得记住的一句话是：

`task` 先把你带到配置入口，配置对象再决定环境怎么运行、算法怎么训练。

---

## 2. `task` 字符串在哪里被注册

当前讨论的任务是：

```text
RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0
```

它注册在：

- [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/__init__.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/__init__.py)

其中关键代码是：

```python
gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeA1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
    },
)
```

这段代码做的不是“立刻创建环境”，而是先登记一条映射关系：

- 任务名是 `RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0`
- 环境类入口是 `isaaclab.envs:ManagerBasedRLEnv`
- 环境配置入口是 `UnitreeA1RoughEnvCfg`
- RSL-RL 算法配置入口是 `UnitreeA1RoughPPORunnerCfg`

所以这里的 `task` 本质上是一个查表键，不是配置对象本身。

---

## 3. Gym 注册表是什么

可以先把 Gym 注册表理解成一个全局映射表：

```text
任务名 -> 这个任务应该怎么创建
```

所以 `gym.register(...)` 的作用不是运行训练，而是先把“创建说明书”放进去。

后面如果代码调用：

```python
gym.make("RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0", ...)
```

Gym 就会去注册表里查这个名字，再按登记好的信息创建环境。

这里虽然项目底层依赖的是 Isaac Sim 和 Isaac Lab，但仍然使用 `gymnasium`，是因为：

1. Gymnasium 提供了一套统一的 RL 环境接口
2. Isaac Lab 的环境可以做成 Gym 兼容环境
3. 这样训练脚本就能统一用 `gym.make(...)` 创建环境

所以不是“改用 Gym 替代 Isaac Lab”，而是：

- Isaac Sim / Isaac Lab 负责物理模拟和环境实现
- Gymnasium 负责统一的环境注册和创建接口

---

## 4. `import robot_lab.tasks` 为什么重要

在训练脚本里有一句：

```python
import robot_lab.tasks
```

这句不是可有可无的。

因为：

- [`../source/robot_lab/robot_lab/tasks/__init__.py`](../source/robot_lab/robot_lab/tasks/__init__.py)
  里会调用 `import_packages(__name__, _BLACKLIST_PKGS)`
- 这会递归导入很多任务相关子包
- 子包一旦被导入，它们内部的 `gym.register(...)` 就会执行

所以这一步的真实作用是：

让各个任务文件里的注册代码真的跑起来。

如果这一步没发生，后面的 `gym.make(task_name, ...)` 和 Hydra 查配置入口都找不到对应任务。

其中 `_BLACKLIST_PKGS = ["utils"]` 的意思不是“代表后续所有子包”，而是：

- 如果递归导入时遇到名字叫 `utils` 的包，就跳过
- 它只是一个排除名单，不是包结构说明

---

## 5. `gym.register(...)` 里几个容易混淆的语法

### `id=...`

这里是在传关键字参数，表示任务字符串。

### `entry_point="isaaclab.envs:ManagerBasedRLEnv"`

这里的冒号不是 Python 类型注解。

它只是字符串里的一种约定格式：

```text
模块路径:对象名
```

也就是：

- 模块是 `isaaclab.envs`
- 对象是 `ManagerBasedRLEnv`

### `f"{__name__}.rough_env_cfg:UnitreeA1RoughEnvCfg"`

这里的 `f"..."` 是格式化字符串。

`__name__` 会先被替换成当前模块名，再拼成完整入口字符串。

所以这类写法的本质仍然是：

```text
模块路径:类名
```

---

## 6. `@hydra_task_config(args_cli.task, args_cli.agent)` 到底做了什么

这句代码最容易让人误解成“直接给 `main` 传参数”，但其实不是。

```python
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    ...
```

它更接近下面这层意思：

```python
main = hydra_task_config(args_cli.task, args_cli.agent)(main)
```

所以发生了两件事：

1. 先把 `args_cli.task` 和 `args_cli.agent` 传给 `hydra_task_config(...)`
2. `hydra_task_config(...)` 返回一个装饰器，这个装饰器再接收原始的 `main`

这里的 `task` 和 `agent` 不是直接对应 `main(env_cfg, agent_cfg)` 里的两个形参。

它们的作用是告诉 Hydra：

- 当前该按哪个任务名查配置入口
- 当前该选哪个 agent 配置入口

这里也要特别纠正一个容易出现的误解：

- 不是“不同 `task` / `agent` 会挑选不同的装饰器类型”
- 而是同一个装配逻辑，根据不同的 `task` / `agent` 去加载不同的配置入口

---

## 7. 为什么最后调用 `main()` 时不用手动传 `env_cfg` 和 `agent_cfg`

因为文件最后执行的已经不是“原始的 `main`”，而是“被装饰器处理后的 `main`”。

原始函数写的是：

```python
def main(env_cfg, agent_cfg):
    ...
```

但装饰器处理之后，名字 `main` 已经指向了一个新函数。这个新函数会先：

1. 根据 `task` 去查注册信息
2. 找到 `env_cfg_entry_point`
3. 根据 `agent` 这个入口名去查对应 agent 配置入口
4. 加载出配置对象
5. 再在内部调用原始的 `main(env_cfg, agent_cfg)`

所以不是参数消失了，而是改成由装饰器先准备，再自动传入。

---

## 8. `hydra_args` 为什么没有直接写进装饰器参数里

训练脚本里有这一段：

```python
args_cli, hydra_args = parser.parse_known_args()
...
sys.argv = [sys.argv[0]] + hydra_args
```

这里要区分两类参数：

### `args_cli`

这是 `argparse` 明确认识并接走的参数，例如：

- `--task`
- `--agent`
- `--num_envs`
- `--seed`

这些值会在 Python 代码里直接访问，比如：

```python
args_cli.task
args_cli.agent
```

### `hydra_args`

这是 `argparse` 没吃掉、但希望交给 Hydra 的参数。

它们没有直接写在：

```python
hydra_task_config(args_cli.task, args_cli.agent)
```

里面，而是先被放回：

```python
sys.argv
```

原因是 Hydra 会在后续流程里自己读取当前进程的 `sys.argv`。

所以这里的参数流向不是：

```text
hydra_args -> 作为函数实参传给 hydra_task_config
```

而是：

```text
hydra_args -> 写回 sys.argv -> Hydra 在内部读取
```

---

## 9. `sys.argv` 为什么 Hydra 能访问

因为 `sys.argv` 不是局部变量，而是当前 Python 进程的全局命令行参数列表。

只要 Hydra 运行在同一个 Python 进程里，它就能读取当前的 `sys.argv`。

所以这里不是“手动把 `sys.argv` 传给 Hydra”，而是：

1. 训练脚本先改写当前进程的 `sys.argv`
2. Hydra 后面在同一进程中读取它
3. 因而拿到了 `hydra_args`

---

## 10. `cli_args.py` 在这条链里的位置

`cli_args.py` 目前至少承担三件事：

1. 往 `parser` 里补充 RSL-RL 相关命令行参数
2. 根据任务名，从注册信息里加载默认的 RSL-RL 配置
3. 用命令行参数覆盖一部分 `agent_cfg`

其中要特别记住一件事：

```python
add_rsl_rl_args(parser)
```

虽然没有返回新的 `parser`，但它操作的是同一个 `parser` 对象。

这是因为：

- `parser` 是可变对象
- `arg_group = parser.add_argument_group(...)` 也仍然属于这个 `parser`
- 往 `arg_group` 里加参数，本质上还是在修改原来的解析器

所以函数结束后，外面的 `parser` 会直接拥有这些新参数。

---

## 11. 找到入口之后，`env_cfg` 到底是什么

当前任务对应的环境配置入口是：

- [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/rough_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/rough_env_cfg.py)

其中核心类是：

```python
class UnitreeA1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
```

这句最重要的含义不是“它叫这个名字”，而是：

- 它不是从零写一个全新环境
- 它是在通用基类 `LocomotionVelocityRoughEnvCfg` 上做 A1 的具体化

所以你可以把它理解成：

- 基类：通用四足速度跟踪训练模板
- 子类：把模板套到 Unitree A1 身上的具体任务版本

---

## 12. 通用环境基类已经提供了哪些配置组

在：

- [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)

里，基类已经把环境拆成几大组：

- `scene`
  - 场景里有什么，例如地形、机器人、传感器、灯光
- `observations`
  - 策略和 critic 看什么
- `actions`
  - 策略输出什么动作
- `commands`
  - 训练时给机器人什么目标命令
- `rewards`
  - 奖励项菜单
- `terminations`
  - episode 结束条件
- `events`
  - startup / reset / interval 时机的随机化与扰动
- `curriculum`
  - 课程学习项

所以 `UnitreeA1RoughEnvCfg` 的工作不是发明这些组，而是往这些组里填 A1 的具体值。

---

## 13. `UnitreeA1RoughEnvCfg` 主要改了什么

### 1. 机器人命名信息

它先定义了：

- `base_link_name = "base"`
- `foot_link_name = ".*_foot"`
- `joint_names = [...]`

这些值像一个“机器人命名词典”。

后面的观测、动作、奖励、接触检测、随机化，很多都要靠这些名字去指向 A1 的 body 和 joints。

### 2. `scene`

它把场景里的机器人换成 A1 资产，并把高度扫描器挂到 A1 的 base 上。

也就是说：

- 基类只说“这里需要机器人和传感器”
- 子类才说“机器人具体是谁、传感器具体挂哪里”

### 3. `observations`

它做了三类事：

- 调整观测缩放系数
- 禁用某些观测项
- 把关节相关观测限制到 A1 的 12 个关节

例如：

```python
self.observations.policy.base_lin_vel = None
self.observations.policy.height_scan = None
```

这里表示：

- 这些观测项在模板里存在
- 但当前 A1 rough policy 配置选择不用它们

### 4. `actions`

它仍然使用基类里的关节位置动作，但进一步规定：

- 髋关节和其他关节使用不同缩放
- 动作作用于哪 12 个关节

这表示：

- 动作类型没变
- 但动作幅度和作用对象被 A1 子类具体化了

### 5. `events`

它配置了 reset 和随机化细节，例如：

- base 位姿和速度随机范围
- 哪些刚体随机质量
- 哪些 body 改质心
- 外力外矩施加在谁身上

这部分的核心作用不是“制造随机数”，而是做训练期扰动和 domain randomization。

### 6. `rewards`

这里最重要的理解不是背参数，而是：

- 基类提供一份奖励项菜单
- 子类决定哪些奖励启用、权重是多少、参数怎么改

最主要的几类奖励包括：

- 速度跟踪奖励
- 稳定性相关惩罚
- 能耗和动作平滑惩罚
- 接触相关惩罚
- 足端行为相关奖励或惩罚

### 7. `terminations` 与 `curriculum`

这里可以看到：

- 某些终止条件被关闭
- 某些课程学习项被关闭

也就是说，子类不只是“改默认值”，也会直接“禁用整个功能项”。

---

## 14. 把奖励设为 `0` 和设为 `None` 有什么区别

这是今天后面额外弄清楚的一点。

在基类里有：

```python
def disable_zero_weight_rewards(self):
    ...
    if reward_attr.weight == 0:
        setattr(self.rewards, attr, None)
```

所以在这个项目里：

- `weight = 0`
  - 表示奖励项对象还存在，只是当前权重为 0
- `reward = None`
  - 表示奖励项被真正移出当前任务配置

从“总奖励数值”角度看，两者很多时候都会表现成“这项不贡献奖励”。

但从配置语义看，它们不同：

- `0` 更像“保留这项，但当前不生效”
- `None` 更像“当前任务根本不启用这项”

所以 `disable_zero_weight_rewards()` 做的事情，不只是“让结果等于 0”，而是把模板中没启用的奖励项清理掉，让当前任务配置更干净。

---

## 15. 找到入口之后，`agent_cfg` 到底是什么

当前任务对应的 RSL-RL 配置入口是：

- [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/agents/rsl_rl_ppo_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/agents/rsl_rl_ppo_cfg.py)

核心类是：

```python
class UnitreeA1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
```

它不是环境配置，而是训练器配置。

也就是说：

- `env_cfg` 决定机器人在什么任务里学
- `agent_cfg` 决定训练器怎么学

---

## 16. `UnitreeA1RoughPPORunnerCfg` 里主要写了什么

这部分可以分成三层来看。

### 1. runner 级参数

例如：

- `num_steps_per_env = 24`
- `max_iterations = 20000`
- `save_interval = 100`
- `experiment_name = "unitree_a1_rough"`

这些字段更像“训练调度参数”，控制采样多少步、训练多少轮、多久存一次。

### 2. `policy` 级参数

例如：

- `actor_hidden_dims=[512, 256, 128]`
- `critic_hidden_dims=[512, 256, 128]`
- `activation="elu"`
- `init_noise_std=1.0`

这部分定义的是神经网络结构和部分策略输出相关设置。

### 3. `algorithm` 级参数

例如：

- `learning_rate`
- `gamma`
- `lam`
- `clip_param`
- `entropy_coef`
- `num_learning_epochs`
- `num_mini_batches`
- `desired_kl`
- `max_grad_norm`

这部分定义的是 PPO 的更新规则。

所以可以简化记成：

- `runner` 管训练调度
- `policy` 管网络长相
- `algorithm` 管 PPO 怎么更新

---

## 17. 到今天为止，应该建立起来的两组对应关系

### 第一组：入口对应关系

- `task`
  - 查到 `env_cfg_entry_point`
  - 查到 `rsl_rl_cfg_entry_point`

- `env_cfg_entry_point`
  - 最终落到 `UnitreeA1RoughEnvCfg`

- `rsl_rl_cfg_entry_point`
  - 最终落到 `UnitreeA1RoughPPORunnerCfg`

### 第二组：对象职责对应关系

- `env_cfg`
  - 回答“训练世界长什么样”

- `agent_cfg`
  - 回答“训练器怎么训练”

这两组关系一旦混在一起，就容易把“任务定义”和“算法定义”看乱。

---

## 18. 当前阶段最值得背下来的四句话

### 第一句

`task` 字符串不是配置对象，它只是查表入口。

### 第二句

`gym.register(...)` 先把任务名和配置入口登记到 Gym 注册表里，真正创建环境是在后面的 `gym.make(...)`。

### 第三句

`@hydra_task_config(args_cli.task, args_cli.agent)` 不是直接给 `main` 传参，而是先根据任务名和 agent 入口装配好配置对象，再把它们传给 `main(env_cfg, agent_cfg)`。

### 第四句

`env_cfg` 管训练世界，`agent_cfg` 管训练器行为；前者不是算法配置，后者也不是环境配置。

---

## 19. 下一步最应该做什么

今天到这里为止，已经完成了两关：

1. 找到配置入口在哪
2. 看清配置对象里面主要装了什么

所以下一步不要继续横向扫更多任务文件，而应该进入下一关：

**看这些配置对象在运行时是怎么被环境和 runner 消费的。**

更具体地说，下一步最适合做的是：

1. 顺着 `ManagerBasedRLEnv` 去看
   - `scene / observations / actions / rewards / events / terminations / curriculum` 最终是谁读取的
   - 为什么叫 `ManagerBasedRLEnv`

2. 顺着 `train.py` 里的这两句去看
   - `env = gym.make(args_cli.task, cfg=env_cfg, ...)`
   - `runner = OnPolicyRunner(env, agent_cfg.to_dict(), ...)`

3. 重点回答两个问题
   - `ObsTerm / RewTerm / EventTerm` 最终由谁在运行时执行
   - `policy / algorithm / runner` 配置最终在 runner 的哪一层生效

也就是说，下一步应该从“静态配置”进入“运行时消费”。

---

## 20. 下一次继续时，最自然的起点

下次继续时，建议直接从下面这个问题开始：

**`env_cfg` 里的 `ObsTerm / RewTerm / EventTerm` 到底是谁在运行时读取和执行？**

一旦这个问题讲顺，你对 `manager-based` 这套环境组织方式就会真正开始成型。
