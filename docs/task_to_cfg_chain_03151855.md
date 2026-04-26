# `task` 到配置对象的装配链路 03151855

这份文档专门复盘今天已经弄清楚的一条主线：

命令行里传入的 `task` 字符串，最后是怎么变成 `main(env_cfg, agent_cfg)` 这两个配置对象的。

这份文档不追求面面俱到，只回答当前最关键的问题：

1. `task` 是什么
2. 任务注册是什么意思
3. `gym.register(...)` 到底登记了什么
4. `@hydra_task_config(...)` 为什么能把参数传进 `main`
5. `args_cli` 和 `hydra_args` 分别走哪条路径

---

## 1. 先看最短结论

对当前这条训练链，最应该记住的是：

`task 字符串`
-> 在 `unitree_a1/__init__.py` 里通过 `gym.register(...)` 注册
-> 登记好 `env_cfg_entry_point` 和 `rsl_rl_cfg_entry_point`
-> `@hydra_task_config(args_cli.task, args_cli.agent)` 根据这些入口装配配置对象
-> 把 `env_cfg` 和 `agent_cfg` 传给 `main(...)`
-> `main(...)` 再创建环境、创建 runner、启动训练

所以 `task` 本身不是环境，也不是算法配置。

它更像一个“查表用的名字”。

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

这就是“任务注册”的含义：

把任务名和它对应的创建规则，登记到 Gym 注册表里。

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

## 11. 当前阶段最值得背下来的三句话

### 第一句

`task` 字符串不是配置对象，它只是查表入口。

### 第二句

`gym.register(...)` 先把任务名和配置入口登记到 Gym 注册表里，真正创建环境是在后面的 `gym.make(...)`。

### 第三句

`@hydra_task_config(args_cli.task, args_cli.agent)` 不是直接给 `main` 传参，而是先根据任务名和 agent 入口装配好配置对象，再把它们传给 `main(env_cfg, agent_cfg)`。

---

## 12. 下一步应该接着看什么

如果这份复盘你已经能顺着讲下来，下一步就该继续往里看：

1. [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/rough_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/rough_env_cfg.py)
   - 搞清楚 `UnitreeA1RoughEnvCfg` 到底在基类上改了什么

2. [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)
   - 搞清楚通用环境基类提供了哪些配置组

3. [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/agents/rsl_rl_ppo_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/agents/rsl_rl_ppo_cfg.py)
   - 搞清楚 `agent_cfg` 本体长什么样

这一步的目标不是背参数，而是继续回答：

“装饰器已经把 `env_cfg` 和 `agent_cfg` 交给 `main` 了，那么这两个对象内部到底装了什么？”
