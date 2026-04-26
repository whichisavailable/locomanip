# Robot_Lab Learning Path 03142344

这份文档不是“课程目录”，而是一份面向项目实战的说明书。

目标只有一个：
把 `robot_lab` 这套工程看成一个真实可运行的训练系统，逐步理解它是怎么把“一个四足机器人任务”变成“一次强化学习训练”的。

如果你的最终目标是“以后用自己的算法训练四足机器人”，那最重要的不是先背很多概念，而是先看清楚这套系统到底怎么流动。

---

## 1. 先建立一个总图

你可以先把整个项目理解成四层：

1. 最外层是“启动训练”
2. 中间层是“装配环境和算法配置”
3. 再往里是“定义任务本身”
4. 最里面才是“具体算法如何更新参数”

在 `robot_lab` 里，它们大致对应：

- 启动训练：[`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py)
- 命令行参数补充：[`cli_args.py`](../scripts/reinforcement_learning/rsl_rl/cli_args.py)
- 任务注册：[`__init__.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/__init__.py)
- 环境配置：[`velocity_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)
- 算法配置：[`rsl_rl_ppo_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/agents/rsl_rl_ppo_cfg.py)

一条最核心的主线可以写成：

`命令 -> train.py -> 读取参数 -> 找到任务配置 -> 创建环境 -> 创建 runner -> 开始 learn()`

你后面看到的 Hydra、`gym.register()`、`env_cfg`、`agent_cfg`，都只是为了把这条主线接起来。

---

## 2. 什么叫“命令行脚本入口”

在这个项目里，所谓“命令行脚本入口”，意思其实很朴素：

你在终端里输入一条命令，Python 就从某个脚本文件开始执行，这个脚本文件就是入口。

在这里，训练入口就是 [`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py)。

入口文件通常做三件事：

1. 接收外部输入
2. 准备运行环境
3. 调用真正的主逻辑

在这个项目里，这三件事分别长这样：

1. 接收外部输入
   - 通过 `argparse.ArgumentParser(...)`
   - 例如 `--task`、`--num_envs`、`--seed`

2. 准备运行环境
   - 启动 Isaac Sim
   - 处理日志目录
   - 设置设备、视频录制、分布式训练等

3. 调用主逻辑
   - 通过 `main(...)`
   - 最后走到 `runner.learn(...)`

你可以把入口脚本想成“总控台”。
它本身不负责定义机器人奖励，也不负责 PPO 更新公式；它负责的是把所有零件装起来，然后按下启动键。

---

## 3. 什么叫“命令行参数”

当你在终端里运行训练脚本时，除了脚本本身，还会额外给一些选项。

例如你可能会给：

- 任务名
- 环境数量
- 随机种子
- 是否从 checkpoint 恢复
- 是否录视频

这些额外选项就是命令行参数。

为什么项目要这样设计？

因为训练时经常要改运行条件，但不想每次都手动改代码。于是就把这些“经常变化的外部控制项”放到命令行里。

在 [`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py) 里，你会看到很多 `parser.add_argument(...)`，这就是在声明：

- 我允许用户传哪些参数
- 每个参数的数据类型是什么
- 默认值是什么

例如：

- `--task` 决定训练哪个环境
- `--num_envs` 决定同时并行多少个环境
- `--seed` 决定随机种子
- `--max_iterations` 决定训练迭代轮数

所以命令行参数不是高级概念，它只是“给脚本传开关和选项”的一种方式。

---

## 4. 为什么会有 `cli_args.py`

如果所有参数都写在 `train.py` 一个文件里，也能运行。
那为什么还要单独有一个 [`cli_args.py`](../scripts/reinforcement_learning/rsl_rl/cli_args.py)？

原因是分工。

`train.py` 负责“整个训练怎么启动”。
`cli_args.py` 负责“RSL-RL 相关参数有哪些，以及这些参数怎么写回算法配置”。

所以这个文件更像一个“参数中转站”。

它主要做两件事：

1. 给解析器补充 RSL-RL 参数
   - `add_rsl_rl_args(parser)`

2. 把命令行参数写进 agent 配置对象
   - `update_rsl_rl_cfg(agent_cfg, args_cli)`

这意味着：

- `train.py` 不用自己知道所有 RSL-RL 参数细节
- RSL-RL 训练和播放脚本都可以复用同一套参数逻辑

所以你看到 `cli_args.py` 时，可以把它理解成：
“训练入口和算法配置之间的转接层”。

---

## 5. 什么叫“配置对象”

这个项目里最容易让初学者发懵的一个词，就是“配置”。

很多人一听配置，就以为只是 `.yaml` 文件。
但在 `robot_lab` 里，配置经常是 Python 类实例，也就是“配置对象”。

例如在 [`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py) 的 `main(...)` 里，你会看到两个输入：

- `env_cfg`
- `agent_cfg`

它们的意思分别是：

- `env_cfg`：环境应该长什么样
- `agent_cfg`：训练器应该怎么训练

更具体一点：

- `env_cfg` 里会管场景、机器人、观测、动作、奖励、终止条件、地形、随机化
- `agent_cfg` 里会管网络结构、PPO 超参数、训练轮数、日志名、checkpoint 恢复

为什么不用一大堆普通变量，而要做成配置对象？

因为强化学习工程里参数非常多，而且层级关系很强。
做成对象以后，可以：

- 分类清楚
- 支持继承
- 更容易复用
- 更方便保存到日志里

所以这里的“配置”不是边角料，它本身就是项目组织代码的核心方式。

---

## 6. 什么叫“任务”

在 `robot_lab` 里，“任务”不是一句抽象的话，而是一个可注册、可加载、可训练的环境定义。

比如一个四足速度跟踪任务，背后其实包含很多东西：

- 机器人是谁
- 地形是什么
- 观察到什么
- 动作控制什么
- 奖励鼓励什么
- 什么时候结束

也就是说，一个任务不是单独一条 reward 函数，而是“整套 MDP 设计”。

当你给训练脚本传 `--task` 时，你实际上是在说：

“请帮我找到这整套任务定义，并据此创建环境。”

所以任务名不是一个标签而已，它后面连着的是一整组配置入口。

---

## 7. 什么叫 `gym.register()`

`gym.register()` 的作用，是给一个任务起名字，并告诉系统：

“当用户说要这个任务时，你应该去哪里找它的配置和创建方式。”

在 [`unitree_a1/__init__.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/__init__.py) 里，你会看到类似这样的结构：

- 一个任务 id
- 一个环境入口
- 一个环境配置入口
- 一个算法配置入口

你可以把它理解成一个“任务索引表”。

比如：

- 用户输入任务名
- 系统去注册表里查
- 查到这个任务对应的 `env_cfg_entry_point`
- 查到这个任务对应的 `rsl_rl_cfg_entry_point`
- 再把这两个配置加载出来

所以 `gym.register()` 不是训练逻辑本身，它更像是“任务路由器”。

这一步非常重要，因为你以后要接自己的算法，首先就得搞清楚：
训练脚本到底是怎么从一个字符串任务名，跳到具体配置类的。

---

## 8. 什么叫“环境配置”

环境配置就是对“这个仿真任务长什么样”的完整描述。

在 [`velocity_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py) 里，环境被拆成很多块，这种拆法很值得注意，因为它对应的是强化学习问题最基本的组成。

### `CommandsCfg`

这部分定义“给机器人什么目标”。

在速度任务里，这通常是：

- 目标前向速度
- 目标侧向速度
- 目标转向速度
- 目标朝向

如果没有 command，机器人根本不知道自己该朝哪个方向学。

### `ActionsCfg`

这部分定义“策略输出的动作到底控制什么”。

在四足 locomotion 里，常见的是：

- 输出关节位置偏移
- 输出关节目标位置
- 输出力矩

动作空间不是一件小事，因为它决定了策略能以什么方式影响机器人。

### `ObservationsCfg`

这部分定义“策略能看到什么”。

例如：

- 机体线速度
- 角速度
- 重力方向投影
- 当前命令
- 关节位置
- 关节速度
- 上一步动作
- 地形扫描

强化学习里的策略不会“理解世界”，它只能根据你提供的 observation 做映射。
所以 observation 其实是在定义“策略可用的信息边界”。

### `RewardsCfg`

这部分定义“什么行为值得鼓励，什么行为要被惩罚”。

例如：

- 跟踪目标速度奖励
- 能耗惩罚
- 姿态稳定奖励
- 足端接触相关奖励或惩罚

奖励函数本质上是在表达你的训练意图。
如果奖励设计得不好，策略可能学到你不想要的取巧行为。

### `TerminationsCfg`

这部分定义“什么时候一个 episode 结束”。

例如：

- 摔倒
- 超时
- 某个关键身体部位接触地面

终止条件会直接影响训练分布，所以它不只是“收尾逻辑”。

### `CurriculumCfg`

这部分定义“训练难度怎么逐步变化”。

例如：

- 先从平地开始
- 再逐渐增加粗糙地形
- 或者逐渐增加命令范围

curriculum 的本质是“不要一上来就把最难的问题丢给策略”。

所以当你读环境配置时，不要把它看成很多零碎类；它其实是在定义一个完整任务世界。

---

## 9. 什么叫“算法配置”

算法配置不是任务世界，而是训练器本身的参数。

在 [`rsl_rl_ppo_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/agents/rsl_rl_ppo_cfg.py) 里，你看到的是：

- 神经网络结构
- 学习率
- PPO clip 参数
- batch 相关参数
- 最大训练轮数
- 日志实验名

这类配置回答的是另一类问题：

- 策略网络多大
- 更新步长多大
- 一次采多少数据
- 总共训练多久

所以你需要明确区分两种“改动”：

1. 改环境配置
   - 你是在改任务本身

2. 改算法配置
   - 你是在改训练器如何学习

很多初学者会把它们混在一起，这会导致定位问题非常困难。

---

## 10. Hydra 在这里到底是什么

你不需要先把 Hydra 当成一门独立课程。

在这个项目里，Hydra 最重要的作用只有一句话：
它负责把任务和算法配置装配成 `main(...)` 所需要的对象。

你可以暂时这样理解 [`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py) 里的这一行：

`@hydra_task_config(args_cli.task, args_cli.agent)`

它的意思接近于：

“根据你输入的任务名和算法入口名，把对应配置找出来、实例化，然后传给下面这个 `main(...)`。”

这里你真正要抓住的，不是 Hydra 的全部语法，而是它在项目中的职责边界：

- `argparse` 接收外部输入
- Hydra 装配配置对象
- `main(...)` 使用这些对象创建环境并训练

只要这个边界清楚，后面再学 Hydra 的细节才不会乱。

---

## 11. 为什么 `train.py` 里既有命令行参数，又有 Hydra

这也是一个非常值得解释的点。

你会发现这个项目不是“全都走命令行参数”，也不是“全都走 Hydra 配置”，而是两者混用。

这是因为它们负责的问题不一样。

命令行参数更适合放这些“运行时临时开关”：

- 任务名
- 是否录视频
- 环境数量
- 是否恢复训练
- seed

Hydra / 配置对象更适合放这些“结构化、层级多、长期稳定的参数”：

- 奖励项
- observation 组成
- 动作定义
- 网络结构
- PPO 超参数

所以这不是重复设计，而是分层设计。

你可以把它理解成：

- CLI 控制“这次怎么跑”
- 配置对象定义“这个系统本身是什么”

这也是为什么 `train.py` 里先解析命令行，再把配置加载进来，然后再用命令行去覆盖一部分配置。

---

## 12. `main(...)` 里真正发生了什么

如果你只想抓住训练执行的核心，那就盯着 [`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py) 的 `main(...)`。

它大致在做这些事：

1. 把命令行参数覆盖到配置对象上
2. 设置环境数量、训练轮数、设备、随机种子
3. 创建日志目录
4. 用 `gym.make(...)` 创建环境
5. 必要时包一层视频录制
6. 用 `RslRlVecEnvWrapper` 适配给 RSL-RL
7. 创建 `OnPolicyRunner` 或 `DistillationRunner`
8. 如果需要，加载 checkpoint
9. 把配置保存到日志目录
10. 调用 `runner.learn(...)`

这里最关键的理解是：

`main(...)` 不是在“发明算法”，而是在“搭训练流水线”。

一旦你明白这一点，你以后写自己的算法入口时，心里就会很清楚：
哪些部分要保留，哪些部分只需要替换 runner。

---

## 13. 什么叫“runner”

在这个项目里，runner 可以理解成“真正执行训练循环的对象”。

环境只负责：

- reset
- step
- 返回 observation / reward / done

而 runner 负责：

- 采样
- 前向推理
- 存储轨迹
- 更新策略
- 保存模型
- 记录训练过程

所以 runner 其实就是“算法执行器”。

在 [`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py) 里：

- `OnPolicyRunner` 对应普通 PPO 训练
- `DistillationRunner` 对应蒸馏训练

如果你以后要接自己的算法，你最有可能替换的就是这部分。

这也是为什么我不建议你一开始就把注意力放在 Hydra 上。
因为从“以后自己接算法”的角度看，最核心的分界线其实是：

- 环境怎么创建
- runner 怎么接上去

---

## 14. 现在最值得你盯住的一条样例链路

这个项目里任务很多，但你前期不要横向铺开。

最好的方法是只盯住一个样例，例如：

`RobotLab-Isaac-Velocity-Flat-Unitree-A1-v0`

然后把它沿着下面这条线走一遍：

1. 在训练命令里看到任务名
2. 在 [`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py) 里看到 `args_cli.task`
3. 在 [`unitree_a1/__init__.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/__init__.py) 里看到这个任务是如何注册的
4. 跳到对应的 `flat_env_cfg.py` / `rough_env_cfg.py`
5. 再跳到通用的 [`velocity_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)
6. 再跳到 [`rsl_rl_ppo_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/agents/rsl_rl_ppo_cfg.py)
7. 最后回到 [`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py) 看它如何把配置交给 runner

只要你能把这条链走顺，你对整个项目就已经不是“看热闹”了，而是开始进入工程理解。

---

## 15. 先改哪里，最容易获得直觉

如果你的目标是尽快形成“我能控制训练行为”的感觉，那最适合先动的，不是 PPO 超参数，而是环境配置里的以下几类东西：

### 先改 command

因为 command 决定机器人被要求完成什么。
例如速度范围变了，策略学到的行为范围就会变。

### 再改 reward

因为 reward 是你训练意图的直接表达。
一个奖励项权重变了，策略就可能明显偏向另一种行为。

### 再改 observation

因为它决定策略能看见什么。
如果策略缺关键信息，再怎么调算法都可能没用。

### 再改 action 定义

因为 action 是策略对机器人施加影响的接口。
动作尺度、控制目标不同，训练难度和行为风格都可能变。

这条顺序背后的逻辑是：
先改“任务目标和反馈”，再改“算法学习细节”。

---

## 16. 如果以后你要接自己的算法，真正要替换什么

你的最终目标不是一直用 RSL-RL，而是可能接自己的算法。

那么你需要先明确：不是整个项目都要推翻。

通常保留的部分有：

- 任务注册
- 环境配置
- 环境创建方式
- 日志目录结构
- checkpoint 组织思路

通常需要替换或重写的部分有：

- 你的训练循环
- 你的策略网络
- 你的数据缓存
- 你的更新逻辑
- 你的 runner

所以未来你做自己的算法时，一个合理思路是：

1. 保留任务和环境不动
2. 仿照现有 `rsl_rl/train.py` 写一个新的训练入口
3. 继续复用 `--task` 和环境创建流程
4. 在创建完环境后，换成你自己的 trainer / runner

这时你会发现，前面理解入口脚本、配置对象、任务注册这些内容，不是绕路，而是在为“替换算法而不重写全项目”做准备。

---

## 17. 一条更适合你的阅读顺序

不要按“先学理论再看代码”的方式推进。
更有效的是按“先看训练怎么跑，再问每个零件是什么”的顺序推进。

建议阅读顺序：

1. [`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py)
   - 看主流程，不要求一遍全懂

2. [`cli_args.py`](../scripts/reinforcement_learning/rsl_rl/cli_args.py)
   - 看参数从哪里来，又怎么写回配置

3. [`unitree_a1/__init__.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/__init__.py)
   - 看任务名如何连接到配置入口

4. [`flat_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/flat_env_cfg.py)
   - 看具体机器人任务如何继承通用配置

5. [`velocity_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)
   - 看任务世界是怎么定义出来的

6. [`rsl_rl_ppo_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/agents/rsl_rl_ppo_cfg.py)
   - 看训练器参数长什么样

7. [`play.py`](../scripts/reinforcement_learning/rsl_rl/play.py)
   - 看训练完的策略是怎么被拿来推理和回放的

这条顺序的核心不是“按文件名读”，而是按控制流读。

---

## 18. 最后给你一个判断标准

什么时候算是真的开始看懂这个项目？

不是你会背 Hydra，也不是你会背 PPO 公式。

而是你能把下面这几句话讲顺：

- 我传一个任务名给训练脚本，系统会通过注册表找到环境配置和算法配置。
- 训练入口先解析命令行参数，再装配配置对象，再创建环境和 runner。
- 环境配置定义任务世界，算法配置定义训练器行为。
- runner 才是真正执行训练循环的地方。
- 如果我要接自己的算法，主要替换的是 runner 和训练循环，而不是整个任务系统。

如果这五句话你已经能讲清楚，后面无论你是继续用 RSL-RL，还是换自己的算法，都会顺很多。

---

## 19. 一句话版本

对这个项目，最重要的不是先“学一堆工具名词”，而是先看清楚：
一个任务名是怎么一路流到环境、配置和训练循环里的。

---

## 20. 配套文档

如果你现在正准备细读训练入口，继续看：

- [`python_basic_terms.md`](./python_basic_terms.md)
- [`train_py_flow_03142344.md`](./train_py_flow_03142344.md)
- [`train_py_terms_03142344.md`](./train_py_terms_03142344.md)
- [`file_overview_03142344.md`](./file_overview_03142344.md)
- [`task_to_cfg_chain_03162353.md`](./task_to_cfg_chain_03162353.md)
- [`agent_cfg_to_runner_chain_03170044.md`](./agent_cfg_to_runner_chain_03170044.md)
- [`robot_lab_learning_recap_03172114.md`](./robot_lab_learning_recap_03172114.md)
