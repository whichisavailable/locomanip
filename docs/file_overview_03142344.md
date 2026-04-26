# 文件概览与用途说明

此文档对 `scripts/` 和 `source/` 两个主要目录下的文件进行扫描，
列出每个文件/子目录的作用以及如何使用它们。对新手快速定位代码
和理解项目结构非常有用。

---

## 📁 scripts 目录

该目录保存了几类脚本：训练/播放、实用工具、数据转换等。

### reinforcement_learning

- **rsl_rl/**
  - `train.py`：RSL‑RL 训练主程序。通过命令行指定任务和超参，然后
    启动 IsaacSim、构建环境、包装 VecEnv、创建 Runner 并调用 `learn()`。
  - `play.py`：使用已训练的检查点执行推理/回放，支持视频录制和键盘控制。
  - `play_cs.py`：
    类似 `play.py`，额外提供加载地图 (`--map`) 以及自定义重置、命令
    范围的代码，通常用于“街区地图”场景。
  - `cli_args.py`：提取和解析 RSL‑RL 特有的命令行参数（如 `--resume`、
    `--experiment_name`），并提供函数 `update_rsl_rl_cfg` 供脚本覆盖 Hydra 配置。

- **cusrl/**
  - `train.py`、`play.py`：CusRL 实验性算法的训练/回放脚本，结构与 rsl_rl 类似。

- **skrl/**
  - `train.py`、`play.py`：基于 SKRL 的实验性示例，同样用于训练/回放。

- `rl_utils.py`：公共辅助函数（如 `camera_follow`），被多个脚本导入。

### tools

- **beyondmimic/**
  - `csv_to_npz.py`：将外部 CSV 格式的动作数据转换为 `.npz`，用于 BeyondMimic 训练。
  - `replay_npz.py`：在 IsaacSim 中重放 `.npz` 数据集。

- `list_envs.py`：打印当前 Isaac Lab 环境注册表中包含 `RobotLab` 前缀的环境。
- `zero_agent.py` / `random_agent.py`：将零动作或随机动作代理运行到指定任务，用于快速健康检查。
- `convert_urdf.py` / `convert_mjcf.py`：将 URDF/MJCF 文件转换为 IsaacSim 支持的 USD 格式。
- `clean_trash.py`：清理临时文件或旧输出。

**使用方式**：大部分脚本可以通过 `python scripts/…` 直接运行，且都有
`--help` 显示可用参数。

---

## 📁 source 目录

该目录包含可安装的 Python 包以及项目核心逻辑。

### 顶层文件

- `setup.py`：安装脚本，从 `config/extension.toml` 中读取元数据并
  调用 `setuptools.setup`。
- `pyproject.toml`：仅声明 build-system（`setuptools`），主要
  用于 `pip install -e source/robot_lab`。
- `config/extension.toml`：包含版本、依赖、描述等扩展相关信息。

### robot_lab 包 (`source/robot_lab/robot_lab`)

- `__init__.py`：包初始化。
- `ui_extension_example.py`：示例 Omniverse UI 扩展，可在 Isaac Sim 中
  通过扩展管理器启用。

#### assets/

负责定义机器人资产的代码。每个文件对应一个制造商/机器人：
`unitree.py`、`deeprobotics.py`、`magiclab.py` 等。它们通常创建
`Articulation`、设置 URDF 路径、网格转换等。

#### tasks/

环境与配置的核心部分。

- **direct/**：直接继承 `gym.Env` 的简单环境，仅有少数示例，如
  `g1_amp`。

- **manager_based/**：使用 IsaacLab 管理器/配置系统的复杂环境。
  结构为 `manager_based/<task>/<subtask>/config/<robot>`。
  其中：
    - `*.yaml` 和 `*_env_cfg.py` 定义任务参数。
    - 每个 `config/<robot>/__init__.py` 使用 `gym.register()` 注册
      对应环境，并指定配置入口点。
    - `agent` 子目录存放算法配置（RSL‑RL、CusRL 等）。

  举例：`manager_based/locomotion/velocity/config/unitree_a1` 包含
  `flat_env_cfg.py`、`rough_env_cfg.py`、agent 配置以及注册逻辑。

- **beyondmimic/**：与参考运动数据结合的特殊任务定义。

##### 其他模块

- 许多 Python 文件在 `tasks` 下提供类定义、观察/动作/奖励 term
  的实现，必须通过阅读具体子目录来理解。

### 🔧 tasks 目录详细介绍

`source/robot_lab/robot_lab/tasks` 是环境定义的核心包，分成两大
工作流：`direct` 和 `manager_based`。

#### direct

这里的环境继承自 `isaaclab.envs.DirectRLEnv`，适合快速原型或
只需少量定制的场景。

- `g1_amp/` 提供一个 **Unitree G1 AMP 舞蹈**示例：
  - `g1_amp_env.py`：Python 类 `G1AmpEnv`，定义观测、奖励、动作
    及运行逻辑；
  - `g1_amp_env_cfg.py`：对应 Hydra 配置类；
  - `motions/` 存放参考运动数据；
  - `agents/` 存放 RL 算法配置（如 PPO）用于此任务。

工作机制：`__init__` 注册环境，调用 `gym.register()` 时指定
`entry_point` 为 `G1AmpEnv` 类。

#### manager_based

这是主要的、功能丰富的工作流，使用 **Hydra+configclass** 来构建
环境，是 IsaacLab 的推荐方式。组织结构：

```
manager_based/
 ├── beyondmimic/           # 基于参考运动的环境
 │    ├── tracking_env_cfg.py
 │    └── config/…
 ├── locomotion/            # 常见的行走/速度追踪任务
 │    ├── velocity/         # 包含四足/轮式/人形子类型
 │    │    ├── config/      # 每种机器人对应一个子目录
 │    │    │    └── unitree_a1/    # 示例机器人
 │    │    │        ├── flat_env_cfg.py
 │    │    │        ├── rough_env_cfg.py
 │    │    │        ├── agent/     # 训练算法配置
 │    │    │        └── __init__.py# gym.register() 调用
 │    │    ├── mdp/         # 包含观察/终止/奖励 term 等通用逻辑
 │    │    └── velocity_env_cfg.py # 基类配置
 │    └── …
 └── …
```

- 每个 `config/<robot>` 目录中的 `__init__.py` 为环境注册点，
  指定了 `env_cfg_entry_point`（环境参数类）、
  `rsl_rl_cfg_entry_point`（RSL-RL 算法配置）、以及其他算法入口。
- `mdp/` 下的文件提供观测、动作、奖励等基类逻辑，供多个机器人
 复用。

使用方式示例：

```python
# 注册位于 velocity/config/unitree_a1/__init__.py
gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Unitree-A1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeA1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1FlatPPORunnerCfg",
        …
    },
)
```

要添加新机器人/任务，只需复制现有目录结构、编写环境配置类
(`*_env_cfg.py`)、实现必要的奖励/观测 term，并在 `__init__.py` 中
注册。

此目录下的源码是理解环境行为与参数的最佳入口点。

### 其他辅助包

还有比如 `data/` 存放机器人 URDF/mesh，`config/` 存放扩展元数据，
`pyproject.toml` 声明构建系统等。

---

## ✅ 如何使用上述文件

1. **调试脚本行为**：修改 `scripts/reinforcement_learning/...` 可以快速
   添加命令行选项或观察训练流程。
2. **扩展环境**：在 `source/robot_lab/robot_lab/tasks/...` 目录下新建
   子目录并写入 `__init__.py` 来注册新环境。
3. **自定义机器人资产**：往 `assets/` 添加新的机器人定义并确保
   URDF/mesh 路径正确。
4. **安装与分发**：`pip install -e source/robot_lab`，`setup.py` 会读取
   `extension.toml` 并生产可使用的 `robot_lab` 包。

---

## 补充说明：如何把这份总览用起来

这份文档的强项是“地图”，也就是告诉你项目里大目录和主要文件各放在哪里。

但如果你的目标不是扫目录，而是顺着一条真实训练链路读懂项目，那么还需要额外记住下面三点。

### 1. 先读哪些文件

如果你现在最关心的是 RSL-RL 的训练入口，建议按这个顺序读：

1. [`../scripts/reinforcement_learning/rsl_rl/train.py`](../scripts/reinforcement_learning/rsl_rl/train.py)
2. [`../scripts/reinforcement_learning/rsl_rl/cli_args.py`](../scripts/reinforcement_learning/rsl_rl/cli_args.py)
3. [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/__init__.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/__init__.py)
4. [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/flat_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/flat_env_cfg.py)
5. [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)
6. [`../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/agents/rsl_rl_ppo_cfg.py`](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/agents/rsl_rl_ppo_cfg.py)

这个顺序比按目录从上往下扫更有效，因为它遵循的是训练控制流。

### 2. 最值得记住的一条链

如果只记一条项目主线，记这条就够了：

`task 字符串 -> gym.register -> env_cfg / agent_cfg 入口 -> Hydra 加载配置对象 -> train.py 创建 env -> runner.learn()`

后面你再看别的算法后端或者别的机器人任务，基本还是沿着这条链在变化。

### 3. 配套文档各自做什么

现在 `docs/` 里的几份文档可以这样分工：

- [`learning_path_03142344.md`](./learning_path_03142344.md)
  - 总学习路线
- [`python_basic_terms.md`](./python_basic_terms.md)
  - Python 工程里的基础名词
- [`train_py_terms_03142344.md`](./train_py_terms_03142344.md)
  - `train.py` 里的高频名字和语法
- [`train_py_flow_03142344.md`](./train_py_flow_03142344.md)
  - `train.py` 的控制流和配置覆盖逻辑
- [`file_overview_03142344.md`](./file_overview_03142344.md)
  - 项目目录和文件分布总览

所以这份 `overview` 负责回答“东西在哪”，训练文档负责回答“东西怎么串起来”。
