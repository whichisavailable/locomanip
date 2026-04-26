# robot_lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## 概述

**robot_lab** 是一个基于 IsaacLab 的机器人强化学习扩展库。它允许你在一个隔离的环境中开发，而不用修改 Isaac Lab 核心仓库。

下表列出了所有可用的环境：

| 类别       | 机器人型号                                                                 | 环境名称 (<ENV_NAME>)                                       | 截图                                                                                                                                     |
|------------|---------------------------------------------------------------------------|-------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| **四足**   | [Anymal D](https://www.anybotics.com/robotics/anymal)                     | RobotLab-Isaac-Velocity-Rough-Anymal-D-v0                   | <img src="./docs/imgs/anymal_d.png" alt="anymal_d" width="75">                                                                 |
|            | [Unitree Go2](https://www.unitree.com/go2)                                | RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0                | <img src="./docs/imgs/unitree_go2.png" alt="unitree_go2" width="75">                                                             |
|            | [Unitree B2](https://www.unitree.com/b2)                                  | RobotLab-Isaac-Velocity-Rough-Unitree-B2-v0                | <img src="./docs/imgs/unitree_b2.png" alt="unitree_b2" width="75">                                                             |
|            | [Unitree A1](https://www.unitree.com/a1)                                  | RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0                | <img src="./docs/imgs/unitree_a1.png" alt="unitree_a1" width="75">                                                             |
|            | [Deeprobotics Lite3](https://www.deeprobotics.cn/robot/index/product1.html) | RobotLab-Isaac-Velocity-Rough-Deeprobotics-Lite3-v0        | <img src="./docs/imgs/deeprobotics_lite3.png" alt="Lite3" width="75">                                                         |
|            | [Zsibot ZSL1](https://www.zsibot.com/zsl1)                                | RobotLab-Isaac-Velocity-Rough-Zsibot-ZSL1-v0               | <img src="./docs/imgs/zsibot_zsl1.png" alt="zsibot_zsl1" width="75">                                                          |
|            | [Magiclab MagicDog](https://www.magiclab.top/dog)                         | RobotLab-Isaac-Velocity-Rough-MagicLab-Dog-v0              | <img src="./docs/imgs/magiclab_magicdog.png" alt="magiclab_magicdog" width="75">                                                |
| **轮式**   | [Unitree Go2W](https://www.unitree.com/go2-w)                             | RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0              | <img src="./docs/imgs/unitree_go2w.png" alt="unitree_go2w" width="75">                                                         |
|            | [Unitree B2W](https://www.unitree.com/b2-w)                               | RobotLab-Isaac-Velocity-Rough-Unitree-B2W-v0               | <img src="./docs/imgs/unitree_b2w.png" alt="unitree_b2w" width="75">                                                           |
|            | [Deeprobotics M20](https://www.deeprobotics.cn/robot/index/lynx.html)     | RobotLab-Isaac-Velocity-Rough-Deeprobotics-M20-v0          | <img src="./docs/imgs/deeprobotics_m20.png" alt="deeprobotics_m20" width="75">                                                 |
|            | [DDTRobot Tita](https://directdrive.com/product_TITA)                     | RobotLab-Isaac-Velocity-Rough-DDTRobot-Tita-v0             | <img src="./docs/imgs/ddtrobot_tita.png" alt="ddtrobot_tita" width="75">                                                       |
|            | [Zsibot ZSL1W](https://www.zsibot.com/zsl1)                               | RobotLab-Isaac-Velocity-Rough-Zsibot-ZSL1W-v0              | <img src="./docs/imgs/zsibot_zsl1w.png" alt="zsibot_zsl1w" width="75">                                                        |
|            | [Magiclab MagicDog-W](https://www.magiclab.top/dog-w)                     | RobotLab-Isaac-Velocity-Rough-MagicLab-Dog-W-v0            | <img src="./docs/imgs/magiclab_magicdog_w.png" alt="magiclab_magicdog_w" width="75">                                            |
| **人形**   | [Unitree G1](https://www.unitree.com/g1)                                  | RobotLab-Isaac-Velocity-Rough-Unitree-G1-v0                | <img src="./docs/imgs/unitree_g1.png" alt="unitree_g1" width="75">                                                             |
|            | [Unitree H1](https://www.unitree.com/h1)                                  | RobotLab-Isaac-Velocity-Rough-Unitree-H1-v0                | <img src="./docs/imgs/unitree_h1.png" alt="unitree_h1" width="75">                                                             |
|            | [FFTAI GR1T1](https://www.fftai.com/products-gr1)                         | RobotLab-Isaac-Velocity-Rough-FFTAI-GR1T1-v0               | <img src="./docs/imgs/fftai_gr1t1.png" alt="fftai_gr1t1" width="75">                                                          |
|            | [FFTAI GR1T2](https://www.fftai.com/products-gr1)                         | RobotLab-Isaac-Velocity-Rough-FFTAI-GR1T2-v0               | <img src="./docs/imgs/fftai_gr1t2.png" alt="fftai_gr1t2" width="75">                                                          |
|            | [Booster T1](https://www.boosterobotics.com/)                             | RobotLab-Isaac-Velocity-Rough-Booster-T1-v0                | <img src="./docs/imgs/booster_t1.png" alt="booster_t1" width="75">                                                            |
|            | [RobotEra Xbot](https://www.robotera.com/)                               | RobotLab-Isaac-Velocity-Rough-RobotEra-Xbot-v0            | <img src="./docs/imgs/robotera_xbot.png" alt="robotera_xbot" width="75">                                                        |
|            | [Openloong Loong](https://www.openloong.net/)                            | RobotLab-Isaac-Velocity-Rough-Openloong-Loong-v0          | <img src="./docs/imgs/openloong_loong.png" alt="openloong_loong" width="75">                                                   |
|            | [RoboParty ATOM01](https://roboparty.cn/)                                | RobotLab-Isaac-Velocity-Rough-RoboParty-ATOM01-v0         | <img src="./docs/imgs/roboparty_atom01.png" alt="roboparty_atom01" width="75">                                                 |
|            | [Magiclab MagicBot-Gen1](https://www.magiclab.top/human)                 | RobotLab-Isaac-Velocity-Rough-MagicLab-Bot-Gen1-v0        | <img src="./docs/imgs/magiclab_magicbot_gen1.png" alt="magiclab_magicbot_gen1" width="75">                                     |
|            | [Magiclab MagicBot-Z1](https://www.magiclab.top/z1)                      | RobotLab-Isaac-Velocity-Rough-MagicLab-Bot-Z1-v0          | <img src="./docs/imgs/magiclab_magicbot_z1.png" alt="magiclab_magicbot_z1" width="75">                                       |

> [!NOTE]
> 如果希望在 gazebo 或真实机器人上运行策略，请使用 [rl_sar](https://github.com/fan-ziqi/rl_sar) 项目。
>
> 在 [Github 讨论](https://github.com/fan-ziqi/robot_lab/discussions) 或 [Discord](http://www.robotsfan.com/dc_robot_lab) 中讨论。

## 版本依赖

| robot_lab 版本 | Isaac Lab 版本          | Isaac Sim 版本              |
|----------------|--------------------------|-----------------------------|
| `main` 分支    | `main` 分支              | Isaac Sim 4.5 / 5.0 / 5.1    |
| `v2.3.2`        | `v2.3.2`                 | Isaac Sim 4.5 / 5.0 / 5.1    |
| `v2.2.2`        | `v2.2.1`                 | Isaac Sim 4.5 / 5.0          |
| `v2.1.1`        | `v2.1.1`                 | Isaac Sim 4.5                |
| `v1.1`          | `v1.4.1`                 | Isaac Sim 4.2                |

## 安装

- 按照 [安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Lab。我们建议使用 conda 安装，这样可以简化在终端调用 Python 脚本。

- 在 Isaac Lab 安装目录之外单独克隆本仓库：

  ```bash
  git clone https://github.com/fan-ziqi/robot_lab.git
  ```

- 使用已安装 Isaac Lab 的 Python 解释器安装库：

  ```bash
  python -m pip install -e source/robot_lab
  ```

- 运行下列命令验证扩展安装是否正确，打印扩展中可用的所有环境：

  ```bash
  python scripts/tools/list_envs.py
  ```

<details>

<summary>设置 IDE（可选，点击展开）</summary>

要设置 IDE，请按照以下说明操作：

- 运行 VSCode 任务：按 `Ctrl+Shift+P`，选择 `Tasks: Run Task`，运行下拉菜单中的 `setup_python_env`。运行此任务时，会提示输入 Isaac Sim 安装的绝对路径。

如果一切执行正确，应该会在 `.vscode` 目录下创建一个 `.python.env` 文件。该文件包含 Isaac Sim 和 Omniverse 提供的所有扩展 Python 路径。这有助于在编写代码时进行智能提示。

</details>

<details>

<summary>设置为 Omniverse 扩展（可选，点击展开）</summary>

我们提供了一个示例 UI 扩展，在启用 `source/robot_lab/robot_lab/ui_extension_example.py` 中定义的扩展时会加载。

要启用扩展，请执行以下步骤：

1. **将仓库的搜索路径加入扩展管理器**：
    - 通过 `Window` -> `Extensions` 打开扩展管理器。
    - 点击 **汉堡图标** (☰)，然后转到 `Settings`。
    - 在 `Extension Search Paths` 中输入 `robot_lab/source` 的绝对路径。
    - 如果还未添加，请在 `Extension Search Paths` 中输入 Isaac Lab 扩展目录的路径 (`IsaacLab/source`)。
    - 点击 **汉堡图标** (☰)，然后点击 `Refresh`。

2. **搜索并启用扩展**：
    - 在 `Third Party` 类别下找到你的扩展。
    - 切换开关以启用它。

</details>

## Docker 设置

<details>

<summary>点击展开</summary>

### 构建 Isaac Lab 基础镜像

目前，我们没有公开的 Isaac Lab Docker 镜像。因此，你需要按照 [这里](https://isaac-sim.github.io/IsaacLab/main/source/deployment/index.html) 的步骤在本地构建镜像。

构建完成后，你可以通过以下命令检查镜像是否存在：

```bash
docker images

# 输出应类似：
#
# REPOSITORY                       TAG       IMAGE ID       CREATED          SIZE
# isaac-lab-base                   latest    28be62af627e   32 minutes ago   18.9GB
```

### 构建 robot_lab 镜像

在上述步骤基础上，你可以构建本项目的 Docker 容器。默认名称为 `robot-lab`，但可以在 [`docker/docker-compose.yaml`](docker/docker-compose.yaml) 中修改。

```bash
cd docker
docker compose --env-file .env.base --file docker-compose.yaml build robot-lab
```

使用与之前相同的命令验证镜像构建是否成功：

```bash
docker images

# 输出应类似：
#
# REPOSITORY                       TAG       IMAGE ID       CREATED             SIZE
# robot-lab                        latest    00b00b647e1b   2 minutes ago       18.9GB
# isaac-lab-base                   latest    892938acb55c   About an hour ago   18.9GB
```

### 运行容器

构建后，下一步通常是启动与服务相关的容器。可以使用：

```bash
docker compose --env-file .env.base --file docker-compose.yaml up
```

这会启动 `docker-compose.yaml` 中定义的服务，包括 robot-lab。

如果想在后台运行，请加 `-d`：

```bash
docker compose --env-file .env.base --file docker-compose.yaml up -d
```

### 与运行中的容器交互

要在运行中的容器内运行命令，可使用：

```bash
docker exec --interactive --tty -e DISPLAY=${DISPLAY} robot-lab /bin/bash
```

### 关闭容器

完成或希望停止运行时，可以停用服务：

```bash
docker compose --env-file .env.base --file docker-compose.yaml down
```

此操作会停止并移除容器，但保留镜像。

</details>

## 试用示例

你可以使用以下命令运行所有环境：

RSL-RL：

```bash
# 训练
python scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK_NAME> --headless

# 播放
python scripts/reinforcement_learning/rsl_rl/play.py --task=<TASK_NAME>
```

CusRL（**实验性**）：

```bash
# 训练
python scripts/reinforcement_learning/cusrl/train.py --task=<TASK_NAME> --headless

# 播放
python scripts/reinforcement_learning/cusrl/play.py --task=<TASK_NAME>
```

使用虚拟代理运行任务（包括输出零或随机行为的代理，可用于确认环境配置正确）：

```bash
# 零动作代理
python scripts/tools/zero_agent.py --task=<TASK_NAME>
# 随机动作代理
python scripts/tools/random_agent.py --task=<TASK_NAME>
```

BeyondMimic（适用于 Unitree G1）：

- 收集参考运动数据集（请遵循原始许可证），我们使用与 Unitree 数据集相同的 .csv 约定

  - Unitree 重定向的 LAFAN1 数据集可在 [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) 获取
  - Sidekicks 来自 [KungfuBot](https://kungfu-bot.github.io/)
  - C Ronaldo 庆祝动作来自 [ASAP](https://github.com/LeCAR-Lab/ASAP)。
  - 平衡动作来自 [HuB](https://hub-robot.github.io/)

- 通过前向运动学将重定向的动作转换为包含最大坐标信息（身体姿态、速度和加速度）：

  ```bash
  python scripts/tools/beyondmimic/csv_to_npz.py -f path_to_input.csv --input_fps 60 --headless
  ```

- 在 Isaac Sim 中重放动作：

  ```bash
  python scripts/tools/beyondmimic/replay_npz.py -f path_to_motion.npz
  ```

- 训练与评估

  ```bash
  # 训练
  python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0 --headless

  # 播放
  python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0 --num_envs 2
  ```

其他（**实验性**）

- 训练 Unitree G1 的 AMP 舞蹈

  ```bash
  # 训练
  python scripts/reinforcement_learning/skrl/train.py --task=RobotLab-Isaac-G1-AMP-Dance-Direct-v0 --algorithm AMP --headless

  # 播放
  python scripts/reinforcement_learning/skrl/play.py --task=RobotLab-Isaac-G1-AMP-Dance-Direct-v0 --algorithm AMP --num_envs=32
  ```

- 训练 Unitree A1 站立动作

  ```bash
  # 训练
  python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Flat-HandStand-Unitree-A1-v0 --headless

  # 播放
  python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-HandStand-Unitree-A1-v0
  ```

- 对 Anymal D 进行对称训练

  ```bash
  # 训练
  python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Rough-Anymal-D-v0 --headless --agent=rsl_rl_with_symmetry_cfg_entry_point --run_name=ppo_with_symmetry_data_augmentation agent.algorithm.symmetry_cfg.use_data_augmentation=true

  # 播放
  python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Rough-Anymal-D-v0 --agent=rsl_rl_with_symmetry_cfg_entry_point --run_name=ppo_with_symmetry_data_augmentation agent.algorithm.symmetry_cfg.use_data_augmentation=true
  ```

- 训练并蒸馏 Anymal D

  ```bash
  # 训练教师模型
  python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Flat-Anymal-D-v0 --headless

  # 将教师模型蒸馏为学生模型
  python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Flat-Anymal-D-v0 --headless --agent=rsl_rl_distillation_cfg_entry_point --load_run teacher_run_folder_name

  # 播放学生模型
  python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-Anymal-D-v0 --num_envs 64 --agent rsl_rl_distillation_cfg_entry_point
```

> [!NOTE]
> 如果在回放过程中想使用键盘控制**单个机器人**，请在命令末尾添加 `--keyboard`。
>
> ```
> 键位说明：
> ====================== ========================= ========================
> 命令                按键（正方向）           按键（负方向）
> ====================== ========================= ========================
> 沿 x 轴移动          小键盘 8 / 方向键上      小键盘 2 / 方向键下
> 沿 y 轴移动          小键盘 4 / 方向键右      小键盘 6 / 方向键左
> 绕 z 轴旋转          小键盘 7 / Z              小键盘 9 / X
> ====================== ========================= ========================
> ```

* 以上配置中可以将 `Rough` 替换为 `Flat`。
* 若要记录训练视频（需要安装 `ffmpeg`），加上 `--video --video_length 200`。
* 使用 32 个环境训练/播放，添加 `--num_envs 32`。
* 指定文件夹或检查点进行播放/训练，添加 `--load_run run_folder_name --checkpoint /PATH/TO/model.pt`。
* 从文件夹或检查点恢复训练，添加 `--resume --load_run run_folder_name --checkpoint /PATH/TO/model.pt`。
* 使用多 GPU 训练（`--nproc_per_node` 为可用 GPU 数量）：
    ```bash
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK_NAME> --headless --distributed
    ```
* 若要跨多台机器进行分布式训练，可在每个节点启动进程。

    对于主节点，命令示例（`--nnodes` 为节点数）：
    ```bash
    python -m torch.distributed.run --nproc_per_node=2 --nnodes=2 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:5555 scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK_NAME> --headless --distributed
    ```
    注意端口 (`5555`) 可以替换为其他可用端口。
    对于非主节点，只需将 `--node_rank` 替换为对应的机器索引：
    ```bash
    python -m torch.distributed.run --nproc_per_node=2 --nnodes=2 --node_rank=1 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=ip_of_master_machine:5555 scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK_NAME> --headless --distributed
    ```

## 添加自己的机器人

基于 Isaac Lab 开发的核心框架，我们提供了用于机器人研究的各种学习环境。 这些环境遵循 OpenAI Gym `0.21.0` 的 `gym.Env` API，并通过 Gym 注册表进行注册。

每个环境的名称格式为 `Isaac-<Task>-<Robot>-v<X>`，其中 `<Task>` 表示环境中要学习的技能，`<Robot>` 表示执行主体的机型，`<X>` 表示环境版本（可用于区分不同的观察或动作空间）。

环境可以通过 Python 类（使用 `configclass` 装饰器封装）或 YAML 文件配置。环境的模板结构通常与环境文件处于同一层级，但其各个实例会放在环境目录内的子文件夹中。目录结构示例如下：

```tree
source/robot_lab/assets/
├── __init__.py
└── unitree.py  # <- 在此定义机器人资产

source/robot_lab/tasks/manager_based/locomotion/
├── __init__.py
└── velocity
    ├── config
    │   └── unitree_a1
    │       ├── agent  # <- 存放学习代理配置
    │       ├── __init__.py  # <- 在此将环境和配置注册到 gym 注册表
    │       ├── flat_env_cfg.py
    │       └── rough_env_cfg.py
    ├── __init__.py
    └── velocity_env_cfg.py  # <- 基础任务配置
```

随后在 `source/robot_lab/tasks/manager_based/locomotion/velocity/config/unitree_a1/__init__.py` 中注册环境：

```python
gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Unitree-A1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeA1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeA1FlatTrainerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeA1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeA1RoughTrainerCfg",
    },
)
```

## Tensorboard

要查看 Tensorboard，运行：

```bash
tensorboard --logdir=logs
```

## 代码格式化

提供了 pre-commit 模板用于自动格式化代码。

安装 pre-commit：

```bash
pip install pre-commit
```

然后可以通过以下命令运行：

```bash
pre-commit run --all-files
```

## 故障排查

### Pylance 缺失扩展索引

在某些 VSCode 版本中，部分扩展可能未被正确索引。此时，请在 `.vscode/settings.json` 中的 `"python.analysis.extraPaths"` 添加扩展路径。

**注意：将 `<path-to-isaac-lab>` 替换为你自己的 IsaacLab 路径。**

```json
{
    "python.languageServer": "Pylance",
    "python.analysis.extraPaths": [
        "${workspaceFolder}/source/robot_lab",
        "/<path-to-isaac-lab>/source/isaaclab",
        "/<path-to-isaac-lab>/source/isaaclab_assets",
        "/<path-to-isaac-lab>/source/isaaclab_mimic",
        "/<path-to-isaac-lab>/source/isaaclab_rl",
        "/<path-to-isaac-lab>/source/isaaclab_tasks",
    ]
}
```

### 清理 USD 缓存

模拟运行期间会在 `/tmp/IsaacLab/usd_{date}_{time}_{random}` 生成临时 USD 文件，这些文件可能占用大量磁盘空间，可以通过：

```bash
rm -rf /tmp/IsaacLab/usd_*
```

## 引用

如果你使用了本代码或其中部分，请引用：

```
@software{fan-ziqi2024robot_lab,
  author = {Ziqi Fan},
  title = {robot_lab: RL Extension Library for Robots, Based on IsaacLab.},
  url = {https://github.com/fan-ziqi/robot_lab},
  year = {2024}
}
```

## 致谢

本项目使用了以下开源代码仓库中的部分代码：

- [linden713/humanoid_amp](https://github.com/linden713/humanoid_amp)
- [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)
