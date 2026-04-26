# Python 基础名词：包、模块、类、脚本

这份文档单独整理 Python 里最容易混淆的几个名词，适合复盘时快速对照。

---

## 1. 四个核心概念

### 包 `package`

包通常是一个目录，用来组织多个模块。

你可以先把它理解成：

- 包是“装代码的文件夹”
- 但这个文件夹不是普通文件夹
- 它可以被 Python 当成代码空间来导入

常见形式：

```text
robot_lab/
    __init__.py
    tasks/
    utils/
```

这里 `robot_lab` 就可以看成一个包。

一句话：

`包 = 装模块的目录`

---

### 模块 `module`

模块通常就是一个 `.py` 文件。

例如：

- `train.py`
- `cli_args.py`
- `utils.py`

都可以叫模块。

模块的作用是把代码拆开管理，避免所有代码都堆在一个文件里。

一句话：

`模块 = 一个 Python 文件`

---


### 脚本 `script`

脚本是“被直接运行的 Python 文件”。

例如：

```bash
python train.py
```

这时 `train.py` 的角色就是脚本。

注意：

- 脚本强调“怎么运行”
- 模块强调“代码怎么组织”

所以一个文件可以同时是模块和脚本。

一句话：

`脚本 = 被直接执行的 Python 文件`

---

## 2. 它们之间的关系

可以用这条链路来记：

`项目 -> 包 -> 模块 -> 类 / 函数 / 变量`

同时：

- 某个模块如果被直接运行
- 它就成了脚本

例如：

```text
my_project/
    tools/
        __init__.py
        parser.py
        train.py
```

这里：

- `tools` 是包
- `parser.py` 和 `train.py` 是模块
- `train.py` 直接运行时又是脚本

---

## 3. 最容易混淆的几组概念

### 包 vs 模块

- 包一般是目录
- 模块一般是文件

关系：

- 包里面可以包含很多模块

---

### 类 vs 对象

例如：

```python
parser = argparse.ArgumentParser()
```

这里：

- `ArgumentParser` 是类
- `parser` 是对象，也叫实例

所以：

- 类是模板
- 对象是按模板创建出来的具体东西

---

### 模块 vs 脚本

`train.py` 这种文件最容易让人混淆。

它从结构上看是模块；
当你运行 `python train.py` 时，它又是脚本。

所以：

- 模块是结构身份
- 脚本是运行身份

---

## 4. 还有哪些相邻名词

### 函数 `function`

最普通的可复用代码块。

```python
def add(a, b):
    return a + b
```

---

### 方法 `method`

定义在类里面的函数。

例如：

```python
parser.parse_known_args()
```

这里 `parse_known_args` 是方法。

---

### 对象 / 实例 `object / instance`

由类创建出来的具体值。

例如：

```python
parser = argparse.ArgumentParser()
```

`parser` 就是对象，也是实例。

---

### 属性 `attribute`

对象身上的成员，通常通过点号访问。

例如：

```python
args_cli.video
app_launcher.app
```

这里 `video`、`app` 都是属性。

---

### 变量 `variable`

变量是一个名字绑定一个值。

例如：

```python
x = 3
name = "robot"
```

---

### 配置 `config / cfg`

这不是 Python 语法专有名词，但工程里很常见。

例如：

- `env_cfg`
- `agent_cfg`

通常表示：

- 不是正在运行的对象
- 而是“如何构造对象”的配置

---

## 5. 用 `train.py` 套一遍

看这几行：

```python
import argparse
import cli_args

parser = argparse.ArgumentParser(...)
app_launcher = AppLauncher(args_cli)
```

可以这样拆：

- `argparse`：模块
- `cli_args`：模块
- `ArgumentParser`：类
- `AppLauncher`：类
- `parser`：对象 / 实例
- `app_launcher`：对象 / 实例

而 [`train.py`](../scripts/reinforcement_learning/rsl_rl/train.py) 本身：

- 是模块
- 直接运行时也是脚本

---

## 6. 速记版

- 包：装模块的目录
- 模块：一个 `.py` 文件
- 类：对象的模板
- 对象：类创建出来的实例
- 脚本：被直接运行的 Python 文件
- 函数：普通可调用代码块
- 方法：类里的函数
- 属性：对象上的成员

---

## 7. 一句话总结

读 Python 工程时，先分清“目录、文件、类、对象、入口文件”这五层，包、模块、类、脚本就不容易混。
