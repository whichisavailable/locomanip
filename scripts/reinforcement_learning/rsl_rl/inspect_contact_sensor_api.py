# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Inspect Isaac Sim contact-sensor APIs and optional task/env contact wiring."""

"""Launch Isaac Sim Simulator first."""

import argparse
import faulthandler
import importlib
import inspect
import os
import platform
import pkgutil
from pathlib import Path
import sys
import traceback
from typing import Any

from isaaclab.app import AppLauncher


# 这里单独加少量 CLI，避免把训练脚本里的 Hydra / agent 参数整套带进来。
parser = argparse.ArgumentParser(description="Inspect Isaac Sim contact sensor APIs.")
# 可选：传任务名时，会额外创建环境并打印 robot/body/sensor 信息。
parser.add_argument("--task", type=str, default=None, help="Optional task name to instantiate for inspection.")
# 可选：创建环境时使用的环境数量，默认只开 1 个，尽量减小开销。
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments when --task is provided.")
# 可选：显式指定一个 contact sensor prim 路径，脚本会尝试读取 raw data。
parser.add_argument(
    "--raw-prim-path",
    type=str,
    default=None,
    help="Optional contact sensor prim path used for get_contact_sensor_raw_data inspection.",
)
# 默认把完整报告直接写到脚本同目录文件里，避免用户从终端长输出中手动筛选。
parser.add_argument(
    "--report-file",
    type=str,
    default=None,
    help="Optional output report file path. Defaults to a text file next to this script.",
)
# 可选：限制打印 body_names 的条数，避免终端过长。
parser.add_argument("--max-body-names", type=int, default=256, help="Maximum number of body names to print.")
# 追加 Isaac Sim / Isaac Lab 的标准启动参数，例如 --device、--headless。
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 在启动应用前就把 stdout/stderr 重定向到报告文件，这样 Isaac Sim 启动日志和后续诊断都能完整落盘。
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_DEFAULT_REPORT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inspect_contact_sensor_api_report.txt")
REPORT_FILE_PATH = os.path.abspath(args_cli.report_file) if args_cli.report_file else _DEFAULT_REPORT_FILE
os.makedirs(os.path.dirname(REPORT_FILE_PATH), exist_ok=True)
print(f"[INFO] Writing inspection report to: {REPORT_FILE_PATH}", file=_ORIGINAL_STDOUT, flush=True)
_REPORT_STREAM = open(REPORT_FILE_PATH, "w", encoding="utf-8", buffering=1)
sys.stdout = _REPORT_STREAM
sys.stderr = _REPORT_STREAM
faulthandler.enable(_REPORT_STREAM, all_threads=True)
print("[TRACE] report stream initialized", flush=True)

# 启动 Omniverse / Isaac Sim 应用。
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[TRACE] AppLauncher initialized", flush=True)
print(f"[TRACE] args.task={args_cli.task}", flush=True)
print(f"[TRACE] args.raw_prim_path={args_cli.raw_prim_path}", flush=True)


def print_header(title: str) -> None:
    """打印明显的分隔标题，便于用户在终端里直接复制关键信息。"""
    print("\n" + "=" * 24 + f" {title} " + "=" * 24)


def safe_repr(value: Any) -> str:
    """把任意对象转成尽量稳定的字符串，避免 __repr__ 异常中断脚本。"""
    try:
        return repr(value)
    except Exception as exc:  # noqa: BLE001
        return f"<repr failed: {exc}>"


def dump_module_imports() -> dict[str, Any]:
    """探测本机 Isaac Sim / Contact Sensor 相关 Python 模块能否导入。"""
    print_header("Runtime")
    print(f"python_executable: {sys.executable}")
    print(f"python_version: {sys.version}")
    print(f"platform: {platform.platform()}")

    imported_modules: dict[str, Any] = {}
    module_names = [
        "isaacsim",
        "isaacsim.sensors.physics",
        "isaacsim.sensors.physics._sensor",
        "omni.isaac.sensor",
        "omni.isaac.sensor._sensor",
    ]

    print_header("Module Imports")
    for mod_name in module_names:
        try:
            module = importlib.import_module(mod_name)
            imported_modules[mod_name] = module
            print(f"[OK] {mod_name}: {module}")
            interesting = [
                name
                for name in dir(module)
                if "Contact" in name or "contact" in name.lower() or "raw" in name.lower() or "sensor" in name.lower()
            ]
            print(f"  dir_interesting({len(interesting)}): {interesting}")
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {mod_name}: {exc}")

    return imported_modules


def dump_extension_state() -> None:
    """打印与 sensor/contact 相关的扩展状态，帮助判断模块导入失败是否因为扩展未启用。"""
    print_header("Extensions")
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        extension_dict = manager.get_extensions()
        interesting_extensions: list[tuple[str, bool]] = []

        for ext_id in extension_dict:
            try:
                ext_dict = manager.get_extension_dict(ext_id)
                ext_name = ext_dict.get("package/packageId") or ext_dict.get("name") or ext_id
                ext_name_str = str(ext_name)
                if any(key in ext_name_str.lower() for key in ("sensor", "contact", "isaacsim", "isaac")):
                    interesting_extensions.append((ext_name_str, manager.is_extension_enabled(ext_id)))
            except Exception:  # noqa: BLE001
                continue

        interesting_extensions.sort(key=lambda item: item[0])
        print(f"interesting_extensions({len(interesting_extensions)}):")
        for ext_name, enabled in interesting_extensions:
            print(f"  - enabled={enabled} :: {ext_name}")
    except Exception as exc:  # noqa: BLE001
        print(f"extension inspection failed: {exc}")


def try_enable_candidate_extensions() -> None:
    """尝试启用可能承载 Contact Sensor Python API 的扩展，然后让用户看到是否启用成功。"""
    print_header("Enable Candidate Extensions")
    candidate_names = [
        "isaacsim.sensors.physics",
        "omni.isaac.sensor",
        "isaacsim.sensor.physics",
    ]
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        all_ext_ids = manager.get_extensions()
        matched_any = False

        for candidate_name in candidate_names:
            matched_ids: list[str] = []
            for ext_id in all_ext_ids:
                try:
                    ext_dict = manager.get_extension_dict(ext_id)
                    ext_name = ext_dict.get("package/packageId") or ext_dict.get("name") or ext_id
                    if str(ext_name) == candidate_name:
                        matched_ids.append(ext_id)
                except Exception:  # noqa: BLE001
                    continue

            if not matched_ids:
                print(f"[MISS] extension not found: {candidate_name}")
                continue

            matched_any = True
            for ext_id in matched_ids:
                try:
                    was_enabled = manager.is_extension_enabled(ext_id)
                    if not was_enabled:
                        manager.set_extension_enabled_immediate(ext_id, True)
                    now_enabled = manager.is_extension_enabled(ext_id)
                    print(f"[EXT] {candidate_name}: ext_id={ext_id}, was_enabled={was_enabled}, now_enabled={now_enabled}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[FAIL] enabling {candidate_name} ({ext_id}) failed: {exc}")

        if not matched_any:
            print("No candidate sensor extension names were found in the current app.")
    except Exception as exc:  # noqa: BLE001
        print(f"candidate extension enable failed: {exc}")


def dump_pkgutil_candidates() -> None:
    """在当前 Python 路径里扫描与 sensor/contact 相关的顶层包，帮助定位模块命名差异。"""
    print_header("Pkgutil Scan")
    try:
        top_level = sorted(
            {
                module.name
                for module in pkgutil.iter_modules()
                if any(key in module.name.lower() for key in ("isaac", "sensor", "omni"))
            }
        )
        print(f"top_level_candidates({len(top_level)}): {top_level}")
    except Exception as exc:  # noqa: BLE001
        print(f"pkgutil top-level scan failed: {exc}")

    for package_name in ["isaacsim", "omni"]:
        try:
            package = importlib.import_module(package_name)
            if not hasattr(package, "__path__"):
                print(f"{package_name} has no __path__; skipping submodule scan.")
                continue
            submods = sorted(
                {
                    module.name
                    for module in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}.")
                    if any(key in module.name.lower() for key in ("sensor", "contact", "physics"))
                }
            )
            print(f"{package_name}_submodules({len(submods)}): {submods}")
        except Exception as exc:  # noqa: BLE001
            print(f"{package_name} submodule scan failed: {exc}")


def dump_installation_scan(imported_modules: dict[str, Any]) -> None:
    """扫描 Isaac Sim 安装目录，判断 contact sensor 扩展/源码是否根本存在于磁盘上。"""
    print_header("Installation Scan")

    isaacsim_mod = imported_modules.get("isaacsim")
    if isaacsim_mod is None:
        print("isaacsim module unavailable; skipping installation scan.")
        return

    try:
        isaacsim_file = Path(isaacsim_mod.__file__).resolve()
        # isaacsim/__init__.py 位于 .../_isaac_sim/python_packages/isaacsim/__init__.py
        install_root = isaacsim_file.parents[2]
        print(f"isaacsim_module_file: {isaacsim_file}")
        print(f"isaacsim_install_root: {install_root}")
    except Exception as exc:  # noqa: BLE001
        print(f"failed to resolve isaacsim install root: {exc}")
        return

    if not install_root.exists():
        print("resolved install root does not exist; skipping.")
        return

    # 优先搜扩展清单文件，看磁盘里到底有没有相关扩展包。
    candidate_extension_names = {
        "isaacsim.sensors.physics",
        "omni.isaac.sensor",
        "isaacsim.sensor.physics",
    }
    matched_extension_manifests: list[tuple[Path, str]] = []
    extension_tomls = list(install_root.rglob("extension.toml"))
    print(f"extension_toml_count: {len(extension_tomls)}")
    for manifest_path in extension_tomls:
        try:
            text = manifest_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        for candidate_name in candidate_extension_names:
            if candidate_name in text:
                matched_extension_manifests.append((manifest_path, candidate_name))

    if matched_extension_manifests:
        print(f"matched_extension_manifests({len(matched_extension_manifests)}):")
        for manifest_path, candidate_name in matched_extension_manifests[:50]:
            print(f"  - candidate={candidate_name} :: {manifest_path}")
    else:
        print("matched_extension_manifests(0)")

    # 再扫关键 API 符号，看安装目录里是否存在 contact sensor 的原生 Python / 绑定代码。
    api_needles = [
        "IsaacSensorCreateContactSensor",
        "get_contact_sensor_raw_data",
        "acquire_contact_sensor_interface",
        "decode_body_name",
        "ContactSensorInterface",
    ]
    matched_api_files: list[tuple[Path, str]] = []
    for file_path in install_root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".py", ".toml", ".json", ".kit", ".md"}:
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        for needle in api_needles:
            if needle in text:
                matched_api_files.append((file_path, needle))

    if matched_api_files:
        print(f"matched_api_files({len(matched_api_files)}):")
        for file_path, needle in matched_api_files[:80]:
            print(f"  - needle={needle} :: {file_path}")
    else:
        print("matched_api_files(0)")


def resolve_contact_sensor_bindings(imported_modules: dict[str, Any]) -> tuple[Any | None, Any | None, str | None]:
    """从不同版本模块路径里解析 ContactSensor 类和底层 _sensor 接口模块。"""
    candidates = [
        ("isaacsim.sensors.physics", "isaacsim.sensors.physics._sensor"),
        ("omni.isaac.sensor", "omni.isaac.sensor._sensor"),
    ]
    for public_mod_name, private_mod_name in candidates:
        public_mod = imported_modules.get(public_mod_name)
        private_mod = imported_modules.get(private_mod_name)
        if public_mod is None or private_mod is None:
            continue
        contact_sensor_cls = getattr(public_mod, "ContactSensor", None)
        if contact_sensor_cls is not None:
            return contact_sensor_cls, private_mod, public_mod_name
    return None, None, None


def dump_contact_sensor_api(contact_sensor_cls: Any | None, sensor_module: Any | None, source_name: str | None) -> Any | None:
    """打印 ContactSensor 类签名和底层 interface 可调用方法。"""
    print_header("Contact Sensor API")
    if contact_sensor_cls is None or sensor_module is None:
        print("ContactSensor binding not found in known module paths.")
        return None

    print(f"binding_source: {source_name}")
    print(f"ContactSensor: {contact_sensor_cls}")
    try:
        print(f"ContactSensor.signature: {inspect.signature(contact_sensor_cls)}")
    except Exception as exc:  # noqa: BLE001
        print(f"ContactSensor.signature failed: {exc}")
        print(f"ContactSensor.dir: {dir(contact_sensor_cls)}")

    acquire_fn = getattr(sensor_module, "acquire_contact_sensor_interface", None)
    print(f"acquire_contact_sensor_interface: {acquire_fn}")
    if acquire_fn is None:
        return None

    try:
        interface = acquire_fn()
        print(f"interface: {interface}")
        interesting_methods = [
            name
            for name in dir(interface)
            if any(key in name.lower() for key in ("raw", "contact", "body", "decode", "sensor"))
        ]
        print(f"interface_methods({len(interesting_methods)}): {interesting_methods}")
        return interface
    except Exception as exc:  # noqa: BLE001
        print(f"acquire_contact_sensor_interface failed: {exc}")
        return None


def dump_command_registry() -> None:
    """探测 Contact Sensor 相关的 Omniverse command 名称。"""
    print_header("Omni Commands")
    try:
        import omni.kit.commands

        command_names = omni.kit.commands.get_commands_list()
        interesting = [
            str(name)
            for name in command_names
            if "contactsensor" in str(name).lower() or "contact" in str(name).lower() or "sensor" in str(name).lower()
        ]
        print(f"contact_related_commands({len(interesting)}): {interesting}")
    except Exception as exc:  # noqa: BLE001
        print(f"omni.kit.commands inspection failed: {exc}")


def dump_env_info() -> None:
    """当用户提供任务名时，创建 1 个环境并打印 robot/body/sensor 信息。"""
    if not args_cli.task:
        return

    print_header("Environment")
    print(f"task: {args_cli.task}")
    print(f"num_envs: {args_cli.num_envs}")

    env = None
    try:
        import gymnasium as gym

        # 导入任务注册，确保 gym registry 里能找到 robot_lab 的任务。
        import robot_lab.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        # 设备优先使用命令行传入值；没传时退回 cuda:0，和训练脚本默认习惯一致。
        device = args_cli.device if args_cli.device is not None else "cuda:0"
        env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()
        unwrapped = env.unwrapped

        print(f"env_type: {type(unwrapped)}")
        print(f"scene_keys: {list(unwrapped.scene.keys())}")

        robot = unwrapped.scene["robot"]
        body_names = list(getattr(robot, "body_names", []))
        print(f"robot_type: {type(robot)}")
        print(f"robot_body_count: {len(body_names)}")
        print(f"robot_body_names: {body_names[: args_cli.max_body_names]}")

        # 重点检查四个 foot body 是否真的存在于当前 USD / articulation 里。
        target_foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        try:
            foot_ids, foot_names = robot.find_bodies(target_foot_names, preserve_order=True)
            print(f"robot.find_bodies(target_foot_names): ids={foot_ids}, names={foot_names}")
        except Exception as exc:  # noqa: BLE001
            print(f"robot.find_bodies(target_foot_names) failed: {exc}")

        # 打印所有 scene sensor 的名字、类型和常见 prim_path 信息，帮助后续定位 raw prim path。
        sensor_names = list(unwrapped.scene.sensors.keys())
        print(f"scene_sensor_names({len(sensor_names)}): {sensor_names}")
        for sensor_name in sensor_names:
            sensor = unwrapped.scene.sensors[sensor_name]
            cfg = getattr(sensor, "cfg", None)
            prim_path = getattr(cfg, "prim_path", None) if cfg is not None else None
            filter_paths = getattr(cfg, "filter_prim_paths_expr", None) if cfg is not None else None
            print(
                f"sensor[{sensor_name}]: type={type(sensor)}, prim_path={prim_path}, "
                f"filter_prim_paths_expr={safe_repr(filter_paths)}"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"environment inspection failed: {exc}")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass


def dump_raw_data(interface: Any | None) -> None:
    """当用户显式给出 sensor prim 路径时，尝试调用 get_contact_sensor_raw_data。"""
    print_header("Raw Data")
    print(f"raw_prim_path: {args_cli.raw_prim_path}")
    if interface is None:
        print("contact sensor interface unavailable; skipping raw-data inspection.")
        return
    if not args_cli.raw_prim_path:
        print("no --raw-prim-path provided; skipping raw-data inspection.")
        return

    raw_fn = getattr(interface, "get_contact_sensor_raw_data", None)
    if raw_fn is None:
        print("interface.get_contact_sensor_raw_data not found.")
        return

    try:
        raw_data = raw_fn(args_cli.raw_prim_path)
        print(f"raw_data_type: {type(raw_data)}")
        print(f"raw_data_len: {len(raw_data)}")
        if len(raw_data) > 0:
            first_item = raw_data[0]
            print(f"first_item_type: {type(first_item)}")
            print(f"first_item_dir: {dir(first_item)}")
            for field_name in ["time", "dt", "body0", "body1", "position", "normal", "impulse"]:
                try:
                    print(f"first_item.{field_name}: {getattr(first_item, field_name)}")
                except Exception as exc:  # noqa: BLE001
                    print(f"first_item.{field_name} failed: {exc}")
        else:
            print("raw_data is empty.")
    except Exception as exc:  # noqa: BLE001
        print(f"get_contact_sensor_raw_data failed: {exc}")


def main() -> None:
    """主入口：依次打印模块、API、命令、环境和 raw data 信息。"""
    dump_extension_state()
    try_enable_candidate_extensions()
    dump_extension_state()
    dump_pkgutil_candidates()
    imported_modules = dump_module_imports()
    dump_installation_scan(imported_modules)
    contact_sensor_cls, sensor_module, source_name = resolve_contact_sensor_bindings(imported_modules)
    interface = dump_contact_sensor_api(contact_sensor_cls, sensor_module, source_name)
    dump_command_registry()
    dump_env_info()
    dump_raw_data(interface)


if __name__ == "__main__":
    try:
        print("[TRACE] entering main()", flush=True)
        main()
        print("[TRACE] main() completed", flush=True)
    except Exception:  # noqa: BLE001
        print("[TRACE] unhandled exception follows", flush=True)
        traceback.print_exc(file=_REPORT_STREAM)
    finally:
        # 无论成功或失败，都显式关闭 Isaac Sim 应用，避免进程残留。
        print("[TRACE] entering finally", flush=True)
        simulation_app.close()
        print("[TRACE] simulation_app.close() completed", flush=True)
        try:
            _REPORT_STREAM.flush()
            os.fsync(_REPORT_STREAM.fileno())
        except Exception:  # noqa: BLE001
            pass
        try:
            _REPORT_STREAM.close()
        except Exception:  # noqa: BLE001
            pass
        sys.stdout = _ORIGINAL_STDOUT
        sys.stderr = _ORIGINAL_STDERR
