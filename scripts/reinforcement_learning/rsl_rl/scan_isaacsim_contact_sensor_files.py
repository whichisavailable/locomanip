"""Scan an Isaac Sim installation for contact-sensor related extensions and APIs.

This script does not launch Isaac Sim. It only scans the on-disk installation to answer:
1. Whether contact sensor related extensions exist on disk.
2. Which module / extension names are present in this installation.
3. Which files mention key raw-data APIs such as acquire_contact_sensor_interface.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path


CANDIDATE_EXTENSION_NAMES = [
    "isaacsim.sensors.physics",
    "omni.isaac.sensor",
    "isaacsim.sensor.physics",
]

API_NEEDLES = [
    "IsaacSensorCreateContactSensor",
    "get_contact_sensor_raw_data",
    "acquire_contact_sensor_interface",
    "decode_body_name",
    "ContactSensorInterface",
    "ContactSensor",
]

TEXT_SUFFIXES = {".py", ".toml", ".json", ".kit", ".md"}


def print_header(title: str) -> None:
    print("\n" + "=" * 24 + f" {title} " + "=" * 24)


def resolve_install_root(explicit_root: str | None) -> Path:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    isaacsim = importlib.import_module("isaacsim")
    isaacsim_file = Path(isaacsim.__file__).resolve()
    return isaacsim_file.parents[2]


def scan_extension_manifests(install_root: Path) -> None:
    print_header("Extension Manifests")
    manifests = list(install_root.rglob("extension.toml"))
    print(f"extension_toml_count: {len(manifests)}")

    matches: list[tuple[Path, str]] = []
    for manifest in manifests:
        try:
            text = manifest.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for candidate in CANDIDATE_EXTENSION_NAMES:
            if candidate in text:
                matches.append((manifest, candidate))

    print(f"matched_extension_manifests({len(matches)}):")
    for manifest, candidate in matches[:100]:
        print(f"  - candidate={candidate} :: {manifest}")


def scan_api_files(install_root: Path) -> None:
    print_header("API Needles")
    matches: list[tuple[Path, str]] = []

    for path in install_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for needle in API_NEEDLES:
            if needle in text:
                matches.append((path, needle))

    print(f"matched_api_files({len(matches)}):")
    for path, needle in matches[:200]:
        print(f"  - needle={needle} :: {path}")


def scan_candidate_dirs(install_root: Path) -> None:
    print_header("Directory Hints")
    hints = []
    for path in install_root.rglob("*"):
        if not path.is_dir():
            continue
        lowered = str(path).lower()
        if "sensor" in lowered or "contact" in lowered:
            hints.append(path)
    print(f"interesting_directories({len(hints)}):")
    for path in hints[:200]:
        print(f"  - {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Isaac Sim installation for contact sensor files.")
    parser.add_argument(
        "--install-root",
        type=str,
        default=None,
        help="Optional Isaac Sim install root. Defaults to the root inferred from import isaacsim.",
    )
    args = parser.parse_args()

    print_header("Install Root")
    install_root = resolve_install_root(args.install_root)
    print(f"install_root: {install_root}")
    print(f"exists: {install_root.exists()}")

    if not install_root.exists():
        raise FileNotFoundError(f"Install root does not exist: {install_root}")

    scan_extension_manifests(install_root)
    scan_api_files(install_root)
    scan_candidate_dirs(install_root)


if __name__ == "__main__":
    main()
