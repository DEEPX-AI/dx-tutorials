"""Shared path configuration for the DX tutorials."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TutorialPaths:
    tutorial_root: Path
    config_path: Path
    dx_all_suite_dir: Path
    dx_runtime_dir: Path
    dx_app_dir: Path
    dx_rt_dir: Path
    dx_stream_dir: Path
    dx_compiler_dir: Path
    dx_workspace_dir: Path
    git_branch: str


def find_tutorial_root(start: str | Path | None = None) -> Path:
    """Find the dx-tutorials root containing config.json."""
    candidates: list[Path] = []
    if start is not None:
        candidates.append(Path(start))
    if os.environ.get("ROOT_PATH"):
        candidates.append(Path(os.environ["ROOT_PATH"]))
    candidates.extend([Path.cwd(), *Path.cwd().parents, Path(__file__).parent])

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "config.json").is_file():
            return resolved

    raise FileNotFoundError("Could not find dx-tutorials/config.json")


def load_tutorial_config(tutorial_root: str | Path | None = None) -> dict[str, Any]:
    """Read the shared tutorial configuration."""
    root = find_tutorial_root(tutorial_root)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "dx_all_suite_dir" not in config:
        raise KeyError(f"Missing 'dx_all_suite_dir' in {config_path}")
    return config


def save_tutorial_config(
    tutorial_root: str | Path | None = None,
    **updates: Any,
) -> Path:
    """Update config.json while preserving fields not included in updates."""
    root = find_tutorial_root(tutorial_root)
    config_path = root / "config.json"
    config = load_tutorial_config(root)

    if "dx_all_suite_dir" in updates:
        resolved_path = Path(updates["dx_all_suite_dir"]).expanduser().resolve()
        try:
            relative_path = resolved_path.relative_to(Path.home())
            updates["dx_all_suite_dir"] = (
                "~" if relative_path == Path(".") else "~/" + relative_path.as_posix()
            )
        except ValueError:
            updates["dx_all_suite_dir"] = str(resolved_path)

    config.setdefault("schema_version", 1)
    config.update(updates)
    config_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    return config_path


def load_tutorial_paths(tutorial_root: str | Path | None = None) -> TutorialPaths:
    """Load the base SDK path and derive all shared component paths."""
    root = find_tutorial_root(tutorial_root)
    config = load_tutorial_config(root)
    dx_all_suite_dir = Path(config["dx_all_suite_dir"]).expanduser().resolve()
    dx_runtime_dir = dx_all_suite_dir / "dx-runtime"

    return TutorialPaths(
        tutorial_root=root,
        config_path=root / "config.json",
        dx_all_suite_dir=dx_all_suite_dir,
        dx_runtime_dir=dx_runtime_dir,
        dx_app_dir=dx_runtime_dir / "dx_app",
        dx_rt_dir=dx_runtime_dir / "dx_rt",
        dx_stream_dir=dx_runtime_dir / "dx_stream",
        dx_compiler_dir=dx_all_suite_dir / "dx-compiler",
        dx_workspace_dir=dx_all_suite_dir / "workspace",
        git_branch=str(config.get("git_branch", "")),
    )


def load_tutorial_path_vars(
    tutorial_root: str | Path | None = None,
) -> dict[str, Path | str]:
    """Return Notebook-friendly uppercase variables for all shared paths."""
    paths = load_tutorial_paths(tutorial_root)
    return {
        "TUTORIAL_ROOT": paths.tutorial_root,
        "CONFIG_PATH": paths.config_path,
        "DX_ALL_SUITE_DIR": paths.dx_all_suite_dir,
        "DX_RUNTIME_DIR": paths.dx_runtime_dir,
        "DX_APP_DIR": paths.dx_app_dir,
        "DX_RT_DIR": paths.dx_rt_dir,
        "DX_STREAM_DIR": paths.dx_stream_dir,
        "DX_COMPILER_DIR": paths.dx_compiler_dir,
        "DX_WORKSPACE_DIR": paths.dx_workspace_dir,
        "DX_ALL_SUITE_BRANCH": paths.git_branch,
    }


def print_tutorial_paths(tutorial_root: str | Path | None = None) -> None:
    """Print the shared SDK paths resolved from config.json."""
    values = load_tutorial_path_vars(tutorial_root)
    for name in (
        "DX_ALL_SUITE_DIR",
        "DX_RUNTIME_DIR",
        "DX_APP_DIR",
        "DX_RT_DIR",
        "DX_STREAM_DIR",
        "DX_COMPILER_DIR",
        "DX_WORKSPACE_DIR",
    ):
        print(f"{name:22} {values[name]}")


# `%run <dx-tutorials>/tutorial_paths.py` exposes all paths in a Notebook at once.
globals().update(load_tutorial_path_vars())
