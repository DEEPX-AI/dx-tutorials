"""Shared path configuration for the DX tutorials.

How the DX-All Suite location is resolved (first match wins):

1. the ``DX_ALL_SUITE_DIR`` environment variable,
2. ``dx-tutorials/config.json`` (written by Tutorial 01 or ``--set``),
3. auto-detection of common install locations,
4. the default location ``~/dx-all-suite`` (reported as "not found").

Notebook usage (identical first code cell in every notebook)::

    from pathlib import Path; import importlib, sys
    sys.path.insert(0, str(next(p for p in [Path.cwd(), *Path.cwd().parents]
                                 if (p / "tutorial_paths.py").is_file())))
    import tutorial_paths; importlib.reload(tutorial_paths)
    from tutorial_paths import setup_tutorial
    paths = setup_tutorial(needs=["dx_com"])

Terminal usage::

    python tutorial_paths.py --show
    python tutorial_paths.py --set ~/my/dx-all-suite
    python tutorial_paths.py --detect

``%run tutorial_paths.py`` keeps working for older notebooks: it defines the
uppercase path variables (``DX_APP_DIR`` ...) in the notebook namespace.
"""

from __future__ import annotations

import argparse
import builtins
import glob
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT_MARKER = "tutorial_paths.py"
CONFIG_NAME = "config.json"
CONFIG_EXAMPLE_NAME = "config.example.json"
DEFAULT_SDK_DIR = Path("~/dx-all-suite")
DEFAULT_GIT_BRANCH = "main"
SDK_REMOTE_HINT = "DEEPX-AI/dx-all-suite"


class TutorialRequirementError(RuntimeError):
    """Raised by ``TutorialContext.require`` when a requirement is missing."""


# ---------------------------------------------------------------------------
# Repository root and configuration
# ---------------------------------------------------------------------------

def find_tutorial_root(start: str | Path | None = None) -> Path:
    """Find the dx-tutorials root (the directory containing tutorial_paths.py)."""
    candidates: list[Path] = []
    if start is not None:
        candidates.append(Path(start))
    if os.environ.get("ROOT_PATH"):
        candidates.append(Path(os.environ["ROOT_PATH"]))
    candidates.extend([Path.cwd(), *Path.cwd().parents, Path(__file__).parent])

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / ROOT_MARKER).is_file():
            return resolved

    raise FileNotFoundError(
        "Could not find the dx-tutorials root. Open the notebook from inside the "
        "dx-tutorials repository, or set the ROOT_PATH environment variable."
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    return data if isinstance(data, dict) else {}


def load_tutorial_config(tutorial_root: str | Path | None = None) -> dict[str, Any]:
    """Read config.json merged over config.example.json. Missing files are allowed."""
    root = find_tutorial_root(tutorial_root)
    config = _read_json(root / CONFIG_EXAMPLE_NAME)
    config.update(_read_json(root / CONFIG_NAME))
    config.setdefault("schema_version", 1)
    config.setdefault("git_branch", DEFAULT_GIT_BRANCH)
    return config


def _as_config_path(path: str | Path) -> str:
    """Store paths under $HOME in portable ``~/...`` form."""
    resolved = Path(path).expanduser().resolve()
    try:
        relative = resolved.relative_to(Path.home())
    except ValueError:
        return str(resolved)
    return "~" if relative == Path(".") else "~/" + relative.as_posix()


def save_tutorial_config(
    tutorial_root: str | Path | None = None,
    **updates: Any,
) -> Path:
    """Update config.json while preserving fields not included in updates."""
    root = find_tutorial_root(tutorial_root)
    config_path = root / CONFIG_NAME
    config = _read_json(config_path)
    config.setdefault("schema_version", 1)

    if "dx_all_suite_dir" in updates:
        updates["dx_all_suite_dir"] = _as_config_path(updates["dx_all_suite_dir"])

    config.update(updates)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# DX-All Suite location
# ---------------------------------------------------------------------------

def _non_empty_dir(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def is_dx_all_suite_dir(path: str | Path) -> bool:
    """A DX-All Suite checkout has populated dx-runtime/ and dx-compiler/ submodules.

    An interrupted clone leaves these as empty directories, which must not count.
    """
    directory = Path(path).expanduser()
    return _non_empty_dir(directory / "dx-runtime") and _non_empty_dir(directory / "dx-compiler")


def candidate_sdk_dirs(tutorial_root: str | Path | None = None) -> list[Path]:
    """Locations searched by auto-detection, in order."""
    root = find_tutorial_root(tutorial_root)
    home = Path.home()
    candidates = [
        DEFAULT_SDK_DIR.expanduser(),
        root.parent / "dx-all-suite",
        home / "Works" / "dx-all-suite",
        home / "works" / "dx-all-suite",
        Path("/opt/dx-all-suite"),
    ]
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def resolve_sdk_dir(tutorial_root: str | Path | None = None) -> tuple[Path, str]:
    """Return (dx_all_suite_dir, source) using the documented priority order."""
    env_value = os.environ.get("DX_ALL_SUITE_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve(), "environment variable DX_ALL_SUITE_DIR"

    # Only the user's own config.json counts here; config.example.json documents
    # the fields but must not stop auto-detection on a fresh clone.
    root = find_tutorial_root(tutorial_root)
    config_value = _read_json(root / CONFIG_NAME).get("dx_all_suite_dir")
    if config_value:
        return Path(config_value).expanduser().resolve(), CONFIG_NAME

    for candidate in candidate_sdk_dirs(tutorial_root):
        if is_dx_all_suite_dir(candidate):
            return candidate, "auto-detected"

    return DEFAULT_SDK_DIR.expanduser().resolve(), "default (not found yet)"


def sdk_git_info(dx_all_suite_dir: str | Path) -> dict[str, str]:
    """Best-effort git remote/branch/tag information for the SDK checkout.

    ``branch`` is the checked-out branch name, or "" for a detached HEAD.
    ``tag`` is the tag that exactly matches HEAD (for example after ``git checkout v2.4.2``), or "".
    """
    directory = Path(dx_all_suite_dir).expanduser()
    info = {"remote": "", "branch": "", "tag": ""}
    if not (directory / ".git").exists() or shutil.which("git") is None:
        return info
    for key, args in (
        ("remote", ["remote", "get-url", "origin"]),
        ("branch", ["branch", "--show-current"]),
        ("tag", ["describe", "--tags", "--exact-match"]),
    ):
        result = subprocess.run(
            ["git", "-C", str(directory), *args],
            text=True, capture_output=True, check=False,
        )
        if result.returncode == 0:
            info[key] = result.stdout.strip()
    return info


# ---------------------------------------------------------------------------
# Path objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TutorialPaths:
    """Shared SDK paths (kept for backward compatibility)."""

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


def load_tutorial_paths(tutorial_root: str | Path | None = None) -> TutorialPaths:
    """Load the base SDK path and derive all shared component paths."""
    root = find_tutorial_root(tutorial_root)
    config = load_tutorial_config(root)
    dx_all_suite_dir, _ = resolve_sdk_dir(root)
    dx_runtime_dir = dx_all_suite_dir / "dx-runtime"

    return TutorialPaths(
        tutorial_root=root,
        config_path=root / CONFIG_NAME,
        dx_all_suite_dir=dx_all_suite_dir,
        dx_runtime_dir=dx_runtime_dir,
        dx_app_dir=dx_runtime_dir / "dx_app",
        dx_rt_dir=dx_runtime_dir / "dx_rt",
        dx_stream_dir=dx_runtime_dir / "dx_stream",
        dx_compiler_dir=dx_all_suite_dir / "dx-compiler",
        dx_workspace_dir=dx_all_suite_dir / "workspace",
        git_branch=str(config.get("git_branch", DEFAULT_GIT_BRANCH)),
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


# ---------------------------------------------------------------------------
# Requirement checks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    hint: str = ""


def _check_sdk(ctx: "TutorialContext") -> CheckResult:
    ok = is_dx_all_suite_dir(ctx.dx_all_suite_dir)
    detail = str(ctx.dx_all_suite_dir)
    if ok:
        info = sdk_git_info(ctx.dx_all_suite_dir)
        if info["branch"]:
            detail += f"  (branch {info['branch']})"
        elif info["tag"]:
            detail += f"  (tag {info['tag']})"
        if info["remote"] and SDK_REMOTE_HINT not in info["remote"]:
            detail += f"  [remote is not {SDK_REMOTE_HINT}]"
    elif ctx.dx_all_suite_dir.exists():
        detail += "  (exists, but dx-runtime/ or dx-compiler/ is missing)"
    else:
        detail += "  (directory does not exist)"
    return CheckResult(
        "sdk", ok, detail,
        "Clone dx-all-suite (Tutorial 01, section 1) or point the tutorials to an "
        "existing checkout: python tutorial_paths.py --set <dir>",
    )


def _check_sdk_branch(ctx: "TutorialContext") -> CheckResult:
    directory = ctx.dx_all_suite_dir
    expected = ctx.git_branch
    if not (directory / ".git").exists():
        return CheckResult("sdk_branch", False, "no git repository yet", "Clone dx-all-suite first (Tutorial 01, section 1)")
    info = sdk_git_info(directory)
    current = info["branch"] or info["tag"]          # a tag checkout (detached HEAD) counts as well
    ok = bool(current) and current == expected
    detail = f"{current or 'detached HEAD'}" + ("" if ok else f"  (expected {expected})")
    hint = (
        f"git -C {directory} fetch --depth 1 origin {expected} && "
        f"git -C {directory} checkout {expected} && "
        f"git -C {directory} submodule update --init --recursive --depth 1"
    )
    return CheckResult("sdk_branch", ok, detail, hint)


def _check_sdk_submodules(ctx: "TutorialContext") -> CheckResult:
    directory = ctx.dx_all_suite_dir
    if not (directory / ".git").exists() or shutil.which("git") is None:
        return CheckResult("sdk_submodules", False, "no git repository yet", "Clone dx-all-suite first (Tutorial 01, section 1)")
    result = subprocess.run(
        ["git", "-C", str(directory), "submodule", "status", "--recursive"],
        text=True, capture_output=True, check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    missing = [line.split()[1] for line in lines if line.startswith("-") and len(line.split()) > 1]
    ok = result.returncode == 0 and not missing
    detail = f"{len(lines)} submodules initialized" if ok else f"not initialized: {', '.join(missing) or 'unknown'}"
    return CheckResult(
        "sdk_submodules", ok, detail,
        f"git -C {directory} submodule update --init --recursive --depth 1",
    )


def _check_dx_com(ctx: "TutorialContext") -> CheckResult:
    ok = ctx.dxcom_path.is_file()
    return CheckResult(
        "dx_com", ok, str(ctx.dxcom_path) if ok else f"not found: {ctx.dxcom_path}",
        "Run ./dx-compiler/install.sh --target=dx_com in the SDK directory (Tutorial 01, section 2.1)",
    )


def _check_calibration_dataset(ctx: "TutorialContext") -> CheckResult:
    directory = ctx.dx_com_dir / "calibration_dataset"
    ok = directory.is_dir() and any(directory.iterdir())
    return CheckResult(
        "calibration_dataset", ok,
        str(directory) if ok else f"not found or empty: {directory}",
        "Run <dx-all-suite>/dx-compiler/example/2-download_sample_calibration_dataset.sh",
    )


def _check_dx_rt(ctx: "TutorialContext") -> CheckResult:
    cli = shutil.which("dxcli") or shutil.which("dxrt-cli")
    return CheckResult(
        "dx_rt", cli is not None, cli or "dxcli / dxrt-cli not found in PATH",
        "Run ./dx-runtime/install.sh --all in the SDK directory (Tutorial 01, section 3)",
    )


def _check_npu(ctx: "TutorialContext") -> CheckResult:
    devices = sorted(glob.glob("/dev/dxrt*"))
    return CheckResult(
        "npu", bool(devices), ", ".join(devices) if devices else "no /dev/dxrt* device",
        "Check the NPU card and driver: lsmod | grep dxrt. A reboot is required after "
        "the driver installation (Tutorial 01, section 3.2)",
    )


def _check_gst_dxstream(ctx: "TutorialContext") -> CheckResult:
    """The DX-STREAM GStreamer plugin must be loadable; set GST_PLUGIN_PATH for this process if needed."""
    inspect = shutil.which("gst-inspect-1.0")
    if inspect is None:
        return CheckResult("gst_dxstream", False, "gst-inspect-1.0 not found", "sudo apt install gstreamer1.0-tools")

    def visible() -> bool:
        return subprocess.run([inspect, "dxstream"], capture_output=True, text=True).returncode == 0

    current = os.environ.get("GST_PLUGIN_PATH", "")
    if visible():
        return CheckResult("gst_dxstream", True, "plugin 'dxstream' loads" + (f" (GST_PLUGIN_PATH={current})" if current else ""))
    for directory in sorted(glob.glob("/usr/local/lib/*/gstreamer-1.0")) + ["/usr/local/lib/gstreamer-1.0"]:
        if glob.glob(directory + "/libgstdxstream.so"):
            os.environ["GST_PLUGIN_PATH"] = directory + (":" + current if current else "")
            if visible():
                return CheckResult("gst_dxstream", True, f"plugin 'dxstream' loads; GST_PLUGIN_PATH={directory} was set for this kernel")
    return CheckResult(
        "gst_dxstream", False, "plugin 'dxstream' is not visible to gst-inspect-1.0",
        "Build DX-STREAM (./dx-runtime/install.sh --all, or Tutorial 04 section 5.7) and run "
        "export GST_PLUGIN_PATH=<prefix>/lib/<arch>/gstreamer-1.0:$GST_PLUGIN_PATH before ./run-jupyter-lab.sh "
        "(build.sh prints the exact line)",
    )


def _check_dir(name: str, attr: str, hint: str) -> Callable[["TutorialContext"], CheckResult]:
    def check(ctx: "TutorialContext") -> CheckResult:
        directory: Path = getattr(ctx, attr)
        ok = directory.is_dir()
        return CheckResult(name, ok, str(directory) if ok else f"not found: {directory}", hint)
    return check


def _check_command(name: str, command: str, hint: str) -> Callable[["TutorialContext"], CheckResult]:
    def check(ctx: "TutorialContext") -> CheckResult:
        found = shutil.which(command)
        return CheckResult(name, found is not None, found or f"{command} not found in PATH", hint)
    return check


CHECKS: dict[str, Callable[["TutorialContext"], CheckResult]] = {
    "sdk": _check_sdk,
    "sdk_branch": _check_sdk_branch,
    "sdk_submodules": _check_sdk_submodules,
    "dx_com": _check_dx_com,
    "calibration_dataset": _check_calibration_dataset,
    "dx_rt": _check_dx_rt,
    "npu": _check_npu,
    "dx_app": _check_dir(
        "dx_app", "dx_app_dir",
        "Install DX-Runtime with --all (Tutorial 01, section 3)",
    ),
    "dx_stream": _check_dir(
        "dx_stream", "dx_stream_dir",
        "Install DX-Runtime with --all (Tutorial 01, section 3)",
    ),
    "gst_dxstream": _check_gst_dxstream,
    "dx_tron": _check_command(
        "dx_tron", "dxtron",
        "Run ./dx-compiler/install.sh --target=dx_tron in the SDK directory (Tutorial 01, section 2.3)",
    ),
    "cmake": _check_command("cmake", "cmake", "sudo apt install cmake"),
    "git": _check_command("git", "git", "sudo apt install git"),
    "uv": _check_command(
        "uv", "uv", "curl -LsSf https://astral.sh/uv/install.sh | sh (see README.md)",
    ),
}


# ---------------------------------------------------------------------------
# Tutorial context
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TutorialContext:
    """Everything a notebook needs to know about where things are."""

    tutorial_root: Path
    config_path: Path
    dx_all_suite_dir: Path
    sdk_source: str
    git_branch: str
    dx_runtime_dir: Path
    dx_app_dir: Path
    dx_rt_dir: Path
    dx_stream_dir: Path
    dx_compiler_dir: Path
    dx_com_dir: Path
    dx_compiler_venv: Path
    dxcom_path: Path
    dx_workspace_dir: Path
    tutorial_dir: Path
    workspace_dir: Path
    app_dir: Path
    assets_dir: Path
    needs: tuple[str, ...] = ()
    results: tuple[CheckResult, ...] = field(default_factory=tuple)

    @property
    def missing(self) -> list[CheckResult]:
        return [result for result in self.results if not result.ok]

    @property
    def ok(self) -> bool:
        return not self.missing

    def check(self, *needs: str) -> list[CheckResult]:
        """Run the named checks and return their results."""
        unknown = [need for need in needs if need not in CHECKS]
        if unknown:
            raise ValueError(f"Unknown requirement(s): {unknown}. Known: {sorted(CHECKS)}")
        return [CHECKS[need](self) for need in needs]

    def require(self, *needs: str) -> None:
        """Raise TutorialRequirementError if any named requirement is missing."""
        missing = [result for result in self.check(*needs) if not result.ok]
        if not missing:
            return
        lines = ["This tutorial cannot continue. Missing requirements:"]
        for result in missing:
            lines.append(f"  - {result.name}: {result.detail}")
            lines.append(f"      how to fix: {result.hint}")
        raise TutorialRequirementError("\n".join(lines))

    def as_notebook_vars(self) -> dict[str, Path | str]:
        """Uppercase names for notebook cells (superset of load_tutorial_path_vars)."""
        return {
            "TUTORIAL_ROOT": self.tutorial_root,
            "CONFIG_PATH": self.config_path,
            "DX_ALL_SUITE_DIR": self.dx_all_suite_dir,
            "DX_ALL_SUITE_BRANCH": self.git_branch,
            "DX_RUNTIME_DIR": self.dx_runtime_dir,
            "DX_APP_DIR": self.dx_app_dir,
            "DX_RT_DIR": self.dx_rt_dir,
            "DX_STREAM_DIR": self.dx_stream_dir,
            "DX_COMPILER_DIR": self.dx_compiler_dir,
            "DX_COM_DIR": self.dx_com_dir,
            "DX_COMPILER_VENV": self.dx_compiler_venv,
            "DXCOM_PATH": self.dxcom_path,
            "DX_WORKSPACE_DIR": self.dx_workspace_dir,
            "TUTORIAL_DIR": self.tutorial_dir,
            "WORKSPACE_DIR": self.workspace_dir,
            "APP_DIR": self.app_dir,
            "ASSETS_DIR": self.assets_dir,
        }

    def status_text(self) -> str:
        """Human-readable status table."""
        try:
            tutorial_label = str(self.tutorial_dir.relative_to(self.tutorial_root))
        except ValueError:
            tutorial_label = str(self.tutorial_dir)
        lines = [
            f"{'dx-tutorials root':<20} {self.tutorial_root}",
            f"{'DX-All Suite':<20} {self.dx_all_suite_dir}   from: {self.sdk_source}",
        ]
        for result in self.results:
            state = "OK" if result.ok else "MISSING"
            lines.append(f"  [{state:<7}] {result.name:<20} {result.detail}")
            if not result.ok and result.hint:
                lines.append(f"            {'':<20} -> {result.hint}")
        lines.append(f"{'This tutorial':<20} {tutorial_label}   (workspace: {self.workspace_dir.name}/)")
        return "\n".join(lines)


def _tutorial_dir_for(root: Path, start: Path | None) -> Path:
    """The notebooks/<Txx-...> directory that contains ``start`` (default: cwd)."""
    current = (start or Path.cwd()).resolve()
    notebooks = (root / "notebooks").resolve()
    try:
        relative = current.relative_to(notebooks)
    except ValueError:
        return current
    return notebooks / relative.parts[0] if relative.parts else current


def build_context(
    tutorial_root: str | Path | None = None,
    tutorial_dir: str | Path | None = None,
    needs: list[str] | tuple[str, ...] | None = None,
) -> TutorialContext:
    """Resolve all paths and run the requested checks (no printing)."""
    root = find_tutorial_root(tutorial_root)
    config = load_tutorial_config(root)
    dx_all_suite_dir, source = resolve_sdk_dir(root)
    dx_runtime_dir = dx_all_suite_dir / "dx-runtime"
    dx_compiler_dir = dx_all_suite_dir / "dx-compiler"
    dx_compiler_venv = dx_compiler_dir / "venv-dx-compiler-local"
    this_tutorial = _tutorial_dir_for(root, Path(tutorial_dir) if tutorial_dir else None)

    ctx = TutorialContext(
        tutorial_root=root,
        config_path=root / CONFIG_NAME,
        dx_all_suite_dir=dx_all_suite_dir,
        sdk_source=source,
        git_branch=str(config.get("git_branch", DEFAULT_GIT_BRANCH)),
        dx_runtime_dir=dx_runtime_dir,
        dx_app_dir=dx_runtime_dir / "dx_app",
        dx_rt_dir=dx_runtime_dir / "dx_rt",
        dx_stream_dir=dx_runtime_dir / "dx_stream",
        dx_compiler_dir=dx_compiler_dir,
        dx_com_dir=dx_compiler_dir / "dx_com",
        dx_compiler_venv=dx_compiler_venv,
        dxcom_path=dx_compiler_venv / "bin" / "dxcom",
        dx_workspace_dir=dx_all_suite_dir / "workspace",
        tutorial_dir=this_tutorial,
        workspace_dir=this_tutorial / "workspace",
        app_dir=this_tutorial / "app",
        assets_dir=this_tutorial / "assets",
    )

    requested = ["sdk"] + [need for need in (needs or []) if need != "sdk"]
    results = tuple(ctx.check(*requested))
    return TutorialContext(**{**ctx.__dict__, "needs": tuple(requested), "results": results})


def _ipython_namespace() -> dict[str, Any] | None:
    get_ipython = getattr(builtins, "get_ipython", None)
    if get_ipython is None:
        return None
    shell = get_ipython()
    return getattr(shell, "user_ns", None) if shell is not None else None


def setup_tutorial(
    needs: list[str] | tuple[str, ...] | None = None,
    *,
    tutorial_root: str | Path | None = None,
    tutorial_dir: str | Path | None = None,
    export_globals: bool = True,
    quiet: bool = False,
) -> TutorialContext:
    """Resolve paths, run checks, print a status table, and return the context.

    ``needs`` lists what this notebook requires, e.g. ``["dx_com"]`` or
    ``["dx_rt", "npu", "cmake"]``. Missing items are reported with a fix, not
    raised; call ``paths.require(...)`` where the notebook really needs them.
    With ``export_globals`` the uppercase names (``DX_APP_DIR`` ...) are also
    defined in the notebook namespace for older cells.
    """
    ctx = build_context(tutorial_root, tutorial_dir, needs)
    if export_globals:
        namespace = _ipython_namespace()
        if namespace is not None:
            namespace.update(ctx.as_notebook_vars())
    if not quiet:
        print(ctx.status_text())
    return ctx


# ---------------------------------------------------------------------------
# Downloads (shared by the tutorials that fetch Model Zoo files from Python)
# ---------------------------------------------------------------------------

def download_file(url: str, destination: "str | Path", chunk_size: int = 1024 * 1024) -> Path:
    """Download ``url`` to ``destination`` unless a non-empty file is already there.

    Rules:
      1. A non-empty destination is reused without any network request
         (delete the file to force a fresh download).
      2. A new download goes to ``<name>.part`` and replaces the destination only
         after the whole file arrived (checked against Content-Length when known),
         so an interrupted download never leaves a truncated final file.
    """
    from urllib.request import Request, urlopen

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Skip download: found {destination} ({destination.stat().st_size:,} bytes)")
        return destination
    if destination.is_file():
        print(f"Existing file is empty; downloading again: {destination}")

    with urlopen(Request(url, method="HEAD"), timeout=30) as response:
        expected_size = int(response.headers.get("Content-Length") or 0)

    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    downloaded = 0
    next_report = 10
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if expected_size:
                    percent = downloaded * 100 // expected_size
                    if percent >= next_report:
                        print(f"{destination.name}: {min(percent, 100)}%")
                        next_report += 10
        if expected_size and downloaded != expected_size:
            raise IOError(f"Incomplete download: expected {expected_size} bytes, received {downloaded}")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    print(f"Saved: {destination} ({downloaded:,} bytes)")
    return destination


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Show or change where the DX tutorials look for DX-All Suite.",
    )
    parser.add_argument("--set", metavar="DIR", help="save DIR as the DX-All Suite location")
    parser.add_argument("--branch", metavar="NAME", help="save the DX-All Suite git branch")
    parser.add_argument("--detect", action="store_true", help="list auto-detection candidates")
    parser.add_argument(
        "--needs", default="",
        help="comma-separated checks to run, e.g. dx_com,dx_rt,npu (exit 1 if missing)",
    )
    parser.add_argument("--show", action="store_true", help="print the resolved paths (default)")
    args = parser.parse_args(argv)

    updates: dict[str, Any] = {}
    if args.set:
        updates["dx_all_suite_dir"] = args.set
    if args.branch:
        updates["git_branch"] = args.branch
    if updates:
        config_path = save_tutorial_config(**updates)
        print(f"Saved {config_path}")

    if args.detect:
        print("Auto-detection candidates (first valid one is used):")
        for candidate in candidate_sdk_dirs():
            state = "valid" if is_dx_all_suite_dir(candidate) else "-"
            print(f"  [{state:<5}] {candidate}")
        print()

    needs = [need.strip() for need in args.needs.split(",") if need.strip()]
    ctx = setup_tutorial(needs=needs, export_globals=False)
    return 0 if ctx.ok else 1


def _running_in_ipython() -> bool:
    return getattr(builtins, "get_ipython", None) is not None


if __name__ == "__main__":
    if _running_in_ipython():
        # `%run tutorial_paths.py` in older notebooks: expose the uppercase variables.
        globals().update(load_tutorial_path_vars())
    else:
        sys.exit(main())
