"""Configuration loading and path resolution.

No hardcoded local paths: every filesystem location is resolved from, in order
of precedence, (1) an environment variable, (2) a YAML config file, or (3) a
deterministic default relative to the auto-detected repo root. The same code
therefore runs unchanged on a laptop (CPU-only tests) and on the GPU server.
See docs/WORKFLOW.md s4 ("No hardcoded local paths").

Config file discovery (first existing wins):
    1. path passed explicitly to load_config()
    2. $TSMC_CONFIG
    3. <repo_root>/configs/paths.yaml        (machine-specific, gitignored)
    4. <repo_root>/configs/paths.example.yaml (committed fallback)

Environment overrides: TSMC_REPO_ROOT, TSMC_DATA_ROOT.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ENV_CONFIG = "TSMC_CONFIG"        # explicit path to a paths.yaml
ENV_REPO_ROOT = "TSMC_REPO_ROOT"  # override repo-root detection
ENV_DATA_ROOT = "TSMC_DATA_ROOT"  # override the bulk-artifact root

_REPO_MARKERS = (".git", "pyproject.toml")


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root by walking up to a known marker.

    Overridable with $TSMC_REPO_ROOT so the package works when installed
    out-of-tree on the server.
    """
    env = os.environ.get(ENV_REPO_ROOT)
    if env:
        return Path(env).expanduser().resolve()
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if any((parent / marker).exists() for marker in _REPO_MARKERS):
            return parent
    # Fallback: src/tsmc/config.py -> repo root is two levels up from src/.
    return Path(__file__).resolve().parents[2]


def _candidate_config_paths(repo_root: Path) -> list[Path]:
    candidates: list[Path] = []
    env = os.environ.get(ENV_CONFIG)
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(repo_root / "configs" / "paths.yaml")
    candidates.append(repo_root / "configs" / "paths.example.yaml")
    return candidates


def load_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the YAML path config. Returns a dict with an added ``_config_path`` key."""
    repo_root = find_repo_root()
    candidates = [Path(config_path)] if config_path else _candidate_config_paths(repo_root)
    for path in candidates:
        if path.is_file():
            with open(path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            data.setdefault("_config_path", str(path.resolve()))
            return data
    raise FileNotFoundError(
        "No config file found. Looked in: "
        + ", ".join(str(p) for p in candidates)
        + ". Copy configs/paths.example.yaml to configs/paths.yaml and edit it."
    )


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved, absolute project paths. Built by :func:`get_paths`."""

    repo_root: Path
    data_root: Path
    mceval_dir: Path
    mceval_data_dir: Path
    configs_dir: Path
    manifest_path: Path
    generations_dir: Path
    compressed_dir: Path
    sft_dir: Path
    weights_dir: Path
    eval_dumps_dir: Path

    @property
    def artifact_dirs(self) -> tuple[Path, ...]:
        """The five gitignored bulk-output directories (created by bootstrap)."""
        return (
            self.generations_dir,
            self.compressed_dir,
            self.sft_dir,
            self.weights_dir,
            self.eval_dumps_dir,
        )


def get_paths(config: dict[str, Any] | None = None) -> ProjectPaths:
    """Resolve all project paths from config + environment, relative to repo root."""
    cfg = config if config is not None else load_config()
    repo_root = find_repo_root()
    paths_cfg = cfg.get("paths") or {}

    def resolve(value: Any, default: Path) -> Path:
        if value is None or value == "":
            return Path(default)
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        return candidate.resolve()

    data_root_raw = os.environ.get(ENV_DATA_ROOT) or paths_cfg.get("data_root")
    data_root = resolve(data_root_raw, repo_root)

    mceval_dir = resolve(paths_cfg.get("mceval_dir"), repo_root / "McEval")
    mceval_data_dir = resolve(paths_cfg.get("mceval_data_dir"), mceval_dir / "data")
    manifest_path = resolve(
        paths_cfg.get("manifest_path"), repo_root / "manifest" / "split_manifest.csv"
    )

    def under_data(key: str, name: str) -> Path:
        return resolve(paths_cfg.get(key), data_root / name)

    return ProjectPaths(
        repo_root=repo_root,
        data_root=data_root,
        mceval_dir=mceval_dir,
        mceval_data_dir=mceval_data_dir,
        configs_dir=repo_root / "configs",
        manifest_path=manifest_path,
        generations_dir=under_data("generations_dir", "generations"),
        compressed_dir=under_data("compressed_dir", "compressed"),
        sft_dir=under_data("sft_dir", "sft"),
        weights_dir=under_data("weights_dir", "weights"),
        eval_dumps_dir=under_data("eval_dumps_dir", "eval_dumps"),
    )


def ensure_dirs(paths: ProjectPaths) -> list[Path]:
    """Create the bulk-artifact directories if absent. Returns those created."""
    created: list[Path] = []
    for directory in paths.artifact_dirs:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)
    return created
