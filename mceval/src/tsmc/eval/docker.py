"""Driver for the pinned McEval Docker image. SERVER-ONLY (needs Docker).

McEval's ``eval_all.py`` hardcodes ``/workspace/MMCodeEval/eval/tmp`` and resolves
``../data`` from CWD, so we run it *inside* the container (which ships the repo,
data, and all 40 language toolchains) rather than forking the path
(docs/WORKFLOW.md s3). We mount our result dir (read-only) and a save dir, then:

    cd /workspace/MMCodeEval/eval && python eval_all.py \
        --result_path /work/result --save_path /work/save

The image is pinned by **sha256 digest** (never a floating tag); the digest is
read from configs/run_metadata.yaml (mceval.docker_digest) or passed explicitly.

This module is import-safe on any machine (no Docker call at import time); only
``run_eval`` invokes ``docker``.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tsmc.config import ProjectPaths, get_paths

DEFAULT_IMAGE = "multilingualnlp/mceval"
CONTAINER_EVAL_DIR = "/workspace/MMCodeEval/eval"
_PLACEHOLDER_DIGESTS = {"", "sha256:TBD", "TBD"}


@dataclass
class DockerEvalConfig:
    digest: str  # "sha256:..." (preferred) or a tag
    image: str = DEFAULT_IMAGE
    # In-container interpreter for eval_all.py. The pinned image's McEval deps
    # (bs4, ...) live in its conda Python (3.8) at /opt/conda/bin/python; the bare
    # /usr/bin/python is Py2 and /usr/bin/python3 (3.6) lacks the deps. Override
    # with --python only if a different image moves it.
    python_exe: str = "/opt/conda/bin/python"
    network: str | None = None  # e.g. "none" to sandbox untrusted execution
    extra_docker_args: tuple[str, ...] = ()

    def image_ref(self) -> str:
        if self.digest.startswith("sha256:"):
            return f"{self.image}@{self.digest}"
        return f"{self.image}:{self.digest}"


def load_digest_from_metadata(paths: ProjectPaths | None = None) -> str | None:
    """Read ``mceval.docker_digest`` from configs/run_metadata.yaml, if present."""
    paths = paths or get_paths()
    meta_path = paths.configs_dir / "run_metadata.yaml"
    if not meta_path.is_file():
        return None
    with open(meta_path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    digest = (data.get("mceval") or {}).get("docker_digest")
    return digest if digest not in _PLACEHOLDER_DIGESTS else None


def save_file_for(result_dir: Path, save_dir: Path) -> Path:
    """eval_all.py writes ``<save_dir>/<basename(result_dir)>.jsonl``."""
    return save_dir / f"{result_dir.name}.jsonl"


def build_command(cfg: DockerEvalConfig, host_result_dir: Path, host_save_dir: Path) -> list[str]:
    # eval_all.py names its output <save>/<basename(result_path)>.jsonl, so the
    # in-container result path must keep the host dir name (else we read the wrong file).
    container_result = f"/work/{host_result_dir.name}"
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{host_result_dir}:{container_result}:ro",
        "-v", f"{host_save_dir}:/work/save",
    ]
    if cfg.network:
        cmd += ["--network", cfg.network]
    cmd += list(cfg.extra_docker_args)
    # Run via an INTERACTIVE shell (bash -ic): the image installs most language
    # toolchains through version managers (rustup ~/.cargo, SDKMAN kotlin, coursier
    # scala, ghcup, nvm, Flutter, Julia, Go) that add themselves to PATH only from
    # ~/.bashrc, which a non-interactive shell skips -> `go`/`cargo`/... not found.
    # We then prepend the conda env so the candidate `python` McEval spawns resolves
    # to conda Py3.8 (the bare /usr/bin/python is Py2).
    inner = (
        "export PATH=/opt/conda/bin:$PATH && "
        f"cd {CONTAINER_EVAL_DIR} && "
        f"{cfg.python_exe} eval_all.py --result_path {container_result} --save_path /work/save"
    )
    cmd += [cfg.image_ref(), "bash", "-ic", inner]
    return cmd


def run_eval(
    cfg: DockerEvalConfig,
    host_result_dir: Path,
    host_save_dir: Path,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run McEval evaluation in the container. SERVER-ONLY (invokes ``docker``)."""
    if cfg.digest in _PLACEHOLDER_DIGESTS:
        raise ValueError(
            "McEval Docker digest is unset/placeholder. Set mceval.docker_digest "
            "in configs/run_metadata.yaml (sha256:...) or pass --digest."
        )
    host_save_dir.mkdir(parents=True, exist_ok=True)
    # eval_all.py resumes by skipping languages already in the save file, so clear
    # a stale one to force a clean run.
    stale = save_file_for(host_result_dir, host_save_dir)
    if stale.exists():
        stale.unlink()
    cmd = build_command(cfg, host_result_dir.resolve(), host_save_dir.resolve())
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def parse_scores(save_file: Path) -> dict[str, dict[str, Any]]:
    """Parse eval_all.py's ``<lang>\\t<json score>`` output into a dict."""
    scores: dict[str, dict[str, Any]] = {}
    with open(save_file, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            lang, _, payload = line.partition("\t")
            scores[lang.strip()] = json.loads(payload)
    return scores


# --- per-problem detail eval (Phase 1.2) --------------------------------------
# Our committed shim tsmc/eval/detail_eval.py runs INSIDE the container and writes
# a per-task_id verdict alongside the aggregate, so we can filter correct
# trajectories. It is mounted read-only; McEval itself is never modified.
_DETAIL_SHIM = Path(__file__).resolve().parent / "detail_eval.py"


def detail_file_for(result_dir: Path, save_dir: Path) -> Path:
    """detail_eval.py writes ``<save_dir>/<basename(result_dir)>_detail.jsonl``."""
    return save_dir / f"{result_dir.name}_detail.jsonl"


def build_detail_command(
    cfg: DockerEvalConfig, host_result_dir: Path, host_save_dir: Path, host_shim: Path
) -> list[str]:
    container_result = f"/work/{host_result_dir.name}"
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{host_result_dir}:{container_result}:ro",
        "-v", f"{host_save_dir}:/work/save",
        "-v", f"{host_shim}:/work/detail_eval.py:ro",
    ]
    if cfg.network:
        cmd += ["--network", cfg.network]
    cmd += list(cfg.extra_docker_args)
    inner = (
        "export PATH=/opt/conda/bin:$PATH && "
        f"cd {CONTAINER_EVAL_DIR} && "
        f"{cfg.python_exe} /work/detail_eval.py "
        f"--result_path {container_result} --save_path /work/save"
    )
    cmd += [cfg.image_ref(), "bash", "-ic", inner]
    return cmd


def run_detail_eval(
    cfg: DockerEvalConfig,
    host_result_dir: Path,
    host_save_dir: Path,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run the per-problem detail eval in the container. SERVER-ONLY (invokes docker)."""
    if cfg.digest in _PLACEHOLDER_DIGESTS:
        raise ValueError(
            "McEval Docker digest is unset/placeholder. Set mceval.docker_digest "
            "in configs/run_metadata.yaml (sha256:...) or pass --digest."
        )
    host_save_dir.mkdir(parents=True, exist_ok=True)
    for stale in (save_file_for(host_result_dir, host_save_dir),
                  detail_file_for(host_result_dir, host_save_dir)):
        if stale.exists():
            stale.unlink()
    cmd = build_detail_command(
        cfg, host_result_dir.resolve(), host_save_dir.resolve(), _DETAIL_SHIM
    )
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def parse_detail(detail_file: Path) -> dict[str, bool]:
    """Parse the detail jsonl into ``{task_id: pass}``."""
    verdicts: dict[str, bool] = {}
    with open(detail_file, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            verdicts[row["task_id"]] = bool(row["pass"])
    return verdicts
