"""Adapter to McEval's own ``extract()`` (CPU-only; for the local smoke test).

McEval re-runs its per-language ``extract()`` on whatever we put in
``raw_generation[0]``. That function is pure regex/string (no language
toolchain), so we can import it and verify *locally* that our contract output
lands on McEval's happy path -- only the execution step needs Docker.

We import McEval's vendored ``eval/extract`` package by adding it to ``sys.path``
(it is not an installable package). Best-effort: raises if the vendored tree is
absent.
"""
from __future__ import annotations

import importlib
import sys
import warnings
from typing import Callable

from tsmc.config import ProjectPaths, get_paths


def mceval_extract_available(paths: ProjectPaths | None = None) -> bool:
    paths = paths or get_paths()
    return (paths.mceval_dir / "eval" / "extract" / "__init__.py").is_file()


def get_mceval_extract(paths: ProjectPaths | None = None) -> Callable[[str, dict, str], str | None]:
    """Return McEval's ``extract(text, item, lang)`` callable.

    Adds ``McEval/eval`` and ``McEval/eval/extract`` to ``sys.path``. Raises
    ``FileNotFoundError`` if the vendored extractor is missing.
    """
    paths = paths or get_paths()
    extract_dir = paths.mceval_dir / "eval" / "extract"
    eval_dir = paths.mceval_dir / "eval"
    if not (extract_dir / "__init__.py").is_file():
        raise FileNotFoundError(f"McEval extractor not found at {extract_dir}")
    for entry in (str(eval_dir), str(extract_dir)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # McEval has harmless invalid-escape warnings
        module = importlib.import_module("extract")
    return module.extract
