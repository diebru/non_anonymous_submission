#!/usr/bin/env python3
"""Contract <-> extractor smoke test (roadmap Phase 0). Trio: Python, C, Rust.

LOCAL (default): confirm our contract output survives McEval's own pure-regex
``extract()`` -- producing output identical to the gold reference -- for
Python + Rust (Family A) and C (Family B). No Docker needed.

--docker (SERVER): also write the contract result files and execute them in the
pinned McEval image, asserting ~100% pass.

Usage:
    python3 scripts/smoke_contract_extractor.py            # local extract check
    python3 scripts/smoke_contract_extractor.py --limit 0  # all trio problems
    python3 scripts/smoke_contract_extractor.py --docker [--digest sha256:...]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc import contract, mceval_data  # noqa: E402
from tsmc.config import get_paths  # noqa: E402
from tsmc.eval import results  # noqa: E402
from tsmc.eval.mceval_adapter import get_mceval_extract, mceval_extract_available  # noqa: E402


def _rich(items):
    return [
        it for it in items
        if it.get("entry_point") and it.get("test")
        and it.get("prompt") is not None and it.get("canonical_solution") is not None
    ]


def run_local(limit: int):
    """Returns (failures, per_lang_stats, contract_items_by_lang)."""
    paths = get_paths()
    if not mceval_extract_available(paths):
        print("McEval extractor is not vendored; cannot run the local extract check.")
        return 1, {}, {}
    extract = get_mceval_extract(paths)

    failures = 0
    per_lang: dict[str, tuple[int, int]] = {}
    contract_items: dict[str, list] = {}
    for lang in results.TRIO:
        items = _rich(mceval_data.load_generation_language(lang, paths))
        items = items[:limit] if limit else items
        ok = 0
        built = []
        for it in items:
            out = results.synthetic_contract_output(it, lang)
            parsed = contract.parse_generation(out, entry_point=it["entry_point"], finish_reason="stop")
            raw = results.wrap_code(parsed.code_snippet, lang)
            contract_extract = extract(raw, it, lang)
            gold_extract = extract(results.gold_raw_generation(it, lang), it, lang)
            good = (
                parsed.status.parser_branch in ("sentinel", "multi_fence")
                and parsed.status.fence_found
                and parsed.status.entry_point_found
                and bool(contract_extract)
                and contract_extract == gold_extract
                and it["entry_point"] in contract_extract
            )
            ok += int(good)
            failures += int(not good)
            built.append(results.build_result_item(it, raw))
        per_lang[lang] = (ok, len(items))
        contract_items[lang] = built
    return failures, per_lang, contract_items


def run_docker(contract_items: dict[str, list], digest: str | None, network: str | None,
               python_exe: str, threshold: float) -> int:
    from tsmc.eval import docker

    paths = get_paths()
    digest = digest or docker.load_digest_from_metadata(paths)
    if not digest:
        print("No McEval Docker digest. Set mceval.docker_digest in "
              "configs/run_metadata.yaml or pass --digest sha256:...")
        return 1
    result_dir = paths.eval_dumps_dir / "phase0_smoke_contract"
    save_dir = paths.eval_dumps_dir / "phase0_smoke_contract_scores"
    results.write_result_dir(contract_items, result_dir)
    cfg = docker.DockerEvalConfig(digest=digest, network=network, python_exe=python_exe)
    print(f"Running McEval image {cfg.image_ref()} on contract outputs ...")
    proc = docker.run_eval(cfg, result_dir, save_dir, check=False)
    if proc.returncode != 0:
        print(proc.stdout[-2000:]); print(proc.stderr[-2000:])
        print("docker run failed."); return 1
    scores = docker.parse_scores(docker.save_file_for(result_dir, save_dir))
    overall = results.report_trio_execution(scores, threshold)
    print("DOCKER:", "PASS -- contract output executes (Python+C; Rust soft)"
          if overall else "FAIL -- a required language (Python/C) is below threshold")
    return 0 if overall else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=5, help="problems per language (0 = all)")
    ap.add_argument("--docker", action="store_true", help="also execute via the pinned image (server)")
    ap.add_argument("--digest", default=None, help="McEval image sha256 digest")
    ap.add_argument("--network", default=None, help="docker --network (e.g. none)")
    ap.add_argument("--python", default="/opt/conda/bin/python", help="in-container interpreter")
    ap.add_argument("--threshold", type=float, default=0.90)
    args = ap.parse_args()

    print("=" * 64)
    print("Contract <-> extractor smoke test (local)")
    print("=" * 64)
    failures, per_lang, contract_items = run_local(args.limit)
    for lang, (ok, n) in per_lang.items():
        mark = "PASS" if ok == n and n > 0 else "FAIL"
        print(f"  [{mark}] {lang}: {ok}/{n} contract-extract == gold-extract")
    print("LOCAL:", "PASS" if failures == 0 and per_lang else "FAIL")
    if failures or not per_lang:
        return 1

    if args.docker:
        print("\n" + "=" * 64)
        print("Execution via pinned McEval image (server)")
        print("=" * 64)
        return run_docker(contract_items, args.digest, args.network, args.python, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
