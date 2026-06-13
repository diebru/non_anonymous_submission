#!/usr/bin/env python3
"""Verify the pinned McEval Docker image on canonical solutions. SERVER-ONLY.

Builds GOLD result files for the trio (Python, C, Rust) -- raw_generation[0] =
the reference program (prompt + canonical_solution) in a fenced block -- runs
McEval's eval_all.py inside the pinned image, and asserts the canonical pass rate
is ~100%. This confirms the image, toolchains, extractor, and executor all work
end to end before we trust any real run.

The image is pinned by sha256 digest, read from configs/run_metadata.yaml
(mceval.docker_digest) or passed via --digest.

Usage (on the server, after `git pull`):
    python3 scripts/verify_mceval_docker.py --digest sha256:<...>
    python3 scripts/verify_mceval_docker.py --limit 0     # all trio problems
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc import mceval_data  # noqa: E402
from tsmc.config import get_paths  # noqa: E402
from tsmc.eval import docker, results  # noqa: E402


def _rich(items):
    return [
        it for it in items
        if it.get("entry_point") and it.get("test")
        and it.get("prompt") is not None and it.get("canonical_solution") is not None
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=10, help="problems per language (0 = all)")
    ap.add_argument("--digest", default=None, help="McEval image sha256 digest")
    ap.add_argument("--network", default=None, help="docker --network (e.g. none)")
    ap.add_argument("--python", default="/opt/conda/bin/python", help="in-container interpreter")
    ap.add_argument("--langs", default="trio",
                    help="'trio' (Python/C/Rust gate), 'all' (every generation language -> "
                         "per-language scoring-health map), or a comma list")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="min canonical accuracy (~0.9 ceiling: McEval's own extractor "
                         "mis-reconstructs a few problems)")
    args = ap.parse_args()

    paths = get_paths()
    digest = args.digest or docker.load_digest_from_metadata(paths)
    if not digest:
        print("No McEval Docker digest. Set mceval.docker_digest in "
              "configs/run_metadata.yaml or pass --digest sha256:...")
        return 1

    if args.langs == "trio":
        langs = list(results.TRIO)
    elif args.langs == "all":
        langs = results.discover_generation_languages(paths)
    else:
        langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    print(f"Languages ({len(langs)}): {langs}")

    # Build GOLD result files (raw_generation[0] = reference program, fenced).
    items_by_lang: dict[str, list] = {}
    for lang in langs:
        try:
            problems = _rich(mceval_data.load_generation_language(lang, paths))
        except FileNotFoundError:
            print(f"  {lang}: no generation file -> skip")
            continue
        problems = problems[:args.limit] if args.limit else problems
        if not problems:
            print(f"  {lang}: 0 rich gold items -> skip (reduced language)")
            continue
        items_by_lang[lang] = [
            results.build_result_item(it, results.gold_raw_generation(it, lang))
            for it in problems
        ]
        print(f"  {lang}: {len(items_by_lang[lang])} gold items")

    tag = "trio" if args.langs == "trio" else ("all" if args.langs == "all" else "custom")
    result_dir = paths.eval_dumps_dir / f"verify_gold_{tag}"
    save_dir = paths.eval_dumps_dir / f"verify_gold_{tag}_scores"
    results.write_result_dir(items_by_lang, result_dir)

    cfg = docker.DockerEvalConfig(digest=digest, network=args.network, python_exe=args.python)
    print(f"\nRunning McEval image {cfg.image_ref()} over {len(items_by_lang)} languages ...")
    # trio gate keeps the proven aggregate path; the all/custom map uses the ROBUST
    # per-language detail eval so one missing toolchain can't abort the whole sweep.
    if args.langs == "trio":
        proc = docker.run_eval(cfg, result_dir, save_dir, check=False)
    else:
        proc = docker.run_detail_eval(cfg, result_dir, save_dir, check=False)
    if proc.returncode != 0:
        print(proc.stdout[-2000:]); print(proc.stderr[-2000:])
        print("docker run failed.")
        return 1

    scores = docker.parse_scores(docker.save_file_for(result_dir, save_dir))
    print("\n" + "=" * 56)
    if args.langs == "trio":
        overall = results.report_trio_execution(scores, args.threshold)
        print("=" * 56)
        print("RESULT:", "PASS -- canonical pass rate OK (Python+C; Rust soft)"
              if overall else "FAIL -- a required language (Python/C) is below threshold")
        return 0 if overall else 1
    # all / custom: per-language scoring-health map (gold)
    results.report_gold_languages(scores, [l for l in langs if l in items_by_lang], args.threshold)
    print("=" * 56)
    print("Gold scoring-health map complete. 'ZERO'/'LOW' languages need a scoring "
          "handler before their MODEL scores can be trusted; 'OK' = ceiling established.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
