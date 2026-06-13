#!/usr/bin/env python3
"""Phase-0 data-side empirical checks (CPU-only; runnable locally).

Resolves the "open Phase-0 empirical checks" that can be answered from the
shipped McEval data alone (docs/PROJECT_ROADMAP.md "Open Phase-0 empirical
checks"), and guards the verified numbers as regressions:

  1. Sentinel-collision scan   - confirm the contract sentinel never appears in
                                 any McEval field (Decision #1 is safe).
  2. Base-problem overlap      - membership Venn over base problems, raw vs
                                 language-case-normalized (surfaces the sql/SQL
                                 casing issue; see docs/phase0_findings.md).
  3. Difficulty distribution   - generation `level` counts; identify the
                                 unlabeled base problems needing a proxy.
  4. Difficulty-proxy calib.   - solution-LOC tertiles per level + global cutoffs
                                 (informational; feeds the Task 0.4 proxy).

The gamma-convention check is NOT here: it needs the TokenSkip runtime (server,
tokenskip_env), not the dataset. It stays open until that environment exists.

Exit code is non-zero if any HARD invariant fails (so this doubles as a CI-style
guard). Usage:
    python3 scripts/phase0_empirical_checks.py [--json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from collections import Counter

# Allow running from a fresh checkout without `pip install -e .`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc import constants as C  # noqa: E402
from tsmc import mceval_data as M  # noqa: E402
from tsmc.config import get_paths  # noqa: E402

# Verified expected values (regression guards). HARD ones fail the run.
EXPECTED = {
    "gen_rows": 2007,
    "expl_rows": 2066,
    "compl_merge_rows": 10128,
    "compl_subtype_rows": {"single": 2998, "multi": 2998, "span": 4132},
    "distinct_base_raw": 2125,
    "distinct_base_norm": 2066,
    "core_all_three": 2007,
    "sql_shared": 59,
    "difficulty": {"easy": 1221, "middle": 401, "hard": 385},
    "sentinel_hits": 0,
    "langs_norm_union": 41,
}


def _nonblank_loc(text: str) -> int:
    return sum(1 for line in (text or "").splitlines() if line.strip())


def run_checks() -> dict:
    paths = get_paths()
    gen = M.load_generation(paths)
    expl = M.load_explanation(paths)
    compl = M.load_completion(paths, subset="merge")
    compl_sub = {
        s: M.load_completion(paths, subset=s) for s in ("single", "multi", "span")
    }

    results: dict = {"checks": {}, "failures": []}

    def record(name: str, value, expected, hard: bool):
        ok = value == expected
        results["checks"].setdefault(name, {})
        results["checks"][name] = {"value": value, "expected": expected, "ok": ok, "hard": hard}
        if not ok and hard:
            results["failures"].append(name)
        return ok

    # ---- Check 1: sentinel collision --------------------------------------
    hits = 0
    fields = 0
    for rows in (gen, expl, compl):
        for row in rows:
            for _key, value in M.iter_string_fields(row):
                fields += 1
                if C.SENTINEL in value:
                    hits += 1
    record("sentinel_hits", hits, EXPECTED["sentinel_hits"], hard=True)
    results["sentinel_fields_scanned"] = fields

    # ---- row counts --------------------------------------------------------
    record("gen_rows", len(gen), EXPECTED["gen_rows"], hard=True)
    record("expl_rows", len(expl), EXPECTED["expl_rows"], hard=True)
    record("compl_merge_rows", len(compl), EXPECTED["compl_merge_rows"], hard=True)
    record(
        "compl_subtype_rows",
        {s: len(v) for s, v in compl_sub.items()},
        EXPECTED["compl_subtype_rows"],
        hard=True,
    )

    # ---- Check 2: base-problem overlap ------------------------------------
    gen_raw = {M.base_problem_id(r["task_id"]) for r in gen}
    expl_raw = {M.base_problem_id(r["task_id"]) for r in expl}
    compl_raw = {M.base_problem_id(r["task_id"]) for r in compl}
    record("distinct_base_raw", len(gen_raw | expl_raw | compl_raw), EXPECTED["distinct_base_raw"], hard=True)

    gen_n = {M.canonical_base_id(r["task_id"]) for r in gen}
    expl_n = {M.canonical_base_id(r["task_id"]) for r in expl}
    compl_n = {M.canonical_base_id(r["task_id"]) for r in compl}
    union_n = gen_n | expl_n | compl_n
    record("distinct_base_norm", len(union_n), EXPECTED["distinct_base_norm"], hard=True)

    membership = Counter()
    for base in union_n:
        tag = "+".join(
            t for t, s in (("gen", gen_n), ("expl", expl_n), ("compl", compl_n)) if base in s
        )
        membership[tag] += 1
    results["membership_norm"] = dict(membership)
    record("core_all_three", membership.get("gen+expl+compl", 0), EXPECTED["core_all_three"], hard=True)
    record("sql_shared", membership.get("expl+compl", 0), EXPECTED["sql_shared"], hard=True)

    # subset relations (normalized)
    results["gen_subset_expl_norm"] = gen_n <= expl_n
    results["gen_subset_compl_norm"] = gen_n <= compl_n
    if not (gen_n <= expl_n and gen_n <= compl_n):
        results["failures"].append("gen_subset_relation")

    # casing evidence (raw)
    results["sql_casing"] = {
        "explanation_only_langs": sorted({b.split("/")[0] for b in (expl_raw - gen_raw - compl_raw)}),
        "completion_only_langs": sorted({b.split("/")[0] for b in (compl_raw - gen_raw - expl_raw)}),
    }

    # ---- languages ---------------------------------------------------------
    langs_norm = {b.split("/")[0] for b in union_n}
    record("langs_norm_union", len(langs_norm), EXPECTED["langs_norm_union"], hard=True)
    results["langs_raw_union"] = len({b.split("/")[0] for b in (gen_raw | expl_raw | compl_raw)})

    # ---- Check 3: difficulty ----------------------------------------------
    diff = Counter(r.get("level") for r in gen)
    record("difficulty", {k: diff[k] for k in ("easy", "middle", "hard")}, EXPECTED["difficulty"], hard=True)
    labeled = {M.canonical_base_id(r["task_id"]) for r in gen if r.get("level")}
    unlabeled = sorted(union_n - labeled)
    results["unlabeled_base_count"] = len(unlabeled)
    results["unlabeled_langs"] = sorted({b.split("/")[0] for b in unlabeled})

    # ---- Check 4: difficulty-proxy calibration (informational) ------------
    per_level_loc: dict[str, list[int]] = {"easy": [], "middle": [], "hard": []}
    for r in gen:
        lvl = r.get("level")
        if lvl in per_level_loc:
            per_level_loc[lvl].append(_nonblank_loc(r.get("canonical_solution", "")))
    calib = {}
    for lvl, vals in per_level_loc.items():
        vals_sorted = sorted(vals)
        q = statistics.quantiles(vals_sorted, n=4)
        calib[lvl] = {"n": len(vals), "median_loc": statistics.median(vals_sorted),
                      "p25_loc": round(q[0], 1), "p75_loc": round(q[2], 1)}
    all_loc = sorted(v for vs in per_level_loc.values() for v in vs)
    tert = statistics.quantiles(all_loc, n=3)  # p33, p66 cutoffs
    results["loc_calibration"] = {
        "per_level": calib,
        "global_tertile_cutoffs_loc": [round(tert[0], 1), round(tert[1], 1)],
        "global_p25_50_75": [round(x, 1) for x in statistics.quantiles(all_loc, n=4)],
    }
    # SQL (unlabeled) LOC from explanation, to see where the proxy would place them
    sql_loc = sorted(
        _nonblank_loc(r.get("canonical_solution", ""))
        for r in expl
        if M.canonical_base_id(r["task_id"]) in set(unlabeled)
    )
    if sql_loc:
        results["unlabeled_loc"] = {
            "n": len(sql_loc),
            "p25_50_75": [round(x, 1) for x in statistics.quantiles(sql_loc, n=4)],
        }
    return results


def print_report(r: dict) -> None:
    c = r["checks"]

    def line(name: str, extra: str = ""):
        chk = c[name]
        mark = "PASS" if chk["ok"] else ("FAIL" if chk["hard"] else "warn")
        print(f"  [{mark}] {name}: {chk['value']}  (expected {chk['expected']}) {extra}")

    print("=" * 72)
    print("Phase-0 empirical checks (data-side)")
    print("=" * 72)

    print("\n[1] Sentinel-collision scan")
    line("sentinel_hits", f"-- scanned {r['sentinel_fields_scanned']:,} string fields")

    print("\n[ ] Row counts")
    for name in ("gen_rows", "expl_rows", "compl_merge_rows", "compl_subtype_rows"):
        line(name)

    print("\n[2] Base-problem overlap")
    line("distinct_base_raw", "<- raw task_id (sql vs SQL split)")
    line("distinct_base_norm", "<- language-case-normalized (canonical)")
    line("core_all_three")
    line("sql_shared", "<- shared SQL base problems (expl+compl, no generation)")
    print(f"       membership (normalized): {r['membership_norm']}")
    print(f"       gen subset of expl/compl (norm): {r['gen_subset_expl_norm']}/{r['gen_subset_compl_norm']}")
    print(f"       casing: explanation-only langs={r['sql_casing']['explanation_only_langs']}, "
          f"completion-only langs={r['sql_casing']['completion_only_langs']}")
    line("langs_norm_union", f"(raw union {r['langs_raw_union']})")

    print("\n[3] Difficulty distribution")
    line("difficulty")
    print(f"       unlabeled base problems (need proxy): {r['unlabeled_base_count']} "
          f"-> langs {r['unlabeled_langs']}")

    print("\n[4] Difficulty-proxy calibration (informational; nonblank solution LOC)")
    cal = r["loc_calibration"]
    for lvl in ("easy", "middle", "hard"):
        pl = cal["per_level"][lvl]
        print(f"       {lvl:6s}: n={pl['n']:4d}  median={pl['median_loc']}  p25/p75={pl['p25_loc']}/{pl['p75_loc']}")
    print(f"       global p25/50/75 LOC : {cal['global_p25_50_75']}")
    print(f"       proposed tertile cuts: {cal['global_tertile_cutoffs_loc']} (p33,p66)")
    if "unlabeled_loc" in r:
        print(f"       unlabeled(SQL) LOC p25/50/75: {r['unlabeled_loc']['p25_50_75']} (n={r['unlabeled_loc']['n']})")

    print("\n" + "=" * 72)
    if r["failures"]:
        print(f"RESULT: FAIL -- hard invariants failed: {r['failures']}")
    else:
        print("RESULT: PASS -- all hard invariants hold")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    results = run_checks()
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print_report(results)
    return 1 if results["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
