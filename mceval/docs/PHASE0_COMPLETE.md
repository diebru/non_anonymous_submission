# Phase 0 — Foundations: completion report

**Status: ✅ COMPLETE and validated end-to-end** (local + server).
Last commit at completion: `06b1379`. Test suite: **79 passing** (stdlib `unittest`).

This document is the self-contained record of everything Phase 0 produced, so the
work is recoverable without the build conversation. Companions:
[`PROJECT_ROADMAP.md`](PROJECT_ROADMAP.md) (design source of truth),
[`phase0_findings.md`](phase0_findings.md) (empirical results in detail),
[`WORKFLOW.md`](WORKFLOW.md) (local→push→server-pull→exec rules).

---

## 1. Goal of Phase 0
Stand up the foundations so later phases can run: a config-driven repo, the
McEval data understood and verified, the CoT/code **contract** and the result
**schema** frozen, a leakage-proof **train/test manifest**, and the **McEval
Docker** evaluator proven to work on a validation trio. No model inference yet.

All of that is done and committed.

---

## 2. What was built (per task)

| Task | Deliverable | Where | Status |
|---|---|---|---|
| **0.1 Repo skeleton** | Config-driven `tsmc` package, no hardcoded paths, CPU-testable; env/metadata templates; artifact-dir bootstrap | `src/tsmc/`, `configs/`, `scripts/{show_config,bootstrap_dirs}.py`, `pyproject.toml` | ✅ |
| **0.2 Empirical checks** | Sentinel-collision scan, base-problem overlap, difficulty distribution + proxy calibration; regression-guarded | `src/tsmc/mceval_data.py`, `scripts/phase0_empirical_checks.py` | ✅ |
| **0.3 Freeze contract + schema** | CoT/code parsing (sentinel, 3-way outcome) + γ-marker; long-format record + validator | `src/tsmc/contract/`, `src/tsmc/schema/` | ✅ frozen |
| **0.4 Manifest generator** | Stratified 80/20 base-problem split, leakage-proof; committed CSV; distributional gate | `src/tsmc/manifest/`, `scripts/build_manifest.py`, `manifest/split_manifest.csv` | ✅ provisionally frozen |
| **0.5 McEval Docker + contract↔extractor** | Result-file builder, McEval-extract adapter, Docker driver; gold + contract verification | `src/tsmc/eval/`, `scripts/{verify_mceval_docker,smoke_contract_extractor}.py` | ✅ validated on server |

### Package map (`src/tsmc/`)
```
config.py        path/config resolution (env > paths.yaml > example); ProjectPaths
constants.py     FROZEN decision-sheet values (sentinel, 12-γ grid, seed 42, enums, model matrix)
mceval_data.py   read generation/explanation/completion; canonical_base_id (sql↔SQL fix)
contract/        prompt.py (gamma_marker, directives) + parser.py (parse_*, three_way_outcome)
schema/          LongFormatRecord, ExtractionStatus, validate_record
manifest/        build_base_problems, assign_splits, validate_manifest, summarize
eval/            results.py (gold/contract builders), mceval_adapter.py, docker.py (SERVER-ONLY)
inference/ compression/ sft/   documented placeholders for Phases 1/2/3 (server-only)
```

### Scripts (`scripts/`, all CPU-runnable except the Docker ones)
- `show_config.py`, `bootstrap_dirs.py` — config sanity / artifact dirs
- `phase0_empirical_checks.py` — the data-side checks (exit 0 = invariants hold)
- `build_manifest.py` — build / `--check` / `--dry-run` the manifest
- `verify_mceval_docker.py` *(server)* — gold canonical pass on the trio
- `smoke_contract_extractor.py` — local extract check + `--docker` execution *(server)*

### Tests (`tests/`, 79 total, stdlib `unittest`)
`test_config.py` (8) · `test_mceval_data.py` (9) · `test_contract.py` (22) ·
`test_schema.py` (15) · `test_manifest.py` (11) · `test_eval.py` (14).
Run: `python3 -m unittest discover -s tests -v`.

---

## 3. Frozen decisions (and where they live in code)
- **Sentinel** `@@@FINAL_CODE_7F3A9@@@` — `constants.SENTINEL` (0 collisions in 126k McEval fields).
- **γ grid** (12, fraction *retained*): 1.0, .95, .9, .85, .8, .7, .6, .5, .4, .3, .2, .1 — `constants.GAMMA_GRID`.
- **γ-marker** `<|eot_id|>{γ}<|eot_id|>`, omitted at γ=1.0 — `contract.gamma_marker` (grounded in TokenSkip; byte-identical SFT↔inference).
- **Three-way outcome** format_fail / exec_fail / pass — `contract.three_way_outcome` (keeps extraction failure ≠ reasoning failure — the central concavity confound).
- **Long-format schema** (one row per problem×task×subtype×model×γ×run) — `schema.validate_record`.
- **Split**: 80/20 on the **canonical (lower-cased) base id**, seed 42 — `manifest/split_manifest.csv` (the frozen manifest is authoritative, not the seed).
- **Model matrix** (Qwen-only): 3B-Instruct ↔ Coder-3B (controlled pair), Coder-3B→7B→14B (size axis) — `constants.MODEL_IDS`.
- **1 run per (model×γ×task)**; greedy temp 0.

---

## 4. Key findings (full detail in `phase0_findings.md`)
1. **SQL casing bug** — the 59 shared problems are `sql/N` (explanation) vs `SQL/N` (completion); raw count inflates to 2,125 and SQL would **leak across the split**. The manifest keys on the **case-normalized** id (`canonical_base_id`) → 2,066, leak-proof.
2. **SQL difficulty** — Decision #4's fallback signals fail (LOC constant=1; #test-cases constant=2 and non-discriminating). Used **docstring-length rank tertiles** internal to SQL (`derived_proxy`).
3. **McEval excludes SQL from execution** (`exclude_langs=['sql']`) → the 59 SQL are kept in the manifest but **execution-unscored** (generation has no SQL, so the primary result is unaffected).
4. **γ-convention resolved from TokenSkip source** — `compression_ratio` = LLMLingua-2 `rate` = fraction **retained**, matching our grid.
5. **McEval execution-environment quirks (server)**: it runs candidate Python via `['python']` (Py2 on a non-interactive shell) → driver prepends `/opt/conda/bin` to PATH so `python` = conda Py3.8; **Rust** cold-recompiles deps per problem under a timeout → soft-gated; its **own extractor caps canonical pass at ~0.9** (e.g. Python/9), a per-problem constant with no γ-bias.

---

## 5. Validation evidence
- **Local:** 79/79 tests pass. `phase0_empirical_checks.py` exit 0. `build_manifest.py` distributional gate PASS. Contract output extracts **identically to gold** for the trio (Python 50/50, C 50/50, Rust 53/53). Python gold/contract **execute** 14/15 locally.
- **Server (the server, pinned image `sha256:4735da5db683f96aef4b3d849881fa7ddf865a480ce55287f0019b4f377c52a5`):**
  `verify_mceval_docker.py` → **`RESULT: PASS`** — gold **Python 0.9, C 1.0**, Rust 0.5 (soft).
  `smoke_contract_extractor.py --docker` → contract **Python 1.0, C 1.0**, Rust 0.0 (soft).
  ⇒ the image, toolchains, McEval extractor/executor, and **our contract output** all work end to end.

---

## 6. Manifest summary (`manifest/split_manifest.csv`, 2,066 rows)
- Columns: `problem_id, split, language, difficulty, difficulty_source, membership` (problem_id/language are lower-case canonical).
- Split: **1,653 train / 413 test**. Membership: **2,007 `gen+expl+compl`** + **59 `expl+compl`** (SQL).
- Difficulty source: 2,007 `level_propagated`, 59 `derived_proxy`. Difficulty totals easy 1240 / middle 421 / hard 405.
- Row-level (variants inherit split): gen 1606/401, expl 1653/413, compl 8098/2030.
- **State: provisionally frozen** (distributional gate passed). Becomes *confirmed-frozen* after the Phase-1 behavioral ±3% gate.

---

## 7. Server environment (verified, for reproducing runs)
- Host `the server`; repo at `<repo-root>`; run CPU scripts in conda env **`tokenskip_env`** (has PyYAML).
- McEval image pinned by digest **`sha256:4735da5db683f96aef4b3d849881fa7ddf865a480ce55287f0019b4f377c52a5`** (record in `configs/run_metadata.yaml` → `mceval.docker_digest`, gitignored).
- In-container interpreter for the harness/subprocesses: **`/opt/conda/bin/python`** (Py3.8 with McEval deps) — now the driver default.
- Reproduce the Phase-0 server gate:
  ```bash
  git pull
  python3 scripts/verify_mceval_docker.py --digest sha256:4735da5db683f96aef4b3d849881fa7ddf865a480ce55287f0019b4f377c52a5
  python3 scripts/smoke_contract_extractor.py --docker --digest sha256:4735da5db683f96aef4b3d849881fa7ddf865a480ce55287f0019b4f377c52a5
  ```

---

## 8. Commit history (Phase 0)
```
06b1379 chore(eval): set canonical threshold to 0.90; print contract RESULT
0f24663 fix(eval): newline-join prompt+canonical so gold/contract is executable
9bfeede fix(eval): conda python on PATH for execution; soft-gate Rust
b0abbdb fix(eval): match McEval output filename + default to conda python
6ae69a4 feat(phase0): Task 0.5 McEval Docker driver + contract<->extractor smoke test
a68513b feat(phase0): Task 0.4 split-manifest generator + committed manifest
25bb153 feat(phase0): Task 0.3 freeze contract + long-format schema
be0c560 feat(phase0): Task 0.2 data-side empirical checks
65c2a19 feat(phase0): Task 0.1 repo skeleton
(pre-0: 79ff55b, 6e66f49, 3981622 — docs + vendored components)
```

---

## 9. What is NOT done (Phase 1 entry conditions)
Phase 0 deliberately stopped before any inference. Phase 1 (next) will:
- build the **vLLM inference harness** (contract prompts, two-pass explanation, completion induced-CoT, per-problem **timestamps** for the later energy join);
- run **baseline γ=1.0** on the **train** split per model, parse to long-format, score with McEval → **filter correct trajectories** (SFT raw material);
- run γ=1.0 on **test** for the **behavioral ±3% gate** → **confirm-freeze** the manifest;
- measure the **completion induced-CoT length** distribution → set `gate_decision` per (model × subtype).

Still-open checks carried into Phase 1: **completion-gate calibration** (needs Phase-1 generations) and a **γ-convention runtime spot-check** (Phase-4 knob validation).

Open inputs to decide at Phase-1 start: first model to run (suggest **Qwen2.5-Coder-3B-Instruct**), and whether to do a **trio-only smoke** of inference→parse→eval before the full 40-language train run.
