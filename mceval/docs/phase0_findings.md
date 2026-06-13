# Phase-0 empirical findings (data-side)

Results of the Task 0.2 checks against the vendored McEval data. Reproduce with:

```bash
python3 scripts/phase0_empirical_checks.py        # human report (exit 0 = all hard invariants hold)
python3 scripts/phase0_empirical_checks.py --json # machine-readable
python3 -m unittest discover -s tests -v          # regression guards
```

These resolve the data-side items in the roadmap's "Open Phase-0 empirical checks".
The **γ-convention** check is *not* here — it needs the TokenSkip runtime
(server, `tokenskip_env`), not the dataset, so it stays open until that env exists.

## 1. Sentinel-collision scan — PASS

The contract sentinel `@@@FINAL_CODE_7F3A9@@@` (Decision #1) appears **0 times**
across **126,195** string fields of all three tasks. The sentinel is safe; no
collision risk in extraction/parsing.

## 2. Base-problem overlap — verified, with a casing caveat

| Quantity | Value |
|---|---|
| Generation rows | 2,007 |
| Explanation rows | 2,066 |
| Completion rows (`merge`) | 10,128 = single 2,998 + multi 2,998 + span 4,132 |
| Distinct base problems — **raw `task_id`** | **2,125** |
| Distinct base problems — **language-case-normalized** | **2,066** |
| Membership (normalized) | `gen+expl+compl`: 2,007 · `expl+compl`: 59 |

Generation's 2,007 base problems are a strict subset of both explanation and
completion (normalized). Explanation and completion share the **same** 2,066 base
problems, of which 59 (the SQL problems) are absent from generation.

### ⚠ Finding: inconsistent SQL language casing in `task_id`
The 59 SQL problems are stored as **`sql/N`** in explanation but **`SQL/N`** in
completion (same problem numbers 1–59; generation has no SQL at all). As raw,
case-sensitive strings they do **not** match, which:

- inflates the distinct base-problem count to **2,125** (the 59 SQL appear twice);
- would, if used directly as the split key, **leak SQL problems across train/test**
  — e.g. `sql/5` (explanation) could land in *train* while `SQL/5` (completion)
  lands in *test*. That is exactly the cross-task leakage the base-problem split
  exists to prevent.

**Required mitigation (binds Task 0.4):** the manifest generator must key on
`tsmc.mceval_data.canonical_base_id` (language lower-cased), **not** the raw
`task_id`. With that normalization the count is the expected **2,066** and the
roadmap's overlap design holds exactly. The roadmap's stated "2,066 shared" was
correct *post-normalization*; the raw data needed this reconciliation.

### Languages
41 distinct languages after normalization (40 generation languages + SQL); 42
raw (the `sql`/`SQL` pair). SQL is present only in explanation + completion.

## 3. Difficulty distribution — PASS

Generation `level`: **easy 1,221 / middle 401 / hard 385** (= 2,007), matching the
decision-sheet grounding stats. The **59 SQL** base problems carry **no `level`**
(they have no generation row) → they are exactly the set needing a derived
difficulty proxy (Decision #4).

## 4. Difficulty-proxy calibration (informational)

Nonblank LOC of `canonical_solution`, per generation `level`:

| Level | n | median LOC | p25 / p75 |
|---|---|---|---|
| easy | 1,221 | 7 | 3 / 10 |
| middle | 401 | 13 | 9 / 18 |
| hard | 385 | 20 | 13 / 27 |

- Global LOC p25/50/75 = **4 / 9 / 15** (the roadmap's "4/10/17" was approximate;
  use **4/9/15**, or the proposed tertile cuts p33/p66 ≈ **6 / 13**).
- LOC separates the three levels cleanly (medians 7 → 13 → 20), so it is a usable
  proxy signal for the labeled distribution.

### ⚠ Finding: LOC proxy collapses for SQL
The 59 unlabeled SQL solutions have LOC p25/50/75 = **1 / 1 / 1** — SQL answers are
essentially one-line queries regardless of difficulty. A LOC-based proxy would
dump **all 59** SQL problems into "easy".

**Resolved in Task 0.4.** Both of Decision #4's fallback signals fail for SQL:
solution LOC is constant (1) and the test-case count is constant (2) — and the
test-case count does not discriminate `level` even in the labeled core (easy /
middle / hard all median ~6–7, identical p25/p75). The only varying SQL signal is
**docstring length** (166–2825 words). The SQL difficulty is therefore assigned by
**rank tertiles of docstring length, internal to SQL** (≈19/20/20 across
easy/middle/hard), all tagged `difficulty_source = derived_proxy`. This keeps the
language×difficulty grid balanced without pretending to calibrate to a signal that
does not exist. Implemented in `tsmc.manifest`.

## 5. γ convention + marker — resolved from TokenSkip source (Task 0.3)

Code-side (not data-side), recorded here for completeness:

- **Convention:** `TokenSkip/LLMLingua.py` compresses with
  `compress_prompt(..., rate=compression_ratio, force_tokens=['Step', ':'],
  force_reserve_digit=True, drop_consecutive=True)`, and `get_average_compress_rate`
  averages `compressed_tokens / origin_tokens`. LLMLingua-2's `rate` is the
  fraction of tokens **retained**, so TokenSkip's `compression_ratio` = **fraction
  retained** — matching our γ (1.0 = full CoT). The compressor force-keeps
  `Step`/`:` and digits (relevant later to Phase-2 reasoning-structure preservation).
- **Marker:** `<|eot_id|>{ratio}<|eot_id|>` appended to the user content and
  **omitted at ratio 1.0** (`get_llamafactory_input.py` train, `evaluation.py`
  inference). For Qwen, `<|eot_id|>` is a literal text delimiter. Frozen in
  `tsmc.contract.gamma_marker`; place it last (before the assistant turn).
- A runtime spot-check during Phase-4 knob validation is still advisable.

## 6. McEval excludes SQL from execution — Task 0.5 finding

`McEval/eval/eval_all.py` hard-codes `exclude_langs = ['sql']`: SQL is **never
executed** by the harness. So the 59 SQL base problems (explanation + completion
only) **cannot receive an execution pass/fail**. Implications:

- SQL stays in the manifest (problem-level split + leakage tracking) but is
  **execution-unscored**; it drops out of any accuracy/behavioral computation.
- Low impact on the primary result: **generation has no SQL**, and SQL is 2.9%
  of base problems and lives only in explanation/completion. Report explanation/
  completion accuracy on the 2,007 executable core (note SQL excluded).

## 7. McEval eval harness I/O contract (Task 0.5)

- Input: `<result_path>/<Lang>.jsonl`, each line = the rich problem record
  (entry_point/signature/prompt/test/canonical_solution) + `raw_generation:[text]`.
  McEval re-runs its own `extract(raw_generation[0], item, lang)` then executes.
- It hard-codes `/workspace/MMCodeEval/eval/tmp` and resolves `../data` from CWD,
  so we run it **inside** the pinned image (`cd /workspace/MMCodeEval/eval && python
  eval_all.py --result_path … --save_path …`); output is `<lang>\t<json score>`.
- `extract()` is pure regex/string → the contract↔extractor round-trip is
  verified **locally** (`tsmc.eval`), shown identical to gold for the trio; only
  execution needs Docker (`scripts/verify_mceval_docker.py`,
  `scripts/smoke_contract_extractor.py --docker`).
- Generation `data/*.jsonl` are rich for most languages (AWK is the reduced
  exception); the rich fields McEval needs are present for the trio.
- **Server run notes (verified on the server):** the pinned image digest is
  `sha256:4735da5db683f96aef4b3d849881fa7ddf865a480ce55287f0019b4f377c52a5`.
  Its McEval deps (bs4, ...) live in the **conda** Python `/opt/conda/bin/python`
  (3.8); the bare `/usr/bin/python` is Py2 and `/usr/bin/python3` (3.6) lacks
  them — so the driver runs eval under `/opt/conda/bin/python` (the default).
  eval_all.py names its output `<save>/<basename(result_path)>.jsonl`, so the
  driver mounts the result dir under its own name. CPU-side PyYAML must be in the
  env running the scripts (it is in `tokenskip_env`).
- **Execution environment (verified):** McEval's `excute.py` runs candidate
  **Python via `['python', path]`** — which is the image's **Py2** on a
  non-interactive shell, so *all* Py3 canonical solutions fail (Python 0/10). Fix:
  the driver prepends `/opt/conda/bin` to `PATH` so `python` → conda Py3.8. After
  this, Python (Family A) and C (Family B) confirm the pipeline.
- **Gold/contract construction needed a newline join:** McEval `prompt` doesn't
  always end in a newline (it can end at the docstring's closing `"""`), so
  `prompt + canonical_solution` glued the body onto that line → `SyntaxError`
  (Python gold 2/10). Fixed with `results.reference_program` (one clean newline);
  Python gold/contract → ~14/15 locally (the lone miss is a McEval extraction
  edge case). A local execution regression test now guards this — the earlier
  parsing-only check missed it because gold and contract were *equally* broken.
- **Canonical ceiling ≈ 0.9 (not 1.0):** McEval's own `extract()` mis-reconstructs
  a small fraction of problems even for reference solutions (e.g. Python/9 → an
  IndentationError in its test reconstruction). Verified gold rates: **Python 0.9,
  C 1.0**; contract (our output) **Python 1.0, C 1.0**. The verification threshold
  is therefore **0.90** for the required languages. These McEval-broken problems
  are per-problem constants → they add a little noise but **no γ-dependent bias**
  to the concavity curves (they fail identically across all models/γ).
- **Phase-0 Docker gate: PASS** — Python + C (Family A + B) execute through the
  pinned image end to end, for both gold and our contract output; Rust soft.
- **Rust is soft-gated:** `excute.py` does `rm -rf target` + `cargo test` per
  problem under a ~35 s timeout, so each problem cold-recompiles all crate deps
  and slow ones time out (gold Rust ≈ 0.5). This is a McEval limitation, not our
  pipeline (our Rust *extraction* matches gold exactly). The verification gate
  therefore **requires Python + C** and treats **Rust** as informational.

## Status of the roadmap's open checks

| Open check | Status |
|---|---|
| Sentinel-collision scan | ✅ resolved — 0 hits |
| Base-problem overlap | ✅ resolved — 2,066 normalized (2,125 raw); **normalize SQL casing** |
| Difficulty-proxy calibration | ✅ calibrated — LOC cuts 4/9/15; **SQL needs a non-LOC signal** |
| γ convention (retained vs removed) | ✅ resolved from TokenSkip source — fraction **retained**; runtime spot-check in Phase 4 |
| Contract↔extractor (trio) | ✅ resolved locally — contract output extracts identically to gold for Python/C/Rust (`tsmc.eval`); execution pending server |
| McEval Docker canonical pass | ⏳ scripts written (`scripts/verify_mceval_docker.py`); **run on server** with the pinned digest |
| Completion-gate calibration | ⏳ open — needs Phase-1 generations |
