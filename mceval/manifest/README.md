# manifest/

Home of the **split manifest** — the one bulk-ish artifact that **is** committed to
git (it is tiny text and *defines the experiment*; see
[`docs/WORKFLOW.md`](../docs/WORKFLOW.md) §5 and
[`docs/PROJECT_ROADMAP.md`](../docs/PROJECT_ROADMAP.md) §6).

## `split_manifest.csv`

Built by `tsmc.manifest` / `scripts/build_manifest.py` (Task 0.4). One line per
base problem (2,066 rows), sorted by (language, number), columns:

```
problem_id, split, language, difficulty, difficulty_source, membership
```

Example row: `awk/1,test_problems,awk,easy,level_propagated,gen+expl+compl`

Regenerate / verify:
```bash
python3 scripts/build_manifest.py          # rebuild (seed 42) + gate + write
python3 scripts/build_manifest.py --check  # validate the committed file only
```

- **Atomic unit:** the **canonical (lower-cased)** base id `lang/N`
  (`tsmc.mceval_data.canonical_base_id`) — this reconciles `sql`↔`SQL` (Task 0.2),
  so `problem_id` and `language` are lower-case. Every task variant sharing that
  prefix (generation, explanation, completion `lang/N-k-sub`) inherits the same
  `split` → blocks cross-task leakage.
- **split** values are `train_problems` / `test_problems` (the schema enum).
- **Composition:** 1,653 train / 413 test; membership 2,007 `gen+expl+compl`
  + 59 `expl+compl` (all SQL, difficulty by `derived_proxy`); rest
  `level_propagated`.
- **Seed:** 42 (provenance). The **frozen manifest is authoritative**, not the seed.
- **Format:** `.csv` on purpose, so it is *not* caught by the bulk-`.jsonl` ignore
  rule in [`.gitignore`](../.gitignore).
- **Freeze loop:** *provisionally* frozen now (distributional gate passes),
  *confirmed-frozen* after the Phase-1 behavioral ±3% gate.
