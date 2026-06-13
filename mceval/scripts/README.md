# scripts/

Thin, runnable entry points. CPU-only utilities here may be run locally; GPU /
Docker / SFT scripts added in later phases are **written locally but executed only
on the server** (see [`docs/WORKFLOW.md`](../docs/WORKFLOW.md) §2).

Each script inserts `src/` on `sys.path`, so they run from a fresh checkout without
`pip install -e .`.

| Script | Runs locally? | Purpose |
|---|---|---|
| `show_config.py` | ✅ | Print resolved paths + frozen constants (read-only). |
| `bootstrap_dirs.py` | ✅ | Create the gitignored artifact dirs under `data_root` (idempotent). |
