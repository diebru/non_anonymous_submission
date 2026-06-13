# configs/

Configuration for the project. **No hardcoded local paths** anywhere in the code —
every path comes from here or from environment variables (see
[`docs/WORKFLOW.md`](../docs/WORKFLOW.md) §4 and [`src/tsmc/config.py`](../src/tsmc/config.py)).

## Files

| File | Committed? | Purpose |
|---|---|---|
| `paths.example.yaml` | ✅ yes | Template for machine-specific paths. |
| `paths.yaml` | ❌ gitignored | Your local/server copy of the above. |
| `run_metadata.example.yaml` | ✅ yes | Template for pinned run provenance (decision sheet #8). |
| `run_metadata.yaml` | ❌ gitignored | The filled-in pinned values for a run. |

## Setup

```bash
cp configs/paths.example.yaml configs/paths.yaml
# edit configs/paths.yaml: on the server set paths.data_root to a big disk
```

## Path resolution precedence (paths)

1. value passed explicitly in code
2. `$TSMC_CONFIG` (explicit config file), then `$TSMC_REPO_ROOT` / `$TSMC_DATA_ROOT`
   (per-setting overrides)
3. `configs/paths.yaml`
4. `configs/paths.example.yaml` (fallback so the package imports out-of-the-box)

Relative paths resolve against the auto-detected repo root; absolute paths are
used as-is. Defaults (when a value is blank): `data_root` = repo root; the four
artifact dirs (`generations/`, `compressed/`, `weights/`, `eval_dumps/`) sit
under `data_root` and are gitignored.

## Quick check

```bash
python scripts/show_config.py      # print the resolved paths + frozen constants
python scripts/bootstrap_dirs.py   # create the artifact dirs under data_root
```
