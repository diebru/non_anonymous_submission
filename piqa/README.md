# piqa

Physical-commonsense two-choice QA; exact-match scoring. Self-contained working directory for the **piqa** arm of the TokenSkip
CoT-compression study.

## Contents
- `configs/piqa_{test,train}.json` — dataset + processing config read by the engine.
- `datasets/piqa/{test,train}.jsonl` — the benchmark data.
- `prepare.py` — (re)download and format the raw data from its public source.
- `run.sh` — single evaluation run (edit the model and compression ratio inside).

## Run
```bash
bash run.sh        # evaluates via ../common/evaluation.py
```

Compression (LLMLingua-2), the SFT-data builders, LoRA merge, and the full
energy-sweep drivers are shared and live in [`../common`](../common).
