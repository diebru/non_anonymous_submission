# math

MATH-500 competition problems; match on the boxed final answer. Self-contained working directory for the **math** arm of the TokenSkip
CoT-compression study.

## Contents
- `configs/math_{test,train}.json` — dataset + processing config read by the engine.
- `datasets/math/{test,train}.jsonl` — the benchmark data.
- `run.sh` — single evaluation run (edit the model and compression ratio inside).

## Run
```bash
bash run.sh        # evaluates via ../common/evaluation.py
```

Compression (LLMLingua-2), the SFT-data builders, LoRA merge, and the full
energy-sweep drivers are shared and live in [`../common`](../common).
