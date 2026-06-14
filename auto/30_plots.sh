#!/usr/bin/env bash
# Phase D — figures. NOTE: aggregating raw run outputs into
# plots/data/{inference_summary,prompt_length_summary}.json is the one manual gap
# (no committed aggregator). The committed summaries reproduce the paper figures as-is.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

log "Installing plot deps + regenerating figures from plots/data/*.json"
run "cd '$REPO_ROOT/plots' && conda run -n $TS_ENV pip install -r requirements.txt"
run "cd '$REPO_ROOT/plots' && conda run -n $TS_ENV python plot_curves_normalized.py"
run "cd '$REPO_ROOT/plots' && conda run -n $TS_ENV python fitting_linear_cot_energy_normalized.py"
log "Figures written under plots/output/ (gitignored)."
cat <<'EOF'

To plot YOUR runs instead of the committed summaries, first aggregate:
  - reasoning: walk <bench>/outputs_sweep/**/samples/metrics.json (accuracy, cot len)
               + the sibling *_gpu.json / *_pdu.json (energy) into
               plots/data/inference_summary.json
  - mceval:    the pipeline already emits curves via build_curves.py / plot_curves.py
Ask me to generate an aggregator (auto/_aggregate.py) if you want it wired up.
EOF
