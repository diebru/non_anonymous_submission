#!/usr/bin/env bash
# Phase C — multilingual-code arm (mceval). Generates configs, then runs the
# orchestrator per model. The pipeline STOPS at two human gates (p1_gate, p4_knob)
# unless MCEVAL_FORCE=1. SFT uses 2 GPUs; the energy sweep is single-GPU by design.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

MCEVAL="$REPO_ROOT/mceval"
: "${MCEVAL_MODELS:=qwen2.5-14b-instruct qwen2.5-7b-instruct qwen2.5-3b-instruct}"  # 14B->7B->3B
: "${MCEVAL_FORCE:=0}"

command -v docker >/dev/null || { warn "docker required for mceval scoring"; exit 1; }
run "conda run -n $TS_ENV pip install -q PyYAML"

log "Generating mceval configs (paths.yaml, run_metadata.yaml) from config.env"
run "conda run -n $TS_ENV python '$AUTO_DIR/_mceval_metadata.py' '$MCEVAL'"
run "cd '$MCEVAL' && conda run -n $TS_ENV python scripts/show_config.py"
run "cd '$MCEVAL' && conda run -n $TS_ENV python scripts/bootstrap_dirs.py"

FORCE_FLAG=""; [[ "$MCEVAL_FORCE" == "1" ]] && FORCE_FLAG="--force"

for M in $MCEVAL_MODELS; do
  log "================  mceval: $M  ================"
  run "cd '$MCEVAL' && conda run -n $TS_ENV python scripts/run_pipeline.py --model '$M' --dry-run"
  if [[ "$MCEVAL_FORCE" == "1" ]]; then
    run "cd '$MCEVAL' && conda run -n $TS_ENV python scripts/run_pipeline.py --model '$M' $FORCE_FLAG"
  else
    cat <<EOF

  >> GATED run for $M. Execute these in order, reviewing each gate's printout:
       cd '$MCEVAL'
       conda run -n $TS_ENV python scripts/run_pipeline.py --model $M                 # stops at p1_gate
       conda run -n $TS_ENV python scripts/run_pipeline.py --model $M --from-stage p1_corpus  # stops at p4_knob, then runs sweep
     (or re-run this script with MCEVAL_FORCE=1 to blow through gates unattended.)
EOF
  fi
done

log "mceval phase dispatched."
