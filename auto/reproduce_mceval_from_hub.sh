#!/usr/bin/env bash
# Reproduce the mceval (multilingual-code) arm from the uploaded LoRA adapters.
# Per model: download mceval adapter -> peft merge -> run_energy_sweep.py (vLLM inference
# + McEval Docker scoring + GPU/PDU energy over the generate window) -> curves.
#
# Single GPU by design (energy). mceval uses the MERGED model for ALL gammas (the contract
# injects the gamma marker, so no base-model special case like the reasoning arm).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

MCEVAL="$REPO_ROOT/mceval"
: "${MCEVAL_RUN_ID:=hubrepro}"
: "${HF_NAMESPACE:?set in config.env}"
PREFIX="${HF_REPO_PREFIX:-tokenskip}"
GPU_INDEX=0
BASE_TOKENS="$(bench_tokens mceval)"     # 1024
PDU_FLAG=""; [[ "${ENABLE_PDU:-1}" == "1" ]] || PDU_FLAG="--no-pdu"

command -v docker >/dev/null || { warn "docker required for mceval scoring"; exit 1; }
docker image inspect "multilingualnlp/mceval@${MCEVAL_DIGEST}" >/dev/null 2>&1 || \
  run "docker pull multilingualnlp/mceval@${MCEVAL_DIGEST}"

# 1) configs from config.env (data_root, digest, model commit, PDU host/community/oid)
run "conda run -n $TS_ENV pip install -q PyYAML"
run "conda run -n $TS_ENV python '$AUTO_DIR/_mceval_metadata.py' '$MCEVAL'"
run "cd '$MCEVAL' && conda run -n $TS_ENV python scripts/show_config.py"
run "cd '$MCEVAL' && conda run -n $TS_ENV python scripts/bootstrap_dirs.py"

for size in ${PUBLISH_SIZES:-3b}; do
  M="qwen2.5-${size}-instruct"                       # mceval model id
  HFBASE="Qwen/Qwen2.5-${size%b}B-Instruct"          # 3b -> Qwen2.5-3B-Instruct
  REPO="$HF_NAMESPACE/${PREFIX}-qwen2.5-${size}"
  DL="$DATA_ROOT/hub_adapters/qwen2.5-${size}"; ADAPTER="$DL/mceval"
  MERGED="$DATA_ROOT/merged/qwen2.5-${size}-mceval"

  log "================  mceval: $M  ================"
  # 2) adapter from the Hub
  if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then
    run "conda run -n $TS_ENV python '$AUTO_DIR/_hf_download.py' '$REPO' '$DL' --include 'mceval/*'"
  fi
  # 3) merge (peft) — skip if already merged
  if [[ ! -f "$MERGED/config.json" ]]; then
    run "cd '$MCEVAL' && conda run -n $TS_ENV python scripts/merge_lora.py --base '$HFBASE' --adapter '$ADAPTER' --output '$MERGED'"
  fi
  # 4) energy-instrumented gamma sweep (SINGLE GPU; Docker scoring; PDU from run_metadata)
  log "energy sweep [1 GPU=$GPU_INDEX, base_tokens=$BASE_TOKENS scaled by gamma] $M -> run-id $MCEVAL_RUN_ID"
  run "cd '$MCEVAL' && conda run -n $TS_ENV python scripts/run_energy_sweep.py \
        --model '$M' --run-id '$MCEVAL_RUN_ID' --model-path '$MERGED' \
        --digest '$MCEVAL_DIGEST' --scale-by-gamma --max-tokens '$BASE_TOKENS' \
        --gpu-index $GPU_INDEX --skip-existing $PDU_FLAG"
  # 5) curves
  run "cd '$MCEVAL' && conda run -n $TS_ENV python scripts/build_curves.py --model '$M' --task generation --split test --run-id '$MCEVAL_RUN_ID'"
  run "cd '$MCEVAL' && conda run -n $TS_ENV python scripts/plot_curves.py  --model '$M' --task generation --split test --run-id '$MCEVAL_RUN_ID'"
done

log "mceval reproduction done. Live-watch a running sweep with:"
log "  cd $MCEVAL && conda run -n $TS_ENV python scripts/watch_sweep.py --model qwen2.5-<size>-instruct --run-id $MCEVAL_RUN_ID --log <logfile>"
