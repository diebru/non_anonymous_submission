#!/usr/bin/env bash
# Reproduce the mceval (multilingual-code) arm. Per model: get a MERGED checkpoint
# (download HF adapter -> peft merge, OR reuse an existing server merge for 8B/Llama),
# then run_energy_sweep.py (vLLM inference + McEval Docker scoring + GPU/PDU energy over
# the generate window) -> curves. Single GPU by design. Merged model for ALL gammas.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

MCEVAL="$REPO_ROOT/mceval"
: "${MCEVAL_RUN_ID:=hubrepro}"
: "${HF_NAMESPACE:?set in config.env}"
PREFIX="${HF_REPO_PREFIX:-tokenskip}"
GPU_INDEX=0
BASE_TOKENS="$(bench_tokens mceval)"                                  # 1024
GAMMAS="${MCEVAL_GAMMAS:-${SWEEP_RATIOS:-1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1}}"  # no 0.95/0.85
PDU_FLAG=""; [[ "${ENABLE_PDU:-1}" == "1" ]] || PDU_FLAG="--no-pdu"

# Per-size resolution. Sets: MC_MODEL, MC_BASE, MC_REPO, MERGED, USE_LOCAL_MERGED.
mceval_spec() {
  local size="$1"
  case "$size" in
    3b|7b|14b)
      MC_MODEL="qwen2.5-${size}-instruct"
      MC_BASE="Qwen/Qwen2.5-${size%b}B-Instruct"
      MC_REPO="$HF_NAMESPACE/${PREFIX}-qwen2.5-${size}"
      MERGED="$DATA_ROOT/merged/qwen2.5-${size}-mceval"
      USE_LOCAL_MERGED=0 ;;
    8b)   # Llama-3.1-8B: adapter not on HF -> reuse the merged model already on the server
      MC_MODEL="llama-3.1-8b-instruct"
      MC_BASE="meta-llama/Llama-3.1-8B-Instruct"
      MC_REPO=""
      MERGED="${MCEVAL_8B_MERGED:-$MCEVAL_WEIGHTS_DIR/llama-3.1-8b-instruct/merged_sft_run01}"
      USE_LOCAL_MERGED=1 ;;
    *) warn "unknown mceval size: $size"; return 1 ;;
  esac
}

command -v docker >/dev/null || { warn "docker required for mceval scoring"; exit 1; }
docker image inspect "multilingualnlp/mceval@${MCEVAL_DIGEST}" >/dev/null 2>&1 || \
  run "docker pull multilingualnlp/mceval@${MCEVAL_DIGEST}"

# configs from config.env (data_root, digest, model commits, PDU host/community/oid)
run "conda run -n $TS_ENV pip install -q PyYAML"
run "conda run -n $TS_ENV python '$AUTO_DIR/_mceval_metadata.py' '$MCEVAL'"
run "cd '$MCEVAL' && conda run -n $TS_ENV python scripts/show_config.py"
run "cd '$MCEVAL' && conda run -n $TS_ENV python scripts/bootstrap_dirs.py"

for size in ${PUBLISH_SIZES:-3b}; do
  mceval_spec "$size" || continue
  log "================  mceval: $MC_MODEL  (gammas: $GAMMAS)  ================"

  if [[ "$USE_LOCAL_MERGED" == "1" ]]; then
    [[ -f "$MERGED/config.json" ]] || { warn "no merged model at $MERGED — skipping $size"; continue; }
    log "using existing server merge: $MERGED"
  else
    DL="$DATA_ROOT/hub_adapters/qwen2.5-${size}"; ADAPTER="$DL/mceval"
    [[ -f "$ADAPTER/adapter_config.json" ]] || \
      run "conda run -n $TS_ENV python '$AUTO_DIR/_hf_download.py' '$MC_REPO' '$DL' --include 'mceval/*'"
    [[ -f "$MERGED/config.json" ]] || \
      run "cd '$MCEVAL' && conda run -n $TS_ENV python scripts/merge_lora.py --base '$MC_BASE' --adapter '$ADAPTER' --output '$MERGED'"
  fi

  log "energy sweep [1 GPU=$GPU_INDEX, base_tokens=$BASE_TOKENS scaled by gamma] $MC_MODEL -> run-id $MCEVAL_RUN_ID"
  run "cd '$MCEVAL' && conda run -n $TS_ENV python scripts/run_energy_sweep.py \
        --model '$MC_MODEL' --run-id '$MCEVAL_RUN_ID' --model-path '$MERGED' \
        --gammas $GAMMAS --digest '$MCEVAL_DIGEST' --scale-by-gamma --max-tokens '$BASE_TOKENS' \
        --gpu-index $GPU_INDEX --skip-existing $PDU_FLAG"
  run "cd '$MCEVAL' && conda run -n $TS_ENV python scripts/build_curves.py --model '$MC_MODEL' --task generation --split test --run-id '$MCEVAL_RUN_ID' --gammas $GAMMAS"
  run "cd '$MCEVAL' && conda run -n $TS_ENV python scripts/plot_curves.py  --model '$MC_MODEL' --task generation --split test --run-id '$MCEVAL_RUN_ID' --gammas $GAMMAS"
done

log "mceval done. Live-watch a running sweep:"
log "  cd $MCEVAL && conda run -n $TS_ENV python scripts/watch_sweep.py --model <id> --run-id $MCEVAL_RUN_ID --log <logfile>"
