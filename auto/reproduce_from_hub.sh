#!/usr/bin/env bash
# Fast reproduction for the 3 Qwen models using the published LoRA adapters.
# Reasoning arm only (gsm8k/math/boolq/piqa): downloads each adapter from the Hub and
# runs the SINGLE-GPU measured sweep with base+adapter (vLLM LoRARequest) -- no train/merge.
# mceval fast-repro is documented at the bottom (needs a quick peft merge first).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

: "${HF_NAMESPACE:?set in config.env}"
PREFIX="${HF_REPO_PREFIX:-tokenskip}"
ADAPTERS_DIR="$DATA_ROOT/hub_adapters"
REASONING_BENCHES="gsm8k math boolq piqa"

# HF_TOKEN is read from the env by huggingface_hub / the `hf` CLI (exported in lib.sh).

for size in ${PUBLISH_SIZES:-3b 7b 14b}; do
  model_spec "qwen2.5-$size" || continue
  BASE="$LOCAL_MODEL"; [[ -f "$BASE/config.json" ]] || BASE="$HF_REPO"     # local pinned dir, else HF id
  REPO="$HF_NAMESPACE/${PREFIX}-qwen2.5-${size}"

  for BENCH in $REASONING_BENCHES; do
    [[ " $BENCHMARKS_TO_RUN " == *" $BENCH "* ]] || continue
    DLROOT="$ADAPTERS_DIR/qwen2.5-${size}"
    ADAPTER="$DLROOT/$BENCH"
    if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then
      log "download adapter $REPO :: $BENCH"
      run "conda run -n $TS_ENV python '$AUTO_DIR/_hf_download.py' '$REPO' '$DLROOT' --include '${BENCH}/*'"
    fi

    # Energy-correct methodology (validated against the reference): gamma<1.0 runs the
    # MERGED model (base+adapter merged via peft -> no vLLM LoRARequest runtime overhead),
    # gamma=1.0 runs the pristine BASE model (the compression model collapses to ~0 CoT
    # without a gamma marker). base+adapter via LoRARequest reproduced ACCURACY but inflated
    # energy ~70% and washed out the per-gamma trend.
    MERGED="$DATA_ROOT/merged/qwen2.5-${size}-${BENCH}"
    if [[ ! -f "$MERGED/config.json" ]]; then
      log "merge adapter -> $MERGED (peft)"
      run "conda run -n $TS_ENV python '$REPO_ROOT/common/merge.py' --base '$BASE' --adapter '$ADAPTER' --output '$MERGED'"
    fi

    log "measured sweep [1 GPU] qwen2.5-$size / $BENCH  (gamma=1.0 base, gamma<1.0 merged)"
    for T in $SWEEP_TOKENS; do for r in $SWEEP_RATIOS; do for k in $(seq 1 "$SWEEP_REPEATS"); do
      base="$REPO_ROOT/$BENCH/outputs_hubrepro/qwen2.5-${size}/$BENCH/tok${T}/run${k}"
      rid="${size}_${BENCH}_tok${T}_ratio${r}_run${k}"
      run "mkdir -p '$base'"
      EVAL_MODEL="$MERGED"
      [[ "$r" == "1.0" ]] && EVAL_MODEL="$BASE"
      [[ "${DRY_RUN:-0}" == "1" ]] && { echo "  + [eval] CUDA_VISIBLE_DEVICES=0 ratio=$r tok=$T model=$EVAL_MODEL -> $base"; continue; }
      python3 "$REPO_ROOT/common/monitor_gpu.py" --run-name "$rid" --output-dir "$base" --interval "$MONITOR_INTERVAL" & GPU_PID=$!
      PDU_PID=""
      [[ "${ENABLE_PDU:-1}" == "1" ]] && { python3 "$REPO_ROOT/common/monitor_pdu.py" --run-name "$rid" --output-dir "$base" --interval "$MONITOR_INTERVAL" & PDU_PID=$!; }
      # run from the benchmark folder so evaluation.py finds configs/ and datasets/ (paths are relative)
      ( cd "$REPO_ROOT/$BENCH" && CUDA_VISIBLE_DEVICES=0 conda run -n "$TS_ENV" python "$REPO_ROOT/common/evaluation.py" \
          --output-dir "$base" --model-path "$EVAL_MODEL" --tokenizer-path "$EVAL_MODEL" \
          --model-size "$MSIZE" --model-type "$MTYPE" --data-type test \
          --max_new_tokens "$T" --eval_batch_size "$EVAL_BATCH_SIZE" \
          --temperature "$TEMPERATURE" --seed "$SEED" --benchmark "$BENCH" \
          --use_vllm --compression_ratio "$r" )
      kill -2 "$GPU_PID" 2>/dev/null || true
      [[ -n "$PDU_PID" ]] && { kill -2 "$PDU_PID" 2>/dev/null || true; }
      sleep 10
    done; done; done
  done
done

cat <<EOF

[mceval fast-repro] adapters are also on the Hub under each repo's mceval/ subfolder. To run:
  conda run -n $TS_ENV python $AUTO_DIR/_hf_download.py $HF_NAMESPACE/${PREFIX}-qwen2.5-7b $ADAPTERS_DIR/qwen2.5-7b --include 'mceval/*'
  # merge (peft) then point the mceval energy sweep at the merged dir:
  conda run -n $TS_ENV python mceval/scripts/merge_lora.py --base Qwen/Qwen2.5-7B-Instruct \\
      --adapter $ADAPTERS_DIR/qwen2.5-7b/mceval --output $DATA_ROOT/merged/qwen2.5-7b-mceval
  cd mceval && conda run -n $TS_ENV python scripts/run_energy_sweep.py --model qwen2.5-7b-instruct \\
      --run-id hubrepro --model-path $DATA_ROOT/merged/qwen2.5-7b-mceval --digest $MCEVAL_DIGEST
EOF
log "Hub reproduction (reasoning arm) complete."
