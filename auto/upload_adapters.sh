#!/usr/bin/env bash
# Upload the 3 Qwen TokenSkip LoRA adapters (5 benches each) to the Hub.
# RUN THIS ON THE SERVER (where the weights live). One private repo per model.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

: "${HF_NAMESPACE:?set HF_NAMESPACE in config.env}"
: "${HF_TOKEN:?set HF_TOKEN in config.env}"
: "${LORA_SAVES_DIR:?}" ; : "${MCEVAL_WEIGHTS_DIR:?}"

# huggingface_hub>=1.x: the library reads HF_TOKEN from the env (exported by lib.sh);
# no CLI login needed. (The old `huggingface-cli` is removed in favor of `hf`.)
run "conda run -n $TS_ENV pip install -q -U 'huggingface_hub'"

log "Previewing what would upload (dry-run):"
run "conda run -n $TS_ENV python '$AUTO_DIR/_upload_adapters.py' \
      --namespace '$HF_NAMESPACE' --prefix '${HF_REPO_PREFIX:-tokenskip}' --private '${HF_PRIVATE:-1}' \
      --sizes '${PUBLISH_SIZES:-3b 7b 14b}' --benches '${PUBLISH_BENCHES:-boolq gsm8k math piqa mceval}' \
      --lora-saves '$LORA_SAVES_DIR' --mceval-weights '$MCEVAL_WEIGHTS_DIR' --dry-run 1"

if [[ "${DRY_RUN:-0}" == "1" ]]; then log "DRY_RUN=1 set globally — stopping before real upload."; exit 0; fi

read -r -p $'\nProceed with the REAL upload to the Hub? [y/N] ' ans
[[ "$ans" == "y" || "$ans" == "Y" ]] || { log "aborted."; exit 0; }

run "conda run -n $TS_ENV python '$AUTO_DIR/_upload_adapters.py' \
      --namespace '$HF_NAMESPACE' --prefix '${HF_REPO_PREFIX:-tokenskip}' --private '${HF_PRIVATE:-1}' \
      --sizes '${PUBLISH_SIZES:-3b 7b 14b}' --benches '${PUBLISH_BENCHES:-boolq gsm8k math piqa mceval}' \
      --lora-saves '$LORA_SAVES_DIR' --mceval-weights '$MCEVAL_WEIGHTS_DIR' --dry-run 0"

log "Done. Repos: ${HF_NAMESPACE}/${HF_REPO_PREFIX:-tokenskip}-qwen2.5-{3b,7b,14b} (subfolders per benchmark)."
