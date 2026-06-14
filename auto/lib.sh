#!/usr/bin/env bash
# Shared helpers + model matrix. Sourced by the phase scripts.
set -euo pipefail

# --- locate + load config ------------------------------------------------
AUTO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$AUTO_DIR/config.env" ]]; then
  echo "ERROR: $AUTO_DIR/config.env not found. Run: cp config.env.example config.env  (then edit it)" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$AUTO_DIR/config.env"

: "${REPO_ROOT:?set in config.env}"
: "${DATA_ROOT:?}" ; : "${HF_HOME:?}" ; : "${MODELS_DIR:?}"
export HF_HOME HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

log()  { echo -e "\n[$(date +%H:%M:%S)] $*"; }
warn() { echo "  !! $*" >&2; }
run()  { # run a command, honoring DRY_RUN
  echo "  + $*"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then return 0; fi
  eval "$@"
}
have_conda_env() { conda env list | awk '{print $1}' | grep -qx "$1"; }

# --- model matrix --------------------------------------------------------
# key | model_folder | size | mtype | hf_repo | rev_var | model_name | template | builder
_MODEL_ROWS=(
  "qwen2.5-3b|Qwen2.5-3b-Instruct|3b|qwen|Qwen/Qwen2.5-3B-Instruct|QWEN3B_SHA|qwen2.5|qwen|qwen"
  "qwen2.5-7b|Qwen2.5-7b-Instruct|7b|qwen|Qwen/Qwen2.5-7B-Instruct|QWEN7B_SHA|qwen2.5|qwen|qwen"
  "qwen2.5-14b|Qwen2.5-14b-Instruct|14b|qwen|Qwen/Qwen2.5-14B-Instruct|QWEN14B_SHA|qwen2.5|qwen|qwen"
  "llama3.1-8b|LLaMA-3.1-8B-Instruct|8b|llama3|meta-llama/Llama-3.1-8B-Instruct|LLAMA8B_SHA|llama3.1|llama3|llama"
)

# Per-benchmark BASE max_new_tokens (evaluation.py scales it by gamma for gamma<1.0).
# Reference budgets: gsm8k 512, math 1024, boolq 1024, piqa 512, mceval 1024.
# Override per bench via TOK_<BENCH> env vars in config.env.
bench_tokens() {
  case "$1" in
    gsm8k)  echo "${TOK_GSM8K:-512}";;
    math)   echo "${TOK_MATH:-1024}";;
    boolq)  echo "${TOK_BOOLQ:-1024}";;
    piqa)   echo "${TOK_PIQA:-512}";;
    mceval) echo "${TOK_MCEVAL:-1024}";;
    *)      echo "${SWEEP_TOKENS:-512}";;
  esac
}

# Populate MF/MSIZE/... globals for a model key. Returns nonzero if unknown.
model_spec() {
  local key="$1" row
  for row in "${_MODEL_ROWS[@]}"; do
    IFS='|' read -r k folder size mtype hf rev_var mname template builder <<<"$row"
    if [[ "$k" == "$key" ]]; then
      MF="$folder"; MSIZE="$size"; MTYPE="$mtype"; HF_REPO="$hf"
      MREV="${!rev_var:-}"; MODEL_NAME="$mname"; MTEMPLATE="$template"; MBUILDER="$builder"
      LOCAL_MODEL="$MODELS_DIR/$folder"
      return 0
    fi
  done
  warn "unknown model key: $key"; return 1
}
