#!/usr/bin/env bash
# Phase A — one-time server setup: envs, deps, LLaMA-Factory, base models, Docker, PDU patch.
# Safe to re-run (idempotent-ish: skips work that already exists).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

cd "$REPO_ROOT"
git pull --ff-only || warn "git pull skipped/failed (ok if not a clone)"

# --- system SNMP tooling (for the PDU monitor) ---------------------------
if [[ "${ENABLE_PDU:-1}" == "1" ]] && ! command -v snmpget >/dev/null; then
  log "Installing SNMP tooling (needs sudo)"
  run "sudo apt-get update && sudo apt-get install -y snmp snmp-mibs-downloader" || warn "install snmp manually"
fi

# --- conda env: tokenskip_env --------------------------------------------
if ! have_conda_env "$TS_ENV"; then
  log "Creating $TS_ENV"
  run "conda create -n $TS_ENV python=3.10 -y"
fi
log "Installing reasoning-arm deps into $TS_ENV"
run "conda run -n $TS_ENV pip install -r '$REPO_ROOT/common/requirements.txt'"
run "conda run -n $TS_ENV pip install -U 'huggingface_hub'"

# --- conda env: llamafactory_env + LLaMA-Factory -------------------------
if ! have_conda_env "$LF_ENV"; then
  log "Creating $LF_ENV"
  run "conda create -n $LF_ENV python=3.10 -y"
fi
if [[ ! -d "$REPO_ROOT/LlamaFactory" ]]; then
  log "Cloning LLaMA-Factory into ./LlamaFactory (sibling of common/)"
  run "git clone https://github.com/hiyouga/LLaMA-Factory '$REPO_ROOT/LlamaFactory'"
fi
log "Installing LLaMA-Factory into $LF_ENV"
run "conda run -n $LF_ENV pip install -e '$REPO_ROOT/LlamaFactory[torch,metrics]'"

# --- download the 4 base models at PINNED commits ------------------------
mkdir -p "$MODELS_DIR" "$DATA_ROOT" "$HF_HOME"
[[ -n "${HF_TOKEN:-}" ]] || warn "HF_TOKEN empty; gated Llama download may fail"
for key in $MODELS_TO_RUN; do
  model_spec "$key" || continue
  if [[ -z "$MREV" || "$MREV" == PUT_* ]]; then warn "$key: commit SHA not set in config.env — skipping download"; continue; fi
  if [[ -f "$LOCAL_MODEL/config.json" ]]; then log "$key already downloaded at $LOCAL_MODEL"; continue; fi
  log "Downloading $HF_REPO @ $MREV -> $LOCAL_MODEL"
  run "conda run -n $TS_ENV hf download '$HF_REPO' --revision '$MREV' --local-dir '$LOCAL_MODEL'"
done

# --- McEval Docker image by digest ---------------------------------------
if command -v docker >/dev/null; then
  log "Pulling McEval image by digest"
  run "docker pull multilingualnlp/mceval@${MCEVAL_DIGEST}"
else
  warn "docker not found; mceval arm (20_mceval.sh) will need it"
fi

# --- point the PDU monitor at the real PDU IP ----------------------------
if [[ "${ENABLE_PDU:-1}" == "1" && "${PDU_IP}" != "192.0.2.1" ]]; then
  log "Patching PDU IP in common/monitor_pdu.py -> $PDU_IP"
  run "sed -i \"s/'192\\.0\\.2\\.1'/'${PDU_IP}'/\" '$REPO_ROOT/common/monitor_pdu.py'"
  run "conda run -n $TS_ENV snmpget -v2c -c '${PDU_SNMP_COMMUNITY}' '${PDU_IP}' '${PDU_OID}'" || warn "SNMP check failed — verify PDU reachability"
fi

log "Setup complete. Next: bash auto/10_reasoning.sh"
