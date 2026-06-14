#!/usr/bin/env python3
"""Fill mceval/configs/run_metadata.yaml + paths.yaml from environment (config.env).

Round-trips the committed *.example.yaml templates so nothing else changes.
"""
import os, sys, yaml

MCEVAL = sys.argv[1]                      # path to mceval/
data_root = os.environ["DATA_ROOT"]
digest = os.environ["MCEVAL_DIGEST"]
pdu_host = os.environ.get("PDU_IP", "192.0.2.1")
community = os.environ.get("PDU_SNMP_COMMUNITY", "public")
oid = os.environ.get("PDU_OID", "PowerNet-MIB::ePDUPhaseStatusActivePower.1")

shas = {
    "qwen2.5-3b-instruct": os.environ.get("QWEN3B_SHA", ""),
    "qwen2.5-7b-instruct": os.environ.get("QWEN7B_SHA", ""),
    "qwen2.5-14b-instruct": os.environ.get("QWEN14B_SHA", ""),
    "llama-3.1-8b-instruct": os.environ.get("LLAMA8B_SHA", ""),
}

# --- run_metadata.yaml ---
with open(os.path.join(MCEVAL, "configs", "run_metadata.example.yaml")) as f:
    meta = yaml.safe_load(f)
for k, sha in shas.items():
    if sha and not sha.startswith("PUT_") and k in meta.get("models", {}):
        meta["models"][k]["commit"] = sha
meta.setdefault("mceval", {})["docker_image"] = "multilingualnlp/mceval"
meta["mceval"]["docker_digest"] = digest
meta.setdefault("energy", {}).update({
    "gpu_index": 0, "sample_interval_s": 0.5,
    "pdu_host": pdu_host, "snmp_community": community, "snmp_oid": oid,
})
with open(os.path.join(MCEVAL, "configs", "run_metadata.yaml"), "w") as f:
    yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

# --- paths.yaml ---
with open(os.path.join(MCEVAL, "configs", "paths.example.yaml")) as f:
    paths = yaml.safe_load(f)
paths.setdefault("paths", {})["data_root"] = data_root
with open(os.path.join(MCEVAL, "configs", "paths.yaml"), "w") as f:
    yaml.safe_dump(paths, f, sort_keys=False, allow_unicode=True)

print("wrote mceval/configs/{run_metadata,paths}.yaml")
