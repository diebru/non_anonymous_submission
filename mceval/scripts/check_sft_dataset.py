#!/usr/bin/env python3
"""Phase-3 gate: verify the SFT dataset renders correctly under the Qwen template.

Runs in ``llamafactory_env`` (needs ``transformers``). It is the last Phase-3 check
before the Phase-4 LoRA SFT: it loads the emitted dataset and renders every example
through ``tokenizer.apply_chat_template`` -- exactly what vLLM does at Phase-4
inference -- then asserts the train-time and inference-time prompts agree and nothing
truncates. Concretely:

  1. dataset_info.json is well-formed and points at the data file (the "load" gate);
  2. each example is messages=[user, assistant] (optionally preceded by a system
     message) with the right roles, non-empty;
  3. the gamma marker SURVIVES apply_chat_template literally (present iff gamma<1):
     this is the train/inference freeze, now checked through the REAL chat template
     (for Qwen, ``<|eot_id|>`` must stay literal text, not a consumed special token);
  3b. (cross-family) the gamma DELIMITER is PLAIN TEXT for this tokenizer -- i.e. it
     does not tokenize into special/added-token ids. The string check in (3) passes
     even on Llama (apply_chat_template with tokenize=False does no special-token
     parsing), but at real tokenization Llama-3 would collapse ``<|eot_id|>`` into the
     end-of-turn control token; this token-level check is what actually catches that;
  4. the assistant target still round-trips through ``contract.parse_generation``;
  5. NO example exceeds --cutoff-len tokens, measured on the FULL templated sequence
     (incl. chat-template + default-system tokens) -- tighter than the build-time
     proxy, so it is the authoritative cutoff_len gate.

Why this is the consistency gate: Phase-4 vLLM inference uses apply_chat_template, and
LLaMA-Factory's ``template: qwen`` is maintained to reproduce the Qwen chat template,
so an example that renders correctly here is consistent on both sides. The decisive
end-to-end confirmation remains Phase-4 knob validation (gamma measurably shortens CoT
after SFT).

Usage (server, llamafactory_env):
    python3 scripts/check_sft_dataset.py --model qwen2.5-coder-3b-instruct --cutoff-len 2048
    #   --tokenizer is auto-resolved from run_metadata hf_repo; override if needed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import GAMMA_GRID, MODEL_IDS, family_of  # noqa: E402
from tsmc.contract import gamma_delimiter, parse_generation  # noqa: E402

_CLEAN_BRANCHES = ("sentinel", "multi_fence")


def _marker_re(delim: str) -> "re.Pattern[str]":
    """Regex matching our gamma marker ``{delim}{gamma}{delim}``, capturing the value."""
    return re.compile(re.escape(delim) + r"([0-9.]+)" + re.escape(delim))


def delimiter_is_plain_text(tok, delim: str) -> tuple[bool, list[int]]:
    """True iff the gamma delimiter tokenizes as ORDINARY text for this tokenizer.

    The failure we guard against (Llama-3): ``<|eot_id|>`` is a real special/added
    token, so the literal marker would be encoded into control-token ids at SFT /
    inference and corrupt gamma control. A Llama-safe delimiter must (a) not encode
    to any special-token id and (b) decode back to itself.
    """
    ids = tok(delim, add_special_tokens=False).input_ids
    special = set(getattr(tok, "all_special_ids", []) or [])
    plain = bool(ids) and not any(i in special for i in ids) and tok.decode(ids) == delim
    return plain, ids


def resolve_tokenizer_repo(model_id: str, override: str | None, paths) -> str | None:
    if override:
        return override
    for name in ("run_metadata.yaml", "run_metadata.example.yaml"):
        meta = paths.configs_dir / name
        if meta.is_file():
            data = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
            repo = ((data.get("models") or {}).get(model_id) or {}).get("hf_repo")
            if repo:
                return repo
    return None


def _stats(values: list[int]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    pct = lambda p: s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]  # noqa: E731
    return {"n": len(s), "min": s[0], "p50": pct(50), "p95": pct(95), "p100": s[-1]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--cutoff-len", type=int, default=2048,
                    help="must match the LoRA yaml; no templated example may exceed it")
    ap.add_argument("--tokenizer", default=None, help="HF id/path (else run_metadata hf_repo)")
    ap.add_argument("--dataset-dir", default=None, help="else <sft_dir>/<model>")
    ap.add_argument("--limit", type=int, default=0, help="check only the first N examples (smoke)")
    args = ap.parse_args()
    paths = get_paths()

    ddir = pathlib.Path(args.dataset_dir) if args.dataset_dir else paths.sft_dir / args.model
    data_path = ddir / "generation_train.jsonl"
    info_path = ddir / "dataset_info.json"

    print("=" * 70)
    print(f"Phase-3 SFT gate | model={args.model} cutoff_len={args.cutoff_len}")
    print(f"dataset_dir={ddir}")
    print("=" * 70)

    errors: list[str] = []

    # --- gate 1: dataset_info.json well-formed + points at the data file ---
    if not info_path.is_file():
        print(f"FAIL: missing {info_path}", file=sys.stderr)
        return 2
    info = json.loads(info_path.read_text(encoding="utf-8"))
    name = f"tsmc_{args.model}_generation"
    entry = info.get(name)
    if not entry:
        errors.append(f"dataset_info.json has no entry '{name}' (got {list(info)})")
    else:
        if entry.get("formatting") != "sharegpt":
            errors.append(f"dataset_info formatting != sharegpt: {entry.get('formatting')}")
        ref = ddir / entry.get("file_name", "")
        if not ref.is_file():
            errors.append(f"dataset_info file_name not found: {ref}")
    if not data_path.is_file():
        print(f"FAIL: missing {data_path}", file=sys.stderr)
        return 2
    print(f"dataset_info.json: entry '{name}' OK")

    # --- tokenizer ---
    repo = resolve_tokenizer_repo(args.model, args.tokenizer, paths)
    if not repo:
        print("FAIL: no tokenizer repo (set models.<id>.hf_repo in run_metadata.yaml "
              "or pass --tokenizer).", file=sys.stderr)
        return 2
    from transformers import AutoTokenizer  # heavy; llamafactory_env

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    print(f"tokenizer: {repo}")

    # --- gate 3b: the gamma delimiter must be PLAIN TEXT for this tokenizer ---
    family = family_of(args.model)
    delim = gamma_delimiter(family)
    marker_re = _marker_re(delim)
    delim_plain, delim_ids = delimiter_is_plain_text(tok, delim)
    print(f"family={family}  gamma delimiter={delim!r} -> token ids {delim_ids}")
    if not delim_plain:
        errors.append(f"gamma delimiter {delim!r} is NOT plain text for this tokenizer "
                      f"(ids {delim_ids} hit a special token) -> it would corrupt gamma "
                      f"control; choose a delimiter the tokenizer treats as ordinary text")

    # --- per-example checks through the REAL template ---
    examples = [json.loads(x) for x in data_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        examples = examples[: args.limit]

    lengths: list[int] = []
    n_marker = n_baseline = 0
    marker_survival_fail = roundtrip_fail = struct_fail = 0
    over_cutoff: list[int] = []

    for i, ex in enumerate(examples):
        msgs = ex.get("messages")
        # Accept an optional leading system message (e.g. the 7B reason-first prompt);
        # the body must still be exactly [user, assistant].
        sys_msgs = ([msgs[0]] if isinstance(msgs, list) and msgs
                    and msgs[0].get("role") == "system" else [])
        body = msgs[len(sys_msgs):] if isinstance(msgs, list) else None
        if not (isinstance(body, list) and len(body) == 2
                and body[0].get("role") == "user" and body[1].get("role") == "assistant"
                and body[0].get("content") and body[1].get("content")
                and all(s.get("content") for s in sys_msgs)):
            struct_fail += 1
            if struct_fail <= 5:
                errors.append(f"ex{i}: bad message structure")
            continue
        user, asst = body[0]["content"], body[1]["content"]
        prompt_msgs = sys_msgs + [body[0]]  # [system?, user] -- as inference renders the prompt

        m = marker_re.search(user)
        gamma = float(m.group(1)) if m else 1.0
        if m:
            n_marker += 1
            if gamma not in GAMMA_GRID or gamma >= 1.0:
                errors.append(f"ex{i}: marker gamma {gamma} not a sub-1 grid value")
        else:
            n_baseline += 1

        # the prompt half, rendered as inference renders it (default system + wrappers)
        prompt_only = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        if m and m.group(0) not in prompt_only:
            marker_survival_fail += 1
            if marker_survival_fail <= 5:
                errors.append(f"ex{i}: gamma marker {m.group(0)} did NOT survive apply_chat_template")
        if not m and delim in prompt_only:
            errors.append(f"ex{i}: baseline example carries a marker after templating")

        pr = parse_generation(asst)
        if pr.status.parser_branch not in _CLEAN_BRANCHES or not pr.code_snippet:
            roundtrip_fail += 1
            if roundtrip_fail <= 5:
                errors.append(f"ex{i}: assistant target does not round-trip ({pr.status.parser_branch})")

        # Full trained-sequence length. Count the RENDERED strings instead of
        # apply_chat_template(tokenize=True), which some transformers versions return
        # as a dict (len()==2) -- count prompt (already templated) + assistant + eos.
        n_tok = (len(tok(prompt_only, add_special_tokens=False).input_ids)
                 + len(tok(asst, add_special_tokens=False).input_ids) + 1)
        lengths.append(n_tok)
        if n_tok > args.cutoff_len:
            over_cutoff.append(n_tok)

    # --- report ---
    length_stats = _stats(lengths)
    print(f"\nexamples checked: {len(examples)}  (marker={n_marker}, baseline/gamma=1={n_baseline})")
    print(f"  templated token length: {length_stats}")
    print(f"  over cutoff_len({args.cutoff_len}): {len(over_cutoff)}"
          + (f"  max={max(over_cutoff)}" if over_cutoff else ""))
    print(f"  delimiter plain text:     {'OK' if delim_plain else 'FAIL (special token)'}")
    print(f"  marker survives template: {'OK' if marker_survival_fail == 0 else f'FAIL x{marker_survival_fail}'}")
    print(f"  assistant round-trip:     {'OK' if roundtrip_fail == 0 else f'FAIL x{roundtrip_fail}'}")
    print(f"  message structure:        {'OK' if struct_fail == 0 else f'FAIL x{struct_fail}'}")

    if over_cutoff:
        errors.append(f"{len(over_cutoff)} example(s) exceed cutoff_len={args.cutoff_len} "
                      f"(max {max(over_cutoff)}) -> raise cutoff_len or they truncate")

    ok = not errors
    print("\n" + ("RESULT: PASS" if ok else "RESULT: FAIL"))
    if not ok:
        for e in errors[:20]:
            print("  - " + e, file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)
    else:
        print(f"  All examples render consistently under the {family} template and fit cutoff_len.")
        print("  Phase 3 gate cleared -> proceed to Phase-4 LoRA SFT.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
