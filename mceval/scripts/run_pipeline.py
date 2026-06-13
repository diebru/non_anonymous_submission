#!/usr/bin/env python3
"""Master orchestrator: run Phases 1->4 end-to-end for ONE model.

Shells out per stage with the correct conda env (``conda run -n <env>``), stops at
the two human checkpoints (the behavioral +-3% gate after Phase 1, the knob gate
after the merge), and is resumable. It is the automation of docs/PIPELINE_RUNBOOK.md
-- every stage maps 1:1 to a command in that recipe, so the runbook stays the source
of truth and this script just sequences it across the three execution environments.

  Environments (docs/PIPELINE_RUNBOOK.md s1):
    tokenskip_env  -> vLLM inference, LLMLingua-2 compression, peft merge, parsing,
                      analysis, AND McEval scoring (score_generations shells to Docker
                      itself). transformers ~= 4.46.
    llamafactory_env -> LoRA SFT (llamafactory-cli train) + the Phase-3 dataset gate
                      (check_sft_dataset uses LlamaFactory's chat template). 5.2.0.

  Gates (STOP unless --force):
    p1_gate  -- behavioral +-3% train-vs-test accuracy (manifest confirm-freeze).
    p4_knob  -- merged-model median CoT must fall monotonically with gamma.
  At a gate the script prints the numbers and halts so a human can judge (the Coder-3B
  behavioral gate was a marginal ~1.2-SE miss that was accepted with a note); resume
  with --from-stage <next> (add --force to auto-pass remaining gates).

  Merge uses scripts/merge_lora.py (peft) in tokenskip_env -- NOT ``llamafactory-cli
  export``, which is broken on the server's transformers 5.2.0 (PIPELINE_RUNBOOK s5).

  7B/14B get --tensor-parallel-size 2 on inference + knob automatically; the energy
  sweep stays single-GPU by design (clean per-GPU attribution; 14B bf16 ~= 28 GB fits
  one A6000). SFT multi-GPU DDP is automatic in LlamaFactory.

Examples:
    # preview every resolved command + the generated LoRA yaml, run nothing:
    python3 scripts/run_pipeline.py --model qwen2.5-14b-instruct --dry-run

    # run it for real; it will STOP at p1_gate and p4_knob for review:
    python3 scripts/run_pipeline.py --model qwen2.5-14b-instruct

    # resume after eyeballing the Phase-1 gate:
    python3 scripts/run_pipeline.py --model qwen2.5-14b-instruct --from-stage p1_corpus

    # one stage only / unattended full run:
    python3 scripts/run_pipeline.py --model qwen2.5-3b-instruct --only p4_curves
    python3 scripts/run_pipeline.py --model qwen2.5-3b-instruct --force

Recommended queue order (non-code ladder): 14B -> 7B -> 3B (largest first, so a
big-model failure surfaces before the cheap runs).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass

import yaml

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import MODEL_IDS, family_of  # noqa: E402

# Model family -> LLaMA-Factory chat template. MUST equal vLLM's apply_chat_template
# for that model, or SFT and inference diverge (the Phase-3 gate checks this).
LF_TEMPLATE = {"qwen": "qwen", "llama3": "llama3"}

# Per-family SFT cutoff_len. Llama-3 tokenizes the same code into MORE tokens than
# Qwen (measured: templated p100 ~2255 for Llama vs ~1211-1333 for Qwen), so the 2048
# that comfortably fit Qwen truncates the longest Llama targets (drops sentinel+code).
# Drives BOTH the LoRA yaml and the p3_check_sft gate from one place so they can't drift.
CUTOFF_LEN = {"qwen": 2048, "llama3": 3072}

# conda env name per logical environment.
ENV_CONDA = {"tokenskip": "tokenskip_env", "llamafactory": "llamafactory_env"}

CORPUS_RUN_ID = "run01"   # default Phase 1-3 + merge run-id (names lora_/merged_sft_<id>)
SWEEP_RUN_ID = "sft01"    # default energy-sweep run-id (kept apart from the base baseline)


@dataclass(frozen=True)
class Stage:
    key: str
    phase: str
    env: str | None   # "tokenskip" | "llamafactory" | None (in-process)
    gate: bool
    desc: str


# Linear pipeline. Order IS the execution order; keys drive --from-stage/--only/--stop-after.
STAGES: tuple[Stage, ...] = (
    Stage("p1_infer",     "1", "tokenskip",    False, "baseline CoT+code generation (train+test)"),
    Stage("p1_score",     "1", "tokenskip",    False, "McEval execution scoring (Docker)"),
    Stage("p1_gate",      "1", "tokenskip",    True,  "behavioral +-3% gate (manifest confirm-freeze)"),
    Stage("p1_corpus",    "1", "tokenskip",    False, "correct-CoT corpus (train)"),
    Stage("p2_compress",  "2", "tokenskip",    False, "LLMLingua-2 multi-gamma compression"),
    Stage("p2_validate",  "2", "tokenskip",    False, "compression monotonicity gate"),
    Stage("p3_build_sft", "3", "tokenskip",    False, "build gamma-control SFT dataset"),
    Stage("p3_check_sft", "3", "llamafactory", False, "SFT dataset gate (per-family cutoff_len)"),
    Stage("p4_yaml",      "4", None,           False, "generate the per-model LoRA yaml"),
    Stage("p4_sft",       "4", "llamafactory", False, "LoRA SFT (llamafactory-cli train)"),
    Stage("p4_merge",     "4", "tokenskip",    False, "merge adapter into base (peft)"),
    Stage("p4_knob",      "4", "tokenskip",    True,  "knob gate (median CoT falls with gamma)"),
    Stage("p4_sweep",     "4", "tokenskip",    False, "12-gamma energy-instrumented sweep (single GPU)"),
    Stage("p4_curves",    "4", "tokenskip",    False, "accuracy/energy/format_fail curves"),
    Stage("p4_plots",     "4", "tokenskip",    False, "curve plots"),
)
STAGE_BY_KEY = {s.key: s for s in STAGES}


# --- per-model resolution ------------------------------------------------------

def resolve_model_meta(model_id: str, paths) -> tuple[str | None, str | None]:
    """(hf_repo, revision) from run_metadata.yaml, falling back to the committed
    .example (so a fresh checkout still resolves the repo). Mirrors run_inference."""
    for name in ("run_metadata.yaml", "run_metadata.example.yaml"):
        meta = paths.configs_dir / name
        if not meta.is_file():
            continue
        data = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
        entry = (data.get("models") or {}).get(model_id) or {}
        repo = entry.get("hf_repo")
        if repo:
            commit = entry.get("commit")
            rev = commit if commit and not str(commit).startswith("TBD") else None
            return repo, rev
    return None, None


def is_big(model_id: str) -> bool:
    """7B/14B need --tensor-parallel-size 2 for inference/knob (runbook s2.4)."""
    return any(tag in model_id for tag in ("7b", "14b"))


def hf_cache_commit(repo_id: str) -> str | None:
    """Best-effort: the commit ``main`` resolves to in the local HF cache, read
    straight off disk (no huggingface_hub dependency) so an UNPINNED merge still
    records which snapshot it used. None if not a repo id / not cached yet."""
    if not repo_id or "/" not in repo_id or pathlib.Path(repo_id).exists():
        return None  # a local dir is inherently pinned to that path
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache:
        hub = pathlib.Path(cache)
    else:
        home = os.environ.get("HF_HOME")
        hub = (pathlib.Path(home) if home else pathlib.Path.home() / ".cache" / "huggingface") / "hub"
    ref = hub / f"models--{repo_id.replace('/', '--')}" / "refs" / "main"
    try:
        return ref.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


@dataclass
class Ctx:
    args: argparse.Namespace
    paths: object
    model: str
    hf_repo: str | None
    commit: str | None
    big: bool
    adapter_dir: str
    merged_dir: str
    dataset_dir: str
    lora_yaml: pathlib.Path


def build_ctx(args, paths) -> Ctx:
    hf_repo, commit = resolve_model_meta(args.model, paths)
    rid = args.run_id
    weights = paths.weights_dir / args.model
    return Ctx(
        args=args, paths=paths, model=args.model, hf_repo=hf_repo, commit=commit,
        big=is_big(args.model),
        adapter_dir=str(weights / f"lora_sft_{rid}"),
        merged_dir=str(weights / f"merged_sft_{rid}"),
        dataset_dir=str(paths.sft_dir / args.model),
        lora_yaml=paths.configs_dir / "llamafactory" / f"{args.model}_lora.yaml",
    )


# --- LoRA yaml generation (Phase 4a) -------------------------------------------

LORA_YAML_TEMPLATE = """\
# LoRA SFT config for {model} (TokenSkip x McEval, Phase 4).
# AUTO-GENERATED by scripts/run_pipeline.py from the per-size template -- review the
# 3 machine paths below (model_name_or_path / dataset_dir / output_dir) before SFT.
# Hyperparameters match the Coder-3B recipe (rank8/alpha16, lr 5e-5, 3 ep, bs1 x ga8,
# bf16, cutoff_len 2048); size only changes the base + paths (DDP is automatic).

### model
model_name_or_path: {base}
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05

### dataset
dataset: tsmc_{model}_generation
dataset_dir: {dataset_dir}
template: {template}
cutoff_len: {cutoff_len}
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: {output_dir}
logging_steps: 10
save_steps: 300
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
optim: adamw_torch
learning_rate: 5.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 300
"""


def render_lora_yaml(ctx: Ctx) -> str:
    base = ctx.hf_repo or f"TBD-FILL-base-for-{ctx.model}"
    fam = family_of(ctx.model)
    return LORA_YAML_TEMPLATE.format(
        model=ctx.model, base=base, dataset_dir=ctx.dataset_dir, output_dir=ctx.adapter_dir,
        template=LF_TEMPLATE[fam], cutoff_len=CUTOFF_LEN[fam],
    )


def do_yaml_stage(ctx: Ctx, dry_run: bool) -> int:
    rendered = render_lora_yaml(ctx)
    target = ctx.lora_yaml
    if dry_run:
        print(f"[in-proc] would write {target} :\n")
        print("\n".join("    " + ln for ln in rendered.splitlines()))
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not ctx.args.regen_yaml:
        print(f"[in-proc] {target} exists -> keeping it (pass --regen-yaml to overwrite).")
    else:
        target.write_text(rendered, encoding="utf-8")
        print(f"[in-proc] wrote {target}")
    print(f"  review before SFT: base={ctx.hf_repo}  dataset_dir={ctx.dataset_dir}")
    print(f"  output_dir={ctx.adapter_dir}")
    return 0


# --- per-stage command builders ------------------------------------------------

def _py(script: str) -> list[str]:
    # python3 -u for live (unbuffered) logs when piped to a file (ops note in memory).
    return ["python3", "-u", str(HERE / script)]


def stage_command(key: str, ctx: Ctx) -> list[str]:
    """Inner argv (executable + args), WITHOUT the conda-run prefix. The yaml stage
    is in-process and never reaches here."""
    a = ctx.args
    M, rid, srid = ctx.model, a.run_id, a.sweep_run_id
    tp = ["--tensor-parallel-size", "2"] if ctx.big else []
    sysarg = ["--system", a.system] if a.system else []  # pinned system prompt (e.g. 7B reason-first)

    if key == "p1_infer":
        cmd = _py("run_inference.py") + ["--task", "generation", "--split", "both",
                                         "--model", M, "--run-id", rid] + tp + sysarg
        if a.trio_only:
            cmd.append("--trio-only")
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        return cmd
    if key == "p1_score":
        cmd = _py("score_generations.py") + ["--task", "generation", "--split", "both",
                                             "--model", M, "--run-id", rid]
        if a.digest:
            cmd += ["--digest", a.digest]
        return cmd
    if key == "p1_gate":
        return _py("phase1_gates.py") + ["--model", M, "--run-id", rid]
    if key == "p1_corpus":
        return _py("build_corpus.py") + ["--model", M, "--run-id", rid, "--split", "train"]
    if key == "p2_compress":
        return _py("compress_corpus.py") + ["--model", M, "--run-id", rid,
                                            "--task", "generation", "--split", "train"]
    if key == "p2_validate":
        return _py("validate_compression.py") + ["--model", M, "--run-id", rid,
                                                 "--tasks", "generation", "--split", "train"]
    if key == "p3_build_sft":
        return _py("build_sft_dataset.py") + ["--model", M, "--run-id", rid, "--count-tokens"] + sysarg
    if key == "p3_check_sft":
        return _py("check_sft_dataset.py") + ["--model", M,
                                              "--cutoff-len", str(CUTOFF_LEN[family_of(M)])]
    if key == "p4_sft":
        return ["llamafactory-cli", "train", str(ctx.lora_yaml)]
    if key == "p4_merge":
        if not ctx.hf_repo:
            raise SystemExit(f"merge needs a base repo for {M}: set models.{M}.hf_repo "
                             "in configs/run_metadata.yaml")
        cmd = _py("merge_lora.py") + ["--base", ctx.hf_repo,
                                      "--adapter", ctx.adapter_dir, "--output", ctx.merged_dir]
        if ctx.commit:
            cmd += ["--revision", ctx.commit]
        return cmd
    if key == "p4_knob":
        return _py("validate_knob.py") + ["--model", M, "--model-path", ctx.merged_dir,
                                          "--limit", "3"] + tp + sysarg
    if key == "p4_sweep":
        cmd = _py("run_energy_sweep.py") + ["--model", M, "--run-id", srid,
                                            "--model-path", ctx.merged_dir]
        if a.digest:
            cmd += ["--digest", a.digest]
        if a.gpu_index is not None:
            cmd += ["--gpu-index", str(a.gpu_index)]
        if a.no_pdu:
            cmd.append("--no-pdu")
        if a.trio_only:
            cmd.append("--trio-only")
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        cmd += sysarg
        return cmd
    if key == "p4_curves":
        return _py("build_curves.py") + ["--model", M, "--task", "generation",
                                         "--split", "test", "--run-id", srid]
    if key == "p4_plots":
        return _py("plot_curves.py") + ["--model", M, "--task", "generation", "--split",
                                        "test", "--run-id", srid, "--corpus-run-id", rid]
    raise KeyError(key)


def wrap(env: str, inner: list[str]) -> list[str]:
    # --no-capture-output so the child's stdout/stderr stream live instead of buffering.
    return ["conda", "run", "--no-capture-output", "-n", ENV_CONDA[env], *inner]


# --- gate inspection -----------------------------------------------------------

def report_phase1_gate(ctx: Ctx) -> bool:
    p = ctx.paths.generations_dir / ctx.model / ctx.args.run_id / "phase1_gates.json"
    if not p.is_file():
        print(f"  (gate file missing: {p})")
        return False
    data = json.loads(p.read_text(encoding="utf-8"))
    print("  behavioral +-3% (healthy train vs test accuracy):")
    for task, g in (data.get("behavioral") or {}).items():
        fmt = lambda x: "n/a" if x is None else f"{x:.4f}"
        verdict = "WITHIN" if g.get("within_tol") else "OUTSIDE"
        print(f"    {task:11} train={fmt(g.get('train_accuracy'))} "
              f"test={fmt(g.get('test_accuracy'))} |d|={fmt(g.get('abs_delta'))} [{verdict}]")
    ok = bool(data.get("manifest_confirm_frozen"))
    print(f"  manifest_confirm_frozen: {ok}")
    return ok


def report_knob_gate(ctx: Ctx) -> bool:
    p = ctx.paths.generations_dir / ctx.model / "knob_validation_merged.json"
    if not p.is_file():
        print(f"  (gate file missing: {p})")
        return False
    data = json.loads(p.read_text(encoding="utf-8"))
    print(f"  median CoT series (gamma 1.0 -> 0.1): {data.get('median_series')}")
    print(f"  shrink top->bottom: {data.get('shrink_top_to_bottom')}  "
          f"monotonic: {data.get('monotonic')}")
    ok = bool(data.get("pass"))
    print(f"  knob pass: {ok}")
    return ok


GATE_REPORTERS = {"p1_gate": report_phase1_gate, "p4_knob": report_knob_gate}


# --- run loop ------------------------------------------------------------------

def select_stages(args) -> list[Stage]:
    keys = [s.key for s in STAGES]
    if args.only:
        return [STAGE_BY_KEY[args.only]]
    start = keys.index(args.from_stage) if args.from_stage else 0
    end = keys.index(args.stop_after) + 1 if args.stop_after else len(keys)
    return list(STAGES[start:end])


def print_plan(ctx: Ctx, selected: list[Stage]) -> None:
    a = ctx.args
    print("=" * 78)
    print(f"PIPELINE | model={ctx.model}  big(TP=2)={ctx.big}  run-id={a.run_id}  "
          f"sweep-run-id={a.sweep_run_id}")
    print(f"base={ctx.hf_repo}  revision={ctx.commit or '(unpinned)'}  "
          f"digest={a.digest or '(from run_metadata)'}")
    print(f"merged={ctx.merged_dir}")
    print(f"mode={'DRY-RUN' if a.dry_run else 'EXECUTE'}  force={a.force}")
    print("-" * 78)
    print("stages to run (>> = will execute; GATE = stops unless --force):")
    for s in selected:
        env = s.env or "in-proc"
        gate = "  [GATE]" if s.gate else ""
        print(f"  >> {s.key:13} P{s.phase} ({env:12}) {s.desc}{gate}")
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODEL_IDS)
    ap.add_argument("--run-id", default=CORPUS_RUN_ID,
                    help=f"Phase 1-3 + merge run-id (default {CORPUS_RUN_ID}; names "
                         "lora_sft_<id>/merged_sft_<id>)")
    ap.add_argument("--sweep-run-id", default=SWEEP_RUN_ID,
                    help=f"energy-sweep run-id (default {SWEEP_RUN_ID}; kept apart from "
                         "the base-model Phase-1 baseline)")
    ap.add_argument("--from-stage", choices=[s.key for s in STAGES], default=None)
    ap.add_argument("--stop-after", choices=[s.key for s in STAGES], default=None)
    ap.add_argument("--only", choices=[s.key for s in STAGES], default=None)
    ap.add_argument("--force", action="store_true", help="do not STOP at gate checkpoints")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every resolved command + the LoRA yaml; run nothing")
    ap.add_argument("--regen-yaml", action="store_true",
                    help="overwrite an existing per-model LoRA yaml")
    ap.add_argument("--system", default=None,
                    help="pinned system prompt threaded through p1_infer/p3_build_sft/p4_knob/"
                         "p4_sweep (e.g. the 7B reason-first prompt; keeps SFT<->inference identical)")
    ap.add_argument("--digest", default=None, help="McEval sha256 (else from run_metadata)")
    ap.add_argument("--gpu-index", type=int, default=None, help="dedicated GPU for the sweep")
    ap.add_argument("--no-pdu", action="store_true", help="sweep: GPU energy only (skip PDU)")
    ap.add_argument("--limit", type=int, default=0, help="smoke: problems/lang on infer + sweep")
    ap.add_argument("--trio-only", action="store_true", help="smoke: Python/C/Rust on infer + sweep")
    args = ap.parse_args()

    paths = get_paths()
    ctx = build_ctx(args, paths)
    selected = select_stages(args)
    print_plan(ctx, selected)

    for s in selected:
        print(f"\n{'#' * 78}\n# [{s.key}] P{s.phase} ({s.env or 'in-proc'}) {s.desc}\n{'#' * 78}")

        # provenance: an unpinned merge still records which base snapshot it used.
        if s.key == "p4_merge" and ctx.commit is None:
            print(f"!! WARNING: commit for {ctx.model} is TBD -> merging UNPINNED (revision=None).")
            sha = hf_cache_commit(ctx.hf_repo)
            if sha:
                print(f"   resolved base snapshot in HF cache = {sha}")
                print(f"   -> paste into run_metadata.yaml models.{ctx.model}.commit to pin it.")
            else:
                print(f"   base {ctx.hf_repo} not in local HF cache yet (resolves main at load).")

        # in-process: LoRA yaml generation
        if s.env is None:
            if do_yaml_stage(ctx, args.dry_run) != 0:
                return 1
            continue

        inner = stage_command(s.key, ctx)
        full = wrap(s.env, inner)
        if args.dry_run:
            print("[env %s] %s" % (ENV_CONDA[s.env], " ".join(full)))
            if s.gate:
                print(f"  [GATE] would STOP after this stage unless --force "
                      f"(reads {s.key} result, then halts).")
            continue

        rc = subprocess.run(full, cwd=REPO_ROOT).returncode

        if not s.gate:
            if rc != 0:
                print(f"\nSTAGE FAILED: [{s.key}] rc={rc}. Fix it, then resume with "
                      f"--from-stage {s.key}.")
                return rc
            continue

        # --- gate stage: rc 2 (or unexpected) = a real stage error -> abort ------
        if rc not in (0, 1):
            print(f"\nSTAGE ERRORED: [{s.key}] rc={rc} (not a gate verdict). Resume with "
                  f"--from-stage {s.key}.")
            return rc
        print(f"\n--- GATE [{s.key}] ---")
        passed = GATE_REPORTERS[s.key](ctx)
        nxt = _next_key(s.key)
        if not args.force:
            verdict = "PASS" if passed else "REVIEW NEEDED"
            print(f"\nGATE CHECKPOINT [{s.key}] {verdict}. Inspect the numbers above.")
            if nxt:
                print(f"To proceed: python3 scripts/run_pipeline.py --model {ctx.model} "
                      f"--from-stage {nxt}  (add --force to auto-pass remaining gates)")
            return 0
        if not passed:
            print(f"\n!! WARNING: gate [{s.key}] did NOT pass but --force given -> continuing.")
        else:
            print(f"\nGATE [{s.key}] PASS (--force) -> continuing.")

    print(f"\n{'=' * 78}\nPipeline finished: {selected[0].key} .. {selected[-1].key} "
          f"for {ctx.model}.\n{'=' * 78}")
    return 0


def _next_key(key: str) -> str | None:
    keys = [s.key for s in STAGES]
    i = keys.index(key)
    return keys[i + 1] if i + 1 < len(keys) else None


if __name__ == "__main__":
    raise SystemExit(main())
