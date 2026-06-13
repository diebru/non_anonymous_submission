"""Phase-1 inference orchestration (roadmap s8, Phase 1). SERVER-ONLY for real runs.

Drives one ``(task, split)`` at a fixed gamma for one model: select problems from
the frozen manifest, render contract prompts, run vLLM, parse with the frozen
contract parser, count CoT tokens, and emit two parallel per-language artifacts:

  result/<Lang>.jsonl        McEval input  (rich record + raw_generation[0])
  trajectories/<Lang>.jsonl  our long-format records (``pass`` PROVISIONAL=false,
                             filled by the Phase-1.2 eval join via mceval_task_id)

Single-pass for generation/completion; two-pass for explanation (stage-1 describe
-> stage-2 reconstruct, the description kept as the compressible ``cot_text``).
A ``plan_only`` path builds + previews prompts on CPU with no model, so prompts
can be validated locally before the GPU run.

All bulk outputs land under the gitignored ``generations_dir`` (docs/WORKFLOW.md s5).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tsmc.config import ProjectPaths, get_paths
from tsmc.constants import family_of
from tsmc.contract import parse_completion, parse_explanation_stage2, parse_generation
from tsmc.eval import results as R
from tsmc.inference import prompts as P
from tsmc.inference.runner import GenOutput, VLLMRunner

# raw_full_output stitch between the two explanation passes (human-readable, never
# parsed -- code_snippet comes from the stage-2 parse, cot_text from stage-1).
STAGE_SEP = "\n\n===== STAGE 2 (reconstruct) =====\n\n"

# task_type -> compression_method (schema cross-field rule, roadmap s7 / Decision #3).
_COMPRESSION_METHOD = {
    "generation": "model_side",
    "completion": "model_side",   # induced-CoT, test-time (same family as generation)
    "explanation": "post_hoc",    # LLMLingua-2 on the description (Phase 2)
}
_PARSERS = {"generation": parse_generation, "completion": parse_completion}


@dataclass
class HarnessConfig:
    model_id: str                 # constants.MODEL_IDS value (long-format model_id)
    gamma: float = 1.0            # Phase 1 baseline = 1.0
    run_id: str = "run01"
    system: str | None = None     # pinned system prompt (None -> chat-template default)


def _gamma_dir(paths: ProjectPaths, cfg: HarnessConfig, task: str, split: str) -> Path:
    gtag = f"gamma{cfg.gamma:g}"
    return paths.generations_dir / cfg.model_id / cfg.run_id / task / split / gtag


def _run_dir(
    paths: ProjectPaths, cfg: HarnessConfig, task: str, split: str, shard_id: int = -1
) -> Path:
    """Gamma dir, or a per-shard subdir when running data-parallel (shard_id>=0)."""
    base = _gamma_dir(paths, cfg, task, split)
    return base if shard_id < 0 else base / f"shard{shard_id}"


def _trajectory(
    unit: P.ProblemUnit,
    *,
    cot_text: str,
    code_snippet: str,
    status: Any,
    cot_token_count: int,
    code_token_count: int,
    raw_full_output: str,
    cfg: HarnessConfig,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Assemble one long-format record. ``pass`` is PROVISIONAL (false) until the
    Phase-1.2 eval join overwrites it and adds the three-way ``outcome``."""
    return {
        "problem_id": unit.problem_id,
        "task_type": unit.task_type,
        "completion_subtype": unit.completion_subtype,
        "model_id": cfg.model_id,
        "gamma": cfg.gamma,
        "run_id": cfg.run_id,
        "raw_full_output": raw_full_output,
        "cot_text": cot_text,
        "code_snippet": code_snippet,
        "cot_token_count": cot_token_count,
        "code_token_count": code_token_count,  # for the completion cot/code gate (Decision #5)
        "compression_ratio": cfg.gamma,
        "pass": False,  # provisional; set by the eval join
        "extraction_status": status.to_dict(),
        "cot_origin": "original" if cfg.gamma >= 1.0 else "compressed",
        "compression_method": _COMPRESSION_METHOD[unit.task_type],
        "gate_decision": None,           # completion gate set in Phase 1.3
        "gate_measured_median": None,
        "split": unit.split,
        "lang": unit.mceval_lang.lower(),
        "difficulty": unit.difficulty,
        "difficulty_source": unit.difficulty_source,
        # non-schema provenance (ignored by validate_record; used by the join + energy)
        "_provenance": provenance,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


# --- plan-only (CPU; no model) -------------------------------------------------

def plan_task(
    task: str,
    split: str,
    cfg: HarnessConfig,
    paths: ProjectPaths | None = None,
    trio_only: bool = False,
    limit: int = 0,
) -> dict[str, Any]:
    """Build + preview prompts without a model (CPU). Writes a prompts preview and
    returns per-language counts. Validates selection + assembly before the GPU run."""
    paths = paths or get_paths()
    units = P.select_units(task, split, paths, trio_only, limit)
    preview = []
    for u in units:
        item = {
            "problem_id": u.problem_id, "task_type": u.task_type,
            "completion_subtype": u.completion_subtype, "lang": u.mceval_lang,
            "mceval_task_id": u.mceval_task_id,
            "user_text": P.stage1_user_text(u, cfg.gamma, family_of(cfg.model_id)),
        }
        if task == "explanation":
            item["two_pass"] = True
            item["stage2_user_text_template"] = P.explanation_stage2_user(u, "<DESCRIPTION>")
        preview.append(item)
    out_dir = _run_dir(paths, cfg, task, split) / "preview"
    _write_jsonl(out_dir / "prompts.jsonl", preview)
    by_lang = {lang: len(us) for lang, us in P.group_by_language(units).items()}
    return {"task": task, "split": split, "n_units": len(units),
            "by_language": by_lang, "preview": str(out_dir / "prompts.jsonl")}


# --- real run (GPU) ------------------------------------------------------------

def _stitch_timing(*outs: GenOutput) -> dict[str, Any]:
    return {
        "arrival_time": outs[0].arrival_time,
        "finished_time": outs[-1].finished_time,
        "n_prompt_tokens": [o.n_prompt_tokens for o in outs],
        "n_output_tokens": [o.n_output_tokens for o in outs],
        "finish_reason": [o.finish_reason for o in outs],
    }


def _run_single_pass(
    task: str, units: list[P.ProblemUnit], runner: VLLMRunner, cfg: HarnessConfig
) -> list[dict[str, Any]]:
    parser = _PARSERS[task]
    family = family_of(cfg.model_id)
    prompts = [runner.render(P.chat_messages(P.reasoning_user_text(u, cfg.gamma, family), cfg.system))
               for u in units]
    outs = runner.generate(prompts)
    rows: list[dict[str, Any]] = []
    for u, o in zip(units, outs):
        pr = parser(o.text, entry_point=u.entry_point, finish_reason=o.finish_reason)
        rows.append(_trajectory(
            u, cot_text=pr.cot_text, code_snippet=pr.code_snippet, status=pr.status,
            cot_token_count=runner.count_tokens(pr.cot_text),
            code_token_count=runner.count_tokens(pr.code_snippet), raw_full_output=o.text,
            cfg=cfg,
            provenance={"mceval_task_id": u.mceval_task_id, "mceval_lang": u.mceval_lang,
                        "timing": _stitch_timing(o)},
        ))
    return rows


def _run_explanation(
    units: list[P.ProblemUnit], runner: VLLMRunner, cfg: HarnessConfig
) -> list[dict[str, Any]]:
    stage1_prompts = [runner.render(P.chat_messages(P.explanation_stage1_user(u), cfg.system))
                      for u in units]
    stage1 = runner.generate(stage1_prompts)
    stage2_prompts = [
        runner.render(P.chat_messages(P.explanation_stage2_user(u, s1.text), cfg.system))
        for u, s1 in zip(units, stage1)
    ]
    stage2 = runner.generate(stage2_prompts)
    rows: list[dict[str, Any]] = []
    for u, s1, s2 in zip(units, stage1, stage2):
        pr = parse_explanation_stage2(s2.text, entry_point=u.entry_point, finish_reason=s2.finish_reason)
        rows.append(_trajectory(
            u, cot_text=s1.text, code_snippet=pr.code_snippet, status=pr.status,
            cot_token_count=runner.count_tokens(s1.text),
            code_token_count=runner.count_tokens(pr.code_snippet),
            raw_full_output=s1.text + STAGE_SEP + s2.text, cfg=cfg,
            provenance={"mceval_task_id": u.mceval_task_id, "mceval_lang": u.mceval_lang,
                        "timing": _stitch_timing(s1, s2)},
        ))
    return rows


def shard_units(units: list[P.ProblemUnit], shards: int, shard_id: int) -> list[P.ProblemUnit]:
    """Striped data-parallel split: shard ``shard_id`` of ``shards`` takes every
    ``shards``-th unit. Striping (not contiguous slicing) balances per-language and
    per-length load across GPUs. ``shards<=1`` returns all units."""
    if shards <= 1:
        return units
    return units[shard_id::shards]


def run_task(
    task: str,
    split: str,
    runner: VLLMRunner,
    cfg: HarnessConfig,
    paths: ProjectPaths | None = None,
    trio_only: bool = False,
    limit: int = 0,
    shards: int = 1,
    shard_id: int = 0,
) -> dict[str, Any]:
    """Run one (task, split) at cfg.gamma. Writes result/ + trajectories/ per
    language and a run_meta.json; returns a summary dict. With ``shards>1`` this
    process handles only its stripe and writes to a ``shard{shard_id}`` subdir
    (merge afterwards with scripts/merge_shards.py)."""
    paths = paths or get_paths()
    started = time.time()
    units = shard_units(P.select_units(task, split, paths, trio_only, limit), shards, shard_id)
    out_shard = shard_id if shards > 1 else -1

    # Wall-clock window around generation only (model load already happened in the
    # caller). join_energy integrates the power curve over THIS window so energy is
    # pure decode -- excludes load and, crucially, the McEval Docker scoring that
    # runs in a later step (the accuracy control stays outside the energy window).
    gen_start = time.time()
    if task == "explanation":
        rows = _run_explanation(units, runner, cfg)
    else:
        rows = _run_single_pass(task, units, runner, cfg)
    gen_end = time.time()

    # partition rows by original-case language; write paired result + trajectory files
    out_dir = _run_dir(paths, cfg, task, split, out_shard)
    rows_by_lang: dict[str, list[dict[str, Any]]] = {}
    for u, row in zip(units, rows):
        rows_by_lang.setdefault(u.mceval_lang, []).append(row)
    unit_by_taskid = {u.mceval_task_id: u for u in units}

    for lang, lang_rows in rows_by_lang.items():
        result_items = [
            R.build_result_item(
                unit_by_taskid[r["_provenance"]["mceval_task_id"]].record,
                R.wrap_code(r["code_snippet"], lang),
            )
            for r in lang_rows
        ]
        _write_jsonl(out_dir / "result" / f"{lang}.jsonl", result_items)
        _write_jsonl(out_dir / "trajectories" / f"{lang}.jsonl", lang_rows)

    summary = {
        "task": task, "split": split, "model_id": cfg.model_id, "gamma": cfg.gamma,
        "run_id": cfg.run_id, "n_units": len(units),
        "shards": shards, "shard_id": shard_id if shards > 1 else None,
        "by_language": {lang: len(rs) for lang, rs in rows_by_lang.items()},
        "elapsed_sec": round(time.time() - started, 1),
        "generate_window": [gen_start, gen_end],  # wall clock; energy-join window
        "out_dir": str(out_dir),
        "runner": asdict(runner.cfg),
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _concat_jsonl(srcs: list[Path], dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(dst, "w", encoding="utf-8") as out:
        for src in srcs:
            with open(src, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        n += 1
    return n


def merge_shards(
    task: str, split: str, cfg: HarnessConfig, paths: ProjectPaths | None = None
) -> dict[str, Any]:
    """Concatenate ``shard*/`` result + trajectory files into the gamma dir, so the
    merged ``result/`` is directly consumable by the Phase-1.2 McEval eval. Idempotent."""
    paths = paths or get_paths()
    gdir = _gamma_dir(paths, cfg, task, split)
    shard_dirs = sorted(d for d in gdir.glob("shard*") if d.is_dir())
    if not shard_dirs:
        return {"task": task, "split": split, "merged_shards": 0, "by_language": {}}
    langs: set[str] = set()
    for d in shard_dirs:
        langs.update(p.name for p in (d / "result").glob("*.jsonl"))
    by_lang: dict[str, int] = {}
    for fname in sorted(langs):
        for sub in ("result", "trajectories"):
            srcs = [d / sub / fname for d in shard_dirs if (d / sub / fname).is_file()]
            n = _concat_jsonl(srcs, gdir / sub / fname)
            if sub == "result":
                by_lang[fname[:-len(".jsonl")]] = n
    summary = {"task": task, "split": split, "merged_shards": len(shard_dirs),
               "by_language": by_lang, "out_dir": str(gdir)}
    with open(gdir / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary
