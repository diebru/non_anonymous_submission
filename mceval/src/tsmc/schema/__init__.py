"""Long-format record schema and validators (roadmap s7). FROZEN in Phase 0.

One row per (problem_id x task_type x completion_subtype x model_id x gamma x
run_id). This module is the single definition of a result record: the field
spec, the ``extraction_status`` struct, dataclasses for in-code construction,
and ``validate_record`` for checking dicts loaded from JSONL.

CPU-only, stdlib only (tokenizer-free): ``cot_token_count`` is populated
downstream by the inference harness using the model tokenizer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tsmc import constants as C

# 'pass' is a Python keyword, so the dataclass field is `passed`; it serializes
# to/from the JSON key "pass" (McEval's verdict name).
PASS_KEY = "pass"


@dataclass
class ExtractionStatus:
    """Diagnostics the parser records for every row (roadmap s4.4)."""

    fence_found: bool
    entry_point_found: bool
    truncated: bool
    parser_branch: str  # one of constants.PARSER_BRANCHES

    def to_dict(self) -> dict[str, Any]:
        return {
            "fence_found": self.fence_found,
            "entry_point_found": self.entry_point_found,
            "truncated": self.truncated,
            "parser_branch": self.parser_branch,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExtractionStatus":
        return cls(
            fence_found=d["fence_found"],
            entry_point_found=d["entry_point_found"],
            truncated=d["truncated"],
            parser_branch=d["parser_branch"],
        )


# Field spec drives the structural part of validation. Each entry:
#   name -> (python type(s), nullable, enum-or-None)
_STR = (str,)
_FIELD_SPEC: dict[str, tuple[tuple[type, ...], bool, tuple[str, ...] | None]] = {
    "problem_id": (_STR, False, None),
    "task_type": (_STR, False, C.TASK_TYPES),
    "completion_subtype": (_STR, True, C.COMPLETION_SUBTYPES),
    "model_id": (_STR, False, C.MODEL_IDS),
    "gamma": ((int, float), False, None),
    "run_id": (_STR, False, None),
    "raw_full_output": (_STR, False, None),
    "cot_text": (_STR, True, None),
    "code_snippet": (_STR, False, None),
    "cot_token_count": ((int,), False, None),
    "compression_ratio": ((int, float), False, None),
    PASS_KEY: ((bool,), False, None),
    "cot_origin": (_STR, False, C.COT_ORIGINS),
    "compression_method": (_STR, False, C.COMPRESSION_METHODS),
    "gate_decision": (_STR, True, C.GATE_DECISIONS),
    "split": (_STR, False, C.SPLITS),
    "lang": (_STR, False, None),
    "difficulty": (_STR, False, C.DIFFICULTY_LEVELS),
    "difficulty_source": (_STR, False, C.DIFFICULTY_SOURCES),
}

FIELD_NAMES: tuple[str, ...] = (*_FIELD_SPEC.keys(), "extraction_status")


@dataclass
class LongFormatRecord:
    """In-code representation of one result row. Use ``to_dict`` for JSONL."""

    problem_id: str
    task_type: str
    model_id: str
    gamma: float
    run_id: str
    raw_full_output: str
    code_snippet: str
    cot_token_count: int
    compression_ratio: float
    passed: bool
    extraction_status: ExtractionStatus
    cot_origin: str
    compression_method: str
    split: str
    lang: str
    difficulty: str
    difficulty_source: str
    completion_subtype: str | None = None
    cot_text: str | None = None
    gate_decision: str | None = None
    gate_measured_median: float | None = None  # companion to gate_decision
    energy: dict[str, Any] | None = None  # reserved (roadmap s7 energy_*)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "problem_id": self.problem_id,
            "task_type": self.task_type,
            "completion_subtype": self.completion_subtype,
            "model_id": self.model_id,
            "gamma": self.gamma,
            "run_id": self.run_id,
            "raw_full_output": self.raw_full_output,
            "cot_text": self.cot_text,
            "code_snippet": self.code_snippet,
            "cot_token_count": self.cot_token_count,
            "compression_ratio": self.compression_ratio,
            PASS_KEY: self.passed,
            "extraction_status": self.extraction_status.to_dict(),
            "cot_origin": self.cot_origin,
            "compression_method": self.compression_method,
            "gate_decision": self.gate_decision,
            "gate_measured_median": self.gate_measured_median,
            "split": self.split,
            "lang": self.lang,
            "difficulty": self.difficulty,
            "difficulty_source": self.difficulty_source,
        }
        if self.energy is not None:
            d["energy"] = self.energy
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LongFormatRecord":
        return cls(
            problem_id=d["problem_id"],
            task_type=d["task_type"],
            completion_subtype=d.get("completion_subtype"),
            model_id=d["model_id"],
            gamma=d["gamma"],
            run_id=d["run_id"],
            raw_full_output=d["raw_full_output"],
            cot_text=d.get("cot_text"),
            code_snippet=d["code_snippet"],
            cot_token_count=d["cot_token_count"],
            compression_ratio=d["compression_ratio"],
            passed=d[PASS_KEY],
            extraction_status=ExtractionStatus.from_dict(d["extraction_status"]),
            cot_origin=d["cot_origin"],
            compression_method=d["compression_method"],
            gate_decision=d.get("gate_decision"),
            gate_measured_median=d.get("gate_measured_median"),
            split=d["split"],
            lang=d["lang"],
            difficulty=d["difficulty"],
            difficulty_source=d.get("difficulty_source"),
            energy=d.get("energy"),
        )


def validate_extraction_status(d: Any, prefix: str = "extraction_status") -> list[str]:
    errors: list[str] = []
    if not isinstance(d, dict):
        return [f"{prefix}: must be a struct/dict, got {type(d).__name__}"]
    for key in ("fence_found", "entry_point_found", "truncated"):
        if not isinstance(d.get(key), bool):
            errors.append(f"{prefix}.{key}: must be bool")
    branch = d.get("parser_branch")
    if branch not in C.PARSER_BRANCHES:
        errors.append(f"{prefix}.parser_branch: {branch!r} not in {C.PARSER_BRANCHES}")
    return errors


def validate_record(d: dict[str, Any]) -> list[str]:
    """Validate a record dict (as loaded from JSONL). Returns error messages."""
    errors: list[str] = []

    # --- structural: presence, type, nullability, enum ---
    for name, (types, nullable, enum) in _FIELD_SPEC.items():
        if name not in d:
            errors.append(f"{name}: missing required field")
            continue
        value = d[name]
        if value is None:
            if not nullable:
                errors.append(f"{name}: must not be null")
            continue
        if not isinstance(value, types):
            errors.append(f"{name}: expected {types}, got {type(value).__name__}")
            continue
        # bool is a subclass of int -> guard numeric fields against booleans
        if types in ((int,), (int, float)) and isinstance(value, bool):
            errors.append(f"{name}: bool is not a valid number")
            continue
        if enum is not None and value not in enum:
            errors.append(f"{name}: {value!r} not in {enum}")

    if "extraction_status" not in d:
        errors.append("extraction_status: missing required field")
    else:
        errors.extend(validate_extraction_status(d["extraction_status"]))

    # --- cross-field invariants (roadmap s7) ---
    task = d.get("task_type")
    subtype = d.get("completion_subtype")
    if task == "completion":
        if subtype is None:
            errors.append("completion_subtype: required when task_type=completion")
    elif subtype is not None:
        errors.append("completion_subtype: must be null unless task_type=completion")

    # gate_decision applies to completion only
    if d.get("gate_decision") is not None and task != "completion":
        errors.append("gate_decision: only valid for task_type=completion")

    # gamma domain and baseline origin
    gamma = d.get("gamma")
    if isinstance(gamma, (int, float)) and not isinstance(gamma, bool):
        if not (0.0 < gamma <= 1.0):
            errors.append(f"gamma: {gamma} out of (0, 1]")
        if gamma == 1.0 and d.get("cot_origin") != "original":
            errors.append("cot_origin: must be 'original' at gamma=1.0")

    # compression_method tied to task (Decision #3): never merge curves across methods
    method = d.get("compression_method")
    if task == "explanation" and method != "post_hoc":
        errors.append("compression_method: explanation must be 'post_hoc'")
    if task == "generation" and method != "model_side":
        errors.append("compression_method: generation must be 'model_side'")

    # cot_token_count non-negative
    ctc = d.get("cot_token_count")
    if isinstance(ctc, int) and not isinstance(ctc, bool) and ctc < 0:
        errors.append("cot_token_count: must be >= 0")

    return errors
