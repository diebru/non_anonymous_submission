"""vLLM generation wrapper. SERVER-ONLY (needs GPU + the tokenskip_env).

Thin, deterministic-as-possible wrapper around vLLM batched generation: renders
chat prompts with the model's own tokenizer/chat template, runs greedy decoding,
and returns per-request text + finish_reason + token counts + timing (recorded
from Phase 1 onward for the later energy join, roadmap s7/s8).

``vllm`` is imported lazily inside the methods so this module stays importable on
a CPU-only machine (the harness can build prompts / dry-run without a GPU). Never
run actual generation locally (docs/WORKFLOW.md s2).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from tsmc.constants import SEED, TEMPERATURE


@dataclass
class GenOutput:
    """One decoded completion + provenance for the long-format / energy join."""

    text: str
    finish_reason: str | None      # "stop" | "length" (-> truncated -> format_fail)
    n_prompt_tokens: int
    n_output_tokens: int
    arrival_time: float | None     # vLLM request metrics (wall clock, seconds)
    finished_time: float | None


@dataclass
class RunnerConfig:
    model_path: str                # local dir or HF repo id (pin commit via metadata)
    revision: str | None = None    # HF commit/revision to pin (None for a local dir)
    tensor_parallel_size: int = 1  # 3B fits one A6000; raise for 14B
    dtype: str = "bfloat16"
    max_model_len: int | None = None      # None -> model default
    gpu_memory_utilization: float = 0.90
    max_tokens: int = 2048         # output budget; finish_reason="length" if exceeded
    temperature: float = TEMPERATURE
    top_p: float = 1.0             # greedy: top_p is inert at temperature 0
    seed: int = SEED
    # decoding-time repetition controls (vLLM SamplingParams; defaults = OFF, so existing
    # runs are unchanged). Used by the decoding-config matrix (docs/EXPERIMENTS.md) to
    # break the low-gamma runaway loops (token regurgitation to max_tokens).
    frequency_penalty: float = 0.0   # subtract penalty*count(token) from logits (>0 penalizes repeats)
    presence_penalty: float = 0.0    # flat penalty if token already present (>0 penalizes)
    repetition_penalty: float = 1.0  # multiplicative; 1.0 = off, >1 penalizes any repeat
    trust_remote_code: bool = True
    lora_path: str | None = None   # local LoRA adapter dir (None -> base model, Phase 1)
    max_lora_rank: int = 16        # must be >= the adapter rank (our Phase-3 LoRA r=8)


class VLLMRunner:
    """Lazily-constructed vLLM engine. Build prompts on CPU; ``.load()`` on GPU."""

    def __init__(self, cfg: RunnerConfig):
        self.cfg = cfg
        self._llm: Any = None
        self._tokenizer: Any = None
        self._lora_request: Any = None  # set in .load() iff cfg.lora_path

    # -- engine lifecycle (GPU) --
    def load(self) -> "VLLMRunner":
        from vllm import LLM  # deferred: GPU-only import

        kwargs: dict[str, Any] = dict(
            model=self.cfg.model_path,
            revision=self.cfg.revision,
            tensor_parallel_size=self.cfg.tensor_parallel_size,
            dtype=self.cfg.dtype,
            max_model_len=self.cfg.max_model_len,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            seed=self.cfg.seed,
            trust_remote_code=self.cfg.trust_remote_code,
        )
        if self.cfg.lora_path:  # Phase-4: serve the base model + our SFT LoRA adapter
            kwargs["enable_lora"] = True
            kwargs["max_lora_rank"] = self.cfg.max_lora_rank
        self._llm = LLM(**kwargs)
        self._tokenizer = self._llm.get_tokenizer()
        if self.cfg.lora_path:
            from vllm.lora.request import LoRARequest  # deferred

            self._lora_request = LoRARequest("tsmc_sft", 1, self.cfg.lora_path)
        return self

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise RuntimeError("Runner not loaded; call .load() first (server/GPU).")
        return self._tokenizer

    # -- prompt rendering / token counting --
    def render(self, messages: list[dict[str, str]]) -> str:
        """Apply the model chat template, leaving the assistant turn open."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def count_tokens(self, text: str) -> int:
        """Token count of free text (no special tokens) -- the cot_token_count x-axis."""
        if not text:
            return 0
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    # -- generation (GPU) --
    def generate(self, prompts: list[str]) -> list[GenOutput]:
        """Greedy batched generation. Returns one GenOutput per prompt, in order."""
        from vllm import SamplingParams  # deferred

        if self._llm is None:
            raise RuntimeError("Runner not loaded; call .load() first (server/GPU).")
        sp = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_tokens,
            seed=self.cfg.seed,
            frequency_penalty=self.cfg.frequency_penalty,
            presence_penalty=self.cfg.presence_penalty,
            repetition_penalty=self.cfg.repetition_penalty,
        )
        gen_kwargs: dict[str, Any] = {}
        if self._lora_request is not None:  # route generation through the LoRA adapter
            gen_kwargs["lora_request"] = self._lora_request
        wall_start = time.time()
        request_outputs = self._llm.generate(prompts, sp, **gen_kwargs)
        wall_end = time.time()

        results: list[GenOutput] = []
        for ro in request_outputs:
            comp = ro.outputs[0]
            metrics = getattr(ro, "metrics", None)
            results.append(GenOutput(
                text=comp.text,
                finish_reason=getattr(comp, "finish_reason", None),
                n_prompt_tokens=len(getattr(ro, "prompt_token_ids", []) or []),
                n_output_tokens=len(getattr(comp, "token_ids", []) or []),
                arrival_time=getattr(metrics, "arrival_time", wall_start) if metrics else wall_start,
                finished_time=getattr(metrics, "finished_time", wall_end) if metrics else wall_end,
            ))
        return results
