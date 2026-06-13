"""tsmc - TokenSkip x McEval project package.

Controllable Chain-of-Thought compression (TokenSkip) applied to multilingual
code tasks, scored by McEval's execution-based harness, to characterize the
trade-off between reasoning length, accuracy, and inference energy.

This package is the single home for project code. Subpackages map to the
Phase 0 -> Phase 4 roadmap (docs/PROJECT_ROADMAP.md s8):

    tsmc.config       - path/config resolution (no hardcoded paths)   [Phase 0]
    tsmc.constants    - frozen decision-sheet values                  [Phase 0]
    tsmc.contract     - CoT/code separation contract + parsing        [Phase 0/1]
    tsmc.manifest     - base-problem stratification + split manifest  [Phase 0]
    tsmc.schema       - long-format record schema + validators        [Phase 0]
    tsmc.inference    - vLLM generation harness (server-only)         [Phase 1/4]
    tsmc.compression  - LLMLingua-2 multi-gamma compression (server)  [Phase 2]
    tsmc.sft          - LLaMA-Factory format/registration (server)    [Phase 3]
    tsmc.eval         - McEval Docker evaluation driver (server)      [Phase 0/4]

Import-light by design: heavy GPU stacks live in the server conda envs, so the
CPU-only modules above can be imported and tested locally.
"""

__version__ = "0.0.0"
