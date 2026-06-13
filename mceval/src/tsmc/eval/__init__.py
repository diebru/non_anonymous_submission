"""McEval evaluation driver (roadmap Phases 0 and 4).

Two layers:
  - CPU-only (local): ``results`` builds McEval result files;
    ``mceval_adapter`` imports McEval's pure-regex ``extract()`` so the
    contract<->extractor round-trip can be verified locally.
  - SERVER-ONLY: ``docker`` runs the pinned McEval image (sha256 digest) to
    execute the extracted code across the language toolchains.

We do not fork McEval to relocate its hardcoded /workspace/MMCodeEval/eval/tmp;
evaluation runs inside the pinned container (docs/WORKFLOW.md s3).
"""
from tsmc.eval import docker, join, language_health, mceval_adapter, results

__all__ = ["results", "mceval_adapter", "docker", "join", "language_health"]
