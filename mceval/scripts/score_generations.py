#!/usr/bin/env python3
"""Score Phase-1 generations and emit final long-format records. SERVER (Docker).

For each requested (task, split) of a model run, this:
  1. runs the per-problem McEval detail eval in the pinned container on
     ``.../gammaG/result/`` (our committed shim; McEval itself untouched);
  2. joins the {task_id: pass} verdicts onto ``.../gammaG/trajectories/``,
     stamping the three-way ``outcome`` (format_fail / exec_fail / pass);
  3. writes ``.../gammaG/records/<Lang>.jsonl`` (final long-format) and prints +
     saves an outcome/accuracy summary (format_fail reported separately so a
     contract artifact never hides inside the accuracy number).

Run AFTER scripts/run_inference.py (which produced result/ + trajectories/).

Usage (server, after `git pull`):
    python3 scripts/score_generations.py --task generation --split both \
        --model qwen2.5-coder-3b-instruct --digest sha256:<...>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import GAMMA_BASELINE, MODEL_IDS  # noqa: E402
from tsmc.eval import docker  # noqa: E402
from tsmc.eval.join import join_language, summarize  # noqa: E402

TASKS = ("generation", "explanation", "completion")


def _gamma_dir(paths, model, run_id, task, split, gamma):
    return paths.generations_dir / model / run_id / task / split / f"gamma{gamma:g}"


def score_one(task, split, args, paths, cfg) -> dict | None:
    gdir = _gamma_dir(paths, args.model, args.run_id, task, split, args.gamma)
    result_dir = gdir / "result"
    traj_dir = gdir / "trajectories"
    if not result_dir.is_dir() or not any(result_dir.glob("*.jsonl")):
        print(f"[skip] {task}/{split}: no result/ at {gdir} (run inference first)")
        return None

    save_dir = paths.eval_dumps_dir / args.model / args.run_id / task / split / f"gamma{args.gamma:g}"
    print(f"[eval] {task}/{split}: detail eval on {result_dir} ...")
    proc = docker.run_detail_eval(cfg, result_dir, save_dir, check=False)
    if proc.returncode != 0:
        print(proc.stdout[-2000:]); print(proc.stderr[-2000:])
        print(f"[FAIL] {task}/{split}: docker detail eval returned {proc.returncode}")
        return None
    verdicts = docker.parse_detail(docker.detail_file_for(result_dir, save_dir))

    records_dir = gdir / "records"
    all_records: list[dict] = []
    for traj_file in sorted(traj_dir.glob("*.jsonl")):
        rows = [json.loads(x) for x in traj_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        finalized = join_language(rows, verdicts)
        records_dir.mkdir(parents=True, exist_ok=True)
        with open(records_dir / traj_file.name, "w", encoding="utf-8") as fh:
            for rec in finalized:
                fh.write(json.dumps(rec) + "\n")
        all_records.extend(finalized)

    summ = summarize(all_records)
    summ.update({"task": task, "split": split, "model_id": args.model,
                 "gamma": args.gamma, "records_dir": str(records_dir),
                 "n_verdicts": len(verdicts)})
    with open(gdir / "score_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=2)
    def _r(x):
        return x if x is None else round(x, 4)
    print(f"[done] {task}/{split}: healthy_acc={_r(summ['healthy_accuracy'])} "
          f"(healthy_scored={summ['healthy_scored']}) | all_acc={_r(summ['accuracy'])} "
          f"format_fail_rate={_r(summ['format_fail_rate'])} "
          f"counts={summ['counts']} (scored={summ['scored']}/{summ['n_records']})")
    return summ


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=(*TASKS, "all"), default="generation")
    ap.add_argument("--split", choices=("train", "test", "both"), default="both")
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--gamma", type=float, default=GAMMA_BASELINE)
    ap.add_argument("--run-id", default="run01")
    ap.add_argument("--digest", default=None, help="McEval image sha256 digest")
    ap.add_argument("--network", default=None, help="docker --network (e.g. none)")
    ap.add_argument("--python", default="/opt/conda/bin/python", help="in-container interpreter")
    args = ap.parse_args()

    paths = get_paths()
    digest = args.digest or docker.load_digest_from_metadata(paths)
    if not digest:
        print("No McEval Docker digest. Set mceval.docker_digest in "
              "configs/run_metadata.yaml or pass --digest sha256:...")
        return 1
    cfg = docker.DockerEvalConfig(digest=digest, network=args.network, python_exe=args.python)

    tasks = list(TASKS) if args.task == "all" else [args.task]
    splits = ["train", "test"] if args.split == "both" else [args.split]
    print("=" * 64)
    print(f"Phase-1 scoring | model={args.model} gamma={args.gamma:g} run={args.run_id}")
    print(f"image={cfg.image_ref()}")
    print("=" * 64)

    summaries = []
    for task in tasks:
        for split in splits:
            s = score_one(task, split, args, paths, cfg)
            if s:
                summaries.append(s)

    # behavioral preview: train vs test HEALTHY accuracy per task (formal gate = Task 1.3)
    print("\n" + "=" * 64)
    print("Behavioral preview (healthy-language accuracy; formal ±3% gate = Task 1.3)")
    by_task: dict[str, dict[str, float]] = {}
    for s in summaries:
        if s["healthy_accuracy"] is not None:
            by_task.setdefault(s["task"], {})[s["split"]] = s["healthy_accuracy"]
    for task, d in by_task.items():
        tr, te = d.get("train"), d.get("test")
        if tr is not None and te is not None:
            print(f"  {task}: train={tr:.4f} test={te:.4f} |Δ|={abs(tr - te):.4f} "
                  f"({'within' if abs(tr - te) <= 0.03 else 'OUTSIDE'} ±3%)")
        else:
            print(f"  {task}: {d}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
