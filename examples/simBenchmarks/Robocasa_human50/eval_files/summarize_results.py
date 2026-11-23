"""Aggregate per-task Human-50 results written by ``simulation_env.py``.

    python -m examples.simBenchmarks.Robocasa_human50.eval_files.summarize_results <ckpt>.eval/robocasa_human50

Merges complete ``<Task>.shard<i>of<K>.json`` sets (``run_eval.sh`` with ``EPISODE_SHARDS``)
into ``<Task>.json``, then prints a per-task table, per-category averages and the 24-task average (the number
reported for RoboCasa Human-50), and writes ``summary.json`` next to the task files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.simBenchmarks.Robocasa_human50.human50_tasks import HUMAN50_TASKS

CATEGORY_NAMES = {
    "kitchen_pnp": "Pick-and-place",
    "kitchen_doors": "Doors",
    "kitchen_drawer": "Drawers",
    "kitchen_sink": "Sink",
    "kitchen_stove": "Stove",
    "kitchen_coffee": "Coffee",
    "kitchen_microwave": "Microwave",
}


def merge_shards(result_dir: Path, task: str) -> bool:
    """Write ``<task>.json`` from a complete set of shard files; return whether it did."""
    shard_files = sorted(result_dir.glob(f"{task}.shard*of*.json"))
    if not shard_files:
        return False
    shards = [json.loads(p.read_text()) for p in shard_files]
    num_shards = shards[0]["shard"][1]
    if sorted(s["shard"][0] for s in shards if s["shard"][1] == num_shards) != list(range(num_shards)):
        return False  # incomplete set
    shards = sorted((s for s in shards if s["shard"][1] == num_shards), key=lambda s: s["shard"][0])
    episodes = sorted((e for s in shards for e in s["episodes"]), key=lambda e: e["episode"])
    merged = {k: v for k, v in shards[0].items() if k not in ("shard", "env_seed")}
    merged.update(
        success_rate=sum(e["success"] for e in episodes) / len(episodes),
        num_success=sum(e["success"] for e in episodes),
        num_episodes=len(episodes),
        num_shards=num_shards,
        env_seeds=[s["env_seed"] for s in shards],
        wall_time_s=max(s["wall_time_s"] for s in shards),
        time_breakdown_s={k: round(sum(s["time_breakdown_s"][k] for s in shards), 1) for k in ("reset", "policy", "sim")},
        episodes=episodes,
    )
    (result_dir / f"{task}.json").write_text(json.dumps(merged, indent=2))
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    rows, missing = {}, []
    for task in HUMAN50_TASKS:
        path = args.result_dir / f"{task}.json"
        if path.is_file() or merge_shards(args.result_dir, task):
            rows[task] = json.loads(path.read_text())
        else:
            missing.append(task)

    categories: dict[str, list[float]] = {}
    print(f"{'task':24s} {'SR':>6s} {'succ/n':>8s}")
    for task, res in rows.items():
        cat = CATEGORY_NAMES[HUMAN50_TASKS[task][0].split("/")[2]]
        categories.setdefault(cat, []).append(res["success_rate"])
        print(f"{task:24s} {100 * res['success_rate']:6.1f} {res['num_success']:>4d}/{res['num_episodes']:<3d}")
    print("-" * 40)
    cat_avg = {c: sum(v) / len(v) for c, v in categories.items()}
    for cat, avg in cat_avg.items():
        print(f"{cat:24s} {100 * avg:6.1f}  ({len(categories[cat])} tasks)")
    overall = sum(r["success_rate"] for r in rows.values()) / len(rows) if rows else 0.0
    print(f"{'AVERAGE':24s} {100 * overall:6.1f}  ({len(rows)}/{len(HUMAN50_TASKS)} tasks)")
    if missing:
        print(f"[warning] missing results for {len(missing)} tasks: {missing}")
    timing = {k: sum(r.get("time_breakdown_s", {}).get(k, 0.0) for r in rows.values()) for k in ("reset", "policy", "sim")}
    if any(timing.values()):
        total = sum(timing.values())
        eps = [e for r in rows.values() for e in r["episodes"]]
        q = sum(e.get("num_queries", 0) for e in eps)
        st = sum(e["steps"] for e in eps)
        print(f"time split (summed over clients): reset {100 * timing['reset'] / total:.0f}%, "
              f"policy {100 * timing['policy'] / total:.0f}% ({1000 * timing['policy'] / max(q, 1):.0f} ms/query), "
              f"sim {100 * timing['sim'] / total:.0f}% ({1000 * timing['sim'] / max(st, 1):.1f} ms/step); "
              f"max task wall time {max(r.get('wall_time_s', 0) for r in rows.values()) / 60:.1f} min")

    summary = {
        "average_success_rate": overall,
        "num_tasks": len(rows),
        "complete": not missing,
        "missing_tasks": missing,
        "category_success_rate": cat_avg,
        "task_success_rate": {t: r["success_rate"] for t, r in rows.items()},
        "num_episodes": {t: r["num_episodes"] for t, r in rows.items()},
        "time_breakdown_s": timing,
    }
    with open(args.result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
