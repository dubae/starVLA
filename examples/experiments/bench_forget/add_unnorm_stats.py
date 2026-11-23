"""Add another benchmark's normalization statistics to a run's dataset_statistics.json.

A stage-2 run (LIBERO, key ``franka``) is evaluated on RoboCasa with the RoboCasa statistics
(key ``new_embodiment``) of the stage-1 run; the eval client then picks them with
``unnorm_key``. Existing keys are never overwritten.

    python -m examples.experiments.bench_forget.add_unnorm_stats \
        playground/Checkpoints/bench_forget_B_full \
        playground/Checkpoints/human50_smolvla_full_50k/dataset_statistics.json new_embodiment
"""

import json
import sys
from pathlib import Path


def main() -> None:
    run_dir, src, key = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    dst = run_dir / "dataset_statistics.json"
    stats = json.loads(dst.read_text())
    src_stats = json.loads(src.read_text())
    if key in stats:
        if stats[key] != src_stats[key]:
            raise ValueError(f"{dst} already has a different `{key}` entry")
        print(f"{dst}: `{key}` already present")
        return
    stats[key] = src_stats[key]
    dst.write_text(json.dumps(stats, indent=2))
    print(f"{dst}: added `{key}` from {src}; keys now {list(stats)}")


if __name__ == "__main__":
    main()
