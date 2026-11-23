#!/usr/bin/env bash
# RoboCasa (v0.x) Human-50 — download the `human_im` HDF5 demos of the 24 atomic tasks
# (~13 GB) with the upstream RoboCasa v0.2 downloader. Run from the StarVLA repo root
# inside the `robocasa` (v0.2) simulation env.
#
# Files land under robocasa's DATASET_BASE_PATH; this script points it at RAW_ROOT
# (default: playground/Datasets/robocasa_human50_raw) through robocasa/macros_private.py:
#   ${RAW_ROOT}/v0.1/single_stage/<category>/<Task>/<date>/demo_gentex_im128_randcams.hdf5
set -euo pipefail

RAW_ROOT=$(realpath -m "${RAW_ROOT:-playground/Datasets/robocasa_human50_raw}")
mkdir -p "${RAW_ROOT}"

# (robocasa prints import warnings on stdout; keep only the last line)
ROBOCASA_DIR=$(python -c "import robocasa, os; print(os.path.dirname(robocasa.__file__))" 2>/dev/null | tail -n 1)
MACROS_PRIVATE="${ROBOCASA_DIR}/macros_private.py"
[[ -f "${MACROS_PRIVATE}" ]] || python "${ROBOCASA_DIR}/scripts/setup_macros.py"
if grep -q "^DATASET_BASE_PATH" "${MACROS_PRIVATE}"; then
  sed -i "s|^DATASET_BASE_PATH.*|DATASET_BASE_PATH = \"${RAW_ROOT}\"|" "${MACROS_PRIVATE}"
else
  echo "DATASET_BASE_PATH = \"${RAW_ROOT}\"" >> "${MACROS_PRIVATE}"
fi
echo "[human50] DATASET_BASE_PATH=${RAW_ROOT} (${MACROS_PRIVATE})"

TASKS=$(python -c "from examples.simBenchmarks.Robocasa_human50.human50_tasks import HUMAN50_TASKS; print(' '.join(HUMAN50_TASKS))")
# The upstream script asks for y/n confirmation once.
echo y | python "${ROBOCASA_DIR}/scripts/download_datasets.py" --tasks ${TASKS} --ds_types human_im

echo "[human50] downloaded files:"
find "${RAW_ROOT}" -name "demo_gentex_im128_randcams.hdf5" | wc -l
