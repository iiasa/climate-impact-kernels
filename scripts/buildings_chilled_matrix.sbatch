#!/bin/bash
# Production CHILLED matrix runner.
#
# Invocation:
#   sbatch buildings_chilled_matrix.sbatch
#     submits the full 400-task array (5 GCM × 4 scenario × 20 archetype combo)
#     at 32 GiB / task with 90-minute wall clock.
#   sbatch --array=0 buildings_chilled_matrix.sbatch
#     smoke-verify a single task (GFDL × ssp126 × sfh × rur × s1 at index 0).
#   sbatch --array=0,80,160,240,320 buildings_chilled_matrix.sbatch
#     run one combo per GCM (sfh/rur/s1 across all GCMs for GCM-spread check).
#
# Lives at ~/job/buildings_chilled_matrix.sbatch on the cluster.

#SBATCH --job-name=chilled_matrix
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-399
#SBATCH --output=/home/raghunathan/job/out/chilled_matrix_%A_%a.out
#SBATCH --error=/home/raghunathan/job/out/chilled_matrix_%A_%a.err

set -euo pipefail

JOB_DIR="/hdrive/u142/raghunathan/work/job/buildings_chilled"
cd "$JOB_DIR"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

VENV_DIR="$JOB_DIR/.venv"
LOCK_FILE="$JOB_DIR/.venv.lock"
exec 9>"$LOCK_FILE"
flock 9
if [ ! -d "$VENV_DIR/bin" ]; then
    uv venv --python 3.13 "$VENV_DIR"
    VIRTUAL_ENV="$VENV_DIR" uv pip install --no-deps wheels/*.whl
    VIRTUAL_ENV="$VENV_DIR" uv pip install numpy pandas xarray pyarrow netcdf4 cftime dask pyyaml
fi
flock -u 9

export WATXENE_PATH=/pdrive/projects/watxene
export CHILLED_ARCHETYPE_PATH=/pdrive/projects/watxene/IMPACTS/buildings/inputs/ALPS2025/arch_input_reg.csv
export CHILLED_PARAMETRIC_PATH=/pdrive/projects/watxene/IMPACTS/buildings/inputs/ALPS2025/par_var.csv
export MESSAGE_R12_YAML_PATH=$HOME/message-ix-models/message_ix_models/data/node/R12.yaml

export CHILLED_OUTPUT_ROOT="${CHILLED_OUTPUT_ROOT:-/pdrive/projects/watxene/IMPACTS/buildings/seasonal}"

echo "[matrix] SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-unset}"
echo "[matrix] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}"
echo "[matrix] --mem from SLURM=${SLURM_MEM_PER_NODE:-unset} MB"
echo "[matrix] CHILLED_OUTPUT_ROOT=$CHILLED_OUTPUT_ROOT"

"$VENV_DIR/bin/python" -u "$JOB_DIR/run_chilled_matrix.py"
