#!/bin/bash
#SBATCH --job-name=cooling_country
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-39
#SBATCH --output=/home/raghunathan/job/cooling_country_monthly/logs/%A_%a.out
#SBATCH --error=/home/raghunathan/job/cooling_country_monthly/logs/%A_%a.err

set -euo pipefail

JOB_DIR="/home/raghunathan/job/cooling_country_monthly"
SOURCE_ROOT="/pdrive/projects/watxene/IMPACTS/thermo-electric/production_monthly_imputed"
OUTPUT_ROOT="/pdrive/projects/watxene/IMPACTS/thermo-electric/production_monthly_country"
MANIFEST_PATH="$JOB_DIR/source_files.txt"

cd "$JOB_DIR"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

mkdir -p "$JOB_DIR/logs"

if [ ! -f "$MANIFEST_PATH" ]; then
    find "$SOURCE_ROOT/wet" "$SOURCE_ROOT/dry" -maxdepth 1 -type f -name '*.nc' | sort > "$MANIFEST_PATH"
fi

SOURCE_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST_PATH")
if [ -z "${SOURCE_PATH:-}" ]; then
    echo "No source file for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

VENV_DIR="$JOB_DIR/.venv"
LOCK_FILE="$JOB_DIR/.venv.lock"
exec 9>"$LOCK_FILE"
flock 9
if [ ! -d "$VENV_DIR/bin" ]; then
    uv venv --python 3.13 "$VENV_DIR"
    VIRTUAL_ENV="$VENV_DIR" uv pip install --no-deps wheels/*.whl
    VIRTUAL_ENV="$VENV_DIR" uv pip install numpy pandas xarray pyarrow netcdf4 cftime dask
fi
LATEST_IMPACTS_WHEEL=$(ls -t wheels/impacts-*.whl | head -n1)
VIRTUAL_ENV="$VENV_DIR" uv pip install --no-deps "$LATEST_IMPACTS_WHEEL" --reinstall
flock -u 9

"$VENV_DIR/bin/python" run_monthly_country_aggregation.py \
    --source-path "$SOURCE_PATH" \
    --output-root "$OUTPUT_ROOT" \
    --region-set country
