#!/bin/bash
#SBATCH --job-name=imputed_wet_val
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/raghunathan/job/cooling_validation/logs/imputed_wet_validation_%j.out
#SBATCH --error=/home/raghunathan/job/cooling_validation/logs/imputed_wet_validation_%j.err

JOB_DIR="/home/raghunathan/job/cooling_validation"
cd "$JOB_DIR"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export WATXENE_ROOT="/pdrive/projects/watxene"
export COOLING_VALIDATION_JOB_ROOT="$JOB_DIR"
export IMPUTED_WET_VALIDATION_OUTPUT="$JOB_DIR/results/imputed_wet_validation"

VENV_DIR="$JOB_DIR/.venv"
if [ ! -d "$VENV_DIR/bin" ]; then
    uv venv --python 3.13 "$VENV_DIR"
    VIRTUAL_ENV="$VENV_DIR" uv pip install --no-deps wheels/*.whl
    VIRTUAL_ENV="$VENV_DIR" uv pip install numpy pandas xarray pyarrow netcdf4 cftime dask
fi

LATEST_IMPACTS_WHEEL=$(ls -t wheels/impacts-*.whl | head -n1)
VIRTUAL_ENV="$VENV_DIR" uv pip install --no-deps "$LATEST_IMPACTS_WHEEL" --reinstall

"$VENV_DIR/bin/python" run_imputed_wet_validation.py
