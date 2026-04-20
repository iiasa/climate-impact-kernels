#!/bin/bash
# Single-combo CHILLED diagnostic runner for ticket #421.
#
# Invocation:
#   sbatch --mem=32G buildings_chilled_diag.sbatch SCENARIO_TAG
# where SCENARIO_TAG selects from the config matrix below.
#
# Lives at ~/job/buildings_chilled_diag.sbatch on the cluster.

#SBATCH --job-name=chilled_diag
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/raghunathan/job/out/chilled_diag_%j.out
#SBATCH --error=/home/raghunathan/job/out/chilled_diag_%j.err

set -euo pipefail

TAG="${1:-future_sync_32g}"

# Config matrix. Each tag sets CHILLED_* + DASK_* vars.
case "$TAG" in
  future_sync_32g)
    export CHILLED_GCM=GFDL-ESM4
    export CHILLED_SCENARIO=ssp126
    export CHILLED_YEAR_START=2015
    export CHILLED_YEAR_END=2100
    export DASK_SCHEDULER=synchronous
    export DASK_NUM_WORKERS=1
    ;;
  hist_sync_32g)
    export CHILLED_GCM=GFDL-ESM4
    export CHILLED_SCENARIO=historical
    export CHILLED_YEAR_START=1850
    export CHILLED_YEAR_END=2014
    export DASK_SCHEDULER=synchronous
    export DASK_NUM_WORKERS=1
    ;;
  future_threaded1_32g)
    export CHILLED_GCM=GFDL-ESM4
    export CHILLED_SCENARIO=ssp126
    export CHILLED_YEAR_START=2015
    export CHILLED_YEAR_END=2100
    export DASK_SCHEDULER=threads
    export DASK_NUM_WORKERS=1
    ;;
  future_sync_48g)
    export CHILLED_GCM=GFDL-ESM4
    export CHILLED_SCENARIO=ssp126
    export CHILLED_YEAR_START=2015
    export CHILLED_YEAR_END=2100
    export DASK_SCHEDULER=synchronous
    export DASK_NUM_WORKERS=1
    ;;
  hist_yb10)
    # Historical horizon with 10-year year-batching. Verified production shape
    # for hist-horizon memory envelope (job 2678844: 19.9 GB MaxRSS on 165 yr).
    export CHILLED_GCM=GFDL-ESM4
    export CHILLED_SCENARIO=historical
    export CHILLED_YEAR_START=1850
    export CHILLED_YEAR_END=2014
    export DASK_SCHEDULER=synchronous
    export DASK_NUM_WORKERS=1
    export CHILLED_YEAR_BATCH=10
    ;;
  future_yb10)
    # Future horizon with 10-year year-batching. Verified production shape
    # for 86 yr ssp* horizon (job 2678836: 33.3 GB MaxRSS).
    export CHILLED_GCM=GFDL-ESM4
    export CHILLED_SCENARIO=ssp126
    export CHILLED_YEAR_START=2015
    export CHILLED_YEAR_END=2100
    export DASK_SCHEDULER=synchronous
    export DASK_NUM_WORKERS=1
    export CHILLED_YEAR_BATCH=10
    ;;
  future_sync_32g_ec1000)
    # Hypothesis: per-output-chunk working set = time × entity × N_live_intermediates × 4B.
    # At entity_chunk=10000 × 31411 time × ~10 dask-backed intermediates ≈ 15 GB per chunk.
    # Drop entity_chunk to 1000 → ~1.5 GB per chunk → fits 32G comfortably.
    export CHILLED_GCM=GFDL-ESM4
    export CHILLED_SCENARIO=ssp126
    export CHILLED_YEAR_START=2015
    export CHILLED_YEAR_END=2100
    export DASK_SCHEDULER=synchronous
    export DASK_NUM_WORKERS=1
    export CHILLED_KERNEL_ENTITY_CHUNK=1000
    export CHILLED_COALESCE_CHUNKS=1
    ;;
  verify_threads_sync)
    # Same workload as the failed future_sync_32g (2678814) but with thread
    # instrumentation + COALESCE_CHUNKS=0 so the dask graph stays the full
    # 72k-chunk monster. Purpose: isolate whether scheduler=synchronous
    # actually prevents thread proliferation on a 256-thread compute node.
    export CHILLED_GCM=GFDL-ESM4
    export CHILLED_SCENARIO=ssp126
    export CHILLED_YEAR_START=2015
    export CHILLED_YEAR_END=2100
    export DASK_SCHEDULER=synchronous
    export DASK_NUM_WORKERS=1
    export CHILLED_COALESCE_CHUNKS=0
    ;;
  *)
    echo "unknown TAG: $TAG" >&2
    exit 64
    ;;
esac

# Single archetype combo (sfh_rur_s1) so a failure here clearly localises
# the production-horizon OOM path.
export CHILLED_BUILDING_TYPE=sfh
export CHILLED_URT=rur
export CHILLED_VINTAGE=s1

export CHILLED_TIME_CHUNK="${CHILLED_TIME_CHUNK:-90}"
export CHILLED_FORCE=1
export CHILLED_OUTPUT_ROOT=/pdrive/projects/watxene/IMPACTS/buildings/seasonal_diag

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

echo "[diag] TAG=$TAG"
echo "[diag] --mem from SLURM=${SLURM_MEM_PER_NODE:-unset} MB"
echo "[diag] SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"

"$VENV_DIR/bin/python" -u "$JOB_DIR/run_chilled_diag.py"
