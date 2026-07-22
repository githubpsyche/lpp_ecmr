#!/bin/bash
#SBATCH --job-name=lpp-stagnation
#SBATCH --account=TALMI-SL3-CPU
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=20G

set -euo pipefail

RUN_DIR="${1:-${LPP_RESTART_RUN_DIR:-}}"
ENV_SCRIPT="$HOME/workspace/cluster_env.sh"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is required." >&2
    exit 1
fi
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
    echo "Error: pass the staged run directory as the first argument." >&2
    exit 1
fi
if [ ! -f "$ENV_SCRIPT" ]; then
    echo "Error: missing environment script: $ENV_SCRIPT" >&2
    exit 1
fi

cd "$RUN_DIR"
RUN_DIR="$PWD"

LPP_SOURCE="$RUN_DIR/source/lpp_ecmr"
JAXCMR_SOURCE="$RUN_DIR/source/jaxcmr"
DATA_PATH="$RUN_DIR/data/TalmiEEG.h5"
OUTPUT_DIR="$RUN_DIR/restarts"
if [ ! -f "$LPP_SOURCE/lpp_ecmr/model_comparison_restarts.py" ]; then
    echo "Error: missing staged lpp_ecmr source: $LPP_SOURCE" >&2
    exit 1
fi
if [ ! -f "$JAXCMR_SOURCE/jaxcmr/fitting/evosax.py" ]; then
    echo "Error: missing staged jaxcmr source: $JAXCMR_SOURCE" >&2
    exit 1
fi
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: missing staged dataset: $DATA_PATH" >&2
    exit 1
fi

source "$ENV_SCRIPT"

export PYTHONPATH="$LPP_SOURCE:$JAXCMR_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export JAX_ENABLE_COMPILATION_CACHE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export JAX_PLATFORM_NAME=cpu
export UV_NO_PROJECT=1

mkdir -p "$OUTPUT_DIR"

echo "$(date --iso-8601=seconds): starting stagnation task $SLURM_ARRAY_TASK_ID"
python -u -m lpp_ecmr.model_comparison_restarts fit \
    --campaign stagnation \
    --task-index "$SLURM_ARRAY_TASK_ID" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR"
echo "$(date --iso-8601=seconds): finished stagnation task $SLURM_ARRAY_TASK_ID"
