#!/bin/bash

set -euo pipefail

THROTTLE="${1:-16}"
RUN_DIR="${2:-}"
if ! [[ "$THROTTLE" =~ ^[1-9][0-9]*$ ]] || [ "$THROTTLE" -gt 48 ]; then
    echo "Usage: $0 [throttle from 1 to 48] STAGED_RUN_DIR" >&2
    exit 1
fi
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
    echo "Error: pass one existing, staged run directory." >&2
    exit 1
fi

RUN_DIR="$(cd "$RUN_DIR" && pwd)"
LOG_DIR="$RUN_DIR/logs"
OUTPUT_DIR="$RUN_DIR/restarts"
DATA_PATH="$RUN_DIR/data/TalmiEEG.h5"
LPP_SOURCE="$RUN_DIR/source/lpp_ecmr"
JAXCMR_SOURCE="$RUN_DIR/source/jaxcmr"
WORKER="$RUN_DIR/run_fit_restart.sbatch"

source "$HOME/workspace/cluster_env.sh"
export PYTHONPATH="$LPP_SOURCE:$JAXCMR_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export JAX_ENABLE_COMPILATION_CACHE=false

if [ ! -f "$WORKER" ]; then
    echo "Error: missing staged worker: $WORKER" >&2
    exit 1
fi
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
python -m lpp_ecmr.model_comparison_restarts preflight \
    --campaign stagnation \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR"

MAIL_ARGS=()
if [ -n "${SBATCH_MAIL_USER:-}" ]; then
    MAIL_ARGS=(--mail-type=FAIL --mail-user="$SBATCH_MAIL_USER")
fi

JOB_ID="$(
    sbatch --parsable \
        --array="0-47%$THROTTLE" \
        --output="$LOG_DIR/%x_%A_%a.out" \
        --error="$LOG_DIR/%x_%A_%a.err" \
        "${MAIL_ARGS[@]}" \
        "$WORKER" \
        "$RUN_DIR"
)"
printf '%s\n' "$JOB_ID" > "$RUN_DIR/submission.txt"

printf 'RUN_DIR=%s\nJOB_ID=%s\n' "$RUN_DIR" "$JOB_ID"
printf 'Check queue: squeue -j %s\n' "$JOB_ID"
printf 'Count results: find %q -maxdepth 1 -name '\''*_restart_*.json'\'' | wc -l\n' \
    "$OUTPUT_DIR"
