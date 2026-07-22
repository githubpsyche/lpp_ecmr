#!/bin/bash

set -euo pipefail

LPP_REPO="${LPP_REPO:-$HOME/workspace/lpp_ecmr}"
JAXCMR_REPO="${JAXCMR_REPO:-$HOME/workspace/jaxcmr}"
RUN_ROOT="${LPP_STAGNATION_RUN_ROOT:-$HOME/lpp_ecmr_runs}"
RUN_NAME="lpp-stagnation-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
STAGING_DIR="$RUN_DIR.staging"

for repo in "$LPP_REPO" "$JAXCMR_REPO"; do
    if ! git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "Error: not a Git checkout: $repo" >&2
        exit 1
    fi
    if [ -n "$(git -C "$repo" status --porcelain --untracked-files=normal)" ]; then
        echo "Error: checkout must be clean before staging: $repo" >&2
        exit 1
    fi
done

if [ ! -f "$LPP_REPO/data/TalmiEEG.h5" ]; then
    echo "Error: missing fitting dataset: $LPP_REPO/data/TalmiEEG.h5" >&2
    exit 1
fi
if [ -e "$RUN_DIR" ] || [ -e "$STAGING_DIR" ]; then
    echo "Error: run destination already exists: $RUN_DIR" >&2
    exit 1
fi

mkdir -p "$RUN_ROOT" "$STAGING_DIR/source/lpp_ecmr" \
    "$STAGING_DIR/source/jaxcmr" "$STAGING_DIR/data" \
    "$STAGING_DIR/logs" "$STAGING_DIR/restarts"
cleanup() {
    rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

git -C "$LPP_REPO" archive HEAD lpp_ecmr \
    | tar -xf - -C "$STAGING_DIR/source/lpp_ecmr"
git -C "$JAXCMR_REPO" archive HEAD jaxcmr \
    | tar -xf - -C "$STAGING_DIR/source/jaxcmr"
cp "$LPP_REPO/data/TalmiEEG.h5" "$STAGING_DIR/data/TalmiEEG.h5"
cp "$LPP_REPO/work/pooled_stagnation_fits/run_fit_restart.sbatch" \
    "$STAGING_DIR/run_fit_restart.sbatch"
cp "$LPP_REPO/work/pooled_stagnation_fits/submit_fit_restarts.sh" \
    "$STAGING_DIR/submit_fit_restarts.sh"
chmod u+x "$STAGING_DIR/run_fit_restart.sbatch" \
    "$STAGING_DIR/submit_fit_restarts.sh"

mv "$STAGING_DIR" "$RUN_DIR"
trap - EXIT

echo "Staged lpp_ecmr $(git -C "$LPP_REPO" rev-parse --short HEAD)" >&2
echo "Staged jaxcmr $(git -C "$JAXCMR_REPO" rev-parse --short HEAD)" >&2
echo "Run directory: $RUN_DIR" >&2
printf '%s\n' "$RUN_DIR"
