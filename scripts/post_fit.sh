#!/bin/bash
# Post-fit pipeline: merge per-subject partial fits, clean up intermediates,
# then submit per-model simulation notebooks.
# Intended to run as a sentinel job after all per-subject fitting jobs complete.
#
# Usage: post_fit.sh <project_dir>

set -euo pipefail

PROJECT_DIR="${1:?Usage: post_fit.sh <project_dir>}"
cd "$PROJECT_DIR"

source "$HOME/workspace/cluster_env.sh"
export UV_NO_PROJECT=1

echo "$(date): Merging partial fits in $PROJECT_DIR"
python scripts/merge_partials.py

echo "$(date): Cleaning up per-subject notebooks"
rm -f "$PROJECT_DIR"/analyses/rendered/fitting_*_sub*.ipynb

echo "$(date): Submitting per-model simulation notebooks"
"$HOME/workspace/sbatch/submit_notebooks.sh" \
    "$PROJECT_DIR/analyses/rendered" \
    "fitting_*.ipynb"

echo "$(date): Post-fit pipeline complete"
