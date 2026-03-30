# Running lpp_ecmr fits on the CSD3 cluster

## First-time setup

### 1. SSH to the cluster

```bash
ssh <your_username>@<cluster_host>
```

### 2. Create a workspace

```bash
mkdir -p "$HOME/workspace"
cd "$HOME/workspace"
```

### 3. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Add to `.bashrc` so future sessions see it:

```bash
echo 'source "$HOME/.local/bin/env"' >> "$HOME/.bashrc"
source "$HOME/.bashrc"
```

### 4. Clone repos

```bash
cd "$HOME/workspace"
git clone <jaxcmr_repo_url>
git clone <lpp_ecmr_repo_url>
git clone <sbatch_repo_url>
```

### 5. Check your CSD3 Slurm account

```bash
mybalance
```

The relevant account is `TALMI-SL3-CPU`. The sbatch scripts use this account and the `icelake-himem` partition by default. SL3 CPU jobs have a 12-hour maximum walltime.

### 6. Create a shared virtual environment

The venv lives outside the repos so it can serve multiple projects.

```bash
uv venv "$HOME/workspace/.venv" --python 3.12
source "$HOME/workspace/.venv/bin/activate"
```

### 7. Install packages

```bash
cd "$HOME/workspace"
uv pip install -e "jaxcmr[dev]"
uv pip install jupyter nbclient pandas papermill
uv pip install -e lpp_ecmr
```

Run these from `~/workspace/` (not from inside a project directory). If uv detects a `pyproject.toml` in the current directory it may try to create a project-local `.venv` instead of using the shared one.

### 8. Create an environment activation script

This script is sourced by Slurm jobs and by manual sessions.

```bash
cat > "$HOME/workspace/cluster_env.sh" <<'EOF'
source "$HOME/.local/bin/env"
source "$HOME/workspace/.venv/bin/activate"
EOF

chmod +x "$HOME/workspace/cluster_env.sh"
```

## First verification

Verify the environment works by checking that both packages import:

```bash
source "$HOME/workspace/cluster_env.sh"
python -c "import jaxcmr; print(jaxcmr.__file__)"
python -c "import lpp_ecmr; print(lpp_ecmr.__file__)"
```

The manual notebook execution and Slurm smoke tests come later, after rendered notebooks have been generated locally and transferred to the cluster (see "Recurring workflow" below).

---

## Recurring workflow

The steps below are what you repeat each time you want to fit models on the cluster.

### 1. Generate rendered notebooks locally

The orchestrator notebooks read from the registries and generate per-subject fitting notebooks plus per-model simulation notebooks. With `prepare_only=True` (the default), they write to `analyses/rendered/` without executing. Set `prepare_only=False` to execute fits locally and sequentially instead.

```bash
cd ~/workspace/lpp_ecmr
papermill analyses/render_model_fitting_single_context.ipynb analyses/render_model_fitting_single_context.ipynb --progress-bar
papermill analyses/render_model_fitting_full_ecmr.ipynb analyses/render_model_fitting_full_ecmr.ipynb --progress-bar
papermill analyses/render_model_fitting_group_level.ipynb analyses/render_model_fitting_group_level.ipynb --progress-bar
```

Check the output:

```bash
ls analyses/rendered/fitting_*.ipynb | wc -l
```

### 2. Get code and notebooks onto the cluster

Either commit and push:

```bash
git add analyses/rendered/
git commit -m "Prepare fitting notebooks for cluster"
git push

# On cluster:
cd ~/workspace/lpp_ecmr && git pull
cd ~/workspace/jaxcmr && git pull
```

Or rsync the rendered notebooks directly:

```bash
rsync -av analyses/rendered/ <cluster>:~/workspace/lpp_ecmr/analyses/rendered/
```

If code in `jaxcmr` or `lpp_ecmr` has changed, pull both repos on the cluster. The editable installs pick up changes automatically.

### 3. Smoke test (first time or after environment changes)

Before batch submission, verify a single per-subject notebook runs via Slurm:

```bash
cd ~/workspace/sbatch
sbatch \
  --output ~/workspace/lpp_ecmr/runs/smoke_%j.out \
  --error ~/workspace/lpp_ecmr/runs/smoke_%j.err \
  run_notebook.sbatch \
  ~/workspace/lpp_ecmr/analyses/rendered/fitting_TalmiEEG_Strength_50_set_likelihood_fixed_term_best_of_3_sub0.ipynb
```

Check with `squeue -u "$USER"`, then inspect `~/workspace/lpp_ecmr/runs/smoke_<jobid>.out` and `.err`. A single-subject Strength fit should finish in under a minute. Skip this step on subsequent runs if nothing has changed in the environment.

### 4. Submit fitting notebooks

```bash
cd ~/workspace/sbatch
./submit_notebooks.sh \
  --sentinel ~/workspace/lpp_ecmr/scripts/post_fit.sh \
  ~/workspace/lpp_ecmr/analyses/rendered \
  "fitting_*.ipynb"
```

This submits one Slurm array job with one task per per-subject notebook. Each task gets 1 CPU, 4GB memory, and a 12-hour walltime. The default throttle is 100 concurrent tasks.

The `--sentinel` flag triggers an automatic post-fit pipeline after all fitting jobs succeed: it merges partial fits (`scripts/merge_partials.py`) and submits the simulation notebooks. Omit `--sentinel` to handle post-fit steps manually.

Runs are stored under the project at `~/workspace/lpp_ecmr/runs/`.

### 5. Monitoring

From inside the lpp_ecmr project directory:

```bash
cd ~/workspace/lpp_ecmr
~/workspace/sbatch/check_run.sh
```

This finds the newest run and shows the Slurm job ID and task counts by state (completed/running/pending/failed). Use `-v` for per-task detail with copy-pasteable notebook paths.

To check a specific run:

```bash
~/workspace/sbatch/check_run.sh ~/workspace/lpp_ecmr/runs/<run_id>
```

Check if the post-fit sentinel is queued:

```bash
squeue -u "$USER" | grep post-fit
```

Logs land in `runs/<run_id>/logs/`.

### 6. After everything completes

If you used `SBATCH_SENTINEL`, the merge and simulation steps run automatically. You'll get an email when the full pipeline finishes (fitting → merge → simulation).

To run manually instead (or if the sentinel wasn't set):

```bash
cd ~/workspace/lpp_ecmr
python scripts/merge_partials.py
cd ~/workspace/sbatch
./submit_notebooks.sh ~/workspace/lpp_ecmr/analyses/rendered "simulation_*.ipynb"
```

### 7. Pull results and run comparisons locally

```bash
rsync -av <cluster>:~/workspace/lpp_ecmr/fits/ fits/
rsync -av <cluster>:~/workspace/lpp_ecmr/simulations/ simulations/
rsync -av <cluster>:~/workspace/lpp_ecmr/figures/fitting/ figures/fitting/
```

Then run group-level and comparison analyses:

```bash
papermill analyses/render_model_fitting_group_level.ipynb analyses/render_model_fitting_group_level.ipynb --progress-bar
papermill analyses/render_model_comparison_exclude_termination.ipynb analyses/render_model_comparison_exclude_termination.ipynb --progress-bar
papermill analyses/render_model_comparison_include_termination.ipynb analyses/render_model_comparison_include_termination.ipynb --progress-bar
```

---

## Walltime

The default walltime is 12 hours per task (`sbatch/run_notebook.sbatch`), which is the SL3 maximum. Per-subject fits are fast — typically seconds to minutes per subject, well within the limit.

## Email notifications

To get emailed when jobs finish or fail, set `SBATCH_MAIL_USER` in your `~/.bashrc`:

```bash
export SBATCH_MAIL_USER="your_email@example.com"
```

`submit_notebooks.sh` picks this up automatically. If unset, no emails are sent.

You get one email when the entire batch finishes, plus immediate emails for any individual task failures.

## Re-running after registry changes

If you add, remove, or rename models in the registries, re-run from step 1 of the recurring workflow. The orchestrator notebooks re-read the registries each time and regenerate all rendered notebooks.
