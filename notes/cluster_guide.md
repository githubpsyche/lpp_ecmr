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

The orchestrator notebooks read from the registries, generate one parameterized fitting notebook per enabled model, and write them to `analyses/rendered/`. They do not execute the fits (`prepare_only=True`).

```bash
cd /path/to/lpp_ecmr
papermill analyses/render_model_fitting_single_context.ipynb analyses/render_model_fitting_single_context.ipynb --progress-bar
papermill analyses/render_model_fitting_full_ecmr.ipynb analyses/render_model_fitting_full_ecmr.ipynb --progress-bar
papermill analyses/render_model_fitting_strength.ipynb analyses/render_model_fitting_strength.ipynb --progress-bar
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

Before batch submission, verify a notebook runs manually, then via Slurm:

```bash
source ~/workspace/cluster_env.sh
cd ~/workspace/lpp_ecmr

# Manual — pick a small notebook (Strength variants are fastest)
papermill analyses/rendered/fitting_TalmiEEG_Strength_50_set_likelihood_fixed_term_best_of_3.ipynb \
  analyses/rendered/fitting_TalmiEEG_Strength_50_set_likelihood_fixed_term_best_of_3.ipynb \
  --progress-bar
```

Then a single Slurm job:

```bash
cd ~/workspace/sbatch
sbatch \
  --output ~/workspace/lpp_ecmr/runs/smoke_%j.out \
  --error ~/workspace/lpp_ecmr/runs/smoke_%j.err \
  run_notebook.sbatch \
  ~/workspace/lpp_ecmr/analyses/rendered/fitting_TalmiEEG_Strength_50_set_likelihood_fixed_term_best_of_3.ipynb
```

Check with `squeue -u "$USER"`, then inspect `~/workspace/lpp_ecmr/runs/smoke_<jobid>.out` and `.err`. Once this works, the full chain is verified. Skip this step on subsequent runs if nothing has changed in the environment.

### 4. Submit all fitting notebooks

```bash
cd ~/workspace/sbatch
./submit_notebooks.sh \
  ~/workspace/lpp_ecmr/analyses/rendered \
  "fitting_*.ipynb"
```

This submits one Slurm array job with one task per notebook. Each task gets 1 CPU, 4GB memory, and a 12-hour walltime. The default throttle is 100 concurrent tasks.

Runs are stored under the project at `~/workspace/lpp_ecmr/runs/`. To check progress:

```bash
~/workspace/sbatch/check_run.sh ~/workspace/lpp_ecmr/runs/<run_id>
```

This shows the status and full notebook path for each task — paths are copy-pasteable.

Other useful commands:

```bash
ls -t ~/workspace/lpp_ecmr/runs/ | head -1              # newest run directory
squeue -u "$USER"                                        # running jobs
```

Logs land in `runs/<run_id>/logs/`.

### 5. After fits complete

Each fitting notebook writes outputs to the lpp_ecmr project root:

- `fits/*.json` -- fitted parameters per subject
- `simulations/*.h5` -- simulated recall data
- `figures/fitting/*.png` -- diagnostic plots

Pull results back locally:

```bash
rsync -av <cluster>:~/workspace/lpp_ecmr/fits/ fits/
rsync -av <cluster>:~/workspace/lpp_ecmr/simulations/ simulations/
rsync -av <cluster>:~/workspace/lpp_ecmr/figures/fitting/ figures/fitting/
```

Then run group-level and comparison analyses locally:

```bash
papermill analyses/render_model_fitting_group_level.ipynb analyses/render_model_fitting_group_level.ipynb --progress-bar
papermill analyses/render_model_comparison_exclude_termination.ipynb analyses/render_model_comparison_exclude_termination.ipynb --progress-bar
papermill analyses/render_model_comparison_include_termination.ipynb analyses/render_model_comparison_include_termination.ipynb --progress-bar
```

---

## Walltime

The default walltime is 12 hours per task (`sbatch/run_notebook.sbatch`), which is the SL3 maximum. Strength models finish in minutes; CMR variants in under an hour; eCMR variants may take 1-3 hours depending on parameter count.

If a model exceeds the walltime, split into per-subject notebooks using the `subject_indices` parameter in the fitting template (see `repfr/scripts/generate_render_notebooks.py` for the pattern, and `repfr/scripts/merge_partials.py` for recombining the partial results).

## Email notifications

To get emailed when jobs finish or fail, set `SBATCH_MAIL_USER` in your `~/.bashrc`:

```bash
export SBATCH_MAIL_USER="your_email@example.com"
```

`submit_notebooks.sh` picks this up automatically. If unset, no emails are sent.

You get one email when the entire batch finishes, plus immediate emails for any individual task failures.

## Re-running after registry changes

If you add, remove, or rename models in the registries, re-run from step 1 of the recurring workflow. The orchestrator notebooks re-read the registries each time and regenerate all rendered notebooks.
